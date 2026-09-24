# -*- coding: utf-8 -*-
"""常驻 session 池自检：LRU 记账 + 后端接线（不碰 GPU，不建真 session）。

为什么单独一份：池子的错法全是**安静的** ——
  * 额度 1 时少换出一份：显存按成员数慢慢长，跑到第 N 个成员才 OOM；
  * LRU 记反（换出刚用过的那份）：命中率掉到 0，只是变慢，日志上看不出错；
  * 键用成员号而不是路径：m3 / latent 的同一个 session 被存成 51 份。
所以每条都要能独立说清「错在哪儿」。前三节只用 SessionPool 本尊（不 import
onnxruntime），第四、五节拿 OnnxInferModel + 桩 session 验「池子接到了哪一步」。

    python -m xmetai_inference.backends.selftest
"""
from __future__ import annotations

import logging
import os

from .session_pool import SessionPool, normalize_budget


_TOTAL_SECTIONS = 5


class _Checker:
    """收集断言结果，跑完所有条目再给总结（与 scheduling/selftest.py 同一风格）。"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.passed = 0
        self.failures = []

    def check(self, ok, label, detail=""):
        if ok:
            self.passed += 1
            if self.verbose:
                print(f"    [通过] {label}")
        else:
            self.failures.append(f"{label}：{detail}" if detail else label)
            if self.verbose:
                print(f"    **[不通过]** {label}" + (f" —— {detail}" if detail else ""))
        return bool(ok)

    def __call__(self, ok, label, detail=""):
        return self.check(ok, label, detail)

    def section(self, index, total, title):
        if self.verbose:
            print(f"\n[{index}/{total}] {title}")


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
class _Session:
    """池子里存的不透明对象：只要能认出「是不是同一份」就够（真身是 ORT session）。"""

    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f"Session({os.path.basename(self.key)})"


def _session_for(key, built):
    """把「新建了一份 session」记进 built，并返回那份桩对象。"""
    built.append(key)
    return _Session(key)


def _acquire(pool, key, built):
    """按键取 session：命中就复用，否则新建 —— 与 onnx._acquire_session 同一口径。"""
    return pool.acquire(key, lambda: _session_for(key, built))


def _names(pool):
    return [os.path.basename(item) for item in pool.keys()]


# ---------------------------------------------------------------------------
# 1) 额度归一
# ---------------------------------------------------------------------------
def _section_normalize(check):
    check.section(1, _TOTAL_SECTIONS, "额度归一：0 / 缺省 / 非法 -> 1，负数 -> 不限")
    check(normalize_budget(0) == 1, "0 -> 1（不是「一份都不留」）")
    check(normalize_budget(None) == 1, "None（老 config 里没这个键）-> 1")
    check(normalize_budget(3) == 3 and normalize_budget("4") == 4,
          "正数与数字字符串原样使用", f"{normalize_budget(3)} / {normalize_budget('4')}")
    check(normalize_budget(-1) == -1 and normalize_budget(-7) == -1,
          "负数 -> -1（不限）")
    check(normalize_budget("abc") == 1 and normalize_budget("") == 1,
          "非法字符串 -> 1（宁可少驻留，不要静默变成不限）")
    check(SessionPool(0).budget == 1 and SessionPool(None).budget == 1,
          "池子构造时自己也过一遍归一（它直接吃 config 的原值）")


# ---------------------------------------------------------------------------
# 2) 额度 1 = 一换一放
# ---------------------------------------------------------------------------
def _section_budget_one(check):
    check.section(2, _TOTAL_SECTIONS, "额度 1 = 一换一放（小显存口径，改造前的资源行为）")
    built = []
    pool = SessionPool(1)
    first, hit = _acquire(pool, "m000.onnx", built)
    check(not hit and len(built) == 1, "第一次必然新建")
    again, hit = _acquire(pool, "m000.onnx", built)
    check(hit and again is first and len(built) == 1,
          "同一份再要一次 -> 命中，不重建")
    other, hit = _acquire(pool, "m001.onnx", built)
    check(not hit and other is not first and len(built) == 2,
          "第 2 份是新的一份（不是把上一份当它用）")
    check(len(pool) == 1 and "m000.onnx" not in pool and "m001.onnx" in pool,
          "额度 1 下驻留恒为 1 份：上一份已还给 ORT", str(_names(pool)))
    back, hit = _acquire(pool, "m000.onnx", built)
    check(not hit and len(built) == 3,
          "额度 1 下切回上一个成员**必须重建**（所以 cli 把最后一个成员的 session 留着）")
    check(pool.stats()["evictions"] == 2 and pool.stats()["builds"] == 3,
          "换出按份记账", str(pool.stats()))
    check(pool.purge() == 1 and len(pool) == 0,
          "purge() 清空 —— 这正是 release() 在额度 1 时做的事")

    # 键的口径：abspath 归一，同名不同目录是两份
    built = []
    pool = SessionPool(3)
    dotted = os.path.join("sub", "m000.onnx")
    _acquire(pool, dotted, built)
    _, hit = _acquire(pool, os.path.abspath(dotted), built)
    check(hit and len(built) == 1, "同一个文件写成相对 / 绝对路径都算一份（键先 abspath）")
    _, hit = _acquire(pool, os.path.join("other", "m000.onnx"), built)
    check(not hit and len(built) == 2,
          "同名不同目录是两份（键是整条路径，不是文件名，也不是成员号）")


# ---------------------------------------------------------------------------
# 3) 额度 N 的 LRU
# ---------------------------------------------------------------------------
def _section_lru(check):
    check.section(3, _TOTAL_SECTIONS, "额度 N：命中的重新入尾，满了换出最久没用过的")
    built = []
    pool = SessionPool(3)
    for name in ("a.onnx", "b.onnx", "c.onnx"):
        _acquire(pool, name, built)
    check(len(pool) == 3 and pool.stats()["evictions"] == 0, "前 3 份都留得下")
    for name in ("a.onnx", "b.onnx", "c.onnx"):
        _acquire(pool, name, built)
    check(pool.stats()["builds"] == 3 and pool.stats()["hits"] == 3,
          "回到同一批 key 不再新建（这就是「显存够就别放手」的收益）", str(pool.stats()))
    check(_names(pool) == ["a.onnx", "b.onnx", "c.onnx"],
          "keys() 按 LRU 排序：刚用过的在最后", str(_names(pool)))
    _acquire(pool, "d.onnx", built)
    check(pool.stats()["evictions"] == 1 and "a.onnx" not in pool and "d.onnx" in pool,
          "第 4 份进来换掉最久没用过的 a（不是刚用过的 c）", str(_names(pool)))
    check(len(pool) == 3, "驻留数恒不超过额度")
    _acquire(pool, "a.onnx", built)
    check(pool.stats()["builds"] == 5 and pool.stats()["evictions"] == 2
          and "b.onnx" not in pool,
          "被换出的那份要重建；这一下又换出新任「最久没用」的 b", str(_names(pool)))


# ---------------------------------------------------------------------------
# 4) 不限额度
# ---------------------------------------------------------------------------
def _section_unbounded(check):
    check.section(4, _TOTAL_SECTIONS, "额度 -1 = 不限：永不换出，purge 是唯一出口")
    built = []
    pool = SessionPool(-1, logger=logging.getLogger("xmetai_inference.backends.selftest"))
    for index in range(12):
        _acquire(pool, "m%03d.onnx" % index, built)
    check(len(pool) == 12 and pool.stats()["evictions"] == 0, "12 份全留（大显存极端档）")
    for index in range(12):
        _acquire(pool, "m%03d.onnx" % index, built)
    check(pool.stats()["hits"] == 12 and pool.stats()["builds"] == 12, "再走一遍全是命中")
    check(pool.purge() == 12 and len(pool) == 0 and pool.stats()["size"] == 0,
          "purge() 把整池还回去（release_all 用它）")
    check(pool.stats()["builds"] == 12 and pool.stats()["hits"] == 12,
          "purge 不清计数：收尾日志报的是整个进程的账", str(pool.stats()))
    small = SessionPool(2, logger=logging.getLogger("xmetai_inference.backends.selftest"))
    for name in ("x.onnx", "y.onnx", "z.onnx"):
        _acquire(small, name, built)
    check(small.stats()["evictions"] == 1 and len(small) == 2,
          "带 logger 的换出路径也走得通（会打一条「池已满」日志）")


# ---------------------------------------------------------------------------
# 5) ONNX 后端接线
# ---------------------------------------------------------------------------
class _StubOutput:
    name = "pred"
    shape = [1, 2, 4, 4]


class _StubMeta:
    custom_metadata_map = {}


class _StubSession:
    """够 OnnxInferModel.load() 取那三个派生属性用的最小 session。"""

    def __init__(self, path):
        self.path = path

    def get_outputs(self):
        return [_StubOutput()]

    def get_inputs(self):
        return []

    def get_modelmeta(self):
        return _StubMeta()

    def get_providers(self):
        return ["CPUExecutionProvider"]


def _section_backend(check):
    check.section(5, _TOTAL_SECTIONS, "ONNX 后端接线：release / release_all / 键口径")
    try:
        from .onnx import OnnxInferModel
    except Exception as error:  # noqa: BLE001 - 没装 onnxruntime 就跳过这一节
        check(True, f"没有 onnxruntime，跳过后端接线（池子本身不依赖它）：{error}")
        return

    def _model(budget):
        model = OnnxInferModel(device_id=0, gpu_mem_fraction=0.7)
        model.gpu_state = False
        model.resident_members = budget
        # 不建真 session：只验「池子接到了哪一步」，权重与 cuDNN 是 ORT 的事。
        model._make_session = _StubSession
        return model

    model = _model(1)
    model.load("m000.onnx")
    check(model.pool_stats()["builds"] == 1 and model.session is not None,
          "load() 走池子建了 1 份")
    model.release()
    check(model.session is None and model.pool_stats()["size"] == 0,
          "额度 1：release() 连池子一起清（与改造前逐位相同的资源行为）")
    model.load("m000.onnx")
    check(model.pool_stats()["builds"] == 2,
          "额度 1：release 之后再 load 必然重建（一换一放的口径没变）")

    model = _model(6)
    for index in range(6):
        model.member_index = index
        model.load("m%03d.onnx" % index)
    check(model.pool_stats()["builds"] == 6 and model.pool_stats()["size"] == 6,
          "6 个成员各建一份", str(model.pool_stats()))
    model.member_index = 0
    model.load("m000.onnx")
    check(model.pool_stats()["hits"] == 1 and model.pool_stats()["builds"] == 6,
          "切回第 0 个成员：池里命中，不重建、不重预热（这就是 B 的收益）")
    model.member_index = 9
    model.load("m003.onnx")
    check(model.pool_stats()["builds"] == 6,
          "键是解析后的路径：成员号变了而路径没变，仍算命中（同一份 session 不会存成多份）")
    model.release()
    check(model.session is None and model.pool_stats()["size"] == 6,
          "额度 > 1：release() 不动池子（下一个任务还能拿回来）")
    model.load("m004.onnx")
    check(model.pool_stats()["hits"] == 3 and model.session is not None,
          "release 之后的 load 依然命中", str(model.pool_stats()))
    model.release_all()
    check(model.session is None and model.pool_stats()["size"] == 0,
          "release_all() 把 session 与整池一起还回去（进程收尾）")


# ---------------------------------------------------------------------------
def run_selftest(*, verbose=True):
    check = _Checker(verbose=verbose)
    if verbose:
        print("常驻 session 池自检：LRU 记账 + 额度语义 + ONNX 后端接线"
              "（不碰 GPU，不建真 session）")
    _section_normalize(check)
    _section_budget_one(check)
    _section_lru(check)
    _section_unbounded(check)
    _section_backend(check)
    print("\n" + "=" * 72)
    print(f"合计：{check.passed} 条通过，{len(check.failures)} 条不通过")
    if check.failures:
        for failure in check.failures:
            print(f"  - {failure}")
        print("结论：常驻 session 池有问题，先别拿 resident_members 调大跑正式预报。")
        return 1
    print("结论：池子可用 —— 额度 1 与改造前的资源行为一致，额度 N 的 LRU 命中"
          "与换出都记对了。")
    return 0


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    return run_selftest(verbose="--quiet" not in args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_selftest"]
