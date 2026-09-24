# -*- coding: utf-8 -*-
"""调度层自检：不碰 onnx / torch / GPU，纯文件系统 + 线程。

跑的是一条真链路：任务表 -> 共享盘账本 -> 多 worker 并发认领 -> 完成标记与续跑，
外加把四种僵尸认领（别人的 launch / 换过进程 / 心跳过期 / 硬租约到点）各造一遍。
这些判据错了的后果是「两个进程跑同一个任务、往同一批 .nc 里写交错字节」，比慢更糟，
所以每一条都要能独立说清「为什么它可以被回收」。

    python -m xmetai_inference.scheduling.selftest
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time

import pandas as pd

from .config import SchedulingConfig
from .feed import TaskFeed
from .plan import TaskPlan, member_blocks, run_payload
from .store import ClaimStore, _write_json, new_launch_id


_TOTAL_SECTIONS = 7


class _Checker:
    """收集断言结果，跑完所有条目再给总结（与 perturbation/selftest.py 同一风格）。"""

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
def _plan(fingerprint_text, *, inits=2, members=3, steps=3, per_task=1):
    times = [pd.Timestamp("2025-01-02 00:00") + pd.Timedelta(days=index)
             for index in range(inits)]
    payload = run_payload(
        model_path="", steps=steps, hour_interval=6, members=members,
        variables=["z500"], inits=times, members_per_task=per_task)
    payload["fixture"] = fingerprint_text
    return TaskPlan.build(
        inits=times, members=members, steps=steps, hour_interval=6,
        members_per_task=per_task, payload=payload)


def _store(root, fingerprint, **kwargs):
    options = {
        "launch_id": "L1",
        "heartbeat_timeout": 30.0,
        "task_ttl_seconds": 0.0,
        "task_retries": 1,
    }
    options.update(kwargs)
    store = ClaimStore(root, fingerprint, **options)
    store.ensure_dirs()
    return store


class _FakeWorker:
    """把 cli 里 worker 的骨架搬到测试里：心跳线程 + 认领循环 + 完成标记。"""

    def __init__(self, store, plan, *, order="init", cache_size=0, depth=2,
                 poll_interval=0.03, heartbeat_interval=0.05, limit=None):
        self.store = store
        self.plan = plan
        self.order = order
        self.cache_size = cache_size
        self.depth = depth
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.limit = limit
        self.done = []
        self.errors = []
        self.stats = {}
        self._stop_beat = threading.Event()

    def _beat(self):
        while not self._stop_beat.wait(self.heartbeat_interval):
            self.store.heartbeat(state="running", done=len(self.done))

    def _read(self, task):
        return {"init": task.init}

    def run(self):
        self.store.heartbeat(state="starting", done=0)
        beater = threading.Thread(target=self._beat, daemon=True)
        beater.start()
        feed = TaskFeed(self.store, self.plan, self._read, order=self.order,
                        depth=self.depth, poll_interval=self.poll_interval,
                        cache_size=self.cache_size)
        feed.start()
        try:
            for task, _sample in feed:
                self.store.mark_done(task.key)
                feed.finish(task.key)
                self.done.append(task.key)
                if self.limit is not None and len(self.done) >= self.limit:
                    break
        except Exception as error:  # noqa: BLE001 - 测试里收集起来给断言看
            self.errors.append(error)
        finally:
            self.stats = dict(feed.stats)
            feed.close(release_claims=True)
            self._stop_beat.set()
            beater.join(timeout=1)
            self.store.clear_heartbeat()
        return self.done


def _run_workers(workers, timeout=60):
    threads = [threading.Thread(target=worker.run) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
    return [worker.done for worker in workers]


# ---------------------------------------------------------------------------
# 1/6 配置解析与校验
# ---------------------------------------------------------------------------
def _section_config(check):
    check.section(1, _TOTAL_SECTIONS, "调度超参数：config 里的 scheduling={...}")

    default = SchedulingConfig.from_config(None)
    check(default.mode == "claim" and default.procs_per_gpu == 1
          and default.resume == "outputs",
          "缺省全自动：mode=claim / procs_per_gpu=1 / resume=outputs",
          f"{default.mode} {default.procs_per_gpu} {default.resume}")

    custom = SchedulingConfig.from_config({
        "mode": "claim", "procs_per_gpu": 2, "resume": "marker",
        "heartbeat_interval": 5, "heartbeat_timeout": 45, "prefetch_tasks": 4,
        "input_cache": 0, "atomic_output": False, "members_per_task": 4,
    })
    check(custom.procs_per_gpu == 2 and custom.resume == "marker"
          and custom.heartbeat_timeout == 45.0 and custom.atomic_output is False,
          "每个调度键都能从 config 生效（含 bool 与 int->float 转换）", str(custom))

    for value, why in (({"resume": "always"}, "非法枚举"),
                       ({"procs_per_gpu": 0}, "越界"),
                       ({"heartbeat_timeout": 1, "heartbeat_interval": 10},
                        "超时必须大于间隔"),
                       ({"nope": 1}, "不认识的键")):
        try:
            SchedulingConfig.from_config(value)
            check(False, f"非法配置要报错（{why}）", str(value))
        except (ValueError, TypeError) as error:
            check(True, f"非法配置报错（{why}）：{str(error)[:40]}...")

    check(SchedulingConfig.from_config({"members_per_task": 6})
          .members_per_task_for(3) == 3,
          "members_per_task 超过成员总数时收敛到总数")
    check(SchedulingConfig.from_config({"claim_order": "auto"})
          .claim_order_for(True) == "member"
          and SchedulingConfig.from_config({"claim_order": "auto"})
          .claim_order_for(False) == "init",
          "认领偏好自动分流：扰动档按成员、非扰动按起报")


# ---------------------------------------------------------------------------
# 2/6 任务表与指纹
# ---------------------------------------------------------------------------
def _section_plan(check):
    check.section(2, _TOTAL_SECTIONS, "任务表：切分、任务键、输出布局、指纹")

    check(member_blocks(5, 2) == [(0, 2), (2, 2), (4, 1)],
          "成员块切分（末块可以不齐）", str(member_blocks(5, 2)))

    plan = _plan("A", inits=2, members=3, steps=4, per_task=1)
    check(len(plan.tasks) == 6, "2 起报 x 3 成员 = 6 个任务", str(len(plan.tasks)))
    check(len({task.key for task in plan.tasks}) == 6, "任务键互不相同")
    check(all(len(plan.output_relpaths(task)) == 4 for task in plan.tasks),
          "每个任务的产物个数只由 steps 决定")

    first = plan.tasks[0]
    check(plan.output_relpaths(first) ==
          [f"20250102/member_000/{step:03d}.nc" for step in range(1, 5)],
          "集合布局 = 起报/member_NNN/步号.nc（下游评测不需要改）",
          str(plan.output_relpaths(first)[:2]))

    det = _plan("A", inits=1, members=1, steps=3, per_task=1)
    check(det.output_relpaths(det.tasks[0]) ==
          [f"20250102/{step:03d}.nc" for step in range(1, 4)],
          "确定性布局 = 起报/步号.nc",
          str(det.output_relpaths(det.tasks[0])))

    coarse = _plan("A", inits=1, members=4, steps=2, per_task=2)
    check(len(coarse.tasks) == 2 and coarse.tasks[0].members() == [0, 1],
          "members_per_task=2 时一个任务装两个成员", str(len(coarse.tasks)))

    check(_plan("A").fingerprint == _plan("A").fingerprint, "同输入 -> 同指纹")
    for label, kwargs in (("起报", {"inits": 3}), ("成员", {"members": 4}),
                          ("步数", {"steps": 5}), ("成员块", {"per_task": 2})):
        check(_plan("A", **kwargs).fingerprint != _plan("A").fingerprint,
              f"指纹对「{label}变了」敏感")
    check(_plan("A").fingerprint != _plan("B").fingerprint,
          "指纹对模型身份敏感（权重换掉 -> 旧产物不再复用）")

    restored = TaskPlan.from_dict(plan.to_dict())
    check([task.key for task in restored.tasks] == [t.key for t in plan.tasks]
          and restored.fingerprint == plan.fingerprint
          and restored.steps == plan.steps,
          "plan.json 往返一致")


# ---------------------------------------------------------------------------
# 3/6 账本：认领 / 完成 / 失败 / 僵尸
# ---------------------------------------------------------------------------
def _section_store(check, work):
    check.section(3, _TOTAL_SECTIONS, "共享盘账本：互斥认领、完成标记、四种僵尸判据")
    root = os.path.join(work, "store")
    plan = _plan("S", inits=4, members=2, per_task=1)   # 8 个键，一个场景分一个
    key = plan.tasks[0].key

    first = _store(root, plan.fingerprint, worker_id="w0")
    second = _store(root, plan.fingerprint, worker_id="w1")
    first.heartbeat(state="run")
    second.heartbeat(state="run")

    check(first.claim(key) is True, "第一个 worker 认领成功")
    check(second.claim(key) is False, "第二个 worker 抢不到同一个任务（mkdir 原子）")
    check(first.claimed(key) and not first.stale(key), "活着的 worker 的认领不是僵尸")

    first.mark_done(key)
    check(first.is_done(key) and key in first.done_keys(), "完成标记落盘")
    check(not first.claimed(key), "完成后认领目录被清掉（不会越积越多）")
    check(first.claim(key) is False and second.claim(key) is False,
          "已完成的任务不会再被任何 worker 认领（防重复跑的硬防线）")

    fail_key = plan.tasks[1].key
    check(first.fail(fail_key, "boom") == 1 and first.attempts(fail_key) == 1,
          "失败次数被记下来")
    check(not first.claimed(fail_key), "失败会释放认领，任务可被重派")
    check(first.failed_keys() == set(), "第 1 次失败还没到上限（task_retries=1）")
    first.fail(fail_key, "boom again")
    check(first.failed_keys() == {fail_key}, "超过上限 -> giveup，不再重派")

    # 僵尸判据 1：认领来自别的 launch
    foreign = plan.tasks[2].key
    other = _store(root, plan.fingerprint, launch_id="L2", worker_id="w9")
    other.heartbeat(state="run")
    check(other.claim(foreign), "另一个 launch 认领成功")
    check(not first.stale(foreign),
          "别人的 launch 还活着 -> 不许回收（防多机互抢）")
    stale_beat = other.read_status()["w9"]
    stale_beat["ts"] = time.time() - 3600
    _write_json(other.worker_path("w9"), stale_beat)
    check(first.stale(foreign), "别人的 launch 心跳过期 -> 可以回收（那台机器死了）")

    # 僵尸判据 2：同一个 worker_id 换过进程
    swapped = plan.tasks[3].key
    check(first.claim(swapped), "重新认领一个任务")
    with open(first.worker_path("w0"), "r", encoding="utf-8") as handle:
        beat = json.load(handle)
    beat["pid"] = int(beat["pid"]) + 999
    _write_json(first.worker_path("w0"), beat)
    check(first.stale(swapped),
          "worker_id 没变但 pid 换了 -> 上一世代遗留，可回收")

    # 僵尸判据 3：心跳过期
    beaten = plan.tasks[4].key
    third = _store(root, plan.fingerprint, worker_id="w2")
    third.heartbeat(state="run")
    check(third.claim(beaten), "第三个 worker 认领成功")
    with open(third.worker_path("w2"), "r", encoding="utf-8") as handle:
        fresh = json.load(handle)
    fresh["ts"] = time.time() - 600
    _write_json(third.worker_path("w2"), fresh)
    check(third.stale(beaten), "心跳过期（超 heartbeat_timeout）-> 可回收")

    # 僵尸判据 4：硬租约（由 task_ttl_seconds 控制，默认关闭）
    fourth = _store(root, plan.fingerprint, worker_id="w3")
    fourth.heartbeat(state="run")
    fourth.claim(plan.tasks[5].key)
    check(not fourth.stale(plan.tasks[5].key),
          "心跳新鲜 + 租约关闭（task_ttl_seconds=0）-> 不回收")
    leased = _store(root, plan.fingerprint, worker_id="w5", task_ttl_seconds=0.01)
    leased.heartbeat(state="run")
    leased.claim(plan.tasks[6].key)
    time.sleep(0.05)
    check(leased.stale(plan.tasks[6].key),
          "心跳还新但租约到点 -> 仍然回收（防活着的 worker 卡死在一个任务上）")

    # 没有心跳文件时不能立刻判死（否则刚认领就被别人抢走）
    bare = _store(root, plan.fingerprint, worker_id="w4")
    bare.claim(plan.tasks[7].key)
    check(not bare.stale(plan.tasks[7].key),
          "认领目录刚建、心跳还没落盘 -> 不判死（宁可多等）")


# ---------------------------------------------------------------------------
# 4/6 并发认领：每个任务恰好被跑一次
# ---------------------------------------------------------------------------
def _section_concurrent(check, work):
    check.section(4, _TOTAL_SECTIONS, "并发认领：4 个 worker 分摊 24 个任务，各跑一次")
    root = os.path.join(work, "concurrent")
    plan = _plan("C", inits=4, members=6, steps=2, per_task=1)
    launch = new_launch_id()
    check(len(plan.tasks) == 24, "4 起报 x 6 成员 = 24 个任务", str(len(plan.tasks)))

    workers = []
    for index in range(4):
        store = _store(root, plan.fingerprint, launch_id=launch,
                       worker_id=f"w{index}", heartbeat_timeout=30.0)
        workers.append(_FakeWorker(store, plan, order="member", cache_size=1,
                                   depth=2, poll_interval=0.02))
    started = time.time()
    _run_workers(workers)
    elapsed = time.time() - started

    collected = [key for worker in workers for key in worker.done]
    check(all(not worker.errors for worker in workers),
          "4 个 worker 都没有异常退出",
          str([str(error) for worker in workers for error in worker.errors][:2]))
    check(len(collected) == 24 and len(set(collected)) == 24,
          "24 个任务恰好被跑一次（没有重复认领）",
          f"共 {len(collected)} 次、去重 {len(set(collected))}")
    check(sum(worker.stats.get("claimed", 0) for worker in workers) == 24,
          "认领次数 == 任务数（认领不肯把别人刚跑完的任务再捞一遍）",
          str([worker.stats.get("claimed") for worker in workers]))
    check(set(collected) == {task.key for task in plan.tasks},
          "覆盖了任务表里的全部任务")
    check(len(workers[0].store.done_keys()) == 24, "完成标记 24 个")
    print(f"    （4 worker / 24 任务耗时 {elapsed:.2f}s）")

    # 续跑：同一份计划再跑一遍，一个任务都不该重算
    second_pass = [
        _FakeWorker(_store(root, plan.fingerprint, launch_id=new_launch_id(),
                           worker_id=f"r{index}", heartbeat_timeout=0.3), plan,
                    order="member", poll_interval=0.02, heartbeat_interval=0.05)
        for index in range(4)
    ]
    _run_workers(second_pass, timeout=30)
    check(all(not worker.done for worker in second_pass),
          "重跑同一指纹：0 个任务被重算（断点续跑）",
          str([len(worker.done) for worker in second_pass]))


# ---------------------------------------------------------------------------
# 5/6 崩溃恢复
# ---------------------------------------------------------------------------
def _section_recovery(check, work):
    check.section(5, _TOTAL_SECTIONS, "崩溃恢复：死掉的 worker 手里的任务被捡走")
    root = os.path.join(work, "recovery")
    plan = _plan("R", inits=2, members=2, per_task=1)
    launch = new_launch_id()

    dying = _store(root, plan.fingerprint, launch_id=launch, worker_id="dying",
                   heartbeat_timeout=0.3)
    dying.heartbeat(state="running")
    victim = plan.tasks[0].key
    check(dying.claim(victim), "垂死 worker 认领了任务")
    check(not dying.stale(victim), "刚认领、心跳还新 -> 不能回收")
    time.sleep(0.45)                      # 模拟被 OOM SIGKILL：日志戛然而止
    check(dying.stale(victim),
          "心跳过期 -> 判死（SIGKILL 的典型形态：没有 traceback）")

    survivor = _store(root, plan.fingerprint, launch_id=launch, worker_id="alive",
                      heartbeat_timeout=0.3)
    worker = _FakeWorker(survivor, plan, order="init", poll_interval=0.02)
    _run_workers([worker], timeout=30)
    check(victim in worker.done,
          "幸存 worker 捡起了死者的任务（这就是崩溃重派）", str(worker.done))
    check(len(worker.done) == 4, "本轮的 4 个任务全部跑完", str(len(worker.done)))

    # 主控收尾：把无主认领强制丢掉，再跑一轮就干净了
    root2 = os.path.join(work, "recovery2")
    plan2 = _plan("R2", inits=1, members=2, per_task=1)
    ghost = _store(root2, plan2.fingerprint, launch_id="L-old", worker_id="ghost")
    ghost.claim(plan2.tasks[0].key)
    supervisor = _store(root2, plan2.fingerprint, launch_id="L-new",
                        worker_id="sup", heartbeat_timeout=0.2)
    pending = [task.key for task in plan2.tasks if not supervisor.is_done(task.key)]
    check(len(pending) == 2, "主控看到 2 个未完成任务（含 1 个无主认领）",
          str(len(pending)))
    check(supervisor.stale(plan2.tasks[0].key),
          "上一次运行的认领（无活心跳）被判为僵尸")
    for key in pending:
        supervisor.force_release(key)
    revived = _FakeWorker(supervisor, plan2, poll_interval=0.02)
    _run_workers([revived], timeout=30)
    check(len(revived.done) == 2, "清掉僵尸认领后一轮跑完", str(len(revived.done)))


# ---------------------------------------------------------------------------
# 6/6 预取偏好与输入缓存
# ---------------------------------------------------------------------------
def _section_feed(check, work):
    check.section(6, _TOTAL_SECTIONS, "预取：认领偏好（省 session 换入 / 省输入重读）")
    plan = _plan("F", inits=3, members=3, steps=1, per_task=1)

    store = _store(os.path.join(work, "feed_member"), plan.fingerprint,
                   worker_id="wm")
    worker = _FakeWorker(store, plan, order="member", poll_interval=0.01)
    _run_workers([worker], timeout=30)
    members_order = [plan.by_key[key].member_start for key in worker.done]
    check(members_order == [0, 0, 0, 1, 1, 1, 2, 2, 2],
          "order=member：同一成员的任务连在一起（0,0,0,1,1,1,2,2,2）",
          str(members_order))

    reads = []

    class _CountingWorker(_FakeWorker):
        def _read(self, task):
            reads.append(task.key)
            return {"init": task.init}

    store2 = _store(os.path.join(work, "feed_init"), plan.fingerprint,
                    worker_id="wi")
    worker2 = _CountingWorker(store2, plan, order="init", cache_size=2,
                              poll_interval=0.01)
    _run_workers([worker2], timeout=30)
    check(len(reads) == 3 and worker2.stats["cache_hit"] == 6,
          "order=init + input_cache=2：3 个起报各读一次，其余 6 次命中缓存",
          f"读 {len(reads)} 次 / 命中 {worker2.stats.get('cache_hit')} 次")
    check(len({plan.by_key[key].init for key in worker2.done}) == 3,
          "3 个起报都被跑到")
    check(worker2.stats["claimed"] == 9, "认领了 9 个任务", str(worker2.stats))

    # 关掉缓存时不再复用（对照组，证明上面那 6 次命中确实来自缓存）
    reads_off = []

    class _CountingWorkerNoCache(_CountingWorker):
        def _read(self, task):
            reads_off.append(task.key)
            return {"init": task.init}

    store3 = _store(os.path.join(work, "feed_nocache"), plan.fingerprint,
                    worker_id="wn")
    worker3 = _CountingWorkerNoCache(store3, plan, order="init", cache_size=0,
                                     poll_interval=0.01)
    _run_workers([worker3], timeout=30)
    check(len(reads_off) == 9 and worker3.stats["cache_hit"] == 0,
          "input_cache=0 时每个任务都真读一次（对照组）",
          f"读 {len(reads_off)} 次 / 命中 {worker3.stats.get('cache_hit')}")


# ---------------------------------------------------------------------------
# 7/7 claim 模式的 cli 编排（桩驱动：把真 cli 的认领循环跑一遍）
# ---------------------------------------------------------------------------
def _section_claim_cli(check, work):
    """用桩把 cli._run_claimed 这条真循环跑一遍：认领 -> 跑 -> flush -> 完成标记。

    为什么值得单独测：认领模式的「谁跑哪个任务」是运行期决定的，最容易出的错不是算错
    而是**流程错**——标记写早了（把半成品当完成）、失败后没把自己从手头摘掉（预取线程
    永远等不到收工）、产物路径与静态分片不一致（下游评测读不到）。这些都不需要 GPU、
    不需要 onnx，用桩就能钉住。

    这一节要 Rollout（= torch），所以放在最后；前六节仍然只有文件系统与线程。
    """
    import argparse
    import threading

    import numpy as np
    import torch

    from .. import cli
    from .heartbeat import Heartbeat

    check.section(7, _TOTAL_SECTIONS, "claim 模式的 cli 编排（桩驱动：认领 / 跳过 / 重试）")

    class _Writer:
        """只记文件名与原始数组，外加 flush 事件（替代 cli._AsyncWriter，不落盘）。"""

        def __init__(self):
            self.names = []
            self.raw = []
            self.on_flush = None

        def put(self, name, ds, raw=None):
            self.names.append(name)
            self.raw.append((name, raw[0] if raw else None))

        def flush(self):
            if self.on_flush is not None:
                self.on_flush()

        def close(self):
            return []

    class _Dataset:
        """按位置取一个样本：字段与形状照抄真数据集（[T,C,H,W] + times 标量）。"""

        def __init__(self, inits, shape=(2, 3, 4, 4), seed=7):
            rng = np.random.default_rng(seed)
            self.times = [pd.Timestamp(value) for value in inits]
            self.blocks = {
                time: torch.as_tensor(rng.standard_normal(shape).astype(np.float32))
                for time in self.times}

        def __getitem__(self, index):
            time = self.times[index]
            return {"inputs": self.blocks[time],
                    "times": torch.tensor(time.value, dtype=torch.int64),
                    "sample_index": index}

    class _Source:
        """只有 dataset 字段：认领模式不迭代 DataLoader。"""

        def __init__(self, dataset):
            self.dataset = dataset

    def _model(*, fail_times=0):
        from xmetai_inference.backends.base import BaseInferModel

        class _Stub(BaseInferModel):
            windowed_output = True
            gpu_state = False
            valid_time_step_offset = 0

            def __init__(self):
                super().__init__()
                self.loads = []
                self.releases = 0
                self.session = None
                self.forwards = 0
                self.fail_left = int(fail_times)
                self.processing = object()   # prepare_dataset_batch 的两个前提
                self.data_source = object()

            def load(self, path):
                self.loads.append(int(self.member_index))
                self.session = object()      # 真后端 load 完会把 session 挂上

            def release(self):
                self.releases += 1
                self.session = None          # 真后端的 release 会把它置空

            def forward(self, x, step, valid_time):
                self.forwards += 1
                if self.fail_left > 0:
                    self.fail_left -= 1
                    raise RuntimeError("桩：故意失败一次")
                return x + 1.0               # 确定、形状不变：便于与静态路径逐位比对

            def to_dataset(self, state, save_names=None, lat=None, lon=None):
                return np.asarray(state)     # 真实现要 xarray，这里只给桩 writer 记下

        return _Stub()

    def _batch(dataset, index):
        """与 cli._run_claimed 里的 read_sample 同一口径：单样本补回 batch 维。"""
        item = dataset[index]
        return {key: (value.unsqueeze(0) if hasattr(value, "unsqueeze") else value)
                for key, value in item.items()}

    sched = SchedulingConfig.from_config({
        "poll_interval": 0.02, "heartbeat_interval": 0.05, "heartbeat_timeout": 5.0,
        "prefetch_tasks": 1, "input_cache": 0, "resume": "outputs", "task_retries": 1,
    })
    plan = _plan("CLI", inits=2, members=1, steps=3, per_task=1)   # 确定性：2 个任务
    inits = sorted({task.init for task in plan.tasks})
    dataset = _Dataset(inits)
    source = _Source(dataset)
    save_names = ["c0"]

    def _run(tag, *, model=None, out=None, store=None, limit=120):
        """在一个线程里跑 cli._run_claimed：挂死时是断言失败，不是测试卡死。"""
        store = store or _store(os.path.join(work, f"plan_{tag}"), plan.fingerprint,
                                worker_id=f"wc-{tag}", heartbeat_timeout=5.0)
        store.write_plan(plan)
        writer = _Writer()
        events = []
        writer.on_flush = lambda: events.append(("flush", ""))
        real_mark = store.mark_done

        def _mark(key):
            events.append(("mark", key))
            return real_mark(key)

        store.mark_done = _mark
        beater = Heartbeat(store, 0.05)
        run_args = argparse.Namespace(
            steps=plan.steps, out=out or os.path.join(work, f"out_{tag}"), model="")
        state = {"events": events, "writer": writer, "store": store}

        def body():
            state["code"] = cli._run_claimed(
                model if model is not None else _model(), None, source, writer,
                run_args, sched, plan, store, beater, inits, save_names, np.zeros(1),
                np.zeros(1), 6, False, None, [], None, 0, 1)

        thread = threading.Thread(target=body, daemon=True)
        thread.start()
        thread.join(timeout=limit)
        state["hung"] = thread.is_alive()
        return state

    # 1) 正常跑完：产物路径与静态分片一致、一个任务一个标记、标记在 flush 之后
    run = _run("a")
    check(not run["hung"], "认领循环跑完并退出（没有卡在预取线程上）")
    check(run.get("code") == 0, "返回码 0", str(run.get("code")))
    expected = {f"{init:%Y%m%d}/{step:03d}.nc" for init in inits
                for step in range(1, plan.steps + 1)}
    check(set(run["writer"].names) == expected,
          f"确定性布局逐字一致（{len(expected)} 个文件）",
          str(sorted(set(run["writer"].names) ^ expected)[:4]))
    check(len(run["store"].done_keys()) == 2, "两个任务各写了一个完成标记",
          str(sorted(run["store"].done_keys())))
    check(not run["store"].claimed(plan.tasks[0].key), "完成后认领目录被清掉")
    flushed_before_mark, pending = True, False
    for kind, _payload in run["events"]:
        if kind == "flush":
            pending = True
        else:
            flushed_before_mark = flushed_before_mark and pending
            pending = False
    marks = [payload for kind, payload in run["events"] if kind == "mark"]
    check(len(marks) == 2 and flushed_before_mark,
          "每个完成标记都写在 flush 之后（不会把半成品标成完成）",
          str(run["events"]))

    # 2) 与静态路径逐位等价：同一份输入、同一份编排，产物必须一模一样
    static = _Writer()
    for init in inits:
        cli._run_plain_block(_model(), None, [_batch(dataset, inits.index(init))],
                             static, argparse.Namespace(steps=plan.steps, out=""),
                             [0], save_names, np.zeros(1), np.zeros(1), 6, 0, 1, 1,
                             total_inits=1)
    check(set(static.names) == expected, "静态路径写的是同一批文件名")
    claim_raw = {name: array for name, array in run["writer"].raw if array is not None}
    static_raw = {name: array for name, array in static.raw if array is not None}
    check(all(np.array_equal(claim_raw[name], static_raw[name]) for name in expected),
          "认领与静态的产物逐位相同（同 members / steps / 成员块）")

    # 3) resume=outputs：产物已经在盘上 -> 补完成标记后跳过，不重算
    skip_out = os.path.join(work, "out_skip")
    init_dir = os.path.join(skip_out, f"{inits[0]:%Y%m%d}")
    os.makedirs(init_dir, exist_ok=True)
    with open(os.path.join(init_dir, f"{plan.steps:03d}.nc"), "w",
              encoding="utf-8") as handle:
        handle.write("{}")
    skip_model = _model()
    skip = _run("skip", model=skip_model, out=skip_out)
    check(not skip["hung"] and len(skip["store"].done_keys()) == 2,
          "产物齐全的任务补了完成标记（两个任务都算完成）",
          str(sorted(skip["store"].done_keys())))
    check(skip_model.forwards == plan.steps,
          f"只算了缺产物的那一个任务（前向 {skip_model.forwards} 次，期望 {plan.steps}）",
          str(skip_model.forwards))
    check(not any(name.startswith(f"{inits[0]:%Y%m%d}/") for name in skip["writer"].names),
          "跳过的那个起报没有被重算（没写它的任何文件）",
          str(skip["writer"].names[:3]))

    # 4) 失败 -> 记一次失败、释放认领 -> 自己重认领一次并跑成
    retry = _run("retry", model=_model(fail_times=1))
    check(not retry["hung"] and retry.get("code") == 0
          and len(retry["store"].done_keys()) == 2,
          "第一次失败之后重认领并跑完（两个任务都完成）",
          str(sorted(retry["store"].done_keys())))
    check(not retry["store"].failed_keys(), "重试成功了 -> 没有 giveup 标记",
          str(sorted(retry["store"].failed_keys())))
    check(sum(retry["store"].attempts(task.key) for task in plan.tasks) == 1,
          "失败被记账一次（failed/<key>/01.json），没有重复记账",
          str([retry["store"].attempts(task.key) for task in plan.tasks]))

    # 5) 成员外层：同一成员连着来就沿用已装载的权重（省一次 1.5 GB 的 onnx 装载）
    member_model = _model()
    member_writer = _Writer()
    member_args = argparse.Namespace(steps=2, out="", model="")
    member_init = inits[0]
    blocks = [_batch(dataset, 0)]

    def _member_block(member_ids, **kwargs):
        return cli._run_member_block(
            member_model, None, blocks, member_writer, member_args, [member_init],
            member_ids, save_names, np.zeros(1), np.zeros(1), 6, 0, 1, **kwargs)

    first = _member_block([3])
    check(first == 3, "成员块跑完返回停在 model 上的成员号", str(first))
    check(member_model.session is not None,
          "最后一个成员的 session 留在 model 上（下一个任务靠 loaded_member 复用它）")
    check(member_model.releases == 0,
          "只有一个成员的块不调用 release()：放手了下次就得重建 session + 重预热 cuDNN",
          str(member_model.releases))
    check(member_model.loads == [3], "第一次装载 member_003", str(member_model.loads))
    member_model.loads.clear()
    again = _member_block([3], loaded_member=first)
    check(member_model.loads == [] and again == 3,
          "同一成员连着跑：不重复装载（按成员认领时省下的就是这一次）",
          str(member_model.loads))
    member_model.loads.clear()
    _member_block([4], loaded_member=3)
    check(member_model.loads == [4], "换了成员必须重新装载", str(member_model.loads))
    member_model.loads.clear()
    member_model.releases = 0
    _member_block([5, 6])
    check(member_model.loads == [5, 6] and member_model.releases == 1,
          "多成员块只在成员之间放手：换成员 = 换 session，最后一个仍留着",
          f"loads={member_model.loads} releases={member_model.releases}")
    check(member_model.session is not None, "多成员块的最后一个 session 也留着")

    class _Segments:
        """够 _run_member_block 用的分段桩（不换段，只为验证「不许复用」）。"""

        def __init__(self):
            self.segments = [argparse.Namespace(spec=None)]

        def specs(self):
            return [self.segments[0].spec]

        def switch_after(self):
            return {}

        def describe(self):
            return "桩分段"

    member_model.loads.clear()
    tail = _member_block([3], segments=_Segments(), loaded_member=3)
    check(member_model.loads == [3] and tail is None,
          "分段 rollout 不复用 session（末段的权重不能当第 0 段用）",
          str(member_model.loads))

    # 6) resume=off 的口径：只认本次运行写的完成标记
    root_off = os.path.join(work, "off")
    key = plan.tasks[0].key
    old = _store(root_off, plan.fingerprint, launch_id="L-old", worker_id="w-old")
    old.claim(key)
    old.mark_done(key)
    fresh = _store(root_off, plan.fingerprint, launch_id="L-new", worker_id="w-new",
                   done_scope="launch")
    check(not fresh.is_done(key) and key not in fresh.done_keys(),
          "resume=off：上一次运行的完成标记不算数（整份重算）")
    check(fresh.claim(key), "旧标记不挡认领")
    fresh.mark_done(key)
    check(fresh.is_done(key), "本次运行里写的标记算数（同一轮不重复跑）")
    newer = _store(root_off, plan.fingerprint, launch_id="L-newer", worker_id="w-newer",
                   done_scope="launch")
    check(not newer.is_done(key), "换一次运行又当成未完成（重算就是重算）")
    ever = _store(root_off, plan.fingerprint, launch_id="L-any", worker_id="w-any")
    check(ever.is_done(key), "resume=outputs 的默认口径仍然跨运行采信（断点续跑）")

    # 7) 主控交给 worker 的东西：任务表 + worker 参数 + 指纹握手
    from ..configs.base import InferConfig
    from .store import ClaimStore as _ClaimStore

    ctl = InferConfig(name="桩配方", model_path="mini.onnx",
                      model_class="iwc_fgvp_gdn2_onnx",
                      times="2025010200..2025010300:24", steps=3, members=4,
                      vars="z500", output_dir=os.path.join(work, "ctl_out"))
    ctl_sched = ctl.scheduling_config(
        {"mode": "claim", "procs_per_gpu": 3, "resume": "marker"})
    check(ctl_sched.procs_per_gpu == 3 and ctl_sched.resume == "marker",
          "命令行覆盖只动 mode / procs_per_gpu / resume，其余仍以 config 为准",
          str(ctl_sched))
    ctl_scope = cli._run_scope(ctl, {})
    ctl_plan, ctl_dir = cli._build_run_plan(ctl, ctl_scope, ctl_sched)
    check(len(ctl_plan.tasks) == 8, "2 起报 x 4 成员 = 8 个任务",
          str(len(ctl_plan.tasks)))
    ctl_argv = cli._build_infer_argv(ctl, {}, scope=ctl_scope, sched=ctl_sched,
                                     plan=ctl_plan, plan_dir=ctl_dir,
                                     intra_op_threads=4)
    check("--schedule" in ctl_argv and "claim" in ctl_argv
          and ctl_argv[ctl_argv.index("--plan-fingerprint") + 1] == ctl_plan.fingerprint,
          "worker 参数里带着任务表指纹（父子不会各算一份）")
    payload = json.loads(ctl_argv[ctl_argv.index("--scheduling-json") + 1])
    check(payload["mode"] == "claim" and payload["procs_per_gpu"] == 3,
          "整份调度配置随参数传下去（含命令行覆盖）", str(sorted(payload))[:60])
    check(cli._worker_slots(2, ["0", "1"], 2)
          == [(0, "0"), (1, "0"), (2, "1"), (3, "1")],
          "每卡多进程：槽位 -> 卡号 的映射")
    _ClaimStore(ctl_dir, ctl_plan.fingerprint, launch_id="L",
                worker_id="w").write_plan(ctl_plan)
    worker_args = argparse.Namespace(plan_dir=ctl_dir,
                                     plan_fingerprint=ctl_plan.fingerprint,
                                     launch_id="L", worker_id="w")
    plan_read, _store_read = cli._claim_setup(worker_args, ctl_sched)
    check([task.key for task in plan_read.tasks] == [t.key for t in ctl_plan.tasks],
          "worker 能把任务表原样读回来（plan.json 往返）")
    # 防呆分支 1：目录名（= 主控给的指纹）与 plan.json 里的指纹不一致
    tampered = os.path.join(work, "ctl_tampered")
    _ClaimStore(tampered, "deadbeef", launch_id="L", worker_id="w").write_plan(ctl_plan)
    worker_args.plan_dir = tampered
    worker_args.plan_fingerprint = "deadbeef"
    try:
        cli._claim_setup(worker_args, ctl_sched)
        check(False, "指纹对不上要报错", "没报错")
    except SystemExit as error:
        check("指纹" in str(error), f"plan.json 与指纹目录不一致时直接停：{str(error)[:32]}...")
    # 防呆分支 2：这个指纹目录下根本没有任务表
    worker_args.plan_dir = os.path.join(work, "ctl_missing")
    try:
        cli._claim_setup(worker_args, ctl_sched)
        check(False, "没有任务表要报错", "没报错")
    except SystemExit as error:
        check("plan.json" in str(error), f"没有任务表时直接停：{str(error)[:32]}...")

# ---------------------------------------------------------------------------
def run_selftest(*, verbose=True, keep=False):
    work = tempfile.mkdtemp(prefix="sched_selftest_")
    check = _Checker(verbose=verbose)
    if verbose:
        print("调度层自检：任务表 / 账本 / 并发认领 / 崩溃恢复（不碰 onnx、不碰 GPU）；最后一节跑 cli 的认领循环（桩驱动，需要 torch）")
        print(f"工作目录：{work}")
    try:
        _section_config(check)
        _section_plan(check)
        _section_store(check, work)
        _section_concurrent(check, work)
        _section_recovery(check, work)
        _section_feed(check, work)
        _section_claim_cli(check, work)
        print("\n" + "=" * 72)
        print(f"合计：{check.passed} 条通过，{len(check.failures)} 条不通过")
        if check.failures:
            for failure in check.failures:
                print(f"  - {failure}")
            print("结论：调度层有问题，先别拿它跑正式预报。")
            return 1
        print("结论：任务级动态认领可用——互斥认领、断点续跑、僵尸回收、崩溃重派、"
              "认领偏好都过了。")
        return 0
    finally:
        if keep:
            print(f"已保留工作目录：{work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    return run_selftest(keep="--keep" in args, verbose="--quiet" not in args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_selftest"]
