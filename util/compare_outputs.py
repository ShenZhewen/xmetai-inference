#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比两个预测输出目录的 NetCDF，判断两次跑的结果是否一致（可复现性检查）。

用途：改了代码之后再跑一遍，用这个确认「结果没变」或者「变了多少」。它**不需要
实况数据**，纯粹是两份预测互比。

匹配规则：按「起报目录 + 步序号 + 变量名」三者都相同才比。任一侧缺的部分跳过并
在末尾列出 —— 所以谁的日期/步数少就以少的为准，不用手工对齐。

容错为什么必要：数值等价的改动在 float32 下并不逐位相同。例如 FengQing 的残差
反演由「反归一化→加残差→再归一化」化简成「归一化空间直接加 res_std/std」，两者
代数等价但有约 20 eps 的舍入差。所以判定不能用「完全相等」，而要按每个场自身的
量级做相对判定。

判定档位（每个 变量×lead 一行）：
    IDENTICAL  逐位相同（max|diff| == 0）
    CLOSE      在容差内 —— 可认为「结果没变」
    DRIFT      超出容差但仍远小于场的自然变率，通常是浮点路径变了
    DIFFER     真的不一样，需要查

容差同时看相对和绝对两把尺子（满足其一即通过）：
    |a-b| <= atol + rtol * |b|
tp 这类大片为 0 的场靠 atol 兜底，z500/msl 这类 1e4~1e5 量级的场靠 rtol。

用法：
    python util/compare_outputs.py DIR_A DIR_B
    python util/compare_outputs.py DIR_A DIR_B --vars z500,t2m --steps 20
    python util/compare_outputs.py DIR_A DIR_B --rtol 1e-4 --quiet
    python util/compare_outputs.py DIR_A DIR_B --out /tmp/cmp    # 存 CSV

退出码：0 = 全部 IDENTICAL 或 CLOSE；1 = 有 DRIFT 或 DIFFER；2 = 没有可比对象。
所以可以直接用在脚本里：``python util/compare_outputs.py A B && echo 一致``
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.eval_common import (
    _discover_steps,
    _parse_init_dir,
    load_prediction_file,
)

log = logging.getLogger(__name__)

# 默认容差：rtol=1e-5 能容忍 float32 往返与代数化简的舍入（约 100 eps 的余量），
# 又足以抓住任何真实的物理量变化。atol 按变量量级给一个很小的绝对地板。
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-6
# 超容差点的占比阈值 —— 主判据，理由见 _classify
EXCEED_FRACTION_CLOSE = 1e-3      # 万分之一以下：随机舍入
EXCEED_FRACTION_DRIFT = 0.5       # 半数以上：多数点都变了
# max|diff| 超过场自身空间变率的这个比例，直接 DIFFER（物理上可分辨）
DRIFT_STD_FRACTION = 1e-2


def _discover_init_dirs(root):
    """目录名 -> 起报时刻。兼容 YYYYMMDD 与 YYYYMMDDHH 两种命名。"""
    found = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        init_time = _parse_init_dir(name)
        if init_time is not None:
            found[name] = init_time
    return found


def _variables_in(path):
    """NetCDF 里的二维变量名（小写）。"""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return {
            str(name).lower()
            for name in ds.data_vars
            if ds[name].ndim == 2
        }


def _classify(max_diff, ref_std, exceed_fraction):
    """给一档判定。

    **主判据是超容差点的占比，不是 max|diff|。** 原因：float32 的舍入误差与场的
    量级成正比 —— z500 在 54000 量级上单次运算的 eps 就有 6.5e-3，rollout 60 步
    累积到 0.1 量级完全正常。所以单看 max|diff| 分不开「累积舍入」和「真的算出了
    不同的场」，两者可以是同一个数量级。

    能分开的是**分布**：舍入是随机散布的少数点，真实的数值改动会让绝大多数点都
    偏离。所以按超容差点占比分档：

        占比 = 0             IDENTICAL
        占比 < 1e-3          CLOSE    随机舍入
        1e-3 ~ 0.5           DRIFT    系统性偏移但未波及全场
        > 0.5                DIFFER   多数点都变了

    另外若 max|diff| 超过场自身空间变率的 1%，无论占比多少都判 DIFFER —— 那已经是
    物理上可分辨的差异。
    """
    if max_diff == 0.0:
        return "IDENTICAL"
    if ref_std > 0 and max_diff > DRIFT_STD_FRACTION * ref_std:
        return "DIFFER"
    if exceed_fraction < EXCEED_FRACTION_CLOSE:
        return "CLOSE"
    if exceed_fraction < EXCEED_FRACTION_DRIFT:
        return "DRIFT"
    return "DIFFER"


def _compare_field(a, b, lat, rtol, atol):
    """两个 (H,W) 场 -> 指标 dict。NaN 位置不一致会被单独标出。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    nan_mismatch = int(np.count_nonzero(finite_a != finite_b))
    both = finite_a & finite_b
    if not both.any():
        return {
            "verdict": "DIFFER",
            "max_abs_diff": float("nan"),
            "rms_diff": float("nan"),
            "max_rel_diff": float("nan"),
            "diff_over_std": float("nan"),
            "n_exceed": -1,
            "exceed_frac": float("nan"),
            "nan_mismatch": nan_mismatch,
            "ref_mean": float("nan"),
            "ref_std": float("nan"),
        }

    diff = np.abs(a - b)
    diff_valid = diff[both]
    max_diff = float(diff_valid.max())

    # 纬度加权 RMS：高纬格点面积小，不加权会高估极区的贡献
    weights = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))[:, None]
    w = np.where(both, np.broadcast_to(weights, a.shape), 0.0)
    denom = float(w.sum())
    rms_diff = float(np.sqrt((w * np.where(both, (a - b) ** 2, 0.0)).sum() / denom)) \
        if denom > 0 else float("nan")

    ref = b[both]
    ref_absmax = float(np.abs(ref).max())
    ref_std = float(ref.std())
    ref_mean = float(ref.mean())

    # 逐点相对差（分母加 atol 避免 0 除）
    max_rel = float((diff_valid / (np.abs(ref) + atol)).max())
    tol = atol + rtol * np.abs(ref)
    n_exceed = int(np.count_nonzero(diff_valid > tol))
    n_valid = int(diff_valid.size)
    exceed_fraction = n_exceed / n_valid if n_valid else 0.0

    return {
        "verdict": _classify(max_diff, ref_std, exceed_fraction),
        "max_abs_diff": max_diff,
        "rms_diff": rms_diff,
        "max_rel_diff": max_rel,
        "diff_over_std": max_diff / ref_std if ref_std > 0 else float("nan"),
        "n_exceed": n_exceed,
        "exceed_frac": exceed_fraction,
        "nan_mismatch": nan_mismatch,
        "ref_mean": ref_mean,
        "ref_std": ref_std,
    }


def compare(root_a, root_b, requested_vars=None, step_limit=None,
            rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, lat=None,
            progress=True, progress_every=50):
    """对比两个目录，返回 (rows, skipped)。

    rows    每个 (起报, step, 变量) 一条记录
    skipped 只在一侧存在的东西（起报/步/变量），供末尾提示

    整年目录会有上万对文件、每个文件几十 MB，所以：变量清单与网格只在第一对上
    探测/校验一次（同一次运行内不会变），并按 progress_every 打进度。想快就用
    ``--vars`` / ``--steps`` 缩小范围。
    """
    inits_a = _discover_init_dirs(root_a)
    inits_b = _discover_init_dirs(root_b)
    if not inits_a:
        raise SystemExit(f"{root_a} 下没有起报目录")
    if not inits_b:
        raise SystemExit(f"{root_b} 下没有起报目录")

    skipped = []
    only_a = sorted(set(inits_a) - set(inits_b))
    only_b = sorted(set(inits_b) - set(inits_a))
    for name in only_a:
        skipped.append(("起报", name, "只在 A 侧"))
    for name in only_b:
        skipped.append(("起报", name, "只在 B 侧"))

    common_inits = sorted(set(inits_a) & set(inits_b))
    if not common_inits:
        raise SystemExit(
            f"两侧没有共同的起报目录。\n  A: {sorted(inits_a)}\n  B: {sorted(inits_b)}")

    # 先数出总工作量，好在开跑前就把规模告诉用户（整年目录很容易上万对）
    total_pairs = 0
    for init_dir in common_inits:
        steps_a = set(_discover_steps(os.path.join(root_a, init_dir)))
        steps_b = set(_discover_steps(os.path.join(root_b, init_dir)))
        common = steps_a & steps_b
        if step_limit is not None:
            common = {s for s in common if s <= step_limit}
        total_pairs += len(common)
    if progress:
        print(f"共同起报 {len(common_inits)} 个，待比 {total_pairs} 对文件"
              f"{'（可用 --vars / --steps 缩小范围）' if total_pairs > 500 else ''}",
              flush=True)

    var_cache = None
    grid_checked = False
    done = 0
    t_start = perf_counter()

    rows = []
    for init_dir in common_inits:
        dir_a = os.path.join(root_a, init_dir)
        dir_b = os.path.join(root_b, init_dir)
        steps_a = set(_discover_steps(dir_a))
        steps_b = set(_discover_steps(dir_b))
        for step in sorted(steps_a - steps_b):
            skipped.append(("步", f"{init_dir}/{step:03d}", "只在 A 侧"))
        for step in sorted(steps_b - steps_a):
            skipped.append(("步", f"{init_dir}/{step:03d}", "只在 B 侧"))

        common_steps = sorted(steps_a & steps_b)
        if step_limit is not None:
            common_steps = [s for s in common_steps if s <= step_limit]

        for step in common_steps:
            path_a = os.path.join(dir_a, f"{step:03d}.nc")
            path_b = os.path.join(dir_b, f"{step:03d}.nc")
            # 变量清单只在第一对文件上探测一次并缓存 —— 同一次运行里所有步的变量
            # 集合是一样的。原先每对都探测两次，加上后面 load 两次，等于每对文件
            # 开 4 次；整年目录（365 起报 × 60 步）下这就是 8.7 万次 NetCDF 打开。
            if var_cache is None:
                vars_a = _variables_in(path_a)
                vars_b = _variables_in(path_b)
                common_vars = vars_a & vars_b
                if requested_vars:
                    for name in sorted(requested_vars - common_vars):
                        skipped.append(
                            ("变量", name, "两侧未同时存在"))
                    common_vars = common_vars & requested_vars
                else:
                    for name in sorted(vars_a ^ vars_b):
                        skipped.append(("变量", name, "只在一侧"))
                var_cache = sorted(common_vars)
            names = var_cache
            if not names:
                continue

            fields_a, lat_a, lon_a = load_prediction_file(path_a, names)
            fields_b, lat_b, lon_b = load_prediction_file(path_b, names)
            # 网格只在第一对文件上校验：同一次运行内所有文件同网格，逐对
            # allclose 是纯浪费（721×1440 两次比较 × 2 万对）。
            if not grid_checked:
                if lat_a.shape != lat_b.shape or lon_a.shape != lon_b.shape:
                    raise SystemExit(f"网格 shape 不一致：{path_a} 与 {path_b}")
                if not (np.allclose(lat_a, lat_b, atol=1e-4)
                        and np.allclose(lon_a, lon_b, atol=1e-4)):
                    raise SystemExit(f"网格坐标不一致：{path_a} 与 {path_b}")
                grid_checked = True

            for name in names:
                metrics = _compare_field(
                    fields_a[name], fields_b[name], lat_a, rtol, atol)
                rows.append({
                    "init": init_dir,
                    "step": step,
                    "var": name,
                    **metrics,
                })

            done += 1
            if progress and (done == 1 or done % progress_every == 0
                             or done == total_pairs):
                elapsed = perf_counter() - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total_pairs - done) / rate if rate > 0 else float("nan")
                print(f"  已比 {done}/{total_pairs} 对文件"
                      f"（{init_dir}/{step:03d}）"
                      f" {rate:.1f} 对/秒，预计还需 {eta / 60:.1f} 分钟",
                      flush=True)
    return rows, skipped


def _print_report(rows, skipped, quiet=False):
    """按 变量×step 聚合打印；只有 --quiet 关掉明细。"""
    if not rows:
        return
    frame = pd.DataFrame(rows)
    order = {"IDENTICAL": 0, "CLOSE": 1, "DRIFT": 2, "DIFFER": 3}

    # 每个变量取最坏一档 + 最大差异
    print("\n=== 按变量汇总（跨全部起报与步）===")
    per_var = (
        frame.assign(_rank=frame["verdict"].map(order))
        .groupby("var", as_index=False)
        .agg(最坏判定=("_rank", "max"),
             max_abs_diff=("max_abs_diff", "max"),
             diff_over_std=("diff_over_std", "max"),
             超容差点占比=("exceed_frac", "max"),
             场变率std=("ref_std", "mean"),
             n=("var", "size"))
        .sort_values("最坏判定", ascending=False)
    )
    inv = {v: k for k, v in order.items()}
    per_var["最坏判定"] = per_var["最坏判定"].map(inv)
    print(per_var.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    if not quiet:
        bad = frame[frame["verdict"].isin(["DRIFT", "DIFFER"])]
        if len(bad):
            print(f"\n=== DRIFT / DIFFER 明细（{len(bad)} 条，最多列 40）===")
            cols = ["init", "step", "var", "verdict", "max_abs_diff",
                    "diff_over_std", "exceed_frac", "nan_mismatch"]
            print(bad[cols].head(40).to_string(
                index=False, float_format=lambda v: f"{v:.6g}"))

    nan_bad = frame[frame["nan_mismatch"] > 0]
    if len(nan_bad):
        print(f"\n!! {len(nan_bad)} 条记录的 NaN 位置不一致 —— "
              "这通常不是浮点漂移，而是缺测处理路径变了")

    if skipped:
        print(f"\n=== 跳过（只在一侧存在）：{len(skipped)} 项 ===")
        by_kind = {}
        for kind, what, why in skipped:
            by_kind.setdefault((kind, why), []).append(what)
        for (kind, why), items in sorted(by_kind.items()):
            shown = ", ".join(items[:8])
            more = f" …… 共 {len(items)} 项" if len(items) > 8 else ""
            print(f"  {kind}（{why}）：{shown}{more}")

    counts = frame["verdict"].value_counts()
    print("\n=== 结论 ===")
    for verdict in ("IDENTICAL", "CLOSE", "DRIFT", "DIFFER"):
        if verdict in counts:
            print(f"  {verdict:10s} {counts[verdict]:5d} / {len(frame)}")
    worst = max(frame["verdict"], key=lambda v: order[v])
    if worst in ("IDENTICAL", "CLOSE"):
        print("\n两次结果一致（差异在容差内），可认为改动没有影响数值。")
    elif worst == "DRIFT":
        print("\n存在漂移：超出容差，但相对于场自身的空间变率仍极小。"
              "若本次改动确实动了浮点路径（如代数化简、算子融合），这是预期的；"
              "否则需要查。")
    else:
        print("\n结果不一致。先看上面 DIFFER 的变量和 lead，"
              "再用 --vars 单独盯那个变量、看是从哪一步开始偏的。")
    return worst


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="对比两个预测输出目录的 NetCDF（可复现性检查，不需要实况）")
    parser.add_argument("dir_a", help="第一个预测输出根目录")
    parser.add_argument("dir_b", help="第二个预测输出根目录（作为参考侧）")
    parser.add_argument(
        "--vars", default=None,
        help="只比指定变量，逗号分隔；缺省比两侧共同的全部二维变量")
    parser.add_argument(
        "--steps", type=int, default=None,
        help="只比步序号不大于该值的预测步")
    parser.add_argument(
        "--rtol", type=float, default=DEFAULT_RTOL,
        help=f"相对容差（默认 {DEFAULT_RTOL:g}）")
    parser.add_argument(
        "--atol", type=float, default=DEFAULT_ATOL,
        help=f"绝对容差，给 tp 这类大片为 0 的场兜底（默认 {DEFAULT_ATOL:g}）")
    parser.add_argument(
        "--out", default=None, help="可选：把明细写成 CSV 到该目录")
    parser.add_argument(
        "--quiet", action="store_true", help="只打汇总，不打逐条明细")
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s")

    for d in (args.dir_a, args.dir_b):
        if not os.path.isdir(d):
            raise SystemExit(f"目录不存在：{d}")
    if args.rtol < 0 or args.atol < 0:
        raise SystemExit("--rtol / --atol 不能为负")

    requested = None
    if args.vars:
        requested = {v.strip().lower() for v in args.vars.split(",") if v.strip()}

    print(f"A（被检侧）: {args.dir_a}")
    print(f"B（参考侧）: {args.dir_b}")
    print(f"容差: |a-b| <= {args.atol:g} + {args.rtol:g}·|b|")

    rows, skipped = compare(
        args.dir_a, args.dir_b,
        requested_vars=requested, step_limit=args.steps,
        rtol=args.rtol, atol=args.atol)

    if not rows:
        print("\n没有可比对象：两侧没有同时存在的 (起报, 步, 变量) 组合。")
        if skipped:
            for kind, what, why in skipped[:20]:
                print(f"  跳过 {kind} {what}（{why}）")
        return 2

    worst = _print_report(rows, skipped, quiet=args.quiet)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "compare_outputs.csv")
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.8g")
        print(f"\n明细已写入 {path}")

    return 0 if worst in ("IDENTICAL", "CLOSE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
