# -*- coding: utf-8 -*-
"""按归档调用序重放 aBt15 rank0：13 成员、6/6/1 分块、step 外层。

为什么需要它：FGVP 图里有 RandomNormalLike、没有外部 seed 输入。图内 ε 的
发生器状态在同一个 session 内随每次前向推进，所以归档 20250702 的 ε 只取决于：

1) 它是不是这个 session 的第一次起报 —— aBt15 的 times 是
   "2025070200..2025073000:24,2025110100..2025113000:24"，20250702 是第一个；
2) 每次前向的 batch 形状与调用顺序 —— rank0 按成员切拿到全局成员
   0,4,...,48（13 个），rollout_batch=6 ⇒ 每步 6/6/1 三次前向，step 在外层。

verify_example.py 的 4 成员单块调用序与归档不同，归档只能做统计对拍。
本脚本用完全相同的调用序重放 rank0，并对归档里有的成员 0/4/8/12 逐成员对拍：

    A vs B    = 同调用序的两个新 session，数值噪声下限；
    A vs 归档 = 链路正确时应落在同一量级（~1e-3）。

其余 9 个成员只参与前向、不保留输出，用来保持 ε 的调用序。

运行（服务器，项目根目录）：
    cd /workspace/szwCode/xmetai-inference2
    python verify_archive_replay.py \
      --model /workspace/data/shenzw/model_onnx/iwc_gssp_im_fp16.ar1-5ff029ed6d.onnx \
      --input /workspace/data/shenzw/inference_test_data/spa_test_data/fgvp_input_20250702.npy

如果 A vs 归档明显大于 A vs B：先看归档的 rank_0.log，确认那次运行的第一个
起报确实是 20250702、rollout_batch 确实是 6；这两个条件不成立时本重放无效。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import example  # noqa: E402
import verify_example as vx  # noqa: E402

# rank0 的成员集：全局成员 0,4,...,48（world_size=4 的 stride 切法）。
RANK_MEMBERS = 13
# cfg.rollout_batch=6 ⇒ chunks = [0..5]、[6..11]、[12]。
ROLLOUT_BATCH = 6
# 归档里有的成员：行号 = 本 rank 内的序号，全局成员号 = row * 4。
KEEP_ROWS = (0, 1, 2, 3)
ARCHIVE_MEMBERS = (0, 4, 8, 12)


def run_pass(model, xi_table, real_input, steps, input_names, label):
    """按归档的调用序跑一遍，只保留 KEEP_ROWS 的 5 个变量。"""
    session = example._make_session(model)
    members = xi_table.shape[0]
    chunks = [list(range(start, min(start + ROLLOUT_BATCH, members)))
              for start in range(0, members, ROLLOUT_BATCH)]
    states = [np.repeat(real_input[None], len(chunk), axis=0)
              for chunk in chunks]
    kept = {(row, step, var): None
            for row in KEEP_ROWS for step in range(steps)
            for var, _ in vx.CAPTURE}

    for step in range(steps):
        valid = example.INIT_TIME + timedelta(
            hours=step * example.HOUR_INTERVAL)
        for ci, rows in enumerate(chunks):
            feeds = {
                "input": states[ci],
                "spw/xi": xi_table[rows, step],
            }
            scalars = {
                "step": float(step),
                "hour": valid.hour / 24.0,
                "doy": min(365, valid.timetuple().tm_yday) / 365.0,
            }
            for name, value in scalars.items():
                if name in input_names:
                    feeds[name] = np.full(
                        (len(rows),), value, dtype=np.float32)
            missing = input_names - set(feeds)
            if missing:
                raise RuntimeError(f"以下输入没有喂：{sorted(missing)}")

            pred = vx._as_pred(session.run(["pred"], feeds)[0])
            for local, row in enumerate(rows):
                if row not in KEEP_ROWS:
                    continue
                for var, channel in vx.CAPTURE:
                    field = pred[local, 0, channel]
                    if var == "tp":
                        field = np.maximum(field, 0.0)
                    # 必须真拷贝：field 是 pred 的视图，ascontiguousarray
                    # 对已连续的输入会原样返回，把整个 1.9GB pred 留住。
                    kept[(row, step, var)] = np.array(
                        field, dtype=np.float32, copy=True)
            states[ci] = np.concatenate(
                [states[ci][:, 1:], pred], axis=1)
            print(f"  [{label}] step {step} 块 {ci}"
                  f"（{len(rows)} 成员）完成")
            del pred
            gc.collect()

    del states
    gc.collect()
    return session, kept


def load_refs(archive_dir, archive_steps, height, width):
    refs = {}
    for member in ARCHIVE_MEMBERS:
        for step_idx, step_no in enumerate(archive_steps):
            path = (archive_dir / f"member_{member:03d}"
                    / f"{step_no:03d}.nc")
            for var, _ in vx.CAPTURE:
                refs[(member, step_idx, var)] = vx._open_archive_var(
                    path, var, height, width)
    return refs


def main():
    # 必须在下面 --rollout-batch 的 default=ROLLOUT_BATCH 之前声明。
    global ROLLOUT_BATCH
    parser = argparse.ArgumentParser(
        description="按归档调用序重放 aBt15 rank0（13 成员，6/6/1）")
    parser.add_argument(
        "--model",
        default="/workspace/data/shenzw/model_onnx/"
                "iwc_gssp_im_fp16.ar1-5ff029ed6d.onnx")
    parser.add_argument(
        "--input",
        default="/workspace/data/shenzw/inference_test_data/spa_test_data/"
                "fgvp_input_20250702.npy")
    parser.add_argument(
        "--archive",
        default="/workspace/data/shenzw/fgvp_ens_ar1state_aBt15")
    parser.add_argument("--archive-init", default="20250702")
    parser.add_argument("--archive-steps", default="1,2,3,4")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--rank-members", type=int, default=RANK_MEMBERS)
    parser.add_argument("--rollout-batch", type=int, default=ROLLOUT_BATCH)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cudnn", default="EXHAUSTIVE")
    parser.add_argument("--ops-library", default=example.OPS_LIBRARY)
    parser.add_argument("--input-index", type=int, default=0)
    parser.add_argument("--rel-rmse", type=float, default=1e-3,
                        help="确定性下限；最终阈值还与 A/B 噪声下限挂钩")
    parser.add_argument("--noise-factor", type=float, default=2.0)
    parser.add_argument("--report", default="verify_archive_replay.json")
    args = parser.parse_args()

    ROLLOUT_BATCH = int(args.rollout_batch)
    if args.rank_members < max(KEEP_ROWS) + 1:
        raise SystemExit("--rank-members 至少要覆盖要比较的行号")

    example.OPS_LIBRARY = args.ops_library
    example.DEVICE_ID = args.device
    example.CUDNN_SEARCH = args.cudnn

    archive_steps = [
        int(token) for token in args.archive_steps.split(",") if token.strip()]
    if len(archive_steps) != args.steps:
        raise SystemExit(
            f"--archive-steps 给了 {len(archive_steps)} 个，"
            f"--steps={args.steps}")

    # 先建一次 session 拿图信息；正式跑 A 时再建（保持两个 session 都是新的）。
    probe = example._make_session(args.model)
    frames, channels, height, width, n_targets, rank, input_names = (
        example._check_graph(probe))
    print(f"[graph] input={frames}x{channels}x{height}x{width} "
          f"T={n_targets} R={rank}")
    print(f"[graph] providers={probe.get_providers()}")
    del probe
    gc.collect()

    real_input = vx.load_input(
        args.input, channels, height, width, args.input_index)
    finite = np.isfinite(real_input)
    print(f"[input] shape={real_input.shape} finite={finite.mean():.6f}")
    if not finite.all():
        raise SystemExit("输入里有非有限值")

    xi_table = vx.build_exact_xi(
        n_targets, rank, args.steps, args.rank_members)
    print(f"[xi] rank0 流 {xi_table.shape[0]} 行；"
          f"chunks={[min(ROLLOUT_BATCH, args.rank_members - start) for start in range(0, args.rank_members, ROLLOUT_BATCH)]}")

    print("\n[pass A]")
    session_a, kept_a = run_pass(
        args.model, xi_table, real_input, args.steps, input_names, "A")
    del session_a
    gc.collect()

    print("\n[pass B]")
    session_b, kept_b = run_pass(
        args.model, xi_table, real_input, args.steps, input_names, "B")
    del session_b
    gc.collect()

    archive_dir = Path(args.archive) / args.archive_init
    refs = load_refs(archive_dir, archive_steps, height, width)

    print("\n[compare] 逐成员逐变量（rel = RMSE / std(归档参考输出)）")
    rows = []
    for member, row in zip(ARCHIVE_MEMBERS, KEEP_ROWS):
        for step_idx in range(args.steps):
            for var, _ in vx.CAPTURE:
                a = kept_a[(row, step_idx, var)]
                b = kept_b[(row, step_idx, var)]
                ref = refs[(member, step_idx, var)]
                m_ab = vx.compare(a, b)
                m_arch = vx.compare(a, ref)
                rows.append({
                    "member": member,
                    "row": row,
                    "step": step_idx,
                    "archive_step": archive_steps[step_idx],
                    "var": var,
                    "a_vs_b": m_ab,
                    "a_vs_archive": m_arch,
                })
                print(f"  m{member} step {step_idx}: {var:5s} "
                      f"A/B={m_ab['rel']:.3e} "
                      f"A/arc={m_arch['rel']:.3e}")

    print("\n[compare] 4 成员集合均值")
    mean_rows = []
    for step_idx in range(args.steps):
        for var, _ in vx.CAPTURE:
            a = np.mean([kept_a[(row, step_idx, var)]
                         for row in KEEP_ROWS], axis=0)
            b = np.mean([kept_b[(row, step_idx, var)]
                         for row in KEEP_ROWS], axis=0)
            ref = np.mean([refs[(member, step_idx, var)]
                           for member in ARCHIVE_MEMBERS], axis=0)
            m_ab = vx.compare(a, b)
            m_arch = vx.compare(a, ref)
            mean_rows.append({
                "step": step_idx,
                "archive_step": archive_steps[step_idx],
                "var": var,
                "a_vs_b": m_ab,
                "a_vs_archive": m_arch,
            })
            print(f"  step {step_idx}: {var:5s} "
                  f"A/B={m_ab['rel']:.3e} "
                  f"A/arc={m_arch['rel']:.3e}")

    print("\n[verdict] 阈值 = max(--rel-rmse, --noise-factor x A/B)")
    verdict = {}
    for var, _ in vx.CAPTURE:
        noise = max(m["a_vs_b"]["rel"] for m in rows
                    if m["var"] == var and math.isfinite(m["a_vs_b"]["rel"]))
        arch = max(m["a_vs_archive"]["rel"] for m in rows
                   if m["var"] == var
                   and math.isfinite(m["a_vs_archive"]["rel"]))
        limit = max(float(args.rel_rmse), args.noise_factor * noise)
        ok = math.isfinite(arch) and arch <= limit
        verdict[var] = {
            "max_a_vs_b": noise,
            "max_a_vs_archive": arch,
            "limit": limit,
            "ok": bool(ok),
        }
        print(f"  {var:5s} A/B={noise:.3e} "
              f"A/arc={arch:.3e} limit={limit:.3e} "
              f"{'OK' if ok else 'MISMATCH'}")

    match = all(item["ok"] for item in verdict.values())
    status = "MATCH" if match else "MISMATCH"
    print(f"\nRESULT: {status}")
    if match:
        print("  归档调用序下 example 链路复现了归档（差异在 A/B 数值噪声量级）。")
    else:
        print("  归档没有按预期复现：先核对 rank_0.log 里的第一个起报是否为"
              " 20250702、rollout_batch 是否为 6；再查 ξ 流/amp/tau/成员映射。")

    report = {
        "status": status,
        "match": match,
        "model": args.model,
        "input": args.input,
        "archive": {
            "root": args.archive,
            "init": args.archive_init,
            "steps": archive_steps,
            "members": list(ARCHIVE_MEMBERS),
        },
        "call_order": {
            "rank_members": args.rank_members,
            "rollout_batch": ROLLOUT_BATCH,
            "chunks": [
                [row for row in range(start, min(
                    start + ROLLOUT_BATCH, args.rank_members))]
                for start in range(0, args.rank_members, ROLLOUT_BATCH)
            ],
            "first_init": "2025070200（cfg.times 的第一个）",
        },
        "threshold_rel_rmse": args.rel_rmse,
        "threshold_noise_factor": args.noise_factor,
        "per_member": rows,
        "ensemble_mean": mean_rows,
        "verdict": verdict,
    }
    out = Path(args.report)
    out.write_text(
        json.dumps(vx._clean(report), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"报告：{out}")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
