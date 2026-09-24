#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SPW 超参数敏感性复盘：把已有产物按「旋钮」重算（perturbation/README 的 2G 节由它产生）。

三个子命令互相独立，**都不跑推理**：

    balance <onnx>                    配平口径算术：energy vs count 各给多少 σ
    members <out_dir> [...]           成员数 M 的隐藏偏差：同一份产物抽 M 个成员，σ̂ 怎么变
    power <curves 目录> --sigma ...   从曲线 CSV 反解 σ 幂律 p = d log σ̂ / d log σ

为什么要有它（2G 节的三条硬结论）：

  * **成员数是个隐藏超参数**：4 vs 6 成员的 σ̂ 差 1-14%，2 成员差 20-30% 而且符号不定
    （实测 B/F 偏低 26-35%、H 偏高 18-31%，而纯采样理论一律偏低 16%）—— 所以跨变体比
    σ̂ 之前，先在同一份产物上用 `members` 量一遍成员数偏差，**只有 ≥2 倍的差别能当结论**。
  * **配平口径不是记号**：同一组上 `energy` 与论文原式 `count` 的 σ 差 0.57-1.22 倍
    （根因是 FGVP 的 6 个 `dt_bias`：9216 个元素占候选能量 45.8%）—— 对比 σ 之前先确认口径。
  * **σ 的响应依赖注入点**：`decoder_group` 是线性旋钮（p≈1.0），`latent_mixer` 早线性、
    长时效饱和（p 从 0.93 掉到 0.37）—— `power` 就是量这个的。

用法：

    python util/spw_hparam.py balance model_artifacts/fgvp/iwc_fgvp_gdn2.onnx
    python util/spw_hparam.py balance model_artifacts/fgvp/iwc_fgvp_gdn2.onnx --extra-exclude dt_bias
    python util/spw_hparam.py members /path/spw_ab/F /path/spw_ab/H --steps 8
    python util/spw_hparam.py power /path/spw_ab/curves --sigma A=0.0251885 B=0.0114498 F=0.0457992

`members` 读的是**推理产物目录**（`{起报}/member_XXX/`），所以也能直接喂 51 成员的正式跑
产物（那就是 2F.6 第 2 条要的「σ̂@51 vs σ̂@6」）。口径与 `perturbation spread` 完全一致
（同一份 `analyse_run`），只是把 M 扫一遍并加上纯采样理论的对照列。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import os.path as osp
import sys

DEFAULT_ONNX = osp.join("model_artifacts", "fgvp", "iwc_fgvp_gdn2.onnx")
DEFAULT_GROUPS = "latent_mixer,decoder_group,vae_encoder_group"


def _c4(m: int) -> float:
    """无偏正态总体下样本 std 的期望因子 E[s_M]/σ（M=2 就是 √(2/π)≈0.798）。"""
    return math.sqrt(2.0 / (m - 1)) * math.exp(math.lgamma(m / 2.0) - math.lgamma((m - 1) / 2.0))


# --------------------------------------------------------------------------- #
# balance：配平口径算术
# --------------------------------------------------------------------------- #
def cmd_balance(args) -> int:
    from xmetai_inference.perturbation.graph import load_model, select_initializers
    from xmetai_inference.perturbation.spec import WeightPerturbSpec

    onnx = args.onnx or DEFAULT_ONNX
    if not osp.isfile(onnx):
        print(f"找不到 onnx：{onnx}（用位置参数指路径）", file=sys.stderr)
        return 2
    src_dir = osp.dirname(osp.abspath(onnx))
    model = load_model(onnx)
    extra = tuple(t.strip() for t in (args.extra_exclude or "").split(",") if t.strip())

    spec = WeightPerturbSpec(mechanism="m1a", groups=tuple(args.groups.split(",")),
                             min_elements=args.min_elements)
    if extra:
        spec.exclude = tuple(spec.exclude) + extra
    sel = select_initializers(model, spec, src_dir)

    total_e, total_n = sel.candidate_energy, sel.candidate_elements
    print(f"onnx            : {onnx}")
    print(f"候选（过滤后）  : {len(sel.selected)} 张量 / {total_n/1e6:.2f}M 元素 / SigmaW2={total_e:.6g}")
    print(f"sigma_full      : {args.sigma_full:g}")
    if extra:
        print(f"额外 exclude    : {list(extra)}")
    print()
    head = (f"{'组':>20} {'张量':>5} {'元素(M)':>9} {'能量占比':>8} "
            f"{'sigma@energy':>13} {'sigma@count':>12} {'count/energy':>13}")
    print(head)
    print("-" * len(head))
    for group in sorted(sel.group_energy, key=lambda g: -sel.group_energy[g]):
        energy, elements = sel.group_energy[group], sel.group_elements[group]
        s_e = args.sigma_full * math.sqrt(total_e / energy)
        s_c = args.sigma_full * math.sqrt(total_n / elements)
        count = sum(1 for c in sel.selected.values() if c.group == group)
        print(f"{group:>20} {count:>5} {elements/1e6:>9.2f} {energy/total_e:>7.1%} "
              f"{s_e:>13.6f} {s_c:>12.6f} {s_c/s_e:>12.2f}x")
    print()
    print("怎么读：sigma@energy 就是 config 里 balance='energy' 实际用的值（本仓库默认），")
    print("        sigma@count 是论文原式。两列差 1 倍以上就是换了一次 sigma 扫描 ——")
    print("        对比不同实现的 sigma 之前先确认口径。count 口径对「元素少但权重大」的组")
    print("        会给出离谱的 sigma（FGVP 的 dt_bias 正是如此：9216 个元素占 45.8% 能量）。")
    return 0


# --------------------------------------------------------------------------- #
# members：成员数 M 的隐藏偏差
# --------------------------------------------------------------------------- #
def cmd_members(args) -> int:
    from xmetai_inference.perturbation.spread import analyse_run

    requested = list(range(2, args.max_m + 1))
    for root in args.paths:
        print("=" * 78)
        print(root)
        print("=" * 78)
        curves = {}
        limits = requested
        for m in requested:
            info = analyse_run(root, var=args.var, init=args.init,
                               steps_spec=str(args.steps), verbose=False,
                               members=f"first:{m}")
            # 该产物**真正能用**的成员数（缺帧的成员会让这个数小于目录数）
            used = max(row["members"] for row in info["rows"])
            if used < m:
                limits = [k for k in requested if k <= used]
                print(f"  （产物每步最多 {used} 个成员：M 列表截到 "
                      f"M=2..{limits[-1]}，再大没有意义）")
                break
            curves[m] = {row["lead_h"]: row["sigma"] for row in info["rows"]}
        best = limits[-1]
        leads = sorted(curves[limits[0]])
        print(f"{'lead':>6} " + " ".join(f"{'M=' + str(m):>10}" for m in limits)
              + f"{'M2/M' + str(best):>10}{'理论':>8}")
        for lead in leads:
            vals = [curves[m].get(lead) for m in limits]
            if any(v is None for v in vals):
                continue
            print(f"{lead:>6.0f} " + " ".join(f"{v:>10.2f}" for v in vals)
                  + f"{vals[0]/vals[-1]:>10.3f}{_c4(2)/_c4(best):>8.3f}")
        ratios = [curves[2][lead] / curves[best][lead] for lead in leads
                  if curves[2].get(lead) and curves[best].get(lead)]
        if ratios:
            print(f"  M=2 相对 M={best}：{min(ratios):.3f} .. {max(ratios):.3f}"
                  f"（纯采样理论 {_c4(2)/_c4(best):.3f}）；中位 "
                  f"{sorted(ratios)[len(ratios)//2]:.3f}")
        print("  纯采样因子 c4(M)：  "
              + " ".join(f"M={m}:{_c4(m):.4f}" for m in limits))
        print("  实测偏离理论越远、甚至反号，说明 sigma_hat 由**哪几个成员被抽到**主导")
        print("  （不是样本量问题）—— 这就是「跨变体比较前必须先对齐成员数」的原因。")
        print()
    return 0


# --------------------------------------------------------------------------- #
# power：sigma 幂律
# --------------------------------------------------------------------------- #
def _read_curves(path):
    """读一个曲线 CSV，或一个目录（目录里每个 *.csv 当一个变体）。"""
    if osp.isdir(path):
        out = {}
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(".csv"):
                out[name[:-4]] = _read_curves(osp.join(path, name))
        return out
    with open(path, newline="", encoding="utf-8") as handle:
        return {int(round(float(row["lead_h"]))): row for row in csv.DictReader(handle)}


def cmd_power(args) -> int:
    table = _read_curves(args.curves)
    sigmas = {}
    for token in args.sigma:
        if "=" not in token:
            print(f"--sigma 要写成 TAG=值，读不懂 {token!r}", file=sys.stderr)
            return 2
        tag, value = token.split("=", 1)
        sigmas[tag.strip()] = float(value)
    missing = [tag for tag in sigmas if tag not in table]
    if missing:
        print(f"这些 TAG 在 {args.curves} 里没有曲线：{missing}（可选：{sorted(table)}）",
              file=sys.stderr)
        return 2
    tags = sorted(sigmas, key=lambda tag: -sigmas[tag])
    pairs = [(hi, lo, sigmas[hi] / sigmas[lo]) for index, hi in enumerate(tags)
             for lo in tags[index + 1:]]
    leads = sorted(set.intersection(*[set(table[tag]) for tag in tags]))
    print(args.curves)
    print("p = d log sigma_hat / d log sigma（1.0 = 线性幅度旋钮，<1 = 边际递减，")
    print("随时效掉得越快 = 长时效越补不动）")
    print()
    if len(pairs) > 6:
        print(f"⚠ 给了 {len(sigmas)} 个 TAG -> {len(pairs)} 个组合，表会很宽；"
              "建议只给**同一组**的两三个 TAG。")
    print("⚠ 只有**同组**的两个 TAG 之间的 p 才有意义：跨组时 p 混着「换了注入点」这个因素")
    print("  （例：D/F 这种比法没有物理含义）。")
    print()
    print(f"{'lead':>6} " + " ".join(f"{hi}/{lo} (x{ratio:g})".rjust(18)
                                     for hi, lo, ratio in pairs))
    for lead in leads:
        cells = []
        for hi, lo, ratio in pairs:
            a = float(table[hi][lead]["sigma"])
            b = float(table[lo][lead]["sigma"])
            cells.append(f"{math.log(a/b)/math.log(ratio):>18.2f}" if a > 0 and b > 0
                         else f"{'-':>18}")
        print(f"{lead:>6} " + " ".join(cells))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SPW 超参数敏感性复盘（balance / members / power）")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("balance", help="配平口径算术：energy vs count 各给多少 sigma")
    b.add_argument("onnx", nargs="?", default=None,
                   help=f"源确定性 onnx（默认 {DEFAULT_ONNX}）")
    b.add_argument("--groups", default=DEFAULT_GROUPS, help="逗号分隔的组前缀")
    b.add_argument("--sigma-full", type=float, default=0.01, help="全权重口径的 sigma")
    b.add_argument("--extra-exclude", default=None,
                   help="在默认 exclude 上再加的项，如 dt_bias")
    b.add_argument("--min-elements", type=int, default=64)
    b.set_defaults(func=cmd_balance)

    m = sub.add_parser("members", help="成员数 M 的影响：同一份产物抽 M 个成员看 sigma_hat")
    m.add_argument("paths", nargs="+", help="推理产物目录（{起报}/member_XXX/）")
    m.add_argument("--max-m", type=int, default=6, help="最大成员数（默认 6）")
    m.add_argument("--steps", default="8", help="算哪些步（整数 = 等距抽这么多个，all = 全算）")
    m.add_argument("--var", default=None, help="只看这个变量")
    m.add_argument("--init", default=None, help="只看这些起报目录，逗号分隔")
    m.set_defaults(func=cmd_members)

    p = sub.add_parser("power", help="从曲线 CSV 反解 sigma 幂律")
    p.add_argument("curves", help="curves 目录或单个曲线 CSV")
    p.add_argument("--sigma", nargs="+", required=True,
                   help="TAG=sigma 列表（TAG = 曲线 CSV 的文件名去掉 .csv）；"
                        "只给同一组的 TAG，如 B=0.0114498 F=0.0457992")
    p.set_defaults(func=cmd_power)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())