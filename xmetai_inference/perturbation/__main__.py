# -*- coding: utf-8 -*-
"""扰动 / 分析命令：probe / self-test / e2e / check / spread / verdict。

造产物的动作已经移出命令行：直接改 `xmetai_inference/weights/spw.py` 或
`weights/spa.py` 末尾 `if __name__ == "__main__"` 里的字面量再运行：

    python -m xmetai_inference.weights.spw     # groups / members
    python -m xmetai_inference.weights.spa     # latent build / audit

本模块只保留需要按次传参的分析与测试入口：

    python -m xmetai_inference.perturbation probe   model.onnx --m1a-members 8
    python -m xmetai_inference.perturbation self-test
    python -m xmetai_inference.perturbation e2e --members 51
    python -m xmetai_inference.perturbation check /path/to/output_dir
    python -m xmetai_inference.perturbation spread out/A out/B --csv out/curves
    python -m xmetai_inference.perturbation verdict F S T V E3
"""
from __future__ import annotations

import argparse
import sys

from ..analysis.verdict import DEFAULT_BASE, DEFAULT_PREFIX, DEFAULT_VAR
from ..weights.spw import DEFAULT_EXCLUDE, WeightPerturbSpec


def _spec_from_args(args) -> WeightPerturbSpec:
    return WeightPerturbSpec(
        members=getattr(args, "members", 0) or 0,
        member_dir=getattr(args, "member_dir", None),
        groups=tuple(token.strip() for token in args.groups.split(",") if token.strip()),
        sigma_full=args.sigma_full,
        sigma=args.sigma,
        balance=args.balance,
        exclude=(tuple(token.strip() for token in args.exclude.split(","))
                 if args.exclude else DEFAULT_EXCLUDE),
        include=(tuple(token.strip() for token in args.include.split(","))
                 if args.include else ()),
        min_elements=args.min_elements,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )


def _cmd_self_test(args) -> int:
    from .selftest import run_self_test

    return run_self_test(verbose=not args.quiet, keep=args.keep)


def _cmd_e2e(args) -> int:
    from .e2e import run_e2e

    return run_e2e(
        members=args.members,
        steps=args.steps,
        verbose=not args.quiet,
        keep=args.keep,
    )


def _cmd_probe(args) -> int:
    """探针：量噪声在输出上的量级与线性度。"""
    from ..analysis.probe import format_member_scan, format_probe, member_scan, probe

    spec = _spec_from_args(args)
    common = dict(provider=args.provider, device=args.device,
                  ops_library=args.ops_library, seed=args.seed,
                  feeds_mode=args.feeds, per_channel=args.per_channel)
    if args.m1a_members:
        print(format_member_scan(member_scan(args.model, spec, args.m1a_members,
                                             runs=1, **common)))
        return 0
    report = probe(args.model, runs=args.runs, reference=args.reference, **common)
    print(format_probe(report, per_channel=args.per_channel))
    return 0 if report.get("ok") else 1


def _cmd_check(args) -> int:
    from ..analysis.enscheck import run_check

    return run_check(
        args.path,
        init=args.init,
        var=args.var,
        stride=args.stride,
        steps_spec=str(args.steps).strip().lower(),
        expect_members=args.expect_members,
        hour_interval=args.hour_interval,
    )


def _cmd_spread(args) -> int:
    from ..analysis.spread import run_spread

    return run_spread(
        args.paths,
        var=args.var,
        init=args.init,
        steps_spec=str(args.steps).strip().lower(),
        stride=args.stride,
        hour_interval=args.hour_interval,
        every=args.every,
        csv_out=args.csv,
        members=args.members,
    )


def _cmd_verdict(args) -> int:
    from ..analysis.verdict import run_verdict

    dirs = {}
    for token in str(args.dir or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise SystemExit("--dir 要写成「臂=路径」，逗号分隔")
        name, path = token.split("=", 1)
        dirs[name.strip()] = path.strip()
    p_sigma = (0.01, 0.04)
    if args.p_sigma:
        parts = [float(part) for part in str(args.p_sigma).split(",") if part.strip()]
        if len(parts) != 2:
            raise SystemExit("--p-sigma 要写成「小,大」两个数")
        p_sigma = (parts[0], parts[1])
    return run_verdict(
        args.arms,
        base=args.base,
        prefix=args.prefix,
        dirs=dirs,
        ref=args.ref,
        var=args.var,
        region=args.region,
        p_from=args.p_from,
        p_sigma=p_sigma,
        out_csv=args.csv,
        check=not args.no_check,
    )


def _add_spec_args(parser) -> None:
    """probe 的注入规格参数。"""
    parser.add_argument("--groups", default="all", help="逗号分隔的组前缀，默认 all")
    parser.add_argument("--sigma", type=float, default=None, help="直接指定 σ（跳过自动配平）")
    parser.add_argument("--sigma-full", type=float, default=0.01, help="全权重口径的 σ")
    parser.add_argument("--balance", default="energy",
                        choices=["energy", "count", "none"], help="组间方差配平口径")
    parser.add_argument("--exclude", default=None,
                        help=f"排除模式，逗号分隔；默认 {','.join(DEFAULT_EXCLUDE)}")
    parser.add_argument("--include", default=None,
                        help="强制纳入模式；命中即视为权重，且不受 min_elements 限制")
    parser.add_argument("--min-elements", type=int, default=64,
                        help="小于该元素数的张量跳过（include 命中的不受此限）")
    parser.add_argument("--seed", type=int, default=None, help="固定随机序列")
    parser.add_argument("--cache-dir", default=None, help="输出目录；默认源模型同目录")


def main(argv=None) -> int:
    # Windows 控制台默认 GBK，输出里的符号会直接抛 UnicodeEncodeError；
    # 能改就改成 UTF-8，改不了（重定向到文件等）就继续用原编码。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="扰动分析 / 测试命令（造产物请直接运行 weights 模块）")
    sub = parser.add_subparsers(dest="command", required=True)

    self_test = sub.add_parser(
        "self-test", help="在合成小图上自检：σ→0 无损 / 注入方差定律 / 外部数据不被复制")
    self_test.add_argument("--keep", action="store_true", help="保留临时目录便于排查")
    self_test.add_argument("--quiet", action="store_true", help="只打印失败项与总结")
    self_test.set_defaults(func=_cmd_self_test)

    e2e = sub.add_parser(
        "e2e", help="端到端预检：真实框架路径下 N 个成员是否互不相同、σ 是否可调、"
                    "各 rank 是否独立")
    e2e.add_argument("--members", type=int, default=51, help="成员数（默认 51）")
    e2e.add_argument("--steps", type=int, default=6, help="rollout 步数（默认 6）")
    e2e.add_argument("--keep", action="store_true", help="保留临时工作目录")
    e2e.add_argument("--quiet", action="store_true", help="只打印失败项与总结")
    e2e.set_defaults(func=_cmd_e2e)

    probe_cmd = sub.add_parser(
        "probe", help="量噪声在输出上的量级与线性度（不需要实况、不需要 rollout）")
    probe_cmd.add_argument("model", help="要探的 onnx（确定性图 / 成员图）")
    probe_cmd.add_argument("--runs", type=int, default=3,
                           help="同输入连跑几次（>=2 才能量出离散度；默认 3）")
    probe_cmd.add_argument("--reference", default=None,
                           help="参照模型：额外报一次「与参照的相对 RMS 差异」（成员图用）")
    probe_cmd.add_argument("--m1a-members", type=int, default=0,
                           help="探一个成员集的全部成员（与源模型比对）")
    probe_cmd.add_argument("--ops-library", default=None, help="自定义算子库")
    probe_cmd.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    probe_cmd.add_argument("--device", type=int, default=0)
    probe_cmd.add_argument("--per-channel", dest="per_channel", action="store_true",
                           help="按倒数第 3 维（[B,F,C,H,W] 的通道）细分")
    probe_cmd.add_argument("--feeds", default="normal", choices=["normal", "ones", "zeros"],
                           help="自动造输入的口径（默认 normal）")
    _add_spec_args(probe_cmd)
    probe_cmd.set_defaults(func=_cmd_probe)

    check = sub.add_parser(
        "check",
        help="跑完之后体检真实输出目录：成员是否互不相同、离散度是否随时效增长")
    check.add_argument("path", help="推理输出顶层目录（下面有 {起报目录}/member_XXX/）")
    check.add_argument("--init", default=None, help="只看这些起报目录，逗号分隔")
    check.add_argument("--var", default=None,
                       help="只体检这个变量（默认取每个文件的第一个数据变量）")
    check.add_argument("--stride", type=int, default=4,
                       help="空间子采样步长，越大越快（默认 4）")
    check.add_argument("--steps", default="12",
                       help="体检哪些步：整数=等距抽这么多个（首尾必含），all=全查")
    check.add_argument("--expect-members", type=int, default=None,
                       help="期望的成员数，不符即判失败")
    check.add_argument("--hour-interval", type=float, default=6.0,
                       help="单步小时数，只用来把步号换算成天数（默认 6）")
    check.set_defaults(func=_cmd_check)

    spread = sub.add_parser(
        "spread",
        help="离散度的标定口径曲线：σ̂ 随时效 + 差值功率落在哪些尺度上（可多个目录/CSV 对比）")
    spread.add_argument("paths", nargs="+",
                        help="推理输出顶层目录（{起报}/member_XXX/）或之前 --csv 出来的曲线文件")
    spread.add_argument("--var", default=None,
                        help="只看这个变量（默认取每个文件的第一个数据变量）")
    spread.add_argument("--init", default=None, help="只看这些起报目录，逗号分隔")
    spread.add_argument("--members", default=None,
                        help="只看这些成员：0-5 / 7 / member_007 / first:6 / last:6，"
                             "逗号分隔；默认全用。用于在同一份产物上量「成员数噪声」")
    spread.add_argument("--steps", default="all",
                        help="算哪些步：整数=等距抽这么多个（首尾必含），all=全算（默认）")
    spread.add_argument("--stride", type=int, default=1,
                        help="空间子采样步长（默认 1，全分辨率；谱要全分辨率才准）")
    spread.add_argument("--hour-interval", type=float, default=6.0,
                        help="单步小时数，用来把步号换算成时效（默认 6）")
    spread.add_argument("--every", type=int, default=None,
                        help="打印时每隔几步显示一行（默认自动抽约 12 行）")
    spread.add_argument("--csv", default=None,
                        help="把曲线写成 CSV：给目录则每个输入一个文件，给 .csv 则只允许一个输入")
    spread.set_defaults(func=_cmd_spread)

    verdict = sub.add_parser(
        "verdict",
        help="判决页：只读评测归档，判一个臂是「改了形状」还是「只改了幅度」（零推理）")
    verdict.add_argument("arms", nargs="+",
                         help="臂名（第一个是基准；也可用 --ref 指定基准）")
    verdict.add_argument("--base", default=DEFAULT_BASE,
                         help="评测归档根目录（默认 %(default)s；环境变量 SPW_EVAL_BASE 可覆盖）")
    verdict.add_argument("--prefix", default=DEFAULT_PREFIX,
                         help="归档目录名前缀（默认 %(default)s）")
    verdict.add_argument("--dir", default=None,
                         help="覆盖某个臂的路径，写成「臂=路径」，逗号分隔")
    verdict.add_argument("--ref", default=None, help="基准臂名（默认第一个位置参数）")
    verdict.add_argument("--var", default=DEFAULT_VAR, help="变量（默认 %(default)s）")
    verdict.add_argument("--region", default="global", help="区域（默认 global）")
    verdict.add_argument("--p-from", dest="p_from", nargs=2, default=None,
                         metavar=("SIGMA_SMALL", "SIGMA_LARGE"),
                         help="从两个「只有 sigma 不同」的推理输出目录现场重测 p(lead)")
    verdict.add_argument("--p-sigma", default=None,
                         help="配合 --p-from：两个目录的 sigma，写成「小,大」（默认 0.01,0.04）")
    verdict.add_argument("--no-check", dest="no_check", action="store_true",
                         help="跳过可比性核对")
    verdict.add_argument("--csv", default=None, help="把 r/k 曲线落盘成 CSV")
    verdict.set_defaults(func=_cmd_verdict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
