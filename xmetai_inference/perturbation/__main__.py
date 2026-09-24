# -*- coding: utf-8 -*-
"""SPW 命令行：members（造成员集）/ groups（选组）/ probe（量级）/ e2e（端到端预检）/ check（真实输出目录的集合体检）。

用法（在仓库根目录，或设 PYTHONPATH 指向仓库根）：

    python -m xmetai_inference.perturbation groups  model.onnx
    python -m xmetai_inference.perturbation members model.onnx --members 51
    python -m xmetai_inference.perturbation probe   model.onnx --m1a-members 8
    python -m xmetai_inference.perturbation self-test
    python -m xmetai_inference.perturbation e2e --members 51
    python -m xmetai_inference.perturbation check /path/to/output_dir
    python -m xmetai_inference.perturbation verdict F S T V E3      # 判决页（零推理）
    python -m xmetai_inference.perturbation latent build model.onnx --tau-days 3   # 潜空间 AR(1)
    python -m xmetai_inference.perturbation latent audit model.ar1-xxxxxxxxxx.onnx

`groups` 用来挑注入点：打印每个名字前缀的张量数、元素数、权重能量占比，
按能量降序，方便对着论文的「张量组 / 尺度」两阶段消融选候选。
`members` 造完之后直接 `xmetai-infer --model <config>` 即可（配置里写 weight_perturb，
框架会按成员号自己换路径）。

`audit` 留在这里只是为了复查磁盘上那批**旧图内随机算子图**（该机制 2026-09-23 淘汰、
2026-09-24 删码）；成员图没有图内噪声节点，audit 对它没有意义。
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys

import numpy as np

from .graph import (
    _FLOAT_DTYPES,
    _onnx,
    consumer_map,
    audit_model,
    format_audit,
    key_parts,
    load_model,
    tensor_energy,
    weight_like,
    _match_pattern,
)
from .latent import (DEFAULT_LENGTH_SCALE, LatentForcingSpec, audit_forcing,
                     build_forcing_model, format_forcing_audit)
from .spec import DEFAULT_EXCLUDE, WeightPerturbSpec
from .verdict import DEFAULT_BASE, DEFAULT_PREFIX, DEFAULT_VAR


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


def _cmd_groups(args) -> int:
    """列出候选注入组。与 select_initializers 用同一套「权重」口径，避免误导。

    `--depth` 控制分桶深度（默认 2：只到 `latent_mixer/layers` 这一级）；`--under` 把统计
    限制到某个子树里，并把桶从子树的下一级开始列 —— 两者配合就是「注入点消融」的细粒度扫描
    （论文 Phase 2 的做法）：先 `--under latent_mixer.layers.0.blocks --depth 5` 看每个 block
    的能量，再挑 key 填进 config 的 `groups`。key 里 `/` 与 `.` 等价（`key_parts` 两种都切）。
    """
    onnx = _onnx()
    model = load_model(args.model)
    src_dir = osp.dirname(osp.abspath(args.model))
    consumers = consumer_map(model.graph)
    memo: dict = {}
    exclude = (tuple(token.strip() for token in args.exclude.split(","))
               if args.exclude else DEFAULT_EXCLUDE)
    include = (tuple(token.strip() for token in args.include.split(","))
               if args.include else ())
    under = key_parts(args.under) if args.under else ()
    depth_max = max(1, min(int(args.depth), 8))
    if under and depth_max <= len(under):
        depth_max = len(under) + 1          # 否则一个桶都列不出来，静默空表最坑
    first_depth = len(under) + 1 if under else 1

    buckets: dict[str, dict] = {}
    chosen: list[tuple[float, str, int]] = []
    weight_total = 0.0
    total_tensors = 0
    total_elements = 0
    excluded_energy = other_energy = 0.0
    excluded_count = other_count = 0
    for tensor in model.graph.initializer:
        name = tensor.name
        dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
        elements = int(np.prod(tensor.dims)) if tensor.dims else 1
        if dtype not in _FLOAT_DTYPES:
            continue
        parts = key_parts(name)
        forced = _match_pattern(parts, include) is not None
        if not forced and elements < args.min_elements:
            continue
        energy = tensor_energy(tensor, src_dir) if args.energy else float(elements)
        if _match_pattern(parts, exclude) is not None:
            excluded_energy += energy
            excluded_count += 1
            continue
        if not forced and not weight_like(name, consumers, memo):
            other_energy += energy
            other_count += 1
            continue
        weight_total += energy
        total_tensors += 1
        total_elements += elements
        if under and parts[: len(under)] != under:
            continue                        # --under：只统计子树
        chosen.append((energy, name, elements))
        for depth in range(first_depth, depth_max + 1):
            key = "/".join(parts[:depth])
            bucket = buckets.setdefault(
                key, {"tensors": 0, "elements": 0, "energy": 0.0})
            bucket["tensors"] += 1
            bucket["elements"] += elements
            bucket["energy"] += energy

    unit = "能量 ΣW^2" if args.energy else "元素数"
    print(f"候选权重合计（{unit}）{weight_total:.6g}，"
          f"{total_tensors} 个张量 / {total_elements/1e6:.2f}M 元素")
    if under:
        share = (sum(item[0] for item in chosen) / weight_total * 100.0
                 if weight_total else 0.0)
        print(f"--under {'/'.join(under)}：命中 {len(chosen)} 个张量 / "
              f"{sum(item[2] for item in chosen)/1e6:.2f}M 元素，占候选能量 {share:.1f}%"
              f"（桶列到第 {depth_max} 级）")
    print(f"{'group':40s} {'tensors':>8s} {'elements':>12s} {'share':>7s}")
    for key, bucket in sorted(buckets.items(), key=lambda kv: -kv[1]["energy"])[: args.top]:
        share = bucket["energy"] / weight_total * 100.0 if weight_total else 0.0
        print(f"{key:40s} {bucket['tensors']:8d} {bucket['elements']/1e6:11.2f}M {share:6.1f}%")
    print("\n能量最大的 10 个候选张量（肉眼确认没有混进常量）：")
    for energy, name, elements in sorted(chosen, reverse=True)[:10]:
        print(f"  {energy:12.4g}  {name:56s} n={elements}")
    if excluded_count:
        share = excluded_energy / (weight_total + excluded_energy) * 100.0
        print(f"\n另命中 exclude 的 {excluded_count} 个张量，能量占比 {share:.1f}%"
              f"（默认排除统计量/烘焙常量，可用 --exclude 覆盖）")
    if other_count:
        print(f"另 {other_count} 个非权重张量（{other_energy:.4g}）未计入；"
              f"确属可学习参数可用 --include 纳入")
    print("\ngroups 支持任意深度前缀（`/` 与 `.` 等价），上面列出的 key 可以直接填进 config 的"
          " `groups`。\n扫子组用：--under <前缀> --depth <级数>；权重名被导出器拍平的图会走"
          "消费者节点名前缀匹配。")
    return 0


def _cmd_audit(args) -> int:
    report = audit_model(args.model, reference=not args.no_reference,
                         structural=not args.no_structural)
    if args.json:
        import json

        payload = {
            "path": report["path"],
            "injected_tensors": report["injected_tensors"],
            "injected_variance": report["injected_variance"],
            "sigma_full_effective": report.get("sigma_full_effective"),
            "reference_energy": report.get("reference_energy"),
            "groups": report["groups"],
            "excluded_tensors": report.get("excluded_tensors"),
            "excluded_energy": report.get("excluded_energy"),
            "source_check": report.get("source_check"),
            "structural": report.get("structural"),
            "metadata": report["metadata"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_audit(report))
    structural = report.get("structural")
    return 1 if structural is not None and not structural["ok"] else 0


def _cmd_self_test(args) -> int:
    from .selftest import run_self_test

    return run_self_test(verbose=not args.quiet, keep=args.keep)


def _cmd_members(args) -> int:
    """把逐成员冻结的扰动权重物理落盘。"""
    from .weights import build_member_set, format_member_report, verify_member_set

    spec = _spec_from_args(args)
    report = build_member_set(args.model, spec, args.members, out_dir=args.out,
                              link=not args.no_link, verbose=True)
    if args.verify:
        info = verify_member_set(args.model, spec, args.members, out_dir=args.out)
        print()
        print(format_member_report(info))
        return 1 if info["problems"] else 0
    print(f"清单：{report['manifest']}")
    print("成员模型可直接当确定性模型跑：它们是成员集目录里的普通确定性 onnx"
          "（<模型名>.mNNN.onnx）。正常不用手动指——config 里给了 weight_perturb，"
          "框架会按成员号自己换路径。")
    return 0


def _cmd_probe(args) -> int:
    """探针：量噪声在输出上的量级与线性度。"""
    from .probe import format_member_scan, format_probe, member_scan, probe

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
    if not report.get("ok"):
        return 1
    return 0


def _cmd_e2e(args) -> int:
    from .e2e import run_e2e

    return run_e2e(
        members=args.members,
        steps=args.steps,
        verbose=not args.quiet,
        keep=args.keep,
    )


def _cmd_check(args) -> int:
    from .enscheck import run_check

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
    from .spread import run_spread

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
    from .verdict import run_verdict

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


def _cmd_latent(args) -> int:
    """潜空间 AR(1) 强迫：build（给图加 spw/xi）/ audit（看改得对不对）。

    与 SPW 的关系：SPW 改**权重**（幅度轴），这条改**图**、拧的是噪声的时间相关（tau）与
    空间尺度（length_scale）。两条正交，config 里同时写就是「冻结的扰动权重 + 时间相关的
    潜空间强迫」。build 出来的图要由**框架**喂 spw/xi（每步 T x R 个数），单独拿它跑推理
    只会得到「sigma=0 的空转」——那正是等价性回归用的对照。
    """
    if args.latent_command == "audit":
        report = audit_forcing(args.model)
        if args.json:
            print(json.dumps(dict(report), ensure_ascii=False, indent=2))
        else:
            for line in format_forcing_audit(report):
                print(line)
        return 0 if report.get("ok") else 1

    spec = LatentForcingSpec(
        targets=tuple(token.strip() for token in str(args.targets).split(",")
                      if token.strip()),
        sigma=args.sigma,
        tau_days=args.tau_days,
        rank=args.rank,
        length_scale=args.length_scale,
        seed=args.seed,
        hour_interval=args.hour_interval,
        cache_dir=args.cache_dir,
    )
    path = build_forcing_model(args.model, spec, args.out, verbose=not args.quiet)
    report = audit_forcing(path)
    if args.json:
        print(json.dumps(dict(report), ensure_ascii=False, indent=2))
    else:
        print()
        for line in format_forcing_audit(report):
            print(line)
    return 0 if report.get("ok") else 1


def _add_spec_args(parser, *, with_out: bool = True) -> None:
    """members / probe 共用的注入规格参数。"""
    if with_out:
        parser.add_argument("--out", default=None, help="输出路径；默认源模型同目录")
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
        description="确定性 ONNX -> 集合模型：权重扰动（SPW）工具")
    sub = parser.add_subparsers(dest="command", required=True)

    groups = sub.add_parser("groups", help="列出可选的注入组（按权重能量排序）")
    groups.add_argument("model")
    groups.add_argument("--min-elements", type=int, default=64)
    groups.add_argument("--exclude", default=None,
                        help=f"排除模式，逗号分隔；默认 {','.join(DEFAULT_EXCLUDE)}")
    groups.add_argument("--include", default=None,
                        help="强制纳入模式（名字不像 weight/bias 的可学习参数）")
    groups.add_argument("--energy", action="store_true", default=True,
                        help="按 Frobenius 能量统计（默认）；--no-energy 时退化为元素数")
    groups.add_argument("--no-energy", dest="energy", action="store_false")
    groups.add_argument("--top", type=int, default=25)
    groups.add_argument("--depth", type=int, default=2,
                        help="分桶深度（默认 2）。扫子组时配合 --under 用，例如 "
                             "--under latent_mixer.layers.0.blocks --depth 5")
    groups.add_argument("--under", default=None,
                        help="只统计该名字前缀（`/` 与 `.` 等价）下面的张量，"
                             "并把桶从它的下一级开始列 —— 注入点的细粒度扫描")
    groups.set_defaults(func=_cmd_groups)

    audit = sub.add_parser("audit", help="静态审计注入后的图（组/σ/注入方差/等效 σ_full）")
    audit.add_argument("model")
    audit.add_argument("--no-reference", action="store_true",
                       help="不读全部权重，跳过「等效 σ_full」那一项")
    audit.add_argument("--json", action="store_true")
    audit.add_argument("--no-structural", action="store_true",
                       help="跳过与源图的逐节点结构核对（默认做）")
    audit.set_defaults(func=_cmd_audit)

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

    members = sub.add_parser(
        "members",
        help="把逐成员冻结的扰动权重物理落盘（论文的主配置；推理零噪声开销）")
    members.add_argument("model", help="源确定性 onnx（权重可以走外部 .data）")
    members.add_argument("--members", type=int, required=True, help="要生成多少个成员")
    members.add_argument("--out", default=None,
                         help="成员集目录；默认落在源模型同目录（文件名带指纹）")
    members.add_argument("--no-link", dest="no_link", action="store_true",
                         help="不用硬链接共享源 .data，整份复制（会占大量磁盘）")
    members.add_argument("--verify", action="store_true", help="生成后立刻复核")
    _add_spec_args(members, with_out=False)
    members.set_defaults(func=_cmd_members)

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
    _add_spec_args(probe_cmd, with_out=False)
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

    latent = sub.add_parser(
        "latent",
        help="潜空间 AR(1) 强迫：给图加一个 spw/xi 输入（噪声的**时间相关**那条腿）")
    latent_sub = latent.add_subparsers(dest="latent_command", required=True,
                                       help="build = 改图 / audit = 审计改完的图")

    latent_build = latent_sub.add_parser(
        "build", help="把 AR(1) 潜空间强迫注入图上（权重一个字节都不动），写出新图")
    latent_build.add_argument("model", help="源确定性 onnx（也可以直接给 m1a 的成员图）")
    latent_build.add_argument("--targets", default="latent_state",
                              help="注入目标：latent_state（默认）/ blocks / 节点名后缀，逗号分隔")
    latent_build.add_argument("--sigma", type=float, default=0.04,
                              help="相对强迫强度（z (x) (1 + sigma*n)），默认 0.04；0 = 空转对照")
    latent_build.add_argument("--tau-days", dest="tau_days", type=float, default=3.0,
                              help="AR(1) 相关时间（天），默认 3（rho = exp(-dt/tau) = 0.92）")
    latent_build.add_argument("--rank", type=int, default=16, help="低秩基的秩 R，默认 16")
    latent_build.add_argument("--length-scale", dest="length_scale", type=float,
                              default=DEFAULT_LENGTH_SCALE,
                              help="空间基的高斯低通相关长度（token 数）：32 = 只扰行星尺度，"
                                   "1 = 所有尺度一起扰（默认 %(default)s）")
    latent_build.add_argument("--hour-interval", dest="hour_interval", type=float, default=6.0,
                              help="一步多少小时（算 rho 用），默认 6")
    latent_build.add_argument("--seed", type=int, default=None, help="驱动侧随机流的种子")
    latent_build.add_argument("--cache-dir", dest="cache_dir", default=None,
                              help="输出目录；默认源模型同目录（外部数据要求同目录，别乱指）")
    latent_build.add_argument("--out", default=None, help="输出文件路径；默认按指纹命名")
    latent_build.add_argument("--quiet", action="store_true", help="只打印审计结果")
    latent_build.add_argument("--json", action="store_true", help="审计结果打成 JSON")
    latent_build.set_defaults(func=_cmd_latent)

    latent_audit = latent_sub.add_parser(
        "audit", help="审计强迫图：spw/xi 是否真在输入里、metadata 是否自洽")
    latent_audit.add_argument("model")
    latent_audit.add_argument("--json", action="store_true", help="打成 JSON")
    latent_audit.set_defaults(func=_cmd_latent)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
