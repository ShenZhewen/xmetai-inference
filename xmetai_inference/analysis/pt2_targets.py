# -*- coding: utf-8 -*-
"""PT2 潜空间注入点体检：给一个 .pt2 路径，看 SPA 能注到哪、注到了哪个节点。

只加载 ExportedProgram，不跑前向。**但需要一个能初始化 CUDA 的节点**：pt2 存档里的
权重按保存时的设备恢复，从 CUDA 导出的模型在 ``torch.export.load`` 阶段就会调 CUDA，
CPU-only 节点上会直接报 “Found no NVIDIA driver”。回答两件事：

  1. **逐层锚点**：``layer_stream:k`` 从 ``p_model_decoder_layers_{k}_mlp_w2_weight``
     沿单链回溯到第 k 层末的残差 ``aten.add``。打印每层的链路、末端节点名、形状、dtype。
     结构检查全绿 **不等于** 注对了层（形状一样、节点数一样），链尾名字必须人眼对一遍。
  2. **候选 target 集审计**：对每个由层号组成的集合实际做一次内存注入，打印
     ``n_targets`` / 新增节点数 / 布局 / 隐含 sigma / 每层的链尾。不跑前向。

用法（服务器仓库根目录）::

    python -m xmetai_inference.analysis.pt2_targets <model.pt2>

    # 只看几个深度
    python -m xmetai_inference.analysis.pt2_targets <model.pt2> \
        --layers 0,5,10,15,20,25,29

    # 审计候选集合（分号分隔；层号用逗号）
    python -m xmetai_inference.analysis.pt2_targets <model.pt2> \
        --sets "early=0,3,6;spread=0,10,20;late=24,27,29"

    # 只看集合，不列全部层
    python -m xmetai_inference.analysis.pt2_targets <model.pt2> --no-structure \
        --sets "spread=0,10,20"
"""
from __future__ import annotations

import argparse
import os
import sys


def _load_ep(path):
    try:
        import torch.export
    except ImportError:  # pragma: no cover - 环境缺依赖时给一句人话
        raise SystemExit("需要 torch：请在装了 torch 的环境里运行")
    try:
        return torch.export.load(path)
    except Exception as error:  # 版本不匹配 / 文件损坏 / 无 CUDA 驱动都走这里
        hint = ""
        try:
            if not torch.cuda.is_available():
                hint = (
                    "\n提示：当前节点 torch.cuda.is_available()=False。pt2 存档的权重"
                    "若是从 CUDA 导出的，torch.export.load 会尝试恢复到 CUDA，没有 "
                    "NVIDIA 驱动时必然失败；换到 GPU 节点/容器再跑（本工具只做图检查，"
                    "不占 GPU 算力）。")
        except Exception:
            pass
        raise SystemExit(f"torch.export.load 失败：{error}{hint}")


def _parse_layers(text):
    """'0,5,10' -> [0,5,10]；空串 -> None（表示全部）。"""
    tokens = [part.strip() for part in str(text or "").replace(" ", ",").split(",")]
    layers = [int(token) for token in tokens if token]
    return layers or None


def _parse_sets(text):
    """'early=0,3,6;spread=0,10,20' -> {'early': [0,3,6], ...}。"""
    sets = {}
    for chunk in str(text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"--sets 每项要写成 名称=层号,层号：{chunk!r}")
        name, layers = chunk.split("=", 1)
        name = name.strip()
        parsed = _parse_layers(layers)
        if not name or not parsed:
            raise SystemExit(f"--sets 这一项不完整：{chunk!r}")
        sets[name] = parsed
    return sets


def _structure(path, only=None):
    """逐层打印回溯链路。返回失败层数。"""
    from ..weights.spa_pt2 import (
        _dtype_of,
        _layer_indices,
        _shape_of,
        _walk_layer,
    )

    ep = _load_ep(path)
    gm = ep.graph_module
    available = _layer_indices(gm)
    print("=" * 78)
    print(f"文件   : {os.path.abspath(path)}")
    print(f"可用层 : {len(available)} 个 {available}")
    if not available:
        print("  ⚠ 图里没有任何 layer_stream 锚点（不是 FuXi 式 decoder 结构）")
        return 1

    wanted = available if only is None else only
    failures = 0
    for layer in wanted:
        if layer not in available:
            print(f"{layer:2d}  ⚠ 层号不在图里（可用 {available[0]}..{available[-1]}）")
            failures += 1
            continue
        try:
            chain = _walk_layer(gm, layer)
        except Exception as error:  # 回溯规则不匹配时保留原因
            print(f"{layer:2d}  ERROR {error}")
            failures += 1
            continue
        out = chain[-1]
        shape = _shape_of(out)
        print(
            f"{layer:2d}  shape={shape} dtype={_dtype_of(out)} "
            f"chain={len(chain)} end={out.name}"
        )
        print("    " + " -> ".join(node.name for node in chain))
    return failures


def _audit_sets(path, sets, spec_kwargs):
    """每个集合重新加载一次（注入是就地改图），打印注入审计。返回失败集数。"""
    from ..weights.spa import LatentForcingSpec
    from ..weights.spa_pt2 import attach_forcing

    failures = 0
    for name, layers in sets.items():
        target = "layer_stream:" + ",".join(str(layer) for layer in layers)
        print(f"\n[set] {name}: {layers}  targets=({target!r},)")
        ep = _load_ep(path)
        spec = LatentForcingSpec(targets=(target,), **spec_kwargs)
        try:
            report = attach_forcing(ep, spec, src_path=path, verbose=False)
        except Exception as error:
            print(f"  FAILED: {error}")
            failures += 1
            del ep
            continue
        layout = report.get("layout") or {}
        print(
            f"  ok={report.get('ok')} n_targets={report.get('n_targets')} "
            f"新增节点={report.get('nodes')} 指纹={report.get('fingerprint')}"
        )
        if layout:
            print(
                f"  布局 tokens={layout.get('tokens')} channels={layout.get('channels')} "
                f"grid={layout.get('grid')} dtype={report.get('dtype')} "
                f"隐含 sigma={report.get('sigma_implied')}"
            )
        chains = report.get("chains") or {}
        for layer in layers:
            chain = chains.get(f"layer_{layer}")
            if chain:
                print(f"  layer_{layer} 链尾: {chain[-1]}  (链路长 {len(chain)})")
            else:
                print(f"  layer_{layer} 链尾: （报告里没有；检查 target 写法）")
        if not report.get("ok"):
            failures += 1
        del ep
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PT2 潜空间注入点体检：layer_stream 回溯链 + 候选 target 集审计")
    parser.add_argument("model", help=".pt2 文件路径")
    parser.add_argument("--layers", default="",
                        help="逗号分隔的层号；默认列出全部可用层")
    parser.add_argument("--sets", default="",
                        help="候选集合，分号分隔，写成 名称=层号,层号（如 "
                             "early=0,3,6;spread=0,10,20）")
    parser.add_argument("--no-structure", action="store_true",
                        help="跳过逐层链路，只审计 --sets")
    parser.add_argument("--sigma", type=float, default=0.04, help="注入 sigma（默认 0.04）")
    parser.add_argument("--tau-days", type=float, default=3.0, help="相关时间（默认 3.0）")
    parser.add_argument("--rank", type=int, default=16, help="低秩基秩（默认 16）")
    parser.add_argument("--length-scale", type=float, default=32.0,
                        help="空间相关长度，token 数（默认 32）")
    parser.add_argument("--grid", default="120,240",
                        help="潜网格 rows,cols（FuXi-2.1 是 120,240）")
    parser.add_argument("--seed", type=int, default=0, help="驱动种子（默认 0）")
    parser.add_argument("--hour-interval", type=float, default=6.0,
                        help="一步小时数（默认 6）")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.model):
        raise SystemExit(f"文件不存在：{args.model}")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    failures = 0
    if not args.no_structure:
        failures += _structure(args.model, _parse_layers(args.layers))

    sets = _parse_sets(args.sets)
    if sets:
        grid = tuple(int(part) for part in args.grid.split(",") if part.strip())
        if len(grid) != 2:
            raise SystemExit(f"--grid 要写成 rows,cols：{args.grid!r}")
        spec_kwargs = {
            "sigma": args.sigma,
            "tau_days": args.tau_days,
            "rank": args.rank,
            "length_scale": args.length_scale,
            "grid": grid,
            "seed": args.seed,
            "hour_interval": args.hour_interval,
        }
        failures += _audit_sets(args.model, sets, spec_kwargs)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
