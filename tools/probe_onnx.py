#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 契约探针：新模型接进推理框架之前，先把 I/O 和图看清楚。

    python tools/probe_onnx.py /workspace/tmp/douzsh/models/fengshun2.0_260901.onnx

只读、不改文件、默认不跑推理。输出分八段：

1. 文件       大小、权重是内嵌还是旁挂（>2GB 的 onnx 会把权重拆成 .data）、同目录邻居
2. 模型级     IR / opset / producer / metadata —— 归一化信息常写在这里
3. 输入输出   名字、dtype、维度；顺带判断 rank 是 [B,C,H,W] 还是 [B,T,C,H,W]
4. 图结构     节点数、算子直方图、自定义算子域（要 .so 注册的那种）、子图
5. 可疑点     Random*/Dropout/If/Loop/Scan、输出是否 = 输入 + 残差
6. 活跃性     从输出反推：随机算子在不在主路径（活着 = 一次前向只出一个成员）
7. 契约细节   图头有没有 Div(Sub(x,mean),std)、图尾有没有反归一化、输出几帧

可选开关：

    --ort              用 onnxruntime 再开一次 session，核对它眼里的 I/O（慢，会读权重）
    --run              真跑一次全零输入（隐含 --ort），打印输出量级和 NaN 计数
    --ops-lib SO       注册自定义算子库；不给就自动扫同目录的 xmetai_onnx_plugins*.so
    --provider P       auto（默认，有 CUDA 就用）/ cuda / cpu
    --json F           报告另存一份 JSON，便于两份模型直接 diff
    --no-ls            不列同目录文件

依赖：numpy 必需；onnx 缺失时退化成只有 onnxruntime 的 I/O 视图。
--ort / --run 会先 ``import torch`` 再导 onnxruntime（torch 带入 nvidia 的 .so 路径，
否则 CUDA provider 找不到 libcudnn）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import onnx
except ImportError:  # 服务器上可能只装了 onnxruntime
    onnx = None


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _elem_name(elem_type):
    try:
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).name
    except Exception:
        return f"type{elem_type}"


def _shape_of(value):
    """graph.input/output 里的维度；静态给数字，动态给符号名，未标注给 '?'。"""
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return []
    dims = []
    for d in tensor.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param or "?")
        else:
            dims.append("?")
    return dims


def _same(a, b) -> bool:
    return a == b and len(a) == len(b)


def _walk_graphs(graph, depth: int = 0):
    """主图 + 所有子图（If / Loop / Scan 的 body 也是 GraphProto）。"""
    yield graph, depth
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _walk_graphs(attr.g, depth + 1)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from _walk_graphs(g, depth + 1)


# ---------------------------------------------------------------------------
# 2. 模型级信息
# ---------------------------------------------------------------------------
def section_meta(model, report):
    g = model.graph
    opsets = [{"domain": o.domain or "ai.onnx", "version": o.version}
              for o in model.opset_import]
    meta = {p.key: p.value for p in model.metadata_props}

    print(f"IR version      : {model.ir_version}")
    print(f"opset           : " + ", ".join(f"{o['domain']}={o['version']}" for o in opsets))
    print(f"producer        : {model.producer_name or '-'} {model.producer_version or ''}".rstrip())
    print(f"graph           : {g.name or '-'}")
    print(f"doc_string      : {(model.doc_string or '-')[:200]}")
    if meta:
        print("metadata_props  :")
        for k, v in meta.items():
            print(f"  {k} = {v[:200]}")
    else:
        print("metadata_props  : <空>")
    if g.doc_string:
        print(f"graph doc       : {g.doc_string[:200]}")

    report["meta"] = {
        "ir_version": model.ir_version,
        "opsets": opsets,
        "producer": model.producer_name,
        "producer_version": model.producer_version,
        "graph_name": g.name,
        "doc_string": model.doc_string,
        "metadata_props": meta,
    }


# ---------------------------------------------------------------------------
# 3. 输入输出
# ---------------------------------------------------------------------------
def section_io(graph, report):
    init_names = {t.name for t in graph.initializer}
    inputs = [v for v in graph.input if v.name not in init_names]
    outputs = list(graph.output)

    def describe(values, title):
        rows = []
        print(f"\n{title}（{len(values)} 个）")
        for i, v in enumerate(values):
            if not v.type.HasField("tensor_type"):
                print(f"  [{i}] {v.name}: 非张量类型 {v.type}")
                continue
            dims = _shape_of(v)
            dtype = _elem_name(v.type.tensor_type.elem_type)
            rank = len(dims)
            note = ""
            if rank == 5:
                note = "  ← rank=5，按 [B,T,C,H,W] 读"
            elif rank == 4:
                note = "  ← rank=4，按 [B,C,H,W] 读"
            elif rank == 3:
                note = "  ← rank=3，按 [B,C,L] 读"
            if 721 in dims and 1440 in dims:
                note += "  ← 含 721×1440，疑似 0.25° 全球网格"
            print(f"  [{i}] {v.name}: {dtype} {tuple(dims)}{note}")
            rows.append({"name": v.name, "dtype": dtype, "shape": dims})
        return rows

    in_rows = describe(inputs, "输入 (graph.input，已剔除 initializer)")
    out_rows = describe(outputs, "输出 (graph.output)")

    # 静态量：谁跟谁同形，输出是不是残差
    in_shapes = [r["shape"] for r in in_rows]
    out_shapes = [r["shape"] for r in out_rows]
    same_shape = any(_same(s, o) for s in in_shapes for o in out_shapes)
    print(f"\n任一输出与任一输入同形 : {'是' if same_shape else '否'}"
          f"{'（单步自回归的典型形态）' if same_shape else ''}")

    report["inputs"] = in_rows
    report["outputs"] = out_rows
    report["io_same_shape"] = same_shape
    return in_rows, out_rows


# ---------------------------------------------------------------------------
# 4. 图结构
# ---------------------------------------------------------------------------
def section_graph(model, report):
    main = model.graph
    graphs = list(_walk_graphs(main))
    sub_count = len(graphs) - 1

    op_counter = Counter()
    domain_counter = Counter()
    for g, _ in graphs:
        for node in g.node:
            op_counter[node.op_type] += 1
            domain_counter[node.domain or "ai.onnx"] += 1

    total_nodes = sum(op_counter.values())
    print(f"主图节点数      : {len(main.node)}")
    print(f"全部节点数      : {total_nodes}（子图 {sub_count} 个）")
    print(f"算子种类        : {len(op_counter)}")
    print("\n算子直方图（前 25）:")
    for op, n in op_counter.most_common(25):
        print(f"  {op:<32} {n:>6}")
    if len(op_counter) > 25:
        print(f"  … 另 {len(op_counter) - 25} 种")

    custom = {d: n for d, n in domain_counter.items() if d not in ("", "ai.onnx")}
    if custom:
        print("\n自定义算子域（这些算子要额外的 .so / 插件才能跑）:")
        for d, n in custom.items():
            ops = sorted({nd.op_type for g, _ in graphs for nd in g.node if (nd.domain or "ai.onnx") == d})
            print(f"  {d}: {n} 个节点，算子 {ops}")
    else:
        print("\n自定义算子域    : 无（纯标准算子）")

    # initializer
    inits = list(main.initializer)
    total_params = int(sum(np.prod(t.dims) if t.dims else 1 for t in inits))
    dtype_counter = Counter(_elem_name(t.data_type) for t in inits)
    top = sorted(inits, key=lambda t: int(np.prod(t.dims) if t.dims else 1), reverse=True)[:5]
    sub_init = sum(len(g.initializer) for g, d in graphs if d > 0)

    print(f"\ninitializer     : {len(inits)} 个（子图里另有 {sub_init} 个）")
    print(f"参数量          : {total_params:,}（{total_params / 1e6:.1f} M，按 float32 约 "
          f"{fmt_bytes(total_params * 4)}）")
    print(f"权重 dtype      : {dict(dtype_counter)}")
    print("最大的 5 个权重:")
    for t in top:
        n = int(np.prod(t.dims) if t.dims else 1)
        print(f"  {t.name:<60} {tuple(t.dims)}  {fmt_bytes(n * 4)}")

    report["graph"] = {
        "main_nodes": len(main.node),
        "total_nodes": total_nodes,
        "subgraphs": sub_count,
        "op_count": dict(op_counter.most_common()),
        "custom_domains": custom,
        "initializers": len(inits),
        "params": total_params,
        "param_dtypes": dict(dtype_counter),
    }
    return graphs, op_counter


# ---------------------------------------------------------------------------
# 5. 可疑点：随机算子、控制流、残差结构、外部数据
# ---------------------------------------------------------------------------
def section_suspects(model, graphs, op_counter, report):
    print("随机 / 状态类算子（同一输入两次结果会不同）:")
    random_ops = sorted({op for op in op_counter if op.startswith("Random") or op in ("Multinomial", "Bernoulli")})
    print(f"  {random_ops if random_ops else '<无>'}")

    print("训练残留（推理时应被裁掉，留着也无害）:")
    leftover = sorted({op for op in op_counter if op in ("Dropout", "BatchNormalization")})
    print(f"  {leftover if leftover else '<无>'}")

    print("控制流算子（图里有子图，改写图时要递归进去）:")
    control = sorted({op for op in op_counter if op in ("If", "Loop", "Scan")})
    print(f"  {control if control else '<无>'}")

    # 残差：输出节点的输入里直接出现图输入
    main = model.graph
    init_names = {t.name for t in main.initializer}
    in_names = {v.name for v in main.input if v.name not in init_names}
    producer = {}
    for node in main.node:
        for out in node.output:
            producer[out] = node
    print("\n输出的生产节点（看是否 Add 回输入 = 残差结构）:")
    residual = []
    for v in main.output:
        node = producer.get(v.name)
        if node is None:
            print(f"  {v.name} <- <图输入，直通>")
            continue
        hits = [x for x in node.input if x in in_names]
        tag = f"  ← 直接吃图输入 {hits}，疑似 输出=输入+残差" if node.op_type in ("Add", "Sum") and hits else ""
        print(f"  {v.name} <- {node.op_type}({node.name or '-'}) in={list(node.input)[:4]}{tag}")
        if tag:
            residual.append(v.name)

    # 每个图输入被谁吃
    consumers = defaultdict(list)
    for node in main.node:
        for i, x in enumerate(node.input):
            consumers[x].append((node.op_type, i))
    print("\n图输入的去向（谁第一个吃了它）:")
    for name in in_names:
        uses = consumers.get(name, [])
        head = ", ".join(f"{op}[{i}]" for op, i in uses[:5])
        print(f"  {name}: {len(uses)} 处 -> {head}{' …' if len(uses) > 5 else ''}")

    # 外部数据
    ext = []
    for g, _ in graphs:
        for t in g.initializer:
            if t.data_location == onnx.TensorProto.EXTERNAL or (not t.raw_data and t.external_data):
                ext.append(t.name)
    print(f"\n外部数据张量    : {len(ext)} 个" + ("（权重旁挂在 .data 文件里）" if ext else ""))

    report["suspects"] = {
        "random_ops": random_ops,
        "dropout_ops": leftover,
        "control_flow": control,
        "residual_outputs": residual,
        "external_tensors": len(ext),
    }


# ---------------------------------------------------------------------------
# 5b. 活跃性：从输出反推，谁真的参与了计算
# ---------------------------------------------------------------------------
def section_liveness(model, report):
    """随机算子「在不在主路径」是决定性的：活着 = 一次前向只出一个成员。"""
    main = model.graph
    producer = {}
    for node in main.node:
        for out in node.output:
            producer[out] = node

    live_nodes, stack = set(), [v.name for v in main.output]
    while stack:
        node = producer.get(stack.pop())
        if node is None or id(node) in live_nodes:
            continue
        live_nodes.add(id(node))
        stack.extend(node.input)

    dead = [node for node in main.node if id(node) not in live_nodes]
    used = {x for node in main.node if id(node) in live_nodes for x in node.input}
    dead_params = sum(
        int(np.prod(t.dims) if t.dims else 1)
        for t in main.initializer if t.name not in used)

    print(f"参与输出的节点  : {len(live_nodes)}/{len(main.node)}"
          f"（死节点 {len(dead)}，死权重 {dead_params:,} 个参数）")

    live_random = [n for n in main.node
                   if id(n) in live_nodes and n.op_type.startswith("Random")]
    if live_random:
        print("⚠ 随机算子在主路径上 —— 同一输入两次前向结果不同：")
        consumers = defaultdict(list)
        for node in main.node:
            for x in node.input:
                consumers[x].append(node.op_type)
        for n in live_random[:5]:
            outs = [f"{o}->{consumers.get(o, ['<无人消费>'])[:3]}" for o in n.output]
            print(f"  {n.op_type}({n.name or '-'}) 输出 {outs}")
    else:
        print("随机算子        : 全都不在主路径上（不参与预测，可忽略）")

    if dead:
        kinds = Counter(n.op_type for n in dead)
        print(f"死节点构成      : {dict(kinds.most_common(5))}")

    report["liveness"] = {
        "live_nodes": len(live_nodes),
        "dead_nodes": len(dead),
        "dead_params": dead_params,
        "random_live": bool(live_random),
    }


# ---------------------------------------------------------------------------
# 5c. 归一化 / 输出的具体形态
# ---------------------------------------------------------------------------
def _forward_chain(consumers, start, depth):
    """从图输入往下走 depth 跳，打印每跳算子与它吃到的常量名。"""
    chain, name = [], start
    for _ in range(depth):
        uses = consumers.get(name)
        if not uses:
            break
        node, _idx = uses[0]
        chain.append(node)
        name = node.output[0] if node.output else ""
    return chain


def _reachable_inputs(producer, name, input_names, limit=20000):
    """某个张量反向可达的图输入（比「顺着 input[0] 走」稳，分支图也对）。"""
    seen, stack, found = set(), [name], []
    while stack and len(seen) < limit:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in input_names:
            found.append(cur)
            continue
        node = producer.get(cur)
        if node is not None:
            stack.extend(node.input)
    return found, len(seen)


def section_contract_details(model, report):
    import re

    main = model.graph
    inits = {t.name: t for t in main.initializer}
    init_names = set(inits)
    consumer_map = defaultdict(list)
    for node in main.node:
        for i, x in enumerate(node.input):
            consumer_map[x].append((node, i))
    producer = {out: node for node in main.node for out in node.output}
    input_names = {v.name for v in main.input if v.name not in init_names}

    # --- 图头链：input 往下 8 跳 ---
    print("\n图头链（input 往下 8 跳，判断输入是否在图内归一化）:")
    for node in _forward_chain(consumer_map, "input", 8):
        consts = [x for x in node.input if x in init_names]
        print(f"  {node.op_type:<18} {node.name or '-':<24}"
              f"{'  const=' + str(consts) if consts else ''}")

    # --- 名字像统计量的权重；shape 是 (C,1,1) 的直接把数值打出来 ---
    pat = re.compile(r"(^|[._/])(mean|std|mu|sigma|climat|mask)([._/]|$)", re.I)
    hits = [t for t in main.initializer if pat.search(t.name)]
    print(f"\n名字像统计量/掩膜的权重: {len(hits)} 个")
    for t in hits[:10]:
        print(f"  {t.name} {tuple(t.dims)}")
    for t in hits:
        # 旁挂权重读不到数值，跳过（模型 >2GB 时常见）
        if t.data_location == onnx.TensorProto.EXTERNAL:
            continue
        if len(t.dims) == 3 and t.dims[-2:] == [1, 1] and t.dims[0] <= 256:
            try:
                values = onnx.numpy_helper.to_array(t).reshape(-1)
            except Exception:  # noqa: BLE001
                continue
            print(f"\n  {t.name} 全部 {values.size} 个值（逐通道，可和 store 的通道统计对表）:")
            for start in range(0, values.size, 12):
                chunk = ", ".join(f"{v:.5g}" for v in values[start:start + 12])
                print(f"    [{start:>3}:{min(start + 12, values.size):>3}] {chunk}")

    # --- mean/std 被谁用：一眼看出归一化在不在图里 ---
    for tname in ("mean", "std"):
        if tname not in init_names:
            continue
        uses = consumer_map.get(tname, [])
        print(f"\n  {tname} 的消费者（{len(uses)} 处）:")
        for node, idx in uses[:10]:
            others = [x for j, x in enumerate(node.input) if j != idx]
            print(f"    {node.op_type:<10} {node.name or '-':<44} 其余输入={others[:2]}")

    # --- 图尾链 ---
    for v in main.output:
        print(f"\n图尾链（{v.name} 往上 5 跳）:")
        name, seen = v.name, 0
        while name in producer and seen < 5:
            node = producer[name]
            consts = [x for x in node.input if x in init_names]
            print(f"  {node.op_type:<18} {node.name or '-':<24} "
                  f"in={list(node.input)[:3]}{'  const=' + str(consts) if consts else ''}")
            name = node.input[0] if node.input else ""
            seen += 1

    # --- ScatterND：三个操作数各自的来源 ---
    for node in main.node:
        if node.op_type != "ScatterND" or not any(o == main.output[0].name for o in node.output):
            continue
        print("\nScatterND（输出拼接方式）:")
        for label, x in zip(("data   ", "indices", "updates"), list(node.input)[:3]):
            src, n_nodes = _reachable_inputs(producer, x, input_names)
            shape = tuple(inits[x].dims) if x in inits else None
            print(f"  {label}: {x}" + (f"  shape={shape}" if shape else "")
                  + f"  ↳ 反向可达图输入 {src or '<无，纯计算>'}（扫了 {n_nodes} 个张量）")
        # 输出帧数只能靠 --run 定：这里给静态线索
        upd = node.input[2]
        node_upd = producer.get(upd)
        if node_upd is not None:
            print(f"  ↳ updates 的生产者: {node_upd.op_type}({node_upd.name or '-'})"
                  f" in={list(node_upd.input)[:3]}")

    report["contract_details"] = {"stat_like_inits": [t.name for t in hits]}


# ---------------------------------------------------------------------------
# 1 / 6. 文件与同目录
# ---------------------------------------------------------------------------
def section_file(path: Path, report, show_neighbors: bool):
    size = path.stat().st_size
    print(f"路径            : {path}")
    print(f"大小            : {fmt_bytes(size)}")
    data_files = sorted(p for p in path.parent.glob(path.name + ".*"))
    if data_files:
        print("旁挂权重文件    : " + ", ".join(f"{p.name}({fmt_bytes(p.stat().st_size)})" for p in data_files))
    else:
        print("旁挂权重文件    : <无，权重全在 .onnx 里>")

    if show_neighbors:
        print("\n同目录文件（看有没有外置 mean/std/通道清单）:")
        entries = sorted(p for p in path.parent.iterdir() if p != path and not p.name.startswith("."))
        for p in entries[:60]:
            if p.is_dir():
                print(f"  {p.name}/")
            else:
                print(f"  {p.name:<60} {fmt_bytes(p.stat().st_size)}")
        if len(entries) > 60:
            print(f"  … 另 {len(entries) - 60} 项")
        hint = [p.name for p in entries if p.suffix.lower() in (".nc", ".npy", ".npz", ".json", ".csv", ".txt")]
        if hint:
            print(f"  ↳ 可能的归一化/通道文件: {hint[:10]}")

    report["file"] = {"path": str(path), "bytes": size,
                      "side_data": [p.name for p in data_files]}


# ---------------------------------------------------------------------------
# shape inference（可选）：中间张量的形状 = 注入点的可用性
# ---------------------------------------------------------------------------
def section_shape_inference(model, report):
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as exc:
        print(f"shape inference 失败（不影响其余部分）: {exc}")
        report["shape_inference"] = {"ok": False, "error": str(exc)}
        return

    def static_dims(v):
        t = v.type.tensor_type
        if not t.HasField("shape"):
            return None
        dims = []
        for d in t.shape.dim:
            if not d.HasField("dim_value"):
                return None
            dims.append(d.dim_value)
        return tuple(dims)

    all_values = list(inferred.graph.value_info) + list(inferred.graph.output)
    known = [v for v in all_values if static_dims(v)]
    unknown_outs = [v.name for v in inferred.graph.output if not static_dims(v)]
    print(f"可推断形状的中间张量 : {len(known)}/{len(all_values)}")
    if unknown_outs:
        print(f"输出维度仍不完整的   : {unknown_outs}")
    report["shape_inference"] = {"ok": True, "resolved": len(known), "total": len(all_values)}


# ---------------------------------------------------------------------------
# onnxruntime（可选）
# ---------------------------------------------------------------------------
def _preload_torch():
    """先 import torch 再碰 onnxruntime。

    torch 的 pip 包在导入时会把它自带的 nvidia-* 库目录挂进动态库搜索路径，
    并预加载 cudnn / cublas。不先导的话，onnxruntime 的 CUDA provider 会报
    ``libcudnn.so.9: cannot open shared object file``；自定义算子 .so 链接了
    libtorch 的话也会缺符号。失败不影响纯 CPU 的图结构分析，直接跳过。
    """
    try:
        import torch

        return torch.__version__
    except Exception:  # noqa: BLE001
        return None


def _preload_ort_lib():
    """RTLD_GLOBAL 预加载版本化的 libonnxruntime，供自定义算子 .so 解析。

    和框架 backends/onnx.py 的做法一致：插件 .so 里链接了 ORT 的符号，
    不先全局加载会报 undefined symbol。
    """
    import ctypes
    import glob

    try:
        import onnxruntime as _ort

        capi = os.path.join(os.path.dirname(_ort.__file__), "capi")
        for pattern in ("libonnxruntime.so.1", "libonnxruntime.so.1.*"):
            matches = sorted(glob.glob(os.path.join(capi, pattern)))
            if matches:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                return matches[0]
    except Exception:  # noqa: BLE001
        pass
    return None


def _plugin_candidates(model_path: Path, ort_version: str):
    """同目录里能注册的插件 .so，按「版本匹配度」排序。

    插件文件名自带适配信息：``ort1_24`` = ORT 1.24、``cpython-311`` = CPython 3.11、
    ``sm120`` = 只给 sm120 显卡。名字对不上的排后面，全都不匹配就按原顺序试。
    """
    import re

    major_minor = ".".join(ort_version.split(".")[:2])
    want_ort = "ort" + major_minor.replace(".", "_")
    want_py = f"cpython-{sys.version_info.major}{sys.version_info.minor}"

    def score(p):
        name = p.name
        s = 0
        if want_ort in name:
            s += 10
        elif re.search(r"ort\d+_\d+", name):
            s -= 5  # 明确写着别的 ORT 版本
        if want_py in name:
            s += 3
        elif re.search(r"cpython-\d+", name):
            s -= 3
        return -s

    return sorted(model_path.parent.glob("xmetai_onnx_plugins*.so"), key=score)


def section_ort(path: Path, report, do_run: bool, ops_lib=None, provider="auto"):
    # 必须赶在 onnxruntime 之前：torch 导入时才会把 nvidia 的 .so 路径挂上
    torch_version = _preload_torch()
    print(f"torch            : {torch_version or '<未安装，跳过预加载>'}")

    import onnxruntime as ort

    print(f"onnxruntime      : {ort.__version__}")
    available = ort.get_available_providers()
    print(f"可用 providers   : {available}")

    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        print("警告             : 没有 CUDAExecutionProvider，退回 CPU")
        providers = ["CPUExecutionProvider"]

    # 候选库：显式给了就只用它；没给就在模型同目录自己挑
    if ops_lib:
        libs = [Path(ops_lib).expanduser()]
        if not libs[0].is_file():
            print(f"自定义算子库     : 找不到 {libs[0]}")
            return
    else:
        libs = _plugin_candidates(path, ort.__version__)
        if libs:
            print(f"自定义算子库     : 自动探测到 {len(libs)} 个候选，按版本匹配度排序尝试")

    attempts = list(libs) if libs else [None]
    sess, last_error = None, None
    for lib in attempts:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if lib is not None:
            print(f"  注册 {lib.name}（先 RTLD_GLOBAL 预加载 libonnxruntime）")
            print(f"    预加载: {_preload_ort_lib() or '<跳过>'}")
            try:
                opts.register_custom_ops_library(str(lib))
            except Exception as exc:  # noqa: BLE001
                print(f"    注册失败: {exc}")
                last_error = exc
                continue
        t0 = time.perf_counter()
        try:
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  ✗ {lib.name if lib else '<无插件>'} 建 session 失败: {exc}")
            continue
        print(f"  ✓ 用时 {time.perf_counter() - t0:.2f}s"
              + (f"（{lib.name}）" if lib else ""))
        break

    if sess is None:
        print(f"session 建立失败 : {last_error}")
        report["ort"] = {"version": ort.__version__,
                         "providers": ort.get_available_providers(),
                         "error": str(last_error)}
        return
    print(f"session 建立     : ok")
    print(f"实际 provider    : {sess.get_providers()}")

    ins = sess.get_inputs()
    outs = sess.get_outputs()
    print(f"ORT 眼里输入     : " + ", ".join(f"{i.name}:{i.type}{i.shape}" for i in ins))
    print(f"ORT 眼里输出     : " + ", ".join(f"{o.name}:{o.type}{o.shape}" for o in outs))
    meta = sess.get_modelmeta()
    if meta.custom_metadata_map:
        print(f"ORT metadata     : {dict(meta.custom_metadata_map)}")

    report["ort"] = {
        "version": ort.__version__,
        "providers": ort.get_available_providers(),
        "inputs": [{"name": i.name, "type": i.type, "shape": i.shape} for i in ins],
        "outputs": [{"name": o.name, "type": o.type, "shape": o.shape} for o in outs],
    }

    if not do_run:
        return

    feeds = {}
    for i in ins:
        dims = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        np_dtype = np.float32 if "float" in i.type else (
            np.int64 if "int64" in i.type else (np.int32 if "int32" in i.type else np.float32))
        feeds[i.name] = np.zeros(dims, dtype=np_dtype)

    print("\n跑一次全零输入（CPU）…")
    t0 = time.perf_counter()
    results = sess.run(None, feeds)
    dt = time.perf_counter() - t0
    print(f"耗时             : {dt:.2f}s")
    stats = []
    for o, arr in zip(outs, results):
        a = np.asarray(arr)
        finite = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, dtype=bool)
        line = (f"  {o.name}: {a.dtype} {a.shape} "
                f"min={a.min() if a.size else float('nan'):.4g} "
                f"max={a.max() if a.size else float('nan'):.4g} "
                f"mean={a.mean() if a.size else float('nan'):.4g} "
                f"nan={int(a.size - finite.sum())}")
        print(line)
        stats.append({"name": o.name, "dtype": str(a.dtype), "shape": list(a.shape),
                      "min": float(a.min()) if a.size else None,
                      "max": float(a.max()) if a.size else None,
                      "mean": float(a.mean()) if a.size else None,
                      "nan": int(a.size - finite.sum())})
    report["run"] = {"seconds": dt, "outputs": stats}


# ---------------------------------------------------------------------------
# 契约摘要
# ---------------------------------------------------------------------------
def section_summary(report):
    hr("契约摘要（接入框架前要确认的几件事）")
    ins, outs = report.get("inputs", []), report.get("outputs", [])
    for r in ins:
        print(f"  输入 {r['name']}: {r['dtype']} {tuple(r['shape'])}")
    for r in outs:
        print(f"  输出 {r['name']}: {r['dtype']} {tuple(r['shape'])}")

    questions = [
        ("输入 rank 4 = [B,C,H,W]；rank 5 = [B,T,C,H,W]（框架 prepare_dataset_batch 出 5 维）", None),
        ("归一化在不在图里：看图 metadata / doc_string，再看同目录有没有 mean.nc、std.nc", None),
        ("输出是物理量还是残差：上面的「输出生产节点」若为 Add 且吃图输入，就是残差",
         "residual_outputs" if report.get("suspects", {}).get("residual_outputs") else None),
        ("单步还是多步：输出与输入同形 → 单步自回归，框架按 step 循环调用", None),
        ("自定义算子域 → 需要额外 .so 注册（参考 fgvp 的 FGVP_OPS_LIBRARY）",
         report.get("graph", {}).get("custom_domains") or None),
    ]
    print("\n需要人工确认：")
    for text, extra in questions:
        mark = "!"
        print(f"  [{mark}] {text}" + (f"  → 本图命中: {extra}" if extra else ""))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="ONNX 模型契约探针（只读）")
    ap.add_argument("model", help=".onnx 路径")
    ap.add_argument("--ort", action="store_true", help="额外用 onnxruntime 开 session 核对 I/O")
    ap.add_argument("--run", action="store_true", help="跑一次全零输入（隐含 --ort）")
    ap.add_argument("--ops-lib", metavar="SO",
                    help="自定义算子库 .so；不给则自动扫模型同目录的 xmetai_onnx_plugins*.so，"
                         "按 ORT/CPython 版本匹配度排序逐个尝试")
    ap.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto",
                    help="--ort 用的执行后端，默认有 CUDA 就用 CUDA")
    ap.add_argument("--json", metavar="F", help="报告另存 JSON")
    ap.add_argument("--no-ls", action="store_true", help="不列同目录文件")
    args = ap.parse_args(argv)

    path = Path(args.model).expanduser()
    if not path.is_file():
        raise SystemExit(f"找不到文件: {path}")

    report = {}
    hr("1. 文件")
    section_file(path, report, show_neighbors=not args.no_ls)

    model = None
    if onnx is not None:
        try:
            model = onnx.load(str(path), load_external_data=False)
        except Exception as exc:
            print(f"\n[!] onnx.load 失败，退化成 onnxruntime 视图: {exc}")

    if model is not None:
        hr("2. 模型级信息")
        section_meta(model, report)

        hr("3. 输入输出")
        section_io(model.graph, report)

        hr("4. 图结构")
        graphs, op_counter = section_graph(model, report)

        hr("5. 可疑点")
        section_suspects(model, graphs, op_counter, report)

        hr("6. 活跃性（从输出反推）")
        section_liveness(model, report)

        hr("7. 归一化与输出形态")
        section_contract_details(model, report)

        hr("附加：shape inference")
        section_shape_inference(model, report)
    else:
        print("\n[!] 没有 onnx 包，跳过图结构；只给 onnxruntime 视图。")
        print("    装法: pip install onnx  （或 pip install -e '.[onnx]'）")

    if args.ort or args.run:
        hr("附加：onnxruntime")
        section_ort(path, report, do_run=args.run,
                    ops_lib=args.ops_lib, provider=args.provider)

    section_summary(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nJSON 已写入 {args.json}")


if __name__ == "__main__":
    main(sys.argv[1:])
