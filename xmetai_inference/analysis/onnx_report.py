# -*- coding: utf-8 -*-
"""ONNX 静态体检：给一个 .onnx 路径，打印接模型前必须确认的信息。

不建 session、不读外部数据、不碰 GPU —— 只读图 proto（``load_external_data=False``），
CPU 上几秒。回答四件事：

  1. **I/O 契约**：输入/输出的名字、dtype、形状；5D 输入会直接标出帧数 / 通道数 / 网格。
  2. **图的构成**：opset、节点数、算子直方图、自定义算子（非 ai.onnx 域）、随机算子个数。
     ``RandomNormalLike`` 之类的数量决定它是不是「图内自带随机轴的集合模型」。
  3. **SPA 状态**：图里有没有 ``spw/xi`` 输入；有的话打印它的形状和 ``spw.ar1.*`` 元数据。
  4. **SPA 锚点候选**：直接喂给 ``LayerNormalization`` 的 ``Add`` 输出。换骨干时先看这节，
     别名表（``weights/spa.py`` 的 ``latent_state``）只对 FGVP 有效，别的骨干要用实测名字。

另附 initializer 体量与外部数据文件清单 —— 带 ``.data`` 的图交付时必须同目录。

用法（服务器仓库根目录）::

    python -m xmetai_inference.analysis.onnx_report <model.onnx>
    python xmetai_inference/analysis/onnx_report.py <model.onnx> --no-infer

    # 确认锚点位置：打印指定张量的生产者/消费者（可给多个，逗号分隔）
    python -m xmetai_inference.analysis.onnx_report <model.onnx> \
        --inspect add_35722,add_35727,add_35732,add_36164

``--no-infer`` 跳过 shape inference（大图更快，但锚点只列名字、不显示形状）。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

try:
    import onnx
except ImportError:  # pragma: no cover - 环境缺依赖时给一句人话
    sys.exit("需要 onnx：pip install onnx（或在带 onnx 的环境里运行）")

_DTYPE = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
    12: "uint32", 13: "uint64", 16: "bfloat16",
}
_ITEMSIZE = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1,
    10: 2, 11: 8, 12: 4, 13: 8, 16: 2,
}
_FORCING_INPUT = "spw/xi"


def _dtype(elem_type):
    return _DTYPE.get(int(elem_type), f"type#{elem_type}")


def _shape(tensor_type):
    """protobuf TensorTypeProto -> ['2', '71', '721', '1440']（符号维原样保留）。"""
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def _human_bytes(count):
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _numel(dims):
    total = 1
    for dim in dims:
        if not str(dim).isdigit():
            return None
        total *= int(dim)
    return total


def _io_line(tag, name, dtype, dims):
    line = f"  {tag:<7} {name:<16} {dtype:<9} ({', '.join(dims)})"
    if len(dims) == 5:
        # 约定 [B, T, C, H, W]
        line += f"   # 帧 T={dims[1]}, 通道 C={dims[2]}, 网格 {dims[3]}x{dims[4]}"
    elif len(dims) == 4:
        line += f"   # 通道 C={dims[1]}, 网格 {dims[2]}x{dims[3]}"
    elif len(dims) == 3:
        line += f"   # 2D token 序列: {dims[1]} x {dims[2]}"
    return line


def _load(path, infer_shapes):
    model = onnx.load(path, load_external_data=False)
    if not infer_shapes:
        return model, {}
    try:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=False)
    except Exception as error:  # 大图 / 自定义算子都可能在推断阶段报错
        print(f"⚠ shape inference 失败，形状只显示图里显式写的：{error}")
        return model, {}
    shapes = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        if value.name:
            shapes[value.name] = _shape(value.type.tensor_type)
    return model, shapes


def _report(path, model, shapes):
    graph = model.graph
    print("=" * 78)
    print(f"文件   : {os.path.abspath(path)}")
    print(f"大小   : {_human_bytes(os.path.getsize(path))}")
    opsets = ", ".join(
        f"{(opset.domain or 'ai.onnx')}={opset.version}"
        for opset in model.opset_import
    )
    print(f"IR/opset: IR {model.ir_version} / {opsets}")
    if model.producer_name:
        print(f"导出器 : {model.producer_name} {model.producer_version}".rstrip())

    initializer_names = {init.name for init in graph.initializer}
    inputs = [value for value in graph.input if value.name not in initializer_names]
    outputs = list(graph.output)
    print(f"节点   : {len(graph.node)}    initializer: {len(graph.initializer)}")

    print("\n[1] I/O 契约")
    if not inputs:
        print("  （没有输入？）")
    for value in inputs:
        print(_io_line("input", value.name, _dtype(value.type.tensor_type.elem_type),
                       _shape(value.type.tensor_type)))
    for value in outputs:
        print(_io_line("output", value.name, _dtype(value.type.tensor_type.elem_type),
                       _shape(value.type.tensor_type)))

    print("\n[2] 图的构成")
    counts = Counter(node.op_type for node in graph.node)
    top = counts.most_common(15)
    print("  算子 top15: " + ", ".join(f"{name}x{count}" for name, count in top))
    custom = Counter(
        f"{node.domain}::{node.op_type}"
        for node in graph.node
        if node.domain not in ("", "ai.onnx")
    )
    if custom:
        print("  自定义算子: " + ", ".join(f"{name}x{count}" for name, count in custom.items()))
    else:
        print("  自定义算子: 无")
    random_ops = Counter(
        node.op_type for node in graph.node if node.op_type.startswith("Random")
    )
    if random_ops:
        print("  随机算子: " + ", ".join(f"{name}x{count}" for name, count in random_ops.items())
              + "   # 图内自带随机轴 -> 集合模型")
    else:
        print("  随机算子: 无   # 确定性图")

    print("\n[3] SPA 状态")
    xi = next((value for value in inputs if value.name == _FORCING_INPUT), None)
    meta = {prop.key: prop.value for prop in model.metadata_props}
    if xi is None:
        print(f"  未注入（没有 {_FORCING_INPUT!r} 输入）")
    else:
        print(f"  已注入：{_FORCING_INPUT} {_dtype(xi.type.tensor_type.elem_type)} "
              f"({', '.join(_shape(xi.type.tensor_type))})")
        ar1_meta = {key: value for key, value in meta.items() if key.startswith("spw.ar1.")}
        for key, value in sorted(ar1_meta.items()):
            print(f"    {key} = {value}")
        if not ar1_meta:
            print("    ⚠ 有 spw/xi 但没有 spw.ar1.* 元数据，可能不是本框架注入的")

    print("\n[4] SPA 锚点候选（Add -> LayerNormalization）")
    ln_inputs = {
        name
        for node in graph.node
        if node.op_type == "LayerNormalization"
        for name in node.input
        if name
    }
    candidates = []
    for node in graph.node:
        if node.op_type != "Add":
            continue
        for name in node.output:
            if name in ln_inputs:
                candidates.append((name, node.name))
    if not candidates:
        print("  没有 Add 直接喂 LayerNormalization 的模式；换骨干时需按图自行选锚点")
    else:
        for name, node_name in candidates:
            dims = shapes.get(name)
            shape_text = f"({', '.join(dims)})" if dims else ""
            print(f"  {name:<22} node={node_name or '<unnamed>':<18} {shape_text}")

    print("\n[5] initializer 体量")
    total_bytes = 0
    external = []
    for init in graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            length = next(
                (item.value for item in init.external_data if item.key == "length"), None
            )
            if length is not None and str(length).isdigit():
                total_bytes += int(length)
            external.append(init)
            continue
        numel = _numel(init.dims)
        itemsize = _ITEMSIZE.get(int(init.data_type))
        if numel is not None and itemsize is not None:
            total_bytes += numel * itemsize
    print(f"  合计约 {_human_bytes(total_bytes)}")
    if external:
        files = Counter()
        for init in external:
            for item in init.external_data:
                if item.key == "location":
                    files[item.value] += 1
        print(f"  外部数据: {len(external)} 个 tensor，文件 "
              + ", ".join(f"{name}（{count} 个 tensor）" for name, count in files.items()))
        print("  ⚠ 交付/推理时 .data 必须与 .onnx 同目录，且路径校验看的是真实路径")
    else:
        print("  外部数据: 无（权重全部内嵌）")

    if meta:
        print("\n[6] metadata_props")
        for key, value in sorted(meta.items()):
            print(f"  {key} = {value}")


def _inspect(graph, shapes, names):
    """打印指定张量的生产者/消费者，用来判断 Add 是投影偏置加法还是残差加法。"""
    init_shapes = {
        init.name: [str(dim) for dim in init.dims] for init in graph.initializer
    }
    producer = {}
    for node in graph.node:
        for output in node.output:
            if output:
                producer[output] = node
    consumers = {}
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name:
                consumers.setdefault(name, []).append((node, index))
    graph_inputs = {value.name for value in graph.input}

    def describe(name):
        dims = shapes.get(name) or init_shapes.get(name)
        shape = f"({', '.join(dims)})" if dims else ""
        if name in init_shapes:
            return f"{name} [initializer {shape}]"
        node = producer.get(name)
        if node is not None:
            return f"{name} <- {node.op_type} node={node.name or '<unnamed>'} {shape}"
        if name in graph_inputs:
            return f"{name} [graph input {shape}]"
        return f"{name} {shape}".rstrip()

    print("\n[7] inspect")
    for name in names:
        print(f"  {name}")
        node = producer.get(name)
        if node is None:
            print("    （不是图内节点的输出；名字可能写错）")
            continue
        print(f"    producer: {node.op_type} node={node.name or '<unnamed>'}")
        for operand in node.input:
            if operand:
                print("      in : " + describe(operand))
        uses = consumers.get(name, [])
        if not uses:
            print("    consumers: 无（图输出或死支路）")
        for consumer, index in uses:
            print(f"    consumer: {consumer.op_type} "
                  f"node={consumer.name or '<unnamed>'} input#{index}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ONNX 静态体检：I/O、通道数、自定义算子、SPA 状态与锚点候选")
    parser.add_argument("model", help=".onnx 文件路径")
    parser.add_argument("--no-infer", action="store_true",
                        help="跳过 shape inference（大图更快；锚点只列名字）")
    parser.add_argument("--inspect", default="",
                        help="逗号分隔的张量名，打印生产者/消费者（确认 SPA 锚点用）")
    args = parser.parse_args(argv)
    if not os.path.isfile(args.model):
        raise SystemExit(f"文件不存在：{args.model}")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    model, shapes = _load(args.model, infer_shapes=not args.no_infer)
    _report(args.model, model, shapes)
    names = [
        name.strip()
        for name in args.inspect.replace(",", " ").split()
        if name.strip()
    ]
    if names:
        _inspect(model.graph, shapes, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
