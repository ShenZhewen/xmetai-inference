# -*- coding: utf-8 -*-
r"""图内自带噪声的**幅度**旋钮：把集合图里那条 σ 乘一个整数/小数 k（权重只动一个 initializer）。

为什么需要它（2026-10-12，fgvp_ens / iwc_gssp_im_fp16 的结构实测）：

    cat_9 [b,16200,1536]            ← 潜状态（Concat 三个 VAE 组出口）
      → pad → ConvTranspose(prior_perturb.mean.*)   → μ = Slice(convolution)
      → pad → ConvTranspose(prior_perturb.logvar.*) → σ = Exp(Mul(Slice(convolution_1), 0.5))
      ε = RandomNormalLike(μ)       ← 全图唯一一个随机算子，scale 属性 = 1.0，每步前向重抽
      normal = μ + σ ⊙ ε            ← 51 个成员的唯一来源

σ = exp(logvar/2) 是逐点的，所以「整幅 logvar 平移 δ」等价于「σ 逐点乘 exp(δ/2)」：

    σ × k   ⇔   prior_perturb.logvar.bias += 2·ln k      ← 精确、与 k 无关的常数

这是**改幅度**那条腿，与 SPW（改权重、`weights.py`）和 latent_forcing（改时间相关、
`latent.py`）都正交：SPW 动的是另一批张量，AR(1) 动的是图结构。三者可以叠。

与 probe 的关系：probe 量的是「注入 σ → 输出响应」的斜率，本工具量的是**图自带**那条
噪声的绝对幅度。集合图上 probe 的曲线会被图内噪声污染，先跑本工具（或先测 σ=0 的噪声底）
再谈标定。

用法：

    P="python -m xmetai_inference.perturbation"
    $P noise-scale <集合图> --k 2.0            # σ ×2，输出 <stem>.k2.onnx
    $P noise-scale <集合图> --k 0.7 --out /tmp/x.onnx
"""
from __future__ import annotations

import math
import os.path as osp

import numpy as np
from onnx import numpy_helper

# σ 与 logvar 的换算：σ = exp(logvar / factor)。图上实测的系数是 0.5（见模块头），
# 所以 factor = 2。写成字面量默认值，图上真去核一遍（_verify 会把实测值打出来）。
DEFAULT_FACTOR = 2.0
# logvar 侧的那个 bias（156,）。名字是集合图导出器的命名，换图要重新确认（--tensor）。
DEFAULT_TENSOR = "prior_perturb.logvar.bias"


def _onnx():
    import onnx

    return onnx


def default_scaled_path(src: str, k: float) -> str:
    """输出路径：与源图**同目录**（图里的外部数据引用必须落在模型目录内，见坑 6）。"""
    stem = osp.splitext(osp.basename(src))[0]
    return osp.join(osp.dirname(osp.abspath(src)), f"{stem}.k{k:g}.onnx")


def _producer_map(graph):
    return {value: node for node in graph.node for value in node.output}


def _scalar(value_name, initializers):
    """名字指向的标量常量 → float；不是标量常量就返回 None。"""
    tensor = initializers.get(value_name)
    if tensor is None or tensor.dims and int(np.prod(tensor.dims)) != 1:
        return None
    return float(numpy_helper.to_array(tensor).reshape(-1)[0])


def verify_structure(model, tensor: str, factor: float) -> dict:
    """核对「这个 initializer 真的是自带噪声的 logvar」这条链，返回实测报告。

    只做能廉价做对的三件事：bias 被 ConvTranspose 消费、从该 ConvTranspose 出发能在
    若干跳内走到一个 Exp（沿途记下乘过的标量常量）、全图只有一个 RandomNormalLike。
    结论写进报告让调用方自己判断，不在这里抛错 —— σ 的换算系数图上写死是 0.5，
    但真按 --factor 算错了只是幅度差一点，不值得拦住一次跑。
    """
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}
    consumers = {}
    for node in graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    producer = _producer_map(graph)

    report = {"tensor": tensor, "factor": factor, "consumers": [], "exp_found": False,
              "observed_factor": None, "random_ops": [], "warnings": []}

    if tensor not in initializers:
        report["warnings"].append(f"{tensor!r} 不是这张图的 initializer")
        return report
    for node in consumers.get(tensor, []):
        report["consumers"].append(f"{node.op_type}:{node.name}")

    # 往前追：ConvTranspose -> ... -> Exp，沿途的 Mul(scalar) 就是那个 factor
    start = [node for node in consumers.get(tensor, []) if node.op_type == "ConvTranspose"]
    seen, frontier = set(), [out for node in start for out in node.output]
    multipliers = []
    for _ in range(24):
        nxt = []
        for value in frontier:
            for node in consumers.get(value, []):
                if node.name in seen:
                    continue
                seen.add(node.name)
                if node.op_type == "Exp":
                    report["exp_found"] = True
                    continue
                if node.op_type == "Mul":
                    for other in node.input:
                        value_of = _scalar(other, initializers)
                        if value_of is not None and other != value:
                            multipliers.append(value_of)
                nxt.extend(node.output)
        if report["exp_found"] or not nxt:
            break
        frontier = nxt
    if multipliers:
        # σ = exp(c·x) ⇔ x 的系数 c；factor = 1/c
        observed = 1.0 / multipliers[0] if multipliers[0] else None
        report["observed_factor"] = observed
        if observed is not None and abs(observed - factor) > 1e-6:
            report["warnings"].append(
                f"图上实测的换算系数是 1/{multipliers[0]:g} = factor {observed:g}，"
                f"与 --factor {factor:g} 不一致；σ×k 的位移量按实测值算会是 "
                f"{observed * math.log(2.0):.4g}")
    elif not report["exp_found"]:
        report["warnings"].append("从该 ConvTranspose 出发没走到 Exp —— 这条链可能不是 σ")

    for node in graph.node:
        if "Random" in node.op_type:
            report["random_ops"].append(f"{node.op_type}:{node.name}->{list(node.output)}")
    if len(report["random_ops"]) != 1:
        report["warnings"].append(
            f"全图有 {len(report['random_ops'])} 个随机算子（期望 1）—— 幅度模型的假设要重核")
    return report


def scale_builtin_noise(path: str, *, k: float, tensor: str = DEFAULT_TENSOR,
                        factor: float = DEFAULT_FACTOR, out: str | None = None) -> dict:
    """把 `tensor` 整幅平移 factor·ln(k)，等价于图内噪声 σ 逐点乘 k。返回报告 dict。"""
    onnx = _onnx()
    if k <= 0:
        raise SystemExit("--k 必须为正（σ 是 exp 出来的，永远为正）")
    model = onnx.load(path)
    report = verify_structure(model, tensor, factor)
    if report["warnings"] and any("不是这张图" in w for w in report["warnings"]):
        raise SystemExit(
            f"{path} 里没有 initializer {tensor!r}。用 --tensor 指定，或先跑 "
            f"`groups` 看这张图的张量名结构。")

    initializers = {t.name: t for t in model.graph.initializer}
    before = numpy_helper.to_array(initializers[tensor])
    delta = factor * math.log(k)
    after = (before.astype(np.float64) + delta).astype(before.dtype)
    initializers[tensor].CopyFrom(numpy_helper.from_array(after, tensor))

    props = {entry.key: entry.value for entry in model.metadata_props}
    props.update({
        "spw.builtin_noise.method": "amplitude-scale",
        "spw.builtin_noise.tensor": tensor,
        "spw.builtin_noise.k": f"{k:g}",
        "spw.builtin_noise.factor": f"{factor:g}",
        "spw.builtin_noise.delta": f"{delta:.6g}",
        "spw.builtin_noise.src": osp.basename(path),
    })
    onnx.helper.set_metadata_props(model, props)

    out = out or default_scaled_path(path, k)
    external = any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
    if external:
        onnx.save_model(model, out, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=osp.basename(out) + ".data", size_threshold=1024)
    else:
        onnx.save_model(model, out)

    report.update({
        "path": osp.abspath(path), "out": osp.abspath(out), "k": k, "delta": delta,
        "external_data": external,
        "n_elements": int(before.size),
        "before": (float(before.min()), float(before.mean()), float(before.max())),
        "after": (float(after.min()), float(after.mean()), float(after.max())),
    })
    return report


def format_noise_scale_report(report: dict) -> list[str]:
    lines = ["图内噪声幅度：σ × %g" % report["k"],
             "  源图   %s" % report["path"],
             "  输出   %s%s" % (report["out"], "（含外部数据 .data）" if report["external_data"] else ""),
             "  张量   %s（%d 个元素）" % (report["tensor"], report["n_elements"])]
    lo, mean, hi = report["before"]
    lo2, mean2, hi2 = report["after"]
    lines.append("  logvar 前  min=%+.6g mean=%+.6g max=%+.6g" % (lo, mean, hi))
    lines.append("  logvar 后  min=%+.6g mean=%+.6g max=%+.6g   (整幅 +%.6g)"
                 % (lo2, mean2, hi2, report["delta"]))
    lines.append("  ⇒ σ 逐点 ×%g（σ = exp(logvar/%g)）" % (report["k"], report["factor"]))
    lines.append("  结构核对  消费它的节点 %s；Exp %s；换算系数实测 %s"
                 % (", ".join(report["consumers"]) or "无",
                    "命中" if report["exp_found"] else "未命中",
                    ("%g" % report["observed_factor"]) if report["observed_factor"] else "未测到"))
    lines.append("  随机算子  %d 个：%s"
                 % (len(report["random_ops"]), ", ".join(report["random_ops"]) or "无"))
    for warning in report["warnings"]:
        lines.append("  ⚠ %s" % warning)
    return lines
