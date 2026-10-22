# -*- coding: utf-8 -*-
"""探针：量「扰动到底把输出改变了多少」。

这一轮失败的根因之一是没有这个工具。当时唯一的加载期自检是「两次同输入前向是否逐位
相同」——它只能抓「噪声完全没生效」，抓不到「生效了但只有应有量级的 1/8」。于是 σ=0.01
的偏差一路静默跑完 51 成员 x 60 步的生产任务，最后在评测里才以「离散度太差」的形式暴
露出来。

探针不需要实况、不需要 rollout：

  * **成员集量级**（``member_scan``）：逐成员与源模型同输入比一次，看每个成员把输出推
    动了多少。成员内是冻结的，所以 runs=1；量级靠与源模型比对得到。
  * **单图离散度**：同一组输入连跑 N 次前向，报跨 run 的无偏样本标准差相对输出 RMS
    rel = std(x_k; ddof=1) / rms(mean_k x_k)。确定性图（含成员图）应为 0。

注意 rel 是**权重扰动在单步输出上的相对放大**，不等于集合的 spread/RMSE 比。后者还要
乘上误差随时效的增长，只能靠短 rollout + 已有的确定性 RMSE 曲线来定标（见 README）。
探针的作用是把这个定标从「一遍 51 成员生产任务」压到「几次前向」。
"""
from __future__ import annotations

import os.path as osp

import numpy as np

from ..weights.spw import WeightPerturbSpec


# ---------------------------------------------------------------------------
# 会话与输入
# ---------------------------------------------------------------------------
def make_session(model_path: str, *, provider: str = "cpu", device: int = 0,
                 ops_library: str | None = None):
    """建 session 的**唯一**实现：探针 / 自检 / e2e 都走这里。

    `enable_mem_pattern=False` 与「日志只报 error」是共同的硬要求（探针要干净输出，
    自检的合成图里还有故意留的无消费者张量）；provider 与自定义算子按需加。
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.log_severity_level = 3          # 只报 error：ORT 的警告会淹掉探针输出
    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        providers.insert(0, ("CUDAExecutionProvider", {"device_id": device}))
    if ops_library:
        from xmetai_inference.backends.onnx import _preload_onnxruntime_lib

        _preload_onnxruntime_lib()
        options.register_custom_ops_library(ops_library)
    return ort.InferenceSession(model_path, sess_options=options, providers=providers)


def make_feeds(session, *, seed: int = 0, mode: str = "normal") -> dict:
    """按图的输入签名造一组输入。

    形状含动态维（None）时无法造——直接用 None 表示「这组输入没法自动造」，调用方报错。
    """
    rng = np.random.default_rng(seed)
    feeds = {}
    for meta in session.get_inputs():
        shape = [dim if isinstance(dim, int) else None for dim in meta.shape]
        if any(dim is None for dim in shape):
            return None
        if "float" in meta.type or "double" in meta.type or "float16" in meta.type:
            if mode == "zeros":
                feeds[meta.name] = np.zeros(shape, dtype=np.float32)
            elif mode == "ones":
                feeds[meta.name] = np.ones(shape, dtype=np.float32)
            else:
                feeds[meta.name] = rng.standard_normal(shape).astype(np.float32)
        else:
            feeds[meta.name] = np.zeros(shape, dtype=np.int64)
    return feeds


def _nan_ops(array):
    value = np.asarray(array, dtype=np.float64)
    return value


def _rms(array) -> float:
    value = _nan_ops(array)
    if value.size == 0:
        return 0.0
    with np.errstate(invalid="ignore"):
        return float(np.sqrt(np.nanmean(np.square(value))))


def _diff_rms(left, right) -> float:
    value = _nan_ops(left) - _nan_ops(right)
    if value.size == 0:
        return 0.0
    with np.errstate(invalid="ignore"):
        return float(np.sqrt(np.nanmean(np.square(value))))


def _channel_axis(shape) -> int | None:
    """[B, F, C, H, W] 惯例下通道轴 = 倒数第 3 维；不足 3 维就没有可拆的通道轴。"""
    return len(shape) - 3 if len(shape) >= 3 else None


def _channel_stats(mean, stack, reference, axis: int, top: int = 8):
    moved = np.moveaxis(_nan_ops(mean), axis, 0)
    stacked = np.moveaxis(_nan_ops(stack), axis, 1)
    rows = []
    for index in range(moved.shape[0]):
        base = moved[index]
        field = _rms(base)
        if stacked.shape[1] > 1:
            with np.errstate(invalid="ignore"):
                    drift = float(np.sqrt(np.nanmean(
                    np.nanmean(np.square(stacked[index] - base), axis=0)
                    * stacked.shape[1] / (stacked.shape[1] - 1))))
        else:
            drift = 0.0
        row = {"channel": index, "field_rms": field, "drift_rms": drift,
               "rel": drift / field if field else float("nan")}
        if reference is not None:
            ref = np.moveaxis(_nan_ops(reference), axis, 0)
            row["rel_reference"] = (_diff_rms(base, ref[index]) / field
                                    if field else float("nan"))
        rows.append(row)
    rows.sort(key=lambda item: -(item["rel"] if np.isfinite(item["rel"]) else -1.0))
    return rows[:top]


# ---------------------------------------------------------------------------
# 单个模型的探测
# ---------------------------------------------------------------------------
def probe(model_path: str, *, runs: int = 3, reference: str | None = None,
          provider: str = "cpu", device: int = 0, ops_library: str | None = None,
          seed: int = 0, feeds_mode: str = "normal", per_channel: bool = False,
          runs_if_reference: int = 1) -> dict:
    """探一个模型：runs 次同输入前向 + 可选与 reference 比对。"""
    session = make_session(model_path, provider=provider, device=device,
                           ops_library=ops_library)
    feeds = make_feeds(session, seed=seed, mode=feeds_mode)
    if feeds is None:
        return {"model": model_path, "ok": False,
                "message": "图的输入形状含动态维，无法自动造输入；请显式提供 feeds",
                "outputs": []}
    reference_arrays = None
    if reference:
        reference_session = make_session(reference, provider=provider, device=device,
                                         ops_library=ops_library)
        reference_arrays = reference_session.run(None, feeds)

    effective_runs = runs if reference is None else max(1, runs_if_reference)
    arrays = [session.run(None, feeds) for _ in range(effective_runs)]
    names = [item.name for item in session.get_outputs()]

    # drift 以「多次前向的均值」为基准，也就是跨 run 的离散度：以第 0 次为基准的话，
    # 两次抽样的差值会带上 √2（x1-x0 = σW(ξ1-ξ0)，rms(ξ1-ξ0)=√2），刻度就偏了。
    outputs = []
    for slot, name in enumerate(names):
        stack = np.stack([_nan_ops(run[slot]) for run in arrays])
        mean = np.nanmean(stack, axis=0)
        field = _rms(mean)
        if effective_runs > 1:
            # 无偏（ddof=1）：M 次抽样的总体标准差会系统偏小 sqrt((M-1)/M) 倍，
            # M=3 时就是 0.816——正好是「探针刻度差 1/0.816」的来源。
            with np.errstate(invalid="ignore"):
                per_element = np.nanmean(np.square(stack - mean), axis=0)
                per_element *= effective_runs / (effective_runs - 1)
                drift = float(np.sqrt(np.nanmean(per_element)))
        else:
            drift = 0.0
        row = {
            "name": name,
            "shape": list(mean.shape),
            "dtype": str(mean.dtype),
            "field_rms": field,
            "drift_rms": drift,
            "rel": drift / field if field else float("nan"),
        }
        if reference_arrays is not None:
            row["reference_rms"] = _diff_rms(mean, reference_arrays[slot])
            row["rel_reference"] = (row["reference_rms"] / field
                                    if field else float("nan"))
        axis = _channel_axis(mean.shape)
        if per_channel and axis is not None:
            row["channels"] = _channel_stats(
                mean, stack, reference_arrays[slot] if reference_arrays else None, axis)
        outputs.append(row)
    return {"model": model_path, "reference": reference, "runs": effective_runs,
            "ok": True, "outputs": outputs}


def format_probe(report: dict, per_channel: bool = False) -> str:
    lines = []
    if not report.get("ok"):
        return f"{osp.basename(report['model'])}: {report.get('message')}"
    lines.append(f"探针：{osp.basename(report['model'])}"
                 f"（同输入 {report['runs']} 次前向；rel 为跨 run 的无偏离散度）")
    if report.get("reference"):
        lines.append(f"  参照：{osp.basename(report['reference'])}")
    lines.append(f"  {'output':22s} {'shape':24s} {'field_rms':>12s} "
                 f"{'drift_rms':>12s} {'rel':>10s}" +
                 (f" {'rel_vs_ref':>11s}" if report.get("reference") else ""))
    for row in report["outputs"]:
        shape = "x".join(str(item) for item in row["shape"])
        line = (f"  {row['name']:22s} {shape:24s} {row['field_rms']:12.4g} "
                f"{row['drift_rms']:12.4g} {row['rel']:10.4g}")
        if report.get("reference"):
            line += f" {row.get('rel_reference', float('nan')):11.4g}"
        lines.append(line)
        if per_channel and row.get("channels"):
            for item in row["channels"]:
                lines.append(f"      ch{item['channel']:<4d} field_rms="
                             f"{item['field_rms']:<10.4g} rel={item['rel']:.4g}")
    if all(row["drift_rms"] == 0.0 for row in report["outputs"]):
        lines.append("  [注意] 同输入多次前向逐位相同：图里没有随机性（确定性模型，"
                     "或噪声被图优化折掉了）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------
def member_scan(src_path: str, spec: WeightPerturbSpec, count: int, *,
                runs: int = 1, provider: str = "cpu", device: int = 0,
                ops_library: str | None = None, seed: int = 0,
                feeds_mode: str = "normal") -> dict:
    """成员集探测：每个成员与源模型同输入比一次，看逐成员的输出差异。

    成员内是冻结的（同图两次前向逐位相同），所以 runs=1；靠与源模型比对得到量级。
    """
    from ..weights.spw import member_model_path

    rows = []
    for index in range(count):
        path = member_model_path(src_path, spec, index)
        if not osp.isfile(path):
            rows.append({"index": index, "model": path, "rel": float("nan"),
                         "missing": True})
            continue
        report = probe(path, runs=runs, reference=src_path, provider=provider,
                       device=device, ops_library=ops_library, seed=seed,
                       feeds_mode=feeds_mode)
        rel = max((row.get("rel_reference", 0.0) for row in report.get("outputs", [])),
                  default=0.0)
        rows.append({"index": index, "model": path, "rel": rel, "report": report})
    return {"rows": rows, "count": count}


def format_member_scan(result: dict) -> str:
    lines = ["逐成员探测（rel = 成员输出相对源模型的 RMS 差异）：",
             f"  {'member':>8s} {'rel':>12s}"]
    values = []
    for row in result["rows"]:
        if row.get("missing"):
            lines.append(f"  {row['index']:8d} {'缺失':>12s}")
            continue
        values.append(row["rel"])
        lines.append(f"  {row['index']:8d} {row['rel']:12.5g}")
    if len(values) >= 2:
        lines.append(f"  逐成员 rel 的极差倍数 = {max(values) / max(min(values), 1e-12):.3f}"
                     "（成员间应彼此相当：太小 = 噪声流没分开，太大 = 有不稳成员）")
    return "\n".join(lines)


__all__ = [
    "format_member_scan",
    "format_probe",
    "make_feeds",
    "make_session",
    "member_scan",
    "probe",
]
