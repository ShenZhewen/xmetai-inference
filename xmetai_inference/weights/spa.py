# -*- coding: utf-8 -*-
"""潜空间 AR(1) 强迫（SPA）：规格、基、ONNX 图改写、生成与审计。

驱动侧按 ``ξ_{t+1} = ρξ_t + sqrt(1-ρ²)ε_t`` 递推（ρ = exp(-Δt/τ)），图内把
``n = Σ_r ξ_r·u_r⊗v_r`` 乘到潜变量上：``z' = z ⊙ (1 + σn)``。空间基经高斯低通
去 DC，``length_scale`` 控制注入尺度。

递推放在驱动侧、图只多一个 ``spw/xi`` 输入（单步 ONNX 图没有可靠的跨步状态）：
同一份图可服务全部成员，与权重扰动正交可叠加。图被改写但权重未动。

造产物：改本文件末尾字面量后运行 ``python -m xmetai_inference.weights.spa``。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

log = logging.getLogger(__name__)


# 底座：spec 样板 + ONNX 读写/外部数据校验（SPW 也从这里 import）
class SpecBase:
    """spec 类的公共样板。子类至少要提供 `budgets()`。"""

    #: from_config 的类型报错里用的名字（不要加类型注解）
    config_error = "spec"

    @classmethod
    def from_config(cls, value: Any):
        """配置字段 -> 规格；None/False 表示不启用。"""
        if value is None or value is False:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        if value is True:
            return cls()
        raise TypeError(
            f"{cls.config_error} 应为 dict / {cls.__name__} / None，"
            f"实际 {type(value).__name__}")

    def budgets(self) -> dict:
        """参与指纹的字段。子类必须实现。"""
        raise NotImplementedError

    def source_identity(self, src_path: str) -> dict:
        """源模型身份：文件换了，成员集 / 改写图就必须失效。"""
        stat = os.stat(src_path)
        return {
            "source": osp.basename(src_path),
            "size": int(stat.st_size),
            "mtime": int(stat.st_mtime),
        }

    def fingerprint_payload(self, src_path: str) -> dict:
        """进指纹的完整载荷。子类可以加键。"""
        return {"spec": self.budgets(),
                "source_identity": self.source_identity(src_path)}

    def fingerprint(self, src_path: str) -> str:
        payload = self.fingerprint_payload(src_path)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:10]


def _onnx():
    """惰性导入 onnx：本能力需要 onnx 包（图改写）。"""
    try:
        import onnx
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "SPA 图改写需要 onnx 包：pip install onnx"
            "（已声明在 onnx extra：pip install -e \".[onnx]\"）"
        ) from error
    return onnx


def load_model(path):
    """读模型结构，但**不**加载外部数据（权重留在 .data 里）。"""
    onnx = _onnx()
    return onnx.load(path, load_external_data=False)


_ATTR_TENSOR = 4
_ATTR_GRAPH = 5
_ATTR_GRAPHS = 10


def iter_graphs(graph):
    """深度优先遍历主图与所有子图（If / Loop / Scan 里同样带着 initializer）。"""
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == _ATTR_GRAPH:
                yield from iter_graphs(attribute.g)
            elif attribute.type == _ATTR_GRAPHS:
                for nested in attribute.graphs:
                    yield from iter_graphs(nested)


def external_tensors(model):
    """图里**全部**携带外部数据的张量：initializer + Constant 节点张量 + 子图。"""
    onnx = _onnx()
    for graph in iter_graphs(model.graph):
        for tensor in graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                yield tensor
        for sparse in graph.sparse_initializer:
            for tensor in (sparse.values, sparse.indices):
                if tensor.data_location == onnx.TensorProto.EXTERNAL:
                    yield tensor
        for node in graph.node:
            for attribute in node.attribute:
                if (attribute.type == _ATTR_TENSOR
                        and attribute.t.data_location == onnx.TensorProto.EXTERNAL):
                    yield attribute.t


_LOCATION_ERRORS = (
    "should be a relative path",
    "points outside the directory",
    "should be stored in",
    "is not regular file",
    "does not exist",
)


def _external_data_outside(onnx, model, model_dir: str) -> bool:
    """是否有外部数据落在模型目录之外。"""
    model_dir = osp.abspath(model_dir)
    for tensor in external_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key != "location" or not entry.value:
                continue
            target = entry.value if osp.isabs(entry.value) else osp.join(model_dir, entry.value)
            target = osp.abspath(target)
            if osp.dirname(target) != model_dir:
                return True
    return False


def _verify_external_refs(onnx, model, model_dir: str) -> list:
    """自己校外部数据引用：文件在不在、offset+length 有没有超出文件范围。"""
    problems = []
    model_dir = osp.abspath(model_dir)
    for tensor in external_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = {entry.key: entry.value for entry in tensor.external_data}
        location = info.get("location")
        if not location:
            problems.append(f"{tensor.name}: 外部数据没写 location")
            continue
        path = location if osp.isabs(location) else osp.join(model_dir, location)
        if not osp.isfile(path):
            problems.append(f"{tensor.name}: {osp.abspath(path)} 不存在")
            continue
        offset = int(info.get("offset") or 0)
        length = int(info.get("length") or 0)
        size = osp.getsize(path)
        if offset > size or (length and offset + length > size):
            problems.append(
                f"{tensor.name}: offset={offset}+length={length} 超出 "
                f"{osp.abspath(path)} 的 {size} 字节")
    return problems


def _check_model_path(onnx, out_path: str, *, relax_locations: bool):
    """checker 校验；relax_locations=True 时只放宽「外部数据位置」那一类错误。"""
    try:
        onnx.checker.check_model(out_path)
    except Exception as error:
        message = str(error)
        if relax_locations and any(token in message for token in _LOCATION_ERRORS):
            return False, message
        raise
    return True, ""


def _share_data(src: str, dst: str, *, link: bool) -> tuple[str, str]:
    """把源 .data 放进图目录：硬链接（首选，零字节）-> 整份复制。

    **刻意不用符号链接**：onnxruntime 会把外部数据解析成真实路径，再校验它是否落在
    模型所在目录内；符号链接指到目录外会被拒绝。硬链接没有这个问题。
    """
    if osp.lexists(dst) and not osp.exists(dst):
        os.unlink(dst)                      # 断掉的符号链接：当成不存在，重做
    elif osp.exists(dst):
        return "exist", ""
    reason = ""
    if link:
        try:
            os.link(src, dst)
            return "link", ""
        except OSError as error:
            reason = f"硬链接失败（{error.strerror or error}）"
    else:
        reason = "调用方要求不共享（link=False）"
    shutil.copy2(src, dst)
    return "copy", reason

# 图里的注入节点一律以 spw/ 开头
FORCING_INPUT = "spw/xi"
NODE_PREFIX = "spw/ar1"

# 目标别名 -> 节点输出名：
#   latent_state  三个 VAE 组送进 decoder 的潜状态（512 通道）
#   blocks        latent_mixer 12 个 block 的残差更新出口（1536 通道）
TARGET_ALIASES = {
    "latent_state": (
        "/latent_mixer/Add_output_0",
        "/latent_mixer/Add_1_output_0",
        "/latent_mixer/Add_2_output_0",
    ),
    "blocks": tuple(
        "/latent_mixer/layers.0/blocks.%d/Add_1_output_0" % index for index in range(12)
    ),
}

# 空间基的相关长度（token 数）。1 = 不平滑（所有尺度一起扰）。
DEFAULT_LENGTH_SCALE = 32.0


@dataclass
class LatentForcingSpec(SpecBase):
    """一次潜空间 AR(1) 强迫的规格。

    sigma = 相对强度 ``z⊙(1+σn)``；tau_days = 相关时间（0=每步重抽，∞=冻结）；
    amp_profile / demean_members 只在出口生效、不进 budgets()，换值复用同一份图。
    cache_dir 默认必须与源图同目录（外部数据路径约束）。
    """

    #: from_config 的类型报错里用的名字（实现在 SpecBase）
    config_error = "latent_forcing"

    targets: Sequence[str] = ("latent_state",)
    sigma: float = 0.04
    tau_days: float = 3.0
    rank: int = 16
    length_scale: float = DEFAULT_LENGTH_SCALE
    tokens: int | None = None
    channels: int | None = None
    grid: Sequence[int] | None = None
    seed: int | None = None
    hour_interval: float = 6.0
    amp_profile: Sequence[Sequence[float]] | None = None
    demean_members: int = 0
    input_name: str = FORCING_INPUT
    cache_dir: str | None = None
    verify_on_load: bool = True

    def __post_init__(self):
        if isinstance(self.targets, str):
            self.targets = tuple(
                token.strip() for token in self.targets.split(",") if token.strip())
        else:
            self.targets = tuple(str(token) for token in (self.targets or ("latent_state",)))
        if not self.targets:
            raise ValueError("latent_forcing.targets 不能为空（用 ('latent_state',)）")
        if self.grid is not None:
            self.grid = tuple(int(value) for value in self.grid)
            if len(self.grid) != 2 or min(self.grid) < 1:
                raise ValueError(f"latent_forcing.grid 应是 2 个正整数，实际 {self.grid}")
        if not (self.sigma >= 0):
            # sigma=0 是**合法**的空转对照：gain 恒为 1，输出与源图逐位相同（等价性回归就用它）。
            raise ValueError(f"latent_forcing.sigma 必须 >= 0，实际 {self.sigma!r}")
        for name, value in (("tau_days", self.tau_days), ("length_scale", self.length_scale)):
            if not (value > 0):
                raise ValueError(f"latent_forcing.{name} 必须 > 0，实际 {value!r}")
        if self.rank < 1:
            raise ValueError(f"latent_forcing.rank 必须 >= 1，实际 {self.rank}")
        for name, value in (("tokens", self.tokens), ("channels", self.channels)):
            if value is not None and int(value) < 1:
                raise ValueError(f"latent_forcing.{name} 必须是正整数，实际 {value!r}")
        if self.hour_interval < 0:
            raise ValueError("latent_forcing.hour_interval 不能为负")
        if int(self.demean_members) < 0:
            raise ValueError("latent_forcing.demean_members 不能为负（0 = 关）")
        if int(self.demean_members) == 1:
            # N=1 时「减全体均值」= 把唯一的成员扣成 0，这不是去均值而是静默关掉强迫。
            raise ValueError(
                "latent_forcing.demean_members=1 无意义（单成员减自己的均值 ⇒ ξ 恒 0）。"
                "要关就用 0。")
        if self.amp_profile is not None:
            # 归一化成 ((step, 倍数), ...)：config 里写成列表、元组、list-of-list 都收，
            # 但内部只认一种形状（amp_at 要按它切片）。
            knots = tuple((float(s), float(v)) for s, v in self.amp_profile)
            if not knots:
                raise ValueError("latent_forcing.amp_profile 不能给空表（不要时给 None）")
            if any(v < 0 for _, v in knots):
                raise ValueError(f"latent_forcing.amp_profile 的倍数不能为负，实际 {knots}")
            if any(b <= a for a, b in zip([s for s, _ in knots],
                                          [s for s, _ in knots][1:])):
                raise ValueError(
                    f"latent_forcing.amp_profile 的 step 必须严格递增，实际 {knots}")
            self.amp_profile = knots

    # ---- 派生量 ----

    def amp_at(self, step: int) -> float:
        """时变幅度 a(t)：分段线性插值，两端外推取端点值；None → 恒 1。"""
        knots = self.amp_profile
        if not knots:
            return 1.0
        x = float(step)
        if x <= knots[0][0]:
            return knots[0][1]
        if x >= knots[-1][0]:
            return knots[-1][1]
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return knots[-1][1]

    def rho(self, hour_interval: float | None = None) -> float:
        """一步的 AR(1) 系数：ρ = exp(-Δt/τ)。"""
        step_hours = float(self.hour_interval if hour_interval is None else hour_interval)
        if step_hours <= 0:
            return 0.0
        return float(np.exp(-(step_hours / 24.0) / self.tau_days))

    def budgets(self) -> dict:
        """参与图指纹的字段；τ / a(t) 只在驱动侧起作用，不进指纹。"""
        return {
            "targets": list(self.targets),
            "sigma": float(self.sigma),
            "rank": int(self.rank),
            "length_scale": float(self.length_scale),
            "tokens": self.tokens,
            "channels": self.channels,
            "grid": list(self.grid) if self.grid is not None else None,
            "seed": self.seed,
            "input_name": self.input_name,
            "format": FORCING_FORMAT,
        }

    def describe(self) -> str:
        amp = ""
        if self.amp_profile:
            amp = " a(t)=%d点 %.2f→%.2f" % (
                len(self.amp_profile), self.amp_profile[0][1], self.amp_profile[-1][1])
        demean = "" if int(self.demean_members) <= 1 else " demean=%d成员" % self.demean_members
        return ("latent-ar1 targets=%s sigma=%g tau=%gd rank=%d length_scale=%g%s%s"
                % (" | ".join(self.targets), self.sigma, self.tau_days, self.rank,
                   self.length_scale, amp, demean))


# 图格式版本：改「图怎么改写」（输入名、节点结构、基归一化、外部数据落位）就 +1，
# 使指纹失效、重建新图。v3：`Split` 在 opset>=18 需显式 `num_outputs`；fp16 图
# 须在增益与目标相乘前 `Cast`。
FORCING_FORMAT = 3


# 目标解析与布局探测
def _graph_outputs(model) -> dict:
    """节点输出名 -> 产出它的节点。"""
    out = {}
    for node in model.graph.node:
        for name in node.output:
            out[name] = node
    return out


def resolve_targets(model, spec: LatentForcingSpec) -> list:
    """把 spec.targets（别名 / 后缀）解析成具体的节点输出名列表（保序、去重）。"""
    produced = _graph_outputs(model)
    resolved = []
    for token in spec.targets:
        if token in TARGET_ALIASES:
            names = list(TARGET_ALIASES[token])
        elif token in produced:
            names = [token]
        else:
            names = [name for name in produced if name.endswith(token)]
        if not names:
            raise ValueError(
                f"latent_forcing.targets 里的 {token!r} 在图上找不到对应的节点输出。"
                f"可用别名：{sorted(TARGET_ALIASES)}；"
                f"图上含 'latent_mixer/Add' 的输出有 "
                f"{sorted(name for name in produced if 'latent_mixer/Add' in name)[:8]}")
        for name in names:
            if name not in produced:
                raise ValueError(
                    f"latent_forcing.targets 里的 {token!r} 解析出的 {name!r} "
                    f"不是任何节点的输出（别名表与骨干不匹配？）")
            if name not in resolved:
                resolved.append(name)
    return resolved


def _type_map(model) -> dict:
    """形状+类型推断：名字 -> (维度列表, 元素类型)。dim_value 或 dim_param。"""
    onnx = _onnx()
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    except Exception:  # noqa: BLE001 - 推断失败不影响显式指定 tokens/channels
        inferred = model
    out = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = value.type.tensor_type.shape.dim
        out[value.name] = (
            [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in dims],
            int(value.type.tensor_type.elem_type))
    return out


def _shape_map(model) -> dict:
    """形状推断：名字 -> 维度列表（dim_value 或 dim_param）。"""
    return {name: dims for name, (dims, _elem) in _type_map(model).items()}


def _grid_from_patch_embed(model) -> tuple | None:
    """从 patch_embed 卷积推潜空间网格：kernel == stride 的那个 Conv + 静态输入空间维。"""
    spatial = None
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if len(dims) >= 4 and all(d.HasField("dim_value") for d in dims[-2:]):
            spatial = (dims[-2].dim_value, dims[-1].dim_value)
    if spatial is None:
        return None
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        attrs = {attr.name: list(attr.ints) for attr in node.attribute}
        kernel = attrs.get("kernel_shape")
        stride = attrs.get("strides") or [1, 1]
        if not kernel or len(kernel) != 2 or len(stride) != 2:
            continue
        if kernel[0] != kernel[1] or kernel[0] != stride[0] or stride[0] != stride[1]:
            continue
        step = kernel[0]
        rows = (spatial[0] - step) // step + 1
        cols = (spatial[1] - step) // step + 1
        if rows > 1 and cols > 1:
            return rows, cols
    return None


def detect_layout(model, target_names, spec: LatentForcingSpec) -> dict:
    """定出 (tokens, channels, grid)。优先级：spec 显式值 > 形状推断 > patch_embed 推算。"""
    shapes = _shape_map(model)
    tokens, channels = spec.tokens, spec.channels
    observed = {}
    for name in target_names:
        shape = shapes.get(name)
        if not shape:
            continue
        observed[name] = shape
        if len(shape) != 3:
            raise ValueError(
                f"latent_forcing 的目标 {name!r} 推断出的形状是 {shape}（{len(shape)} 维）；"
                "只有 3 维的 [batch, tokens, channels] 张量才能被注入。")
        if channels is None and isinstance(shape[2], int):
            channels = shape[2]
        if tokens is None and isinstance(shape[1], int):
            tokens = shape[1]
    grid = spec.grid
    if tokens is None:
        grid_guess = _grid_from_patch_embed(model)
        if grid_guess is not None:
            grid = grid or grid_guess
            tokens = grid_guess[0] * grid_guess[1]
    if channels is None:
        raise ValueError(
            "探测不出潜变量通道数：请显式给 latent_forcing.channels"
            f"（已观察到的目标形状：{observed}）")
    if tokens is None:
        raise ValueError(
            "探测不出潜变量 token 数：请显式给 latent_forcing.tokens"
            f"（已观察到的目标形状：{observed}）")
    if grid is None:
        cols = 180 if tokens % 180 == 0 else int(round(np.sqrt(tokens)))
        grid = (tokens // cols, cols) if tokens % cols == 0 else (1, tokens)
    if int(grid[0]) * int(grid[1]) != int(tokens):
        raise ValueError(
            f"latent_forcing.grid={tuple(grid)} 的乘积 {grid[0] * grid[1]} "
            f"与 tokens={tokens} 不一致")
    if int(channels) < 1 or int(tokens) < 1:
        raise ValueError("channels / tokens 必须为正")
    return {"tokens": int(tokens), "channels": int(channels),
            "grid": (int(grid[0]), int(grid[1])), "observed": observed}


# 随机基
def _lowpass(field: np.ndarray, sigma_tokens: float) -> np.ndarray:
    """循环高斯低通（FFT）。sigma_tokens = 相关长度（格点数）。"""
    if sigma_tokens <= 0.5:
        return field
    rows, cols = field.shape
    fy = np.fft.fftfreq(rows)[:, None]
    fx = np.fft.fftfreq(cols)[None, :]
    kernel = np.exp(-2.0 * (np.pi ** 2) * sigma_tokens ** 2 * (fy ** 2 + fx ** 2))
    return np.fft.ifft2(np.fft.fft2(field) * kernel).real


def _unit_columns(matrix: np.ndarray) -> np.ndarray:
    """QR 正交化：返回列正交且单位范数的同形矩阵（列数 <= 行数）。"""
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return q


def build_basis(tokens: int, channels: int, rank: int, length_scale: float,
                grid: tuple, seed: int | None, sigma: float) -> dict:
    """构造空间基 V[N,R] 与通道基 U[C,R]，把 σ 与归一化常数折叠进 U。

    返回 {"Vt": [R,N], "Ut": [R,C], "report": {...}}；图里算
    ``n = Σ_r ξ_r · Ut[r] ⊗ Vt[r]``。U 的缩放取 ``σ·sqrt(C·N/R)``，使 RMS(n) ≈ σ，
    与 SPW 的 σ 同量纲。
    """
    if rank > channels:
        raise ValueError(
            f"latent_forcing.rank={rank} 不能大于通道数 {channels}"
            "（通道基最多只有 channels 个正交方向）")
    rows, cols = int(grid[0]), int(grid[1])
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed or 0), 104729, tokens, channels, int(length_scale * 1000)]))
    # 空间基：低通随机场 -> 去 DC（防整场偏移）-> 正交化。
    # length_scale 相对网格过大时，低通会压掉除 DC 外的全部模，去 DC 后撑不起基；
    # 逐个方向检查残余能量，不达标就退回白噪声。
    space = []
    fallbacks = 0
    for _ in range(rank):
        white = rng.standard_normal((rows, cols))
        flat = _lowpass(white, float(length_scale)).reshape(-1)
        centred = flat - flat.mean()
        if float(centred @ centred) < 1e-8 * float(white.reshape(-1) @ white.reshape(-1)):
            centred = white.reshape(-1)
            centred = centred - centred.mean()
            fallbacks += 1
        space.append(centred)
    if fallbacks:
        log.warning(
            "latent_forcing：%d/%d 个空间基方向在 %dx%d 的网格上退化了（length_scale=%g 的"
            "低通把除 DC 以外的模全压没），这些方向已退回白噪声。请把 length_scale 调到 %d 以下。",
            fallbacks, rank, rows, cols, float(length_scale), max(1, min(rows, cols) // 4))
    basis_v = np.stack(space, axis=1)
    basis_v = _unit_columns(basis_v)
    # 通道基：随机正交
    basis_u = _unit_columns(rng.standard_normal((channels, rank)))
    scale = float(sigma) * float(np.sqrt(channels * tokens / rank))
    return {"Vt": basis_v.T.astype(np.float32),
            "Ut": (basis_u.T * scale).astype(np.float32),
            "report": {"tokens": tokens, "channels": channels, "grid": (rows, cols),
                       "rank": rank, "length_scale": float(length_scale),
                       "sigma": float(sigma), "scale": scale,
                       "fallbacks": int(fallbacks)}}


# 图改写
def _initializer(onnx, name: str, array: np.ndarray):
    return name, onnx.numpy_helper.from_array(np.ascontiguousarray(array), name)


def inject_forcing(onnx, model, targets, basis, spec: LatentForcingSpec) -> int:
    """把 AR(1) 强迫接到目标张量上：`z' = z ⊙ (1 + σ·U diag(ξ) Vᵀ)`，返回新增节点数。

    链为 ``spw/xi -> Split -> Reshape -> Mul(Vt) -> Transpose -> MatMul(Ut) -> Add(1)
    -> Mul(z)``。MatMul 收缩最后一维（R），所以必须先 Mul、再转置、后 MatMul。
    链插在每个目标自己的第一个消费者之前，不依赖目标节点是否相邻。
    """
    graph = model.graph
    rank = int(spec.rank)
    # 目标可能是 fp16 图，而链上的常量与 spw/xi 都是 fp32：链全程保持 fp32，
    # 只在最后往目标类型 Cast 一次。类型须在改图之前推断。
    _types = _type_map(model)
    target_elem = {}
    for _name in targets:
        _elem = _types.get(_name, (None, 0))[1] or onnx.TensorProto.FLOAT
        target_elem[_name] = _elem
    one_name, one_tensor = _initializer(
        onnx, NODE_PREFIX + "/one", np.ones((1,), dtype=np.float32))
    shape_name, shape_tensor = _initializer(
        onnx, NODE_PREFIX + "/xi_shape", np.asarray([0, rank, 1], dtype=np.int64))
    graph.initializer.extend([one_tensor, shape_tensor])
    graph.input.append(onnx.helper.make_tensor_value_info(
        spec.input_name, onnx.TensorProto.FLOAT,
        [NODE_PREFIX + "_batch", len(targets), rank]))

    slices = ["%s/xi_%d" % (NODE_PREFIX, index) for index in range(len(targets))]
    # opset >= 18 的 Split 必须显式给 num_outputs；13-17 加了反成未知属性，
    # 所以按图自己声明的版本决定。
    opset = max((int(entry.version) for entry in model.opset_import
                 if entry.domain in ("", "ai.onnx")), default=0)
    split_attrs = {"num_outputs": len(targets)} if opset >= 18 else {}
    split = onnx.helper.make_node(
        "Split", [spec.input_name], slices, name=NODE_PREFIX + "/Split", axis=1,
        **split_attrs)

    nodes = list(graph.node)
    added = 0
    for index, name in enumerate(targets):
        stem = "%s/t%d" % (NODE_PREFIX, index)
        v_name, v_tensor = _initializer(onnx, stem + "/Vt", basis["Vt"])
        u_name, u_tensor = _initializer(onnx, stem + "/Ut", basis["Ut"])
        graph.initializer.extend([v_tensor, u_tensor])
        # 1) 先定位第一个消费者（必须在改接**之前**做，否则消费者已经被改掉、找不到位置）
        positions = [i for i, node in enumerate(nodes) if name in list(node.input)]
        if not positions:
            raise ValueError(
                f"latent_forcing 的目标 {name!r} 在图上没有任何消费者，注入无处安放"
                "（它是图的输出节点？请换一个目标）")
        at = min(positions)
        # 2) 把消费者改接到本目标的输出（此刻链还没插进去，不会误改自己的输入）
        for node in nodes:
            for slot, value in enumerate(node.input):
                if value == name:
                    node.input[slot] = stem + "/out"
        # 3) 把链插到第一个消费者之前
        chain = [
            onnx.helper.make_node(
                "Reshape", [slices[index], shape_name], [stem + "/xi"],
                name=stem + "/Reshape"),
            onnx.helper.make_node(
                "Mul", [v_name, stem + "/xi"], [stem + "/field_r"], name=stem + "/MulV"),
            onnx.helper.make_node(
                "Transpose", [stem + "/field_r"], [stem + "/field_nr"],
                name=stem + "/Transpose", perm=[0, 2, 1]),
            onnx.helper.make_node(
                "MatMul", [stem + "/field_nr", u_name], [stem + "/field"],
                name=stem + "/MatMulU"),
            onnx.helper.make_node(
                "Add", [stem + "/field", one_name], [stem + "/gain"],
                name=stem + "/AddOne"),
        ]
        # 仅非 fp32 目标需要先把增益降类型
        gain = stem + "/gain"
        if target_elem[name] != onnx.TensorProto.FLOAT:
            chain.append(onnx.helper.make_node(
                "Cast", [gain], [stem + "/gain_t"], name=stem + "/CastGain",
                to=target_elem[name]))
            gain = stem + "/gain_t"
        chain.append(onnx.helper.make_node(
            "Mul", [name, gain], [stem + "/out"], name=stem + "/MulZ"))
        nodes[at:at] = chain
        added += len(chain)
    nodes.insert(0, split)
    del graph.node[:]
    graph.node.extend(nodes)
    return added


def _stamp_forcing(onnx, model, src_path, spec, targets, layout, report):
    """把「这张图是怎么被强迫出来的」写进 metadata（与 _stamp 同一套约定）。"""
    props = {entry.key: entry.value for entry in model.metadata_props}
    props.update({
        "spw.ar1.method": "latent-ar1-multiplicative",
        "spw.ar1.spec": spec.describe(),
        "spw.ar1.spec_json": json.dumps(spec.budgets(), sort_keys=True, ensure_ascii=False),
        "spw.ar1.targets": " | ".join(targets),
        "spw.ar1.input": spec.input_name,
        "spw.ar1.tokens": str(layout["tokens"]),
        "spw.ar1.channels": str(layout["channels"]),
        "spw.ar1.grid": "%dx%d" % tuple(layout["grid"]),
        "spw.ar1.rank": str(int(spec.rank)),
        "spw.ar1.length_scale": "%.6g" % float(spec.length_scale),
        "spw.ar1.sigma": "%.6g" % float(spec.sigma),
        "spw.ar1.tau_days": "%.6g" % float(spec.tau_days),
        "spw.ar1.rho": "%.6g" % spec.rho(),
        "spw.ar1.seed": str(spec.seed if spec.seed is not None else ""),
        "spw.source": osp.basename(src_path),
        "spw.source_fingerprint": spec.fingerprint(src_path),
    })
    onnx.helper.set_metadata_props(model, props)


def default_forcing_path(src_path: str, spec: LatentForcingSpec) -> str:
    """强迫图的落盘路径：默认与源图同目录。

    图的外部数据（.data）必须落在图所在目录内，否则 ORT 建会话时 realpath 校验失败。
    """
    directory = (spec.cache_dir or os.environ.get("XMETAI_AR1_CACHE_DIR")
                 or osp.dirname(osp.abspath(src_path)))
    stem = osp.splitext(osp.basename(src_path))[0]
    return osp.join(osp.abspath(directory), f"{stem}.ar1-{spec.fingerprint(src_path)}.onnx")


def _localize_external_data(onnx, model, src_dir: str, out_dir: str) -> dict:
    """out_dir != src_dir 时：把外部 .data 放进 out_dir，location 改成裸文件名。

    ORT 会把外部数据解析成真实路径并校验它落在图目录内，相对路径与符号链接都过不了，
    所以优先硬链接；不可用（跨卷 / protected_hardlinks）就复制一次、各臂共用。
    必须遍历 external_tensors：漏掉 Constant 节点的外置张量不会报错，只会建会话失败。

    返回 {basename: 方式}，方式 ∈ link / copy / exist。
    """
    placed = {}
    for tensor in external_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key != "location" or not entry.value:
                continue
            target = entry.value if osp.isabs(entry.value) else osp.join(src_dir, entry.value)
            target = osp.abspath(target)
            base = osp.basename(target)
            dst = osp.join(out_dir, base)
            if base not in placed:
                if osp.exists(dst) and os.path.samefile(dst, target):
                    placed[base] = "exist"        # 上一臂已放过（硬链接/复制），直接共用
                else:
                    # 同名但不是同一份文件（陈旧残留）：删掉重放。宁可重做一次也不能错读。
                    if osp.lexists(dst):
                        os.remove(dst)
                    method, _reason = _share_data(target, dst, link=True)
                    placed[base] = method
            if entry.value != base:
                entry.value = base
    return placed


def _relax_needed(onnx, model, out_dir: str, message: str = "") -> bool:
    """要不要放宽 checker 的外部数据定位校验：.data 不在图目录内，或是硬链接共享。"""
    if "hard link" in message:
        return True
    if _external_data_outside(onnx, model, out_dir):
        return True
    return False


def build_forcing_model(src_path: str, spec: LatentForcingSpec, out_path: str | None = None,
                        *, verbose: bool = True) -> str:
    """读源图 -> 解析目标 -> 算基 -> 注入 -> 落盘 -> 校验，返回新图路径。"""
    onnx = _onnx()
    src_path = osp.abspath(src_path)
    src_dir = osp.dirname(src_path)
    model = load_model(src_path)
    targets = resolve_targets(model, spec)
    layout = detect_layout(model, targets, spec)
    basis = build_basis(layout["tokens"], layout["channels"], spec.rank,
                        spec.length_scale, layout["grid"], spec.seed, spec.sigma)
    added = inject_forcing(onnx, model, targets, basis, spec)
    _stamp_forcing(onnx, model, src_path, spec, targets, layout, basis["report"])

    out_path = osp.abspath(out_path) if out_path else default_forcing_path(src_path, spec)
    out_dir = osp.dirname(out_path)
    shared = {}
    if out_dir != src_dir:
        # 跨目录：.data 必须真实落在图目录内（ORT 的 realpath 校验）。
        shared = _localize_external_data(onnx, model, src_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tmp = f"{out_path}.tmp{os.getpid()}"
    onnx.save(model, tmp)
    os.replace(tmp, out_path)
    try:
        broken = _verify_external_refs(onnx, model, out_dir)
        if broken:
            raise RuntimeError("外部数据引用有问题：" + "; ".join(broken[:5]))
        try:
            checked = _check_model_path(onnx, out_path, relax_locations=False)[0]
            relaxed = ""
        except Exception as error:  # noqa: BLE001 - 只看是不是定位类问题
            if not _relax_needed(onnx, model, out_dir, str(error)):
                raise
            checked = _check_model_path(onnx, out_path, relax_locations=True)[0]
            relaxed = str(error).splitlines()[0][:160]
    except Exception:
        os.remove(out_path)
        raise
    if verbose:
        print(f"已注入潜空间强迫：{len(targets)} 个目标 / 每目标 {added // len(targets)} 个节点"
              f" -> {out_path}")
        print(f"  {spec.describe()}")
        print("  布局：tokens=%d channels=%d grid=%dx%d rank=%d length_scale=%g"
              % (layout["tokens"], layout["channels"], layout["grid"][0], layout["grid"][1],
                 int(spec.rank), float(spec.length_scale)))
        # 归一化自检：‖Ut‖_F/sqrt(C*N) 是基隐含的 sigma，必须等于设定值。
        if basis["report"].get("fallbacks"):
            print("  警告：%d/%d 个空间基方向退化（length_scale 相对网格太大），已退回白噪声"
                  % (basis["report"]["fallbacks"], int(spec.rank)))
        norm_v = float(np.linalg.norm(basis["Vt"]))
        sigma_implied = float(np.linalg.norm(basis["Ut"])) / float(
            np.sqrt(layout["channels"] * layout["tokens"]))
        print("  基：‖Vt‖=%.4f（应 ≈ sqrt(rank)=%.4f）；基隐含 sigma=%.6f（设定 %.6f）"
              % (norm_v, np.sqrt(spec.rank), sigma_implied, spec.sigma))
        print("  ρ(Δt=%gh) = %.4f（τ=%g 天）" % (spec.hour_interval, spec.rho(), spec.tau_days))
        if shared:
            print("  外部数据落位（%s）：%s" % (
                out_dir, ", ".join("%s=%s" % item for item in sorted(shared.items()))))
        if not checked:
            print("  提示：checker 对「外部数据位置」的要求已放宽（.data 不复制、仍指向源文件）："
                  f"{relaxed}")
    return out_path


def resolve_forcing_path(src_path: str, spec: LatentForcingSpec | None,
                         *, verbose: bool = False) -> str:
    """有 spec 就返回（必要时先构建）带潜空间强迫的图，否则原样返回。"""
    if spec is None:
        return src_path
    target = default_forcing_path(src_path, spec)
    if osp.isfile(target):
        return target
    return build_forcing_model(src_path, spec, target, verbose=verbose)


def resolve_stack_path(src_path: str, weight_spec, latent_spec,
                       member_index: int = 0, *, verbose: bool = False) -> str:
    """框架侧唯一入口：先按 SPW 机制解析权重路径，再在其上叠加潜空间强迫。

    两步都按指纹缓存；叠加正交——强迫改图结构、SPW 改权重，可以同时开。
    """
    # 函数内导入，避免 spa -> spw 的顶层循环（spw 顶层 import 本文件的 SpecBase）。
    from .spw import resolve_effective_path

    path = resolve_effective_path(src_path, weight_spec, member_index, verbose=verbose)
    return resolve_forcing_path(path, latent_spec, verbose=verbose)


# 审计
class Audit(dict):
    """审计结果：dict + 一行摘要。"""

    def summary(self) -> str:
        return ("latent-ar1 %s targets=%d sigma=%s tau=%sd rank=%s length_scale=%s "
                "input=%s" % ("OK" if self.get("ok") else "FAIL",
                              self.get("n_targets", 0), self.get("sigma"),
                              self.get("tau_days"), self.get("rank"),
                              self.get("length_scale"), self.get("input")))


def audit_forcing(path: str) -> dict:
    """读强迫图的 metadata 与结构，确认「驱动侧喂的 ξ 真的被消费了」。"""
    model = load_model(path)
    meta = {entry.key: entry.value for entry in model.metadata_props}
    inputs = [value.name for value in model.graph.input]
    forcing_input = meta.get("spw.ar1.input", FORCING_INPUT)
    targets = [name for name in meta.get("spw.ar1.targets", "").split(" | ") if name]
    produced = _graph_outputs(model)
    report = {
        "path": osp.abspath(path),
        "input": forcing_input,
        "input_present": forcing_input in inputs,
        "n_targets": len(targets),
        "targets": targets,
        "targets_present": all(name in produced for name in targets),
        "nodes": len([node for node in model.graph.node
                      if node.name.startswith(NODE_PREFIX)]),
        "rank": meta.get("spw.ar1.rank"),
        "sigma": meta.get("spw.ar1.sigma"),
        "tau_days": meta.get("spw.ar1.tau_days"),
        "length_scale": meta.get("spw.ar1.length_scale"),
        "rho": meta.get("spw.ar1.rho"),
        "tokens": meta.get("spw.ar1.tokens"),
        "channels": meta.get("spw.ar1.channels"),
        "grid": meta.get("spw.ar1.grid"),
        "seed": meta.get("spw.ar1.seed"),
        "source": meta.get("spw.source"),
    }
    report["ok"] = bool(report["input_present"] and report["targets_present"]
                        and report["n_targets"] > 0)
    return Audit(report)


def format_forcing_audit(report: Mapping) -> list:
    lines = ["潜空间 AR(1) 强迫审计：%s" % report.get("path")]
    lines.append("  新增输入          %s（在图中：%s）"
                 % (report.get("input"), report.get("input_present")))
    lines.append("  注入目标          %d 个（都在图中：%s）"
                 % (report.get("n_targets", 0), report.get("targets_present")))
    for name in report.get("targets", [])[:4]:
        lines.append("    - %s" % name)
    if len(report.get("targets", [])) > 4:
        lines.append("    - ...（共 %d 个）" % len(report["targets"]))
    lines.append("  强迫节点数        %s" % report.get("nodes"))
    lines.append("  布局              tokens=%s channels=%s grid=%s"
                 % (report.get("tokens"), report.get("channels"), report.get("grid")))
    lines.append("  参数              sigma=%s tau=%s 天（ρ=%s）rank=%s length_scale=%s"
                 % (report.get("sigma"), report.get("tau_days"), report.get("rho"),
                    report.get("rank"), report.get("length_scale")))
    lines.append("  种子（驱动侧）    %s" % report.get("seed"))
    lines.append("  源图              %s" % report.get("source"))
    lines.append("  ⇒ %s" % ("结构自洽" if report.get("ok") else "**结构不自洽，别拿它跑推理**"))
    return lines


__all__ = [
    "DEFAULT_LENGTH_SCALE",
    "FORCING_FORMAT",
    "FORCING_INPUT",
    "LatentForcingSpec",
    "NODE_PREFIX",
    "TARGET_ALIASES",
    "audit_forcing",
    "build_basis",
    "build_forcing_model",
    "default_forcing_path",
    "detect_layout",
    "format_forcing_audit",
    "inject_forcing",
    "resolve_forcing_path",
    "resolve_stack_path",
    "resolve_targets",
]


if __name__ == "__main__":
    # 造产物入口：改下面的字面量后运行 `python -m xmetai_inference.weights.spa`。
    # 没有命令行参数；ACTION 选 build / audit。
    ACTION = "build"                       # "build" | "audit"
    MODEL = "model.onnx"                   # ← 改成源模型路径（onnx 或 .pt2）
    OUT = None                             # 输出文件路径；None = 按指纹命名
    CACHE_DIR = None                       # 输出目录；None = 源图同目录

    TARGETS = ("latent_state",)            # 或 ("blocks",)、节点名后缀
    SIGMA = 0.04                           # 0 = 空转对照（输出应与源图逐位相同）
    TAU_DAYS = 3.0
    RANK = 16
    LENGTH_SCALE = DEFAULT_LENGTH_SCALE
    SEED = None
    HOUR_INTERVAL = 6.0
    GRID = None                            # pt2 必填：(rows, cols)，如 (120, 240)
    QUIET = False

    if ACTION == "audit":
        report = audit_forcing(MODEL)
        for line in format_forcing_audit(report):
            print(line)
        raise SystemExit(0 if report.get("ok") else 1)

    spec = LatentForcingSpec(
        targets=TARGETS, sigma=SIGMA, tau_days=TAU_DAYS, rank=RANK,
        length_scale=LENGTH_SCALE, seed=SEED, hour_interval=HOUR_INTERVAL, grid=GRID,
        cache_dir=CACHE_DIR)

    if str(MODEL).endswith(".pt2"):
        import torch.export

        from .spa_pt2 import attach_forcing

        report = attach_forcing(torch.export.load(MODEL), spec,
                                src_path=MODEL, verbose=not QUIET)
        layout = report.get("layout") or {}
        print(f"pt2 注入：目标 {report.get('n_targets')} 个，ok={report.get('ok')}")
        if layout:
            print(f"  布局 tokens={layout.get('tokens')} channels={layout.get('channels')} "
                  f"grid={layout.get('grid')}")
    else:
        path = build_forcing_model(MODEL, spec, OUT, verbose=not QUIET)
        report = audit_forcing(path)
        for line in format_forcing_audit(report):
            print(line)
    # sigma=0 的 pt2 空转对照没有 "ok"（不改图），跳过即算通过。
    raise SystemExit(0 if report.get("ok", report.get("skipped", False)) else 1)
