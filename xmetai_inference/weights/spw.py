# -*- coding: utf-8 -*-
"""权重扰动（SPW）：规格、选张量、配平，以及逐成员冻结权重落盘。

按 ``W~ = W ⊙ (1 + σξ)`` 为每个成员生成一份普通的确定性 ONNX。
造产物直接运行 ``python -m xmetai_inference.weights.spw``（改本文件末尾的字面量）。
"""
from __future__ import annotations

import json
import os
import os.path as osp
import re
import shutil
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from .spa import SpecBase

# 默认排除归一化统计量与烘焙进图的常量/掩码：它们量级极大，混进能量口径会把
# 组间配平整体抬偏。按名字路径分量整段匹配，所以 "mean" 不会误伤 mean_proj。
DEFAULT_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")

BALANCE_MODES = ("energy", "count", "none")

# 噪声在权重张量内部的相关结构；三者每元素方差都是 1，只换结构、不换预算。
#   iid      逐元素独立（默认）
#   channel  每个输出通道共享一个 ξ（向量退化为 iid）
#   scalar   整个张量共享一个 ξ（随机自由度最少）
NOISE_STRUCTS = ("iid", "channel", "scalar")

# 冻结的指纹键取值：成员集尺寸由它决定，删键会让既有成员集全部失配。
NOISE_BLOCK_FROZEN = 8

# 成员集落盘格式版本：凡改变成员图写法（location 口径、布局、噪声派生）就 +1，
# 使新版本落到新目录，旧产物不会被静默复用。
MEMBER_FORMAT = 3
@dataclass
class WeightPerturbSpec(SpecBase):
    """一次 SPW 运行的注入规格。

    groups 按名字路径分量前缀匹配，"all" = 全部；sigma_full 是自动配平前的全候选口径。
    seed 决定成员集（SeedSequence([seed, index])，不按 rank 偏移——成员集多卡共享）。
    """

    #: from_config 的类型报错里用的名字（实现在 SpecBase）
    config_error = "weight_perturb"

    groups: Sequence[str] = ("all",)
    sigma_full: float = 0.01
    sigma: float | None = None
    balance: str = "energy"
    exclude: Sequence[str] = DEFAULT_EXCLUDE
    include: Sequence[str] = ()
    min_elements: int = 64
    seed: int | None = None
    noise_struct: str = "iid"
    members: int = 0
    member_dir: str | None = None
    cache_dir: str | None = None

    def __post_init__(self):
        if isinstance(self.groups, str):
            self.groups = tuple(
                token.strip() for token in self.groups.split(",") if token.strip()
            )
        else:
            self.groups = tuple(str(token) for token in (self.groups or ("all",)))
        if not self.groups:
            raise ValueError("weight_perturb.groups 不能为空（用 ('all',) 表示全权重）")
        self.exclude = tuple(str(pattern) for pattern in (self.exclude or ()))
        self.include = tuple(str(pattern) for pattern in (self.include or ()))
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError(f"weight_perturb.sigma 必须 > 0，实际 {self.sigma}")
        if self.sigma_full <= 0:
            raise ValueError(f"weight_perturb.sigma_full 必须 > 0，实际 {self.sigma_full}")
        if self.balance not in BALANCE_MODES:
            raise ValueError(
                f"weight_perturb.balance 只能是 {BALANCE_MODES}，实际 {self.balance!r}")
        if self.min_elements < 0:
            raise ValueError("weight_perturb.min_elements 不能为负")
        if self.noise_struct not in NOISE_STRUCTS:
            raise ValueError(
                f"weight_perturb.noise_struct 只能是 {NOISE_STRUCTS}，"
                f"实际 {self.noise_struct!r}")
        if self.members < 0:
            raise ValueError("weight_perturb.members 不能为负")

    def budgets(self) -> dict:
        """参与指纹与审计的字段（cache_dir 这类不影响结果的项不参与）。"""
        return {
            "groups": list(self.groups),
            "sigma_full": self.sigma_full,
            "sigma": self.sigma,
            "balance": self.balance,
            "exclude": list(self.exclude),
            "include": list(self.include),
            "min_elements": self.min_elements,
            "seed": self.seed,
            # 冻结指纹键：删掉会让已落盘的成员集全部失配，取值保持当年固定值
            "rank_seed": False,
            "noise_op": "random_normal_like",
            # 默认结构不进指纹（否则既有成员集会整份重生成）；非默认值必须进来，
            # 否则 channel 会命中 iid 的成员集目录。
            **({"noise_struct": self.noise_struct, "noise_block": NOISE_BLOCK_FROZEN}
               if self.noise_struct != "iid" else {}),
        }

    def fingerprint_payload(self, src_path: str) -> dict:
        payload = super().fingerprint_payload(src_path)
        # 成员集是落盘产物，格式变了就必须换目录
        payload["member_format"] = MEMBER_FORMAT
        return payload

    def describe(self) -> str:
        # 用 " | " 连接：token 内部可能含逗号（并集写法）
        groups = " | ".join(self.groups)
        amplitude = (f"sigma={self.sigma}" if self.sigma is not None
                     else f"sigma_full={self.sigma_full} (balance={self.balance})")
        extra = f" include={list(self.include)}" if self.include else ""
        struct = "" if self.noise_struct == "iid" else f" noise_struct={self.noise_struct}"
        return (f"groups={groups} {amplitude} exclude={list(self.exclude)}"
                f"{extra}{struct}")


# ---------------------------------------------------------------------------
# 选张量底座
# ---------------------------------------------------------------------------
_CHUNK = 1 << 22
_FLOAT_DTYPES = ("FLOAT", "FLOAT16", "DOUBLE", "BFLOAT16")

# 可学习参数的判定：叶子名命中（或 *_weight 形式），或经布局算子走到核心算子的
# 权重槽位。这样能把 mean/std/mask 等烘焙常量排除在能量口径之外。
_WEIGHT_LEAF_NAMES = ("weight", "bias", "scale", "shift", "gamma", "beta", "embedding")
# 核心算子 -> 权重所在的输入槽；只认这些槽位，避免把激活槽的数据误判成权重。
_WEIGHT_SLOTS = {
    "Conv": (1,),
    "ConvTranspose": (1,),
    "Gemm": (1,),                 # A=0, B=1, C=2(bias)
    "MatMul": (0, 1),
    "Einsum": (0, 1, 2, 3),
    "LayerNormalization": (1, 2),  # scale, bias
    "BatchNormalization": (1, 2),  # scale, B；mean/var 是统计量，不算权重
    "InstanceNormalization": (1, 2),
    "GroupNormalization": (1, 2),
    "LSTM": tuple(range(1, 9)),
    "GRU": tuple(range(1, 9)),
    "RNN": tuple(range(1, 9)),
}
_LAYOUT_OPS = (
    "Transpose", "Reshape", "Unsqueeze", "Squeeze", "Cast", "Slice",
    "Concat", "Identity", "Flatten", "Expand",
)
_MAX_LAYOUT_DEPTH = 6


def _onnx():
    """惰性导入 onnx：本能力需要 onnx 包（图改写），而包主依赖只有 onnxruntime。"""
    try:
        import onnx
    except ImportError as error:  # pragma: no cover - 环境缺失时的明确提示
        raise ImportError(
            "权重扰动（图内注入）需要 onnx 包：pip install onnx"
            "（已声明在 onnx extra：pip install -e \".[onnx]\"）"
        ) from error
    return onnx


def load_model(path):
    """读模型结构，但**不**加载外部数据（权重留在 .data 里，避免 2.8 GB 进内存）。"""
    onnx = _onnx()
    return onnx.load(path, load_external_data=False)


# onnx.AttributeProto 的取值；写常量是为了导入期不必先拿到 onnx 包。
_ATTR_TENSOR = 4
_ATTR_GRAPH = 5
_ATTR_GRAPHS = 10


def iter_graphs(graph):
    """深度优先遍历主图与所有子图（If / Loop / Scan 的子图里同样带着 initializer）。"""
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == _ATTR_GRAPH:
                yield from iter_graphs(attribute.g)
            elif attribute.type == _ATTR_GRAPHS:
                for nested in attribute.graphs:
                    yield from iter_graphs(nested)


def external_tensors(model):
    """图里全部携带外部数据的张量：initializer + Constant 节点属性 + 子图。

    不能用 graph.initializer 替代：导出器会把 Constant 节点里的大张量也外置，
    漏掉它们不会报错，只会让成员图继续指向源 .data。
    """
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



# ---------------------------------------------------------------------------
# 名字与拓扑
# ---------------------------------------------------------------------------
def key_parts(name) -> tuple[str, ...]:
    """把名字切成路径分量（"." 与 "/" 等价），用于前缀匹配。"""
    cleaned = str(name).strip("/")
    return tuple(token for token in re.split(r"[./]+", cleaned) if token)


def group_alternatives(token) -> tuple[tuple[str, ...], ...]:
    """把一个 group token 展开成多个前缀的并集（`,` 或 `|` 分隔）。

    并集仍算一个组、只吃一份预算；拆成多个 token 则会被逐组配平，总方差翻倍。
    """
    pieces = [piece.strip() for piece in re.split(r"[,|]", str(token))]
    parts = [key_parts(piece) for piece in pieces if piece]
    if not parts:
        raise ValueError(f"空的 group token：{token!r}")
    return tuple(parts)


def _any_prefix(parts: tuple[str, ...], alternatives) -> bool:
    """parts 是否以 alternatives 里的**任意一个**前缀开头。"""
    return any(parts[:len(alt)] == alt for alt in alternatives)


def consumer_map(graph) -> dict[str, list[Any]]:
    """value 名 -> 消费它的节点列表。"""
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    return consumers


def _match_pattern(parts: tuple[str, ...], patterns) -> str | None:
    """按名字的路径分量整段匹配（`mean` 不会误伤 `mean_proj`）。"""
    for pattern in patterns:
        token = key_parts(pattern)
        size = len(token)
        if not size:
            continue
        for start in range(len(parts) - size + 1):
            if parts[start:start + size] == token:
                return pattern
    return None


def _leaf_is_weight(leaf: str) -> bool:
    """叶子名判定：`weight`/`bias`/`scale`… 以及 `dt_bias` 这种 `*_weight` 形式。"""
    leaf = leaf.lower()
    return leaf in _WEIGHT_LEAF_NAMES or any(
        leaf.endswith("_" + name) for name in _WEIGHT_LEAF_NAMES)


def weight_like(value, consumers, memo: dict | None = None) -> bool:
    """判断一个 initializer 是可学习参数还是烘焙常量。

    叶子名像 weight/bias，或经布局算子走到核心算子的权重槽位，任一成立即算权重。
    """
    memo = {} if memo is None else memo
    if value in memo:
        return memo[value]
    parts = key_parts(value)
    if parts and _leaf_is_weight(parts[-1]):
        memo[value] = True
        return True
    stack = [(value, 0)]
    seen = set()
    result = False
    while stack and not result:
        current, depth = stack.pop()
        if current in seen or depth > _MAX_LAYOUT_DEPTH:
            continue
        seen.add(current)
        for node in consumers.get(current, ()):
            slots = _WEIGHT_SLOTS.get(node.op_type)
            if slots is not None:
                for position, feed in enumerate(node.input):
                    if feed == current and position in slots:
                        result = True
                        break
                continue
            if node.op_type in _LAYOUT_OPS:
                for produced in node.output:
                    stack.append((produced, depth + 1))
    memo[value] = result
    return result


# ---------------------------------------------------------------------------
# 权重能量（配平的依据）
# ---------------------------------------------------------------------------
def tensor_array(tensor, src_dir: str):
    """取张量数值：外部数据用 memmap 只读映射，不把 2.8 GB 拉进内存。"""
    onnx = _onnx()
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        external = {entry.key: entry.value for entry in tensor.external_data}
        location = external.get("location") or ""
        path = location if osp.isabs(location) else osp.join(src_dir, location)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
        shape = tuple(int(dim) for dim in tensor.dims)
        offset = int(external.get("offset") or 0)
        return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    return onnx.numpy_helper.to_array(tensor)


def tensor_energy(tensor, src_dir: str) -> float:
    """Σw²（Frobenius 能量的平方），分块累加以免 float32 溢出/精度损失。"""
    array = tensor_array(tensor, src_dir)
    total = 0.0
    if array.dtype == np.float32 and array.size <= _CHUNK:
        return float(np.sum(np.square(array.astype(np.float64))))
    flat = np.asarray(array).reshape(-1)
    for start in range(0, flat.size, _CHUNK):
        chunk = np.asarray(flat[start:start + _CHUNK], dtype=np.float64)
        total += float(np.sum(chunk * chunk))
    return total


# ---------------------------------------------------------------------------
# 选张量
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    tensor: Any
    elements: int
    energy: float
    dtype: str
    group: str = ""
    matched_by: str = ""


@dataclass
class Selection:
    selected: dict[str, Candidate] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    candidate_energy: float = 0.0      # 全部候选（过滤后、分组前）的 Σ‖W‖²
    candidate_elements: int = 0
    group_energy: dict[str, float] = field(default_factory=dict)
    group_elements: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        for group in sorted(self.group_energy):
            energy = self.group_energy[group]
            share = (energy / self.candidate_energy * 100.0
                     if self.candidate_energy else 0.0)
            count = sum(1 for item in self.selected.values() if item.group == group)
            lines.append(
                f"  {group:24s} 张量={count:4d} 元素={self.group_elements[group]/1e6:8.2f}M "
                f"ΣW^2={energy:.6g} 占比={share:5.1f}%")
        return "\n".join(lines)


def select_initializers(
    model,
    spec: WeightPerturbSpec,
    src_dir: str,
    *,
    with_energy: bool = True,
) -> Selection:
    """按 spec 选出要扰动的 initializer，并算好配平所需的能量统计。

    过滤顺序：浮点 → 元素数别太小 → include/exclude → 非图输入 → 有权重结构 →
    有消费者。被跳过的都记进 skipped。
    """
    graph = model.graph
    onnx = _onnx()
    consumers = consumer_map(graph)
    graph_inputs = {value.name for value in graph.input}
    selection = Selection()
    groups = [group_alternatives(token) for token in spec.groups]
    match_all = any(("all",) in alternatives for alternatives in groups)
    weight_memo: dict = {}

    for tensor in graph.initializer:
        name = tensor.name
        elements = int(np.prod(tensor.dims)) if tensor.dims else 1
        dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
        parts = key_parts(name)

        if dtype not in _FLOAT_DTYPES:
            selection.skipped[name] = f"非浮点 dtype={dtype}"
            continue
        # include 是强制声明，越过 min_elements；exclude 仍然优先
        forced = _match_pattern(parts, spec.include) is not None
        if not forced and elements < spec.min_elements:
            selection.skipped[name] = (
                f"元素数 {elements} < min_elements={spec.min_elements}"
                "（确属可学习参数请写进 include，它不受这条限制）")
            continue
        hit = _match_pattern(parts, spec.exclude)
        if hit is not None:
            selection.skipped[name] = f"命中 exclude={hit!r}（统计量/烘焙常量等）"
            continue
        if name in graph_inputs:
            selection.skipped[name] = "是图输入（可覆盖输入），扰动无意义"
            continue
        node_consumers = consumers.get(name, [])
        if not node_consumers:
            selection.skipped[name] = "无消费者（死张量）"
            continue
        if not forced and not weight_like(name, consumers, weight_memo):
            selection.skipped[name] = (
                "非权重张量（名字不是 weight/bias/*_bias，也没经布局算子走到核心算子的"
                "权重槽位）：多半是统计量、掩码或坐标常量；确属可学习参数请用 include 纳入")
            continue

        energy = tensor_energy(tensor, src_dir) if with_energy else float("nan")
        candidate = Candidate(
            name=name, tensor=tensor, elements=elements, energy=energy, dtype=dtype)
        selection.candidate_energy += energy
        selection.candidate_elements += elements

        group = None
        matched_by = ""
        if match_all:
            group, matched_by = "all", "all"
        else:
            for token, alternatives in zip(spec.groups, groups):
                if _any_prefix(parts, alternatives):
                    group, matched_by = token, "name"
                    break
            if group is None:
                for token, alternatives in zip(spec.groups, groups):
                    for node in node_consumers:
                        if _any_prefix(key_parts(node.name), alternatives):
                            group, matched_by = token, "node"
                            break
                    if group is not None:
                        break
        if group is None:
            selection.skipped[name] = f"未命中任何 group={list(spec.groups)}"
            continue

        candidate.group = group
        candidate.matched_by = matched_by
        selection.selected[name] = candidate
        selection.group_energy[group] = selection.group_energy.get(group, 0.0) + energy
        selection.group_elements[group] = (
            selection.group_elements.get(group, 0) + elements)

    if not selection.selected:
        raise ValueError(
            "没有任何张量被选中：检查 groups / exclude / min_elements。"
            f"候选能量={selection.candidate_energy:.3e}，"
            f"跳过 {len(selection.skipped)} 个张量")
    return selection


def resolve_sigmas(selection: Selection, spec: WeightPerturbSpec) -> dict[str, float]:
    """每个组实际使用的 σ。

    给了 spec.sigma 就直接用；否则按总方差配平：
    ``σ_group = σ_full·√(E_total/E_group)``（energy）或 ``√(N_total/N_group)``（count）。
    """
    sigmas: dict[str, float] = {}
    if spec.sigma is not None:
        return {group: float(spec.sigma) for group in selection.group_energy}
    if spec.balance == "none":
        return {group: float(spec.sigma_full) for group in selection.group_energy}
    for group, energy in selection.group_energy.items():
        if spec.balance == "energy":
            if energy <= 0:
                raise ValueError(f"组 {group!r} 的权重能量为 0，无法配平")
            ratio = selection.candidate_energy / energy
        else:
            elements = selection.group_elements[group]
            if elements <= 0:
                raise ValueError(f"组 {group!r} 的元素数为 0，无法配平")
            ratio = selection.candidate_elements / elements
        sigmas[group] = float(spec.sigma_full * np.sqrt(ratio))
    return sigmas


# checker 对「外部数据位置」的几条要求：这些在 .data 与图分处两个目录时不成立，
# 而 onnxruntime 本身接受这种引用。放宽后由 _verify_external_refs 顶上。
_LOCATION_ERRORS = (
    "should be a relative path",
    "points outside the directory",
    "should be stored in",
    "is not regular file",
    "does not exist",
)


def _external_data_outside(onnx, model, model_dir: str) -> bool:
    """是否有外部数据落在模型目录之外（决定要不要放宽 checker 的定位规则）。"""
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
    """自校外部数据：文件是否存在、offset+length 是否超出文件范围。"""
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
    """checker 校验；relax_locations 时只放宽「外部数据位置」这类错误。"""
    try:
        onnx.checker.check_model(out_path)
    except Exception as error:
        message = str(error)
        if relax_locations and any(token in message for token in _LOCATION_ERRORS):
            return False, message
        raise
    return True, ""


def resolve_effective_path(src_path: str, spec: WeightPerturbSpec | None,
                           member_index: int = 0, *, verbose: bool = False) -> str:
    """把源模型路径换成实际要加载的路径：无扰动原样返回，否则返回成员模型。

    成员必须已落盘，这里不触发构建。
    """
    if spec is None:
        return src_path
    return resolve_member_path(src_path, spec, member_index)


# ---------------------------------------------------------------------------
# 成员集落盘
# ---------------------------------------------------------------------------
BASE_DATA = "base.data"                      # 成员集里共享的源 .data（硬链接，或复制一次）
PERTURBED_DATA_FMT = "perturbed.{tag}.data"  # 逐成员的扰动权重
MEMBER_MODEL_FMT = "{stem}.{tag}.onnx"       # 逐成员的图
MANIFEST_NAME = "spw_members.json"
_BASE_PREFIX = "base"                        # base.data / base_N.data
_PERTURBED_PREFIX = "perturbed"
_MEMBER_TAG_FMT = "m{:03d}"


def _base_spec(spec: WeightPerturbSpec) -> WeightPerturbSpec:
    """种子显式化（None -> 0）。

    与进程 rank 无关：成员集是多卡共享的落盘产物，成员独立性来自
    ``SeedSequence([seed, index])``；把 rank 叠进种子会让各 rank 各造一份。
    """
    seed = 0 if spec.seed is None else int(spec.seed)
    if spec.seed == seed:
        return spec
    return replace(spec, seed=seed)


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def set_dir_for(src_path: str, spec: WeightPerturbSpec, *, out_dir: str | None = None) -> str:
    """成员集目录：<目录>/<模型名>.spw-members-<指纹>/。

    优先级 out_dir -> member_dir -> cache_dir -> XMETAI_SPW_CACHE_DIR -> 源模型同目录。
    目录名不含成员数，先造 5 个再补到 51 个会落在同一目录并合并清单。
    """
    spec = _base_spec(spec)
    directory = (out_dir or spec.member_dir or spec.cache_dir
                 or os.environ.get("XMETAI_SPW_CACHE_DIR")
                 or osp.dirname(osp.abspath(src_path)))
    stem = osp.splitext(osp.basename(src_path))[0]
    name = f"{stem}.spw-members-{spec.fingerprint(src_path)}"
    return osp.join(osp.abspath(directory), name)


def member_tag(index: int) -> str:
    """成员号在文件名里的写法（``m000``）：成员共用成员集目录，靠它区分。"""
    return _MEMBER_TAG_FMT.format(int(index))


def _perturbed_name(index: int) -> str:
    return PERTURBED_DATA_FMT.format(tag=member_tag(index))


def member_dir_for(src_path: str, spec: WeightPerturbSpec, index: int,
                   *, out_dir: str | None = None) -> str:
    """成员图及其 base/perturbed 数据所在目录（即成员集目录）。

    扁平布局是为了让 onnxruntime 的「外部数据须在模型目录内」校验与共享 base 共存。
    index 参数保留以兼容调用方。
    """
    return set_dir_for(src_path, spec, out_dir=out_dir)


def member_model_path(src_path: str, spec: WeightPerturbSpec, index: int,
                      *, out_dir: str | None = None) -> str:
    """第 index 个成员的 .onnx 路径。框架侧就靠它把 model_path 换成成员模型。"""
    stem = osp.splitext(osp.basename(src_path))[0]
    name = MEMBER_MODEL_FMT.format(stem=stem, tag=member_tag(index))
    return osp.join(member_dir_for(src_path, spec, index, out_dir=out_dir), name)


def member_perturbed_path(src_path: str, spec: WeightPerturbSpec, index: int,
                          *, out_dir: str | None = None) -> str:
    """第 index 个成员自己那份被扰动权重的落盘路径。"""
    return osp.join(member_dir_for(src_path, spec, index, out_dir=out_dir),
                    _perturbed_name(index))


def member_base_path(src_path: str, spec: WeightPerturbSpec, *,
                     out_dir: str | None = None) -> str:
    """成员集里共享的源 .data（硬链接，或唯一的那份复制）。"""
    return osp.join(set_dir_for(src_path, spec, out_dir=out_dir), BASE_DATA)

def member_exists(src_path: str, spec: WeightPerturbSpec, index: int, *,
                  out_dir: str | None = None) -> bool:
    """该成员的 .onnx 是否已经在盘上（只查文件，不触发任何构建）。"""
    return osp.isfile(member_model_path(src_path, spec, index, out_dir=out_dir))


def resolve_member_path(src_path: str, spec: WeightPerturbSpec, index: int, *,
                        out_dir: str | None = None) -> str:
    """返回第 index 个成员的模型路径；不在盘上就抛出构建指引。

    不自动构建：可能写几十 GB，必须由调用方显式触发。
    """
    path = member_model_path(src_path, spec, index, out_dir=out_dir)
    if osp.isfile(path):
        return path
    raise FileNotFoundError(
        "M1a 成员模型不存在：" + path + "\n"
        "  成员是预先落盘的冻结权重，需要先生成（当前规格：" + spec.describe() + "）。"
        "单卡可以先跑：\n"
        "    python -m xmetai_inference.weights.spw\n"
        "  运行前把该文件末尾的 MODEL 改成 " + osp.abspath(src_path)
        + "、ACTION=\"members\"、MEMBERS=<成员数>、GROUPS=(\""
        + ",".join(spec.groups) + "\",)。\n"
        "  直接跑推理也行：cli 会在建 session 之前按 rank 分片自动生成。")


# ---------------------------------------------------------------------------
# 源布局与字节序
# ---------------------------------------------------------------------------
def _external_location(tensor):
    for entry in tensor.external_data:
        if entry.key == "location" and entry.value:
            return entry.value
    return None


def _location_closure(proto) -> dict:
    """成员图的外部数据引用分布：location -> 张量个数。

    用来断言所有引用都落在成员目录内；只查可达性不够，漏改的 location 仍可能指向源文件。
    """
    counts: dict = {}
    for tensor in external_tensors(proto):
        location = _external_location(tensor)
        counts[location] = counts.get(location, 0) + 1
    return counts



def _source_layout(onnx, model) -> dict:
    """源图的外部数据位置 -> 成员目录里的 base 文件名（base.data / base_N.data）。"""
    locations: list[str] = []
    for tensor in external_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        location = _external_location(tensor)
        if location and location not in locations:
            locations.append(location)
    if not locations:
        return {}
    if len(locations) == 1:
        return {locations[0]: "base.data"}
    return {location: f"base_{i}.data" for i, location in enumerate(locations)}


def _raw_bytes(tensor, src_dir: str):
    """外部数据里的原始字节（不解释、不重排）。inline 张量返回 None。"""
    onnx = _onnx()
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        return None
    info = {entry.key: entry.value for entry in tensor.external_data}
    location = info.get("location") or ""
    if not location:
        return None
    path = location if osp.isabs(location) else osp.join(src_dir, location)
    offset = int(info.get("offset") or 0)
    length = int(info.get("length") or 0)
    if length <= 0:
        return None
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(length)


def _assert_c_order(tensor, src_dir: str, array, name: str):
    """确认源数据是 C 序：读回的字节必须与文件原始字节逐位相同。

    非 C 序被按 C 序读写等于转置权重，不会报错，只会让预报变差。
    """
    raw = _raw_bytes(tensor, src_dir)
    if raw is None:
        return
    if np.asarray(array).tobytes() != raw:
        raise RuntimeError(
            f"张量 {name} 的外部数据不是 C 序（按 C 序读回来的字节与文件里的原始字节不同）。"
            "按现有实现改写会把权重转置，已中止。请先用 tensor_array 的口径核对这份导出。")


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def _noise_shape(shape, struct: str):
    """按结构给出独立抽样的形状；三种结构每元素方差都是 1，因此预算不变。

    ndim<2 的张量在 channel 下退化为 iid。
    """
    if struct == "scalar":
        # scalar 对一维张量也不退化，必须在下面的 len(shape)<2 之前判
        return (1,) * len(shape)
    if struct == "iid" or len(shape) < 2:
        return shape
    if struct == "channel":
        return (shape[0],) + (1,) * (len(shape) - 1)
    raise ValueError(f"未知的 noise_struct：{struct!r}")


def _expand_noise(shape, struct: str, rng):
    """按结构抽 ξ 并展开到 shape。"""
    draw_shape = _noise_shape(shape, struct)
    if draw_shape == shape:
        return rng.standard_normal(shape)
    return np.broadcast_to(rng.standard_normal(draw_shape), shape)


def _draw(array, sigma: float, rng, struct: str = "iid") -> np.ndarray:
    """W~ = W ⊙ (1 + σξ)；fp32 走快路，其它 dtype 在 float64 上算再落回。"""
    base = np.asarray(array)
    if base.dtype == np.float32:
        noise = _expand_noise(base.shape, struct, rng).astype(np.float32)
        out = base.astype(np.float32, copy=True)
        out *= (np.float32(1.0) + np.float32(sigma) * noise)
        return out
    work = base.astype(np.float64)
    work *= (1.0 + float(sigma) * _expand_noise(base.shape, struct, rng))
    return work.astype(base.dtype)


def _share_data(src: str, dst: str, *, link: bool) -> tuple[str, str]:
    """把源 .data 放进成员集目录：硬链接优先，失败则整份复制一次。

    刻意不用符号链接：onnxruntime 会校验外部数据的真实路径必须在模型目录内，
    指向目录外的符号链接会被拒绝。返回 (方式, 失败原因)。
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


def _rebind_unperturbed(onnx, proto, base_map, selected):
    """把未被扰动的外部张量也改到成员目录里的 base_*.data，使目录自包含。

    范围用 external_tensors（含 Constant 节点），不能退回 graph.initializer；
    offset/length 不用动，base 与源 .data 字节布局相同。
    """
    for tensor in external_tensors(proto):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        if tensor.name in selected:
            continue                      # 稍后会被改写成 PERTURBED_DATA
        location = _external_location(tensor)
        base_name = base_map.get(location)
        if not base_name:
            raise RuntimeError(
                f"张量 {tensor.name} 的外部数据位置 {location!r} 不在源布局里"
                f"（源布局 {sorted(base_map)}）——成员目录无法自包含，已中止")
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = base_name


def _rewrite_member(onnx, proto, src_dir: str, member_dir: str, order, sigma_of,
                    seed: int, index: int, base_map, perturbed_name: str,
                    struct: str = "iid"):
    """把 proto 里被选中的张量换成扰动后的值；外部数据写进成员目录的 perturbed.*.data。"""
    by_name = {tensor.name: tensor for tensor in proto.graph.initializer}
    _rebind_unperturbed(onnx, proto, base_map, set(order))
    data_path = osp.join(member_dir, perturbed_name)
    children = np.random.SeedSequence([int(seed), int(index)]).spawn(len(order))
    records = []
    cursor = 0
    with open(data_path, "wb") as handle:
        for name, child in zip(order, children):
            tensor = by_name[name]
            is_external = tensor.data_location == onnx.TensorProto.EXTERNAL
            elements = int(np.prod(tensor.dims)) if tensor.dims else 1
            array = tensor_array(tensor, src_dir)
            _assert_c_order(tensor, src_dir, array, name)
            new = _draw(array, sigma_of[name], np.random.default_rng(child),
                        struct=struct)
            raw = new.tobytes()
            if is_external:
                del tensor.external_data[:]
                for key, value in (("location", perturbed_name),
                                   ("offset", str(cursor)),
                                   ("length", str(len(raw)))):
                    entry = tensor.external_data.add()
                    entry.key, entry.value = key, value
                handle.write(raw)
                cursor += len(raw)
            else:
                # 小张量留在图内：外置会导致会话初始化失败
                tensor.CopyFrom(onnx.numpy_helper.from_array(new, name=name))
            records.append({"name": name, "sigma": sigma_of[name],
                            "elements": elements, "external": is_external,
                            "bytes": len(raw)})
    return records, cursor


def _stamp_member(onnx, proto, src_path: str, spec, index: int, sigmas, tensors: int,
                  member_dir: str):
    props = {entry.key: entry.value for entry in proto.metadata_props}
    props.update({
        "spw.method": "m1a-frozen-per-member-weights",
        # 落盘 schema 键，不要改名：已落盘的成员集与加载期自检都靠它
        "spw.mechanism": "m1a",
        "spw.member": str(index),
        "spw.spec": spec.describe(),
        # 用 " | " 连接：group token 内部可能含逗号
        "spw.sigma_per_group": " | ".join(
            f"{key}={value:.6g}" for key, value in sorted(sigmas.items())),
        "spw.tensors": str(tensors),
        "spw.seed": str(spec.seed if spec.seed is not None else ""),
        "spw.source": osp.basename(src_path),
        "spw.source_fingerprint": spec.fingerprint(src_path),
        "spw.spec_json": json.dumps(spec.budgets(), sort_keys=True, ensure_ascii=False),
        "spw.member_dir": member_dir,
    })
    onnx.helper.set_metadata_props(proto, props)


def _merge_manifest(manifest: str, report: dict) -> dict:
    """把本次生成的成员并进已有清单（多 rank 分片生成时各写各的，不能互相抹掉）。"""
    try:
        with open(manifest, encoding="utf-8") as handle:
            old = json.load(handle)
    except (OSError, ValueError):
        return report
    if not isinstance(old, dict) or old.get("fingerprint") != report.get("fingerprint"):
        return report                       # 规格或源模型变了：旧清单作废
    merged = {int(item["index"]): item for item in (old.get("member_reports") or [])}
    merged.update({int(item["index"]): item for item in report["member_reports"]})
    report["member_reports"] = [merged[key] for key in sorted(merged)]
    report["members"] = max(int(old.get("members") or 0), int(report["members"]))
    warnings = list(old.get("warnings") or [])
    warnings += [w for w in report["warnings"] if w not in warnings]
    report["warnings"] = warnings
    report["built"] = sorted(set(old.get("built") or []) | set(report.get("built") or []))
    return report


def build_member_set(src_path: str, spec: WeightPerturbSpec, count: int, *,
                     indices: "list[int] | None" = None,
                     out_dir: str | None = None, link: bool = True,
                     verbose: bool = True) -> dict:
    """生成 count 个成员中的一部分（indices），返回报告 dict。

    幂等：已存在的成员直接复用。多卡时各 rank 按 ``range(rank, count, world_size)``
    分片，indices 必须互不相交；清单按成员号合并。base.data 只建一次。
    """
    onnx = _onnx()
    src_path = osp.abspath(src_path)
    src_dir = osp.dirname(src_path)
    spec = _base_spec(spec)
    if count <= 0:
        raise ValueError(f"build_member_set 需要 count > 0，实际 {count}")
    todo = list(range(count)) if indices is None else sorted({int(i) for i in indices})
    if not todo:
        raise ValueError("build_member_set 的 indices 为空，没有要生成的成员")
    out_of_range = [i for i in todo if not 0 <= i < count]
    if out_of_range:
        raise ValueError(
            f"build_member_set 的 indices 越界（count={count}）：{out_of_range[:8]}")

    model = load_model(src_path)
    selection = select_initializers(model, spec, src_dir)
    sigmas = resolve_sigmas(selection, spec)
    order = [tensor.name for tensor in model.graph.initializer
             if tensor.name in selection.selected]
    sigma_of = {name: float(sigmas[selection.selected[name].group]) for name in order}
    base_map = _source_layout(onnx, model)

    set_dir = set_dir_for(src_path, spec, out_dir=out_dir)
    os.makedirs(set_dir, exist_ok=True)

    # 成员图与 base/perturbed 必须同目录；硬链接不可行时复制，扁平布局让复制只发生一次
    share: dict[str, str] = {}
    warnings: list[str] = []
    for location, base_name in base_map.items():
        source = location if osp.isabs(location) else osp.join(src_dir, location)
        mode, reason = _share_data(osp.abspath(source),
                                   osp.join(set_dir, base_name), link=link)
        share[base_name] = mode
        if mode == "copy":
            warnings.append(
                f"{base_name}: 无法与源 .data 共享字节（{reason}），已整份复制一次"
                f"（{osp.getsize(source) / 2**30:.1f} GB）——所有成员共用这一份，不会按"
                "成员数翻倍。想省掉这份复制就得让硬链接可用（与源文件同卷、且源文件属于"
                "当前用户），排查清单见 README 第 10 节第 5 条。")
    if any(mode == "link" for mode in share.values()):
        warnings.insert(0, "base_*.data 用硬链接共享源 .data，onnx 的 checker 会因此报"
                           "「multiple hard links」并拒绝——已按既有做法放宽该类定位"
                           "校验，引用可达性由自校验承担。")

    members = []
    for index in todo:
        perturbed_name = _perturbed_name(index)
        proto = onnx.ModelProto()
        proto.CopyFrom(model)
        records, perturbed_bytes = _rewrite_member(
            onnx, proto, src_dir, set_dir, order, sigma_of,
            int(spec.seed or 0), index, base_map, perturbed_name,
            struct=spec.noise_struct)
        _stamp_member(onnx, proto, src_path, spec, index, sigmas, len(order), set_dir)
        out_path = member_model_path(src_path, spec, index, out_dir=out_dir)
        tmp = f"{out_path}.tmp{os.getpid()}"
        onnx.save(proto, tmp)
        os.replace(tmp, out_path)
        broken = _verify_external_refs(onnx, proto, set_dir)
        if broken:
            raise RuntimeError(f"成员 {index} 的外部数据引用有问题：" + "; ".join(broken[:5]))
        if _external_data_outside(onnx, proto, set_dir):
            raise RuntimeError(f"成员 {index} 有外部数据落在成员集目录之外")
        # 放宽 checker 的外部数据定位检查：base.data 是硬链接，checker 会拒绝；
        # 可达性与字节范围由 _verify_external_refs 承担。
        _check_model_path(onnx, out_path, relax_locations=True)
        members.append({
            "index": index, "dir": set_dir, "model": out_path,
            "perturbed": osp.join(set_dir, perturbed_name),
            "perturbed_bytes": perturbed_bytes,
            "files": sorted(os.listdir(set_dir)),
        })
        if verbose:
            print(f"  {osp.basename(out_path)}: {len(records)} 个张量 / "
                  f"{perturbed_bytes / 2**20:.1f} MB 被扰动权重")

    report = {
        "source": src_path,
        "set_dir": set_dir,
        "spec": spec.describe(),
        "fingerprint": spec.fingerprint(src_path),
        "seed": int(spec.seed or 0),
        "members": count,
        "tensors_per_member": len(order),
        "sigmas": {key: float(value) for key, value in sorted(sigmas.items())},
        "base_map": base_map,
        "share": share,
        "perturbed_elements": sum(selection.selected[name].elements for name in order),
        "member_reports": members,
        "warnings": warnings,
        "built": todo,
    }
    manifest = osp.join(set_dir, MANIFEST_NAME)
    report = _merge_manifest(manifest, report)
    with open(manifest, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    report["manifest"] = manifest
    if verbose:
        total = sum(item["perturbed_bytes"] for item in members)
        shard = "全部" if len(todo) == count else f"本次 {len(todo)}/{count} 个"
        print(f"成员集就绪（{shard}）-> {set_dir}")
        print(f"  被扰动元素 {report['perturbed_elements'] / 1e6:.2f}M / 成员，"
              f"本次生成 {len(members)} 个 x {total / 2**30:.2f} GB")
        if share:
            print("  共享的源 .data：" + ", ".join(
                f"{name}={mode}" for name, mode in sorted(share.items())))
        for group, sigma in sorted(sigmas.items()):
            print(f"  sigma[{group}] = {sigma:.6g}")
        for warning in warnings:
            print(f"  [警告] {warning}")
    return report


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def verify_member_set(src_path: str, spec: WeightPerturbSpec, count: int, *,
                      out_dir: str | None = None, sample: int = 8) -> dict:
    """复核成员集：文件齐全、引用可达，且成员之间权重确实不同。

    同时量权重级相对扰动 rms(W~-W)/rms(W)，应接近该组配平后的 σ。
    """
    onnx = _onnx()
    src_path = osp.abspath(src_path)
    src_dir = osp.dirname(src_path)
    spec = _base_spec(spec)
    model = load_model(src_path)
    selection = select_initializers(model, spec, src_dir)
    sigmas = resolve_sigmas(selection, spec)

    problems: list[str] = []
    ratios: dict[str, list[float]] = {}
    names = list(selection.selected)
    step = max(1, len(names) // max(1, sample))
    picked = names[::step][:sample]

    source_values = {}
    for name in picked:
        tensor = next(t for t in model.graph.initializer if t.name == name)
        source_values[name] = np.asarray(tensor_array(tensor, src_dir))

    members = []
    set_dir = member_dir_for(src_path, spec, 0, out_dir=out_dir)
    real_dir = osp.realpath(set_dir)
    for index in range(count):
        model_path = member_model_path(src_path, spec, index, out_dir=out_dir)
        if not osp.isfile(model_path):
            problems.append(f"member_{index:03d}: 缺少 {model_path}")
            continue
        proto = load_model(model_path)
        for broken in _verify_external_refs(onnx, proto, set_dir):
            problems.append(f"member_{index:03d}: {broken}")
        # 还要确认引用是成员目录内的裸文件名：源 .data 若仍在原地，可达性检查发现不了
        closure = _location_closure(proto)
        stray = sorted(str(location) for location in closure
                       if not location or osp.dirname(location)
                       or not (location.startswith(_BASE_PREFIX)
                               or location.startswith(_PERTURBED_PREFIX)))
        if stray:
            problems.append(
                f"member_{index:03d}: 外部数据引用不在成员集目录内：{stray}"
                "（成员不是自包含的，onnxruntime 建会话时会去找源 .data）")
        # onnxruntime 校验真实路径必须在模型目录内，符号链接会被拒，提前拦下
        for location in sorted((key for key in closure if key), key=str):
            real = osp.realpath(osp.join(set_dir, location))
            if real != osp.join(real_dir, location):
                problems.append(
                    f"member_{index:03d}: {location} 的真实路径在成员集目录之外"
                    f"（{real}）——新版 onnxruntime 的外部数据路径校验会拒绝它")
        by_name = {tensor.name: tensor for tensor in proto.graph.initializer}
        for name in picked:
            tensor = by_name.get(name)
            if tensor is None:
                problems.append(f"member_{index:03d}: 少了张量 {name}")
                continue
            got = np.asarray(tensor_array(tensor, set_dir))
            want = source_values[name]
            if got.shape != want.shape:
                problems.append(f"member_{index:03d}: {name} 形状变了")
                continue
            if np.array_equal(got, want):
                problems.append(f"member_{index:03d}: {name} 与源权重逐位相同（扰动没生效）")
                continue
            ratio = float(np.sqrt(np.mean((got - want) ** 2))
                          / (np.sqrt(np.mean(want ** 2)) or 1.0))
            ratios.setdefault(selection.selected[name].group, []).append(ratio)
        perturbed_path = member_perturbed_path(src_path, spec, index, out_dir=out_dir)
        members.append({"index": index, "model": model_path,
                        "locations": closure,
                        "perturbed_bytes": osp.getsize(perturbed_path)
                        if osp.isfile(perturbed_path) else 0})

    measured = {}
    for group, values in sorted(ratios.items()):
        values.sort()
        measured[group] = {
            "measured_median": values[len(values) // 2],
            "expected_sigma": float(sigmas[group]),
            "ratio": values[len(values) // 2] / float(sigmas[group]),
        }
    return {"problems": problems, "members": members, "amplitude": measured,
            "sampled_tensors": picked, "count": count}


def format_member_report(info: dict) -> str:
    lines = []
    lines.append(f"成员校验：{len(info['members'])}/{info['count']} 个成员就位")
    for item in info["members"]:
        lines.append(f"  member_{item['index']:03d}  被扰动权重 "
                     f"{item['perturbed_bytes'] / 2**20:.1f} MB")
        for location, number in sorted((item.get("locations") or {}).items(),
                                       key=lambda pair: str(pair[0])):
            lines.append(f"      外部数据引用 {str(location):16s} x{number}")
    if info["amplitude"]:
        lines.append("权重级相对扰动 rms(W~-W)/rms(W)（抽样，应当≈该组配平后的 σ）：")
        lines.append(f"  {'group':26s} {'measured':>10s} {'sigma':>10s} {'ratio':>7s}")
        for group, item in info["amplitude"].items():
            lines.append(f"  {group:26s} {item['measured_median']:10.5f} "
                         f"{item['expected_sigma']:10.5f} {item['ratio']:7.3f}")
    if info["problems"]:
        lines.append(f"问题 {len(info['problems'])} 条：")
        for problem in info["problems"][:20]:
            lines.append(f"  - {problem}")
    else:
        lines.append("引用、形状、逐成员差异：全部通过")
    return "\n".join(lines)


def print_group_report(path: str, *, min_elements: int = 64,
                       exclude: Sequence[str] = DEFAULT_EXCLUDE, include: Sequence[str] = (),
                       energy: bool = True, top: int = 25, depth: int = 2,
                       under: str | None = None) -> None:
    """列出候选注入组。

    depth 控制分桶深度；under 把统计限制到某个子树并从下一级开始列，
    两者配合可做注入点消融扫描。
    """
    onnx = _onnx()
    model = load_model(path)
    src_dir = osp.dirname(osp.abspath(path))
    consumers = consumer_map(model.graph)
    memo: dict = {}
    under_parts = key_parts(under) if under else ()
    depth_max = max(1, min(int(depth), 8))
    if under_parts and depth_max <= len(under_parts):
        depth_max = len(under_parts) + 1          # 否则一个桶都列不出来，静默空表最坑
    first_depth = len(under_parts) + 1 if under_parts else 1

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
        if not forced and elements < min_elements:
            continue
        value = tensor_energy(tensor, src_dir) if energy else float(elements)
        if _match_pattern(parts, exclude) is not None:
            excluded_energy += value
            excluded_count += 1
            continue
        if not forced and not weight_like(name, consumers, memo):
            other_energy += value
            other_count += 1
            continue
        weight_total += value
        total_tensors += 1
        total_elements += elements
        if under_parts and parts[: len(under_parts)] != under_parts:
            continue                        # under：只统计子树
        chosen.append((value, name, elements))
        for level in range(first_depth, depth_max + 1):
            key = "/".join(parts[:level])
            bucket = buckets.setdefault(
                key, {"tensors": 0, "elements": 0, "energy": 0.0})
            bucket["tensors"] += 1
            bucket["elements"] += elements
            bucket["energy"] += value

    unit = "能量 ΣW^2" if energy else "元素数"
    print(f"候选权重合计（{unit}）{weight_total:.6g}，"
          f"{total_tensors} 个张量 / {total_elements/1e6:.2f}M 元素")
    if under_parts:
        share = (sum(item[0] for item in chosen) / weight_total * 100.0
                 if weight_total else 0.0)
        print(f"under={'/'.join(under_parts)}：命中 {len(chosen)} 个张量 / "
              f"{sum(item[2] for item in chosen)/1e6:.2f}M 元素，占候选能量 {share:.1f}%"
              f"（桶列到第 {depth_max} 级）")
    print(f"{'group':40s} {'tensors':>8s} {'elements':>12s} {'share':>7s}")
    for key, bucket in sorted(buckets.items(), key=lambda kv: -kv[1]["energy"])[: top]:
        share = bucket["energy"] / weight_total * 100.0 if weight_total else 0.0
        print(f"{key:40s} {bucket['tensors']:8d} {bucket['elements']/1e6:11.2f}M {share:6.1f}%")
    print("\n能量最大的 10 个候选张量（肉眼确认没有混进常量）：")
    for value, name, elements in sorted(chosen, reverse=True)[:10]:
        print(f"  {value:12.4g}  {name:56s} n={elements}")
    if excluded_count:
        share = excluded_energy / (weight_total + excluded_energy) * 100.0
        print(f"\n另命中 exclude 的 {excluded_count} 个张量，能量占比 {share:.1f}%"
              f"（默认排除统计量/烘焙常量，可用 exclude 覆盖）")
    if other_count:
        print(f"另 {other_count} 个非权重张量（{other_energy:.4g}）未计入；"
              f"确属可学习参数可用 include 纳入")
    print("\ngroups 支持任意深度前缀（`/` 与 `.` 等价），上面列出的 key 可以直接填进 spec 的"
          " `groups`。\n扫子组用：under=<前缀> depth=<级数>；权重名被导出器拍平的图会走"
          "消费者节点名前缀匹配。")


__all__ = [
    "BALANCE_MODES",
    "BASE_DATA",
    "Candidate",
    "DEFAULT_EXCLUDE",
    "MANIFEST_NAME",
    "MEMBER_FORMAT",
    "NOISE_BLOCK_FROZEN",
    "NOISE_STRUCTS",
    "PERTURBED_DATA_FMT",
    "Selection",
    "SpecBase",
    "WeightPerturbSpec",
    "build_member_set",
    "consumer_map",
    "external_tensors",
    "format_member_report",
    "group_alternatives",
    "iter_graphs",
    "key_parts",
    "load_model",
    "member_base_path",
    "member_dir_for",
    "member_exists",
    "member_model_path",
    "member_perturbed_path",
    "member_tag",
    "print_group_report",
    "resolve_effective_path",
    "resolve_member_path",
    "resolve_sigmas",
    "select_initializers",
    "set_dir_for",
    "tensor_array",
    "tensor_energy",
    "verify_member_set",
    "weight_like",
]


if __name__ == "__main__":
    # 造产物入口：改字面量后运行 python -m xmetai_inference.weights.spw
    # ACTION = "groups"（列候选组）| "members"（生成成员集）
    ACTION = "members"          # "groups" | "members"
    MODEL = "model.onnx"        # ← 改成源模型路径
    OUT = None                  # 输出目录/文件；None = 源模型同目录
    CACHE_DIR = None            # 落盘目录；None = 源模型同目录

    # ---- members：逐成员冻结权重 ----
    MEMBERS = 51
    GROUPS = ("all",)           # 前缀元组，或逗号分隔的字符串
    SIGMA = None                # 给了就跳过自动配平
    SIGMA_FULL = 0.01
    BALANCE = "energy"          # "energy" | "count" | "none"
    INCLUDE = ()
    MIN_ELEMENTS = 64
    SEED = None
    VERIFY = True               # 造完立刻复核引用 / 形状 / 逐成员差异

    # ---- groups：列候选注入组 ----
    ENERGY = True
    TOP = 25
    DEPTH = 2
    UNDER = None

    if ACTION == "groups":
        print_group_report(MODEL, min_elements=MIN_ELEMENTS, include=INCLUDE,
                           energy=ENERGY, top=TOP, depth=DEPTH, under=UNDER)
    elif ACTION == "members":
        spec = WeightPerturbSpec(
            members=MEMBERS, groups=GROUPS, sigma=SIGMA, sigma_full=SIGMA_FULL,
            balance=BALANCE, include=INCLUDE, min_elements=MIN_ELEMENTS, seed=SEED,
            cache_dir=CACHE_DIR)
        report = build_member_set(MODEL, spec, MEMBERS, out_dir=OUT, verbose=True)
        print(f"清单：{report['manifest']}")
        print("成员模型可直接当确定性模型跑：它们是成员集目录里的普通确定性 onnx"
              "（<模型名>.mNNN.onnx）。")
        if VERIFY:
            info = verify_member_set(MODEL, spec, MEMBERS, out_dir=OUT)
            print()
            print(format_member_report(info))
            raise SystemExit(1 if info["problems"] else 0)
    else:
        raise SystemExit(f"ACTION 应为 groups / members，实际 {ACTION!r}")
