# -*- coding: utf-8 -*-
"""权重扰动的共享底座：选张量、配平、外部数据。

**这里不再做注入**。注入只有一条实现，在 weights.py：逐成员按 W̃ = W ⊙ (1 + σξ) 算出
每个成员的一套权重、落成一份普通的确定性 onnx（m1a，论文的主配置）。本模块留下的是两条
路径都要用的东西：

  * 选张量：key_parts / weight_like / group_alternatives / select_initializers
    —— 判定「什么算可学习参数」、按名字分组、并按 energy/count 配平（resolve_sigmas）；
  * 外部数据：external_tensors / _repoint_external_data / _verify_external_refs
    —— 成员图对外只引用 base.data 与 perturbed.mNNN.data，不复制那 2.8 GB。

选张量的口径（乘性而非加性：注入方差 = σ²‖W‖²_F，σ 无量纲，可跨层/跨组配平）是论文
口径，weights.py 直接复用同一套 Selection。

（历史上这里还有第三条路 M3「图内随机算子」——插入 RandomNormalLike(W) 每次前向重抽。
它 2026-09-23 被实测淘汰，代码已于 2026-09-24 删除；当年为它留下的审计/结构核对工具
（audit_model / reference_budget / structural_diff / find_source_for 与 `audit` 子命令）
已于 2026-09-28 一并删除，M3 的结论与判据留在 README 的「M3 定格」一节。）
"""
from __future__ import annotations

import os.path as osp
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spec import WeightPerturbSpec

_CHUNK = 1 << 22
_FLOAT_DTYPES = ("FLOAT", "FLOAT16", "DOUBLE", "BFLOAT16")

# 「可学习参数」的结构判定：名字像 weight/bias，或经布局算子走到核心计算算子。
# 目的是把 mean/std/attn_mask/坐标常量这类**烘焙进图的常量**排除掉——它们不是权重，
# 但量级极大（FGVP 里 mean 一项就占 Σ‖W‖² 的 96%），一旦混进能量口径，
# 组间配平 σ_group = σ_full·√(E_total/E_group) 会被整体抬高好几倍。
# 名字判定：命中这些叶子名（或其 *_xxx 形式，如 dt_bias）就算可学习参数。
_WEIGHT_LEAF_NAMES = ("weight", "bias", "scale", "shift", "gamma", "beta", "embedding")
# 核心算子 -> 「权重落在第几个输入槽」。只认这些槽位，可以排掉一类误判：
# 例如把常量场喂给 Conv 的第 0 个输入（激活槽）时，它看着走到了卷积，其实只是数据。
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


# onnx.AttributeProto 里的取值。写成常量，是为了让下面这些函数在导入期不必先拿到 onnx 包。
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
    """图里**全部**携带外部数据的张量：initializer + Constant 节点的张量属性 + 子图。

    为什么不能只扫 graph.initializer —— 2026-09-23 在服务器上踩实的坑：导出器开
    convert_attribute=True 时，**Constant 节点里的大张量也会被外置**（FGVP 有 4 个
    attention 常量：[16200,16] x2、[16200,32] x2），而它们的 location 同样按「相对图
    文件的路径」解析。漏掉它们的后果不是报错，而是**静默漏改**：成员图改完之后仍指向
    <stem>.data，onnxruntime 建会话时才报

        filesystem error: cannot get file size: No such file or directory
        [.../member_000/<stem>.data]

    凡是「改名 location / 检查引用可达性」的地方都必须走这个口径，不要再各写一份
    graph.initializer 循环。
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
    """把 initializer 名或节点名切成路径分量，用于前缀匹配。

    `latent_mixer.layers.0.blocks.0.mixer.mixer.q_proj.weight` 与
    `/latent_mixer/layers.0/blocks.0/mixer/GatedDeltaNet2Fn` 都切成
    (latent_mixer, layers, 0, blocks, 0, mixer, ...)，于是「按架构组选张量」
    对「名字完整的图」和「权重名被导出器拍平、只剩节点名可用的图」都成立。
    """
    cleaned = str(name).strip("/")
    return tuple(token for token in re.split(r"[./]+", cleaned) if token)


def group_alternatives(token) -> tuple[tuple[str, ...], ...]:
    """一个 group token 写成「多个前缀的并集」时的展开（`,` 或 `|` 分隔）。

    为什么需要：`latent_mixer` 的 12 个 block 是**交错**的（0/2/4/6/8/10 是
    GatedDeltaNet 的 mixer，1/3/5/7/9/11 是 attn+mlp），没有任何一个前缀能同时覆盖
    「全部块类型」的某一半。而把一个组拆成 6 个 token 写进 `groups` 会让
    `resolve_sigmas` 逐组配平 —— 每组都拿满一份预算，总注入方差变成 6 倍（G 已经踩过
    2 倍的那个坑）。并集写法让「6 个块」仍然是**一个组**，预算守恒。

        groups=("latent_mixer.layers.0.blocks.0,...blocks.2,...blocks.4",)

    `key_parts` 会把 `.` 与 `/` 都切成路径分量，所以逗号/竖线是唯一的分隔语义，
    include/exclude 不受影响（它们是「模式列表」，逐个整体匹配）。
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
    """判断一个 initializer 是「可学习参数」还是烘焙进图的常量。

    两条判据任一成立即算权重：
      1) 名字最后一段是 weight/bias —— 覆盖 Conv/Linear/LayerNorm 等常规命名，
         也把「被 Add 消费的 bias」这类靠算子看不出来的参数保住；
      2) 经布局算子（Transpose/Reshape/Cast/Slice/…）走到核心算子的**权重槽位**
         （见 _WEIGHT_SLOTS）—— 覆盖导出器把名字拍平成 onnx::Conv_28820 的图；
         只认权重槽位，喂进激活槽的常量场不会被误判。

    于是 mean/std（被 Sub/Mul 消费的归一化统计量）、attn_mask（被 Add 消费的掩码）、
    坐标常量都会被排除在外。这不是洁癖：那些常量的量级比权重高好几个数量级，
    混进能量口径会把组间配平整体抬偏。
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

    候选过滤顺序：dtype 必须是浮点 → 元素数 ≥ min_elements → 是「可学习参数」
    （见 weight_like）→ 不被 exclude 命中 → 不是图输入（可覆盖输入）→ 有消费者。
    任一不满足都记进 skipped，不静默丢弃。
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
        # include 先算：它是「明确当权重看」的强制声明，必须能越过「元素数太小」这条
        # 启发式（GatedDeltaNet 的 A_log 只有 24 个元素，但它控制潜变量记忆衰减——
        # 是全模型里唯一一个扰动会随步累积的参数，正是 min_elements 会误伤的典型）。
        # exclude 仍然优先：显式排除比显式纳入更具体。
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

    给了 spec.sigma 就直接用它；否则按方差预算配平：让该组注入的**总方差**等于
    「全部候选张量按 σ_full 注入」的总方差，即

        σ_group = σ_full · √(E_total / E_group)        （energy 口径，默认）
        σ_group = σ_full · √(N_total / N_group)        （count 口径，论文原式）

    这样扫「注入点」时各组的总方差一致，消融才可比。
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


def _repoint_external_data(onnx, model, src_dir: str, out_dir: str):
    """新图落到别的目录时，把外部数据引用重新指向 .data 的真实位置。

    优先写「从新图目录指向 .data 的相对路径」：onnx.checker 明确要求 location 必须是
    相对路径，而且相对路径在同一挂载点内跨机器可迁移。只有在源与目标不在同一个盘 /
    挂载点时才退回绝对路径（Windows 跨盘符时 osp.relpath 会抛错），此时由调用方放宽
    checker 的那一条检查——onnxruntime 本身是接受绝对路径的。

    返回 (改动的引用处数, 是否用了绝对路径)。
    """
    changed = 0
    absolute = False
    for tensor in external_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key != "location" or not entry.value:
                continue
            target = entry.value if osp.isabs(entry.value) else osp.join(src_dir, entry.value)
            target = osp.abspath(target)
            try:
                new_value = osp.relpath(target, out_dir)
            except ValueError:
                new_value, absolute = target, True
            if new_value != entry.value:
                entry.value = new_value
                changed += 1
    return changed, absolute


# checker 对「外部数据位置」的几条要求：必须相对路径、必须落在图所在目录内、必须存
# 在于该目录。它们只在 .data 与图分处两个目录时才不成立，而那种布局恰恰是我们刻意要
# 的（不复制 2.8 GB 权重）；onnxruntime 本身接受这种引用。所以这几条单独放宽，并用
# _verify_external_refs 做一套更贴近实际的校验顶上。
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
    """自己校外部数据引用：文件在不在、offset+length 有没有超出文件范围。

    比 checker 的「必须在同目录」更有意义：它验证的是「这份图真的还能读到那 2.8 GB
    权重」，而且允许 .data 放在别的目录。
    """
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
    """checker 校验；relax_locations=True 时只放宽「外部数据位置」那一类错误。

    返回 (是否完全通过, 放宽掉的错误信息)。放宽掉的定位校验由 _verify_external_refs
    顶上——它同时覆盖「文件在不在、范围对不对」，不是简单地跳过检查。
    """
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
    """框架侧唯一入口：把「源模型路径」换成「实际要加载的路径」。

      * spec 为 None  -> 原样返回（不加扰动）
      * 否则          -> 第 member_index 个成员的模型（必须已落盘）

    放在这里而不是后端里，是为了让 onnx 后端、自检、e2e 用同一条分流。惰性导入是
    为了躲开 graph <- weights 的循环（weights 顶层 import graph）。verbose 只为兼容
    latent.py 的调用点而留着，这里不做任何按需构建（见 weights.resolve_member_path）。
    """
    if spec is None:
        return src_path
    from .weights import resolve_member_path

    return resolve_member_path(src_path, spec, member_index)
