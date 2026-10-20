# -*- coding: utf-8 -*-
"""权重扰动（SPW）：规格、选张量底座、逐成员冻结权重落盘。

本文件是 2026-10-20 从 `perturbation/spec.py` + `perturbation/graph.py` +
`perturbation/weights.py` 合并而来，依次包含：

  1. **规格**（原 spec.py）：`WeightPerturbSpec` —— 选哪组张量、多大 σ、怎么配平、
     缓存与指纹；`SpecBase` 复用 `weights/spa.py` 的实现。
  2. **共享底座**（原 graph.py）：`weight_like` / `select_initializers` /
     `resolve_sigmas` / 外部数据核对 / `resolve_effective_path`。
  3. **m1a 落盘**（原 weights.py）：按 `W~ = W ⊙ (1 + σξ)` 为每个成员算一套权重，
     落成一份普通的确定性 ONNX（成员内整条轨迹冻结）。

图内噪声幅度（σ × k）那条能力已于 2026-10-20 移除——实测无收益（实验记录见
`docs/PAPER.md` §3.3.10 的 `ens_k20`）。

造产物不再走命令行：改本文件末尾 `if __name__ == "__main__"` 里的字面量，直接运行

    python -m xmetai_inference.weights.spw

SPA（潜空间 AR(1)）的 ONNX 版在 `weights/spa.py`，PT2 版在 `weights/spa_pt2.py`；
推理、探针、体检与评测分析仍在 `perturbation/`。
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

# 默认排除项：归一化统计量 + 烘焙进图的常量/掩码。它们不是可学习参数，而且量级
# 远大于权重（FGVP 里一个 `mean` 就占全部候选 ΣW^2 的 96%，`const` 占 90% 量级的
# 局面也是同类问题），混进能量口径会把组间配平整体抬偏；扰动它们还会让整个场
# 一起偏移——正是论文报告的主失效模式（区域平均过离散）。
# 匹配规则见 graph._is_excluded：按名字的路径分量整段匹配，
# 所以 "mean" 命中 `mean`、`foo.mean`，但不会误伤 `mean_proj.weight`。
DEFAULT_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")

BALANCE_MODES = ("energy", "count", "none")

# 噪声在**权重张量内部**的相关结构。论文的 Phase 3 是「把注入点收到粗尺度」（SFNO 取
# 最低十个球谐模）。FGVP 的 latent_mixer 里没有谱域可切，但「让噪声只在少数自由度上
# 变化」这件事可以在权重张量内部直接做 —— 而且**注入的总方差不变**（每个元素仍然
# E[xi^2]=1），所以它是「同预算、换结构」的对照，不是「换幅度」：
#
#   iid      逐元素独立（论文原式，我们的历史基线）
#   channel  每个**输出通道**共享一个 xi（对 ndim>=2 的张量；ndim<2 退化为 iid）。
#            「整通道一致」= 空间上均匀的一阶扰动，是权重空间里最接近「只有粗尺度」的做法
#   scalar   整张量共享**一个** xi（向量也一样，不退化）。每个成员、每个被扰张量只有一个
#            随机数，是随机自由度最少的极端：注入方差仍是 sigma^2 x ||W||^2（每元素
#            E[xi^2]=1），但一个成员张成的扰动子空间秩 <= #被扰张量，所以 N 个成员给出的
#            离散度**本身**就是高方差统计量（单个张量的注入方差比 = sigma^2 x chi^2_1）。
#            它是「同预算、最小秩」的对照臂，也是论文 Phase 3 那类「噪声只落在少数自由度」
#            在没有谱域可切时的另一种极端写法。
NOISE_STRUCTS = ("iid", "channel", "scalar")

# 冻结的指纹键取值：`block` 结构（块边长 8）于 2026-09-28 删除，但当年用 channel 建出来的
# 成员集（S2B 的 51 成员 / 76 GB）指纹里带着 "noise_block": 8，删键会让它们全部失配、
# 被迫重新生成。所以 budgets() 仍按当年的固定值写这个键 —— 与 rank_seed / noise_op 同一个理由。
NOISE_BLOCK_FROZEN = 8

# m1a 成员集的**落盘格式版本**。凡是改动「成员图怎么写」（location 口径、落盘布局、噪声
# 派生方式）就 +1：它进指纹，于是新版本必然落在新的成员集目录里，旧版本的产物不会被静默
# 复用。这条闸门是拿真金白银换来的：
#   1 = 初版：只改 graph.initializer 的 location —— 漏掉被外置的 Constant 节点张量，
#       成员图继续指向 <stem>.data，onnxruntime 建会话时报 cannot get file size。
#   2 = 2026-09-23：location 口径改成 graph.external_tensors（含 Constant 节点），
#       成员图对外只引用 base.data / perturbed.mNNN.data。
#   3 = 2026-09-23：成员集改**扁平**布局（成员图与 base/perturbed 同目录，base 只建一次），
#       并且不再使用符号链接——新版 onnxruntime 会把外部数据解析成真实路径、再校验它必须
#       落在模型所在目录内，符号链接（指向目录外）会被直接拒绝。

MEMBER_FORMAT = 3
@dataclass
class WeightPerturbSpec(SpecBase):
    """一次 SPW 运行的注入规格。

    groups         选中的张量组；按名字路径分量做前缀匹配（`.` 与 `/` 等价），
                   "all" = 全部候选。例：("all",)、("latent_mixer",)、
                   ("decoder_group", "vae_encoder_group")。
                   一个 token 内部可以用 `,`（或 `|`）写**多个前缀的并集**，例如
                   ("latent_mixer.layers.0.blocks.0,...blocks.2,...blocks.4",) ——
                   FGVP 的 12 个 block 是交错的（偶数块是 GatedDeltaNet、奇数是
                   attn+mlp），没有单前缀能覆盖某一半，只能用并集。
                   **并集算一个组**，所以只吃一份方差预算；把它拆成多个 token 写则会
                   逐组配平 -> 总注入方差 = token 数 x 预算（G 已经踩过 2 倍的那个坑）。
                   见 graph.group_alternatives。
    sigma_full     在「全部候选张量」口径下的相对强度（论文的 σ，默认 0.01）
    sigma          直接指定所用的 σ；给了就跳过自动配平（做幅度扫描时用）
    balance        自动配平方式："energy"（按 Frobenius 能量，默认）、
                   "count"（论文原始口径，按元素数）、"none"（不配平，各组同 σ）
    exclude        排除模式（按名字路径分量匹配），默认排除归一化统计量/烘焙常量
    include        强制纳入模式：命中即视为权重，绕过结构判定。
                   用于名字不像 weight/bias 的可学习参数（如 GatedDeltaNet 的 A_log）
    min_elements   小于该元素数的张量跳过（默认 64，滤掉标量/形状常量）
    seed           基础随机种子（None = 0）。逐成员独立性来自 SeedSequence([seed, index])：
                   不按进程 rank 偏移——成员集是多卡共享的落盘产物，把 rank 叠进种子会让
                   各 rank 各造一份目录、并让相邻成员的随机流重叠。
    noise_struct   噪声在张量内部的相关结构，见 NOISE_STRUCTS。默认 "iid"（论文原式）。
                   注意：三种结构的**每元素方差都是 1**，所以 sigma^2 * ||W||^2 的注入方差
                   定律与组间 energy 配平在三种结构下都成立 —— 换结构不换预算。
    cache_dir      改写后的 .onnx 落盘目录；None = 与原模型同目录（外部数据引用最稳）
    members       要生成的成员数（0 = 由调用方从 config 的 members 取）
    member_dir     成员集落盘目录；None = 与原模型同目录（同 cache_dir 的优先级）
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
            # 下面两个键是**冻结的指纹 schema**，不是 dataclass 字段的镜像：它们对应的
            # 那两个自由度（按 rank 偏移种子、噪声算子写法）随 M3 机制一起删掉了，但键必须
            # 留着——指纹决定成员集目录名 `spw-members-<指纹>/`，删键会让已经落盘的成员集
            # （S2B 的 51 成员 / 76 GB）全部失配，被迫重生成。取值就是它们当年的固定值。
            "rank_seed": False,
            "noise_op": "random_normal_like",
            # noise_struct 是后加的字段：**默认值不参与指纹**，否则既有成员集（F / prod 的
            # 51 成员）会因为多出一个 "iid" 键而指纹改变、被迫整份重生成。非默认值必须进来
            # —— 否则 channel 会命中 iid 的成员集目录，静默复用错权重。
            # noise_block 的取值走 NOISE_BLOCK_FROZEN（冻结键，见那里的说明）。
            **({"noise_struct": self.noise_struct, "noise_block": NOISE_BLOCK_FROZEN}
               if self.noise_struct != "iid" else {}),
        }

    def fingerprint_payload(self, src_path: str) -> dict:
        payload = super().fingerprint_payload(src_path)
        # 成员集是**落盘产物**，格式一旦变就必须换目录。见 MEMBER_FORMAT。
        payload["member_format"] = MEMBER_FORMAT
        return payload

    def describe(self) -> str:
        # 用 " | " 连接：单个 token 内部本来就可能含逗号（并集写法），再用逗号连接
        # 两个 token 会读不出边界。
        groups = " | ".join(self.groups)
        amplitude = (f"sigma={self.sigma}" if self.sigma is not None
                     else f"sigma_full={self.sigma_full} (balance={self.balance})")
        extra = f" include={list(self.include)}" if self.include else ""
        struct = "" if self.noise_struct == "iid" else f" noise_struct={self.noise_struct}"
        return (f"groups={groups} {amplitude} exclude={list(self.exclude)}"
                f"{extra}{struct}")


# ---------------------------------------------------------------------------
# 选张量底座（原 graph.py）
# ---------------------------------------------------------------------------
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

    放在这里而不是后端里，是为了让 onnx 后端、自检、e2e 用同一条分流。
    verbose 只为兼容 weights/spa.py 的调用点而留着，这里不做任何按需构建
    （见 weights.spw.resolve_member_path）。
    """
    if spec is None:
        return src_path
    return resolve_member_path(src_path, spec, member_index)


# ---------------------------------------------------------------------------
# m1a 成员集落盘（原 weights.py）
# ---------------------------------------------------------------------------
BASE_DATA = "base.data"                      # 成员集里共享的源 .data（硬链接，或复制一次）
PERTURBED_DATA_FMT = "perturbed.{tag}.data"  # 逐成员的扰动权重
MEMBER_MODEL_FMT = "{stem}.{tag}.onnx"       # 逐成员的图
MANIFEST_NAME = "spw_members.json"
_BASE_PREFIX = "base"                        # base.data / base_N.data
_PERTURBED_PREFIX = "perturbed"
_MEMBER_TAG_FMT = "m{:03d}"


def _base_spec(spec: WeightPerturbSpec) -> WeightPerturbSpec:
    """归一成落盘用的规范形态：种子显式化（None -> 0）。

    **与进程 rank 无关**——成员集是**多卡共享**的落盘产物，逐成员的独立性来自
    ``SeedSequence([seed, index])``。历史上这里还兼着「丢掉 m3 的 rank 偏移」，
    那个语义会同时坏掉两件事：

      * **目录指纹带上 rank** -> 每个 rank 各造一份成员集，磁盘翻 world_size 倍；更糟的是
        分片生成时「等其它 rank 把剩下的造完」永远等不到（大家都在自己的目录里造），
        最后以 1 小时超时报错收场。
      * **各 rank 用不同基础 seed 造成员** -> 同一个成员的内容取决于「是谁造的」，而且
        相邻成员的随机流会重叠。

    m3 已删（2026-09-24），`seed=None -> 0` 是这里唯一还剩的归一。
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

    目录优先级 out_dir -> member_dir -> cache_dir -> XMETAI_SPW_CACHE_DIR -> 源模型同目录，
    文件名同样带 spec+源文件的指纹，所以换 σ、
    换组、权重更新都会自然落到新目录（目录名**不含成员数**，所以先生成 5 个再补到 51 个
    会落在同一个目录里并把清单合并）。

    指纹里的种子取**基础 seed**，不含进程 rank（见 _base_spec）——多卡共享同一份成员集。
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
    """成员图与它依赖的 base/perturbed 数据所在目录 —— 即**成员集目录本身**。

    不再逐成员开子目录（原因见模块 docstring）：onnxruntime 会校验外部数据的真实路径
    必须落在模型所在目录内，而 base 的字节又要被所有成员共享——只有硬链接同时满足这两
    条。保留 index 参数是为了不动调用方签名。
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
    """框架侧入口：返回第 index 个成员的模型路径；不在盘上就抛出可执行的构建命令。

    这里**不自动构建**：一次要造 count 个成员、
    写几十 GB，只能由调用方显式触发（cli 会在跑推理前按 rank 分片造好）。挂在「加载
    模型」这条必然会跑的路径上自动触发，等于把磁盘和时间的账单藏进加载里。
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
    """成员图里的外部数据引用分布：location -> 引用它的张量个数。

    口径是 graph.external_tensors（含被外置的 Constant 节点张量）。verify_member_set 拿它
    断言「所有引用都落在成员目录内」——只查可达性是不够的：漏改的 location 会退化成源
    .data 的名字，而源 .data 只要还在原地就照样可达，于是问题一路静默到建会话才炸。
    """
    counts: dict = {}
    for tensor in external_tensors(proto):
        location = _external_location(tensor)
        counts[location] = counts.get(location, 0) + 1
    return counts



def _source_layout(onnx, model) -> dict:
    """源图里出现过的外部数据位置 -> 成员目录里的 base 文件名。

    统一改名（base.data / base_N.data）而不是沿用原名：源图的 location 可能是绝对路径或
    指向别的目录，统一成固定名字后成员目录才是真正自包含的（图里的相对路径一定能解析）。

    口径是 external_tensors（含 Constant 节点的张量属性），不是 graph.initializer：
    被外置的常量同样按「相对图文件的路径」解析，漏掉它们成员目录就不自包含。
    """
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
    """确认源数据就是 C 序：memmap 读回来的字节必须与文件里的原始字节逐位相同。

    不做这一步的风险：Fortran 序的外部数据被按 C 序读出来、再按 C 序写回去，等于把权重
    转置。它不会报错、不会 NaN，只会让预报变差——正是这一轮最该避免的失败方式。
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
    """按结构给出**独立抽样**的形状；用它广播就得到 ξ。

    三种结构都保证每个元素 E[ξ]=0、Var(ξ)=1，所以 `σ^2 x ‖W‖^2` 的注入方差定律
    与组间 energy 配平在三种结构下都成立 —— 这是「同预算、换结构」可比的前提。
    ndim<2 的张量（bias / norm 权重 / dt_bias）在 channel 下退化为 iid：它们本来
    就是逐通道的向量，再「按通道共享」就等于全张量共享一个 ξ —— 那种「整张量共享」已
    作为 scalar 结构单独实现（见 spec.NOISE_STRUCTS），不必靠 channel 的退化去近似。
    """
    if struct == "scalar":
        # 必须在 `len(shape) < 2` 那道退化之前判：scalar 的意义就是**整张量一个 xi**，
        # 对 bias / dt_bias 这类一维张量也一样（channel 对向量退化成 iid，scalar 不退）。
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
    """W~ = W ⊙ (1 + σ ξ)。fp32 走 fp32 快路，其它 dtype 在 float64 上算再落回。

    struct 决定 ξ 在张量内部的相关结构（见 spec.NOISE_STRUCTS）。三种结构的每元素方差都是 1，
    所以它只改「能量注到哪些自由度」，不改「注入多少能量」。
    """
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
    """把源 .data 放进成员集目录：硬链接（首选，零字节）-> 整份复制。

    **刻意不用符号链接**：新版本 onnxruntime 会把外部数据解析成**真实路径**，再校验它
    是否落在模型所在目录内；符号链接指到目录外会被直接拒绝（2026-09-23 服务器实测）：

        External data path: "base.data" (resolved path: ".../models/<stem>.data")
        escapes both model directory: ".../member_001" and real model directory: ...

    硬链接没有这个问题：另一个名字的真实路径仍在目录内。

    硬链接会在两种真实场景下失败：跨文件系统（EXDEV），以及 fs.protected_hardlinks=1
    （多数发行版的默认值）时**源文件不属于当前进程用户**（EPERM——容器里挂载一份别人
    拥有的只读权重目录就是这么撞上的，2026-09-23 服务器正是这个）。失败就整份复制，
    因为成员集是扁平布局，这份复制只发生**一次**。

    返回 (方式, 失败原因)。方式 ∈ link / copy / exist。
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
    """把**未被扰动**的外部张量也改到成员目录里的 base_*.data。

    源图的 location 可能是绝对路径、也可能带子目录；不改的话成员目录就不是自包含的，
    图会去源目录找 .data（本地测试时表现为「引用不存在」，换台机器就是加载失败）。
    offset/length 不用动——base 文件是源 .data 的硬链接，字节布局逐位相同。

    范围是 external_tensors（含 Constant 节点的张量属性）：2026-09-23 服务器上就是因为
    只扫 initializer，漏掉了 4 个被外置的 attention 常量，成员图改完仍指向 <stem>.data，
    onnxruntime 建会话时报 cannot get file size。改这里时别再退回 initializer。
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
                # 小张量必须留在图内：onnxruntime 会话初始化时要读它们，外置会加载失败。
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
        # **落盘 schema，不要改名**：磁盘上那批成员集（S2B 的 51 个 / 76 GB）都写着这个
        # 键，backends/onnx.py 与 selftest 靠它认「这张图是成员图」。spec 里那个同名的
        # mechanism 字段已经删了，这里留着的是历史数据的标记。
        "spw.mechanism": "m1a",
        "spw.member": str(index),
        "spw.spec": spec.describe(),
        # 用 " | " 连接各条目：并集 group token 内部本来含逗号（见 graph.group_alternatives），
        # 再用逗号连接会读不出条目边界。
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
    """生成 count 个成员中的一部分（indices），返回报告 dict（含逐成员路径）。

    已存在的文件按同名同内容复用，所以重复调用是幂等的。

    indices 缺省 = 全部成员。多卡并行时每个 rank 造自己那一份
    （``range(rank, count, world_size)``）：每个成员的 ``perturbed.mNNN.data`` 是直接以
    "wb" 打开的，两个 rank 同时造同一个成员会写出交错的字节，所以分片必须互不相交。清单
    （spw_members.json）会与已有内容按成员号合并，分片生成也是自洽的。

    ``base.data`` 只建**一次**（与成员号无关）：分片生成时谁先到谁建，其余调用看到的是
    "exist"——不是竞态，内容逐位相同。
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

    # 成员图与 base/perturbed 必须同在 set_dir：onnxruntime 会校验外部数据的真实路径要落在
    # 模型所在目录内，所以「共享一份 base」只能靠硬链接；硬链接不可行时退到复制，而扁平
    # 布局让这份复制**只发生一次**（不是每成员一次）。
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
        # 放宽 checker 的「外部数据定位」那几条：base.data 是源 .data 的硬链接，而 onnx 的
        # checker 会以「多个硬链接 = 潜在硬链接攻击」为由拒绝这样的文件。我们不用复制
        # 2.8 GB 来换它闭嘴——引用可达性与字节范围由 _verify_external_refs 自校验顶上，
        # location 又都是本目录内的裸文件名（相对路径天然成立）。
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
    """独立复核成员集：文件齐全、引用可达、以及**成员之间的权重真的不同**。

    还会静态量一遍「权重级相对扰动」rms(W~-W)/rms(W)，应当≈该组配平后的 σ。这是能在
    本地、不跑模型就看出来的量级证据；探针（probe）量的是它在输出上的放大。
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
        # 可达还不够，位置还必须落在成员集目录内（裸文件名）。漏改一行 location，引用就
        # 退化成源 .data 的名字；而源 .data 只要还在原地，可达性检查什么都发现不了。
        closure = _location_closure(proto)
        stray = sorted(str(location) for location in closure
                       if not location or osp.dirname(location)
                       or not (location.startswith(_BASE_PREFIX)
                               or location.startswith(_PERTURBED_PREFIX)))
        if stray:
            problems.append(
                f"member_{index:03d}: 外部数据引用不在成员集目录内：{stray}"
                "（成员不是自包含的，onnxruntime 建会话时会去找源 .data）")
        # 新版 onnxruntime 会校验外部数据的**真实路径**必须落在模型所在目录内，于是符号
        # 链接（以及带 .. 的引用）会被直接拒绝——这里提前拦下，别等建会话才报。
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
    """列出候选注入组（原 CLI `groups` 的打印逻辑）。

    `depth` 控制分桶深度（默认 2：只到 `latent_mixer/layers` 这一级）；`under` 把统计
    限制到某个子树里，并把桶从子树的下一级开始列 —— 两者配合就是「注入点消融」的细粒度
    扫描：先 `under="latent_mixer.layers.0.blocks", depth=5` 看每个 block 的能量，再挑 key
    填进 `WeightPerturbSpec.groups`。key 里 `/` 与 `.` 等价（`key_parts` 两种都切）。
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
    # ------------------------------------------------------------------
    # 造产物入口：改下面的字面量，然后直接运行
    #
    #     python -m xmetai_inference.weights.spw
    #
    # 没有命令行参数。ACTION 选要做的事。
    # ------------------------------------------------------------------
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
