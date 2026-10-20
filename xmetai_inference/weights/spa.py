# -*- coding: utf-8 -*-
"""潜空间 AR(1) 强迫：SPW 之外的第二条腿（噪声的**时间相关**）。

本文件即原 `perturbation/latent.py`，2026-10-20 起迁到 `weights/spa.py`：
SPA 的规格 / 基 / ONNX 图改写 / 生成 / 审计在图手术这一半都在这里；权重扰动（SPW）
在 `weights/spw.py`，PT2 版图手术在 `weights/spa_pt2.py`。造产物入口见文末「用法」。

推理时的驱动（`Ar1Driver` / `LatentStepStream` / `lag1_autocorr`）不在本文件：
它们在 `backends/spa.py`，由 ONNX 与 PT2 后端共用。

## 为什么需要它（一句话）

权重扰动 `W~ = W ⊙ (1 + σξ)` 的噪声在时间上是**无记忆**的 iid，而实测（README 2L.4 /
2M.1）表明：σ 只是一个**幅度旋钮** —— `spread(lead) ≈ f(σ)^p(lead)`，`p` 只由注入点在骨干上的
位置决定，跟 σ、秩、注入时刻**都无关**；`p(lead)` 从 +6 h 的 1.007 衰到 +342 h 的 0.466。
所以单靠乘性权重噪声，短时效和长时效永远补不齐（要补 0.594 的标定比需要 `k ≈ 3.4` 倍等效
σ，而五个臂最大只给到 2.8）。**缺的是增长率，不是幅度**。

物理集合（IFS-ENS）用的是 SPPT：扰动**参数化倾向项**。神经网络的倾向项对应物就是
「每一步的潜变量残差更新」。而倾向项扰动与权重扰动有一处本质差别：它在**时间上累积**
（SPPT 的噪声过程带 AR(1) 相关），权重扰动是无记忆的白噪声 —— 后者每一步都被 Δt 平滑掉
（m3 的失败诊断：每步重抽的白噪声只有冻结权重的 1/3–1/8 离散度）。

本模块实现的就是这一条：**用 AR(1) 过程驱动的、空间上低秩相关的潜变量强迫**。

## 公式

时间（驱动侧，每 6 h 一步）：

    ξ_{t+1} = ρ ξ_t + sqrt(1 - ρ²) ε_t,      ε_t ~ N(0, I),  ρ = exp(-Δt / τ)

`ξ_t ∈ R^R`（R = `rank`，默认 16），所以是**平稳、单位方差、相关时间 τ** 的随机过程；
τ = 3 天时 ρ = exp(-0.25/3) = 0.9200（每个 6 h 步的相关），自相关按 `ρ^k` 衰减。

空间 + 通道（图内，烘焙成常量）：

    n[b, c, tok] = Σ_r ξ_t[b, r] · u_r[c] · v_r[tok]

其中 `{u_r}` 是通道空间的一组**正交单位**随机基（R 维），`{v_r}` 是**空间**正交单位基 ——
`v_r` 由白噪声在 (rows, cols) 网格上做高斯低通（相关长度 `length_scale` 个 token）得到，
再去掉 DC 分量（**这一条很重要**：不去掉就是「整场一致偏移」，正是论文报告的主失效模式
「区域均值过离散」）。强制 `‖u_r‖ = ‖v_r‖ = 1`（QR 正交化）后：

    RMS_{c,tok}(n) = sqrt(Σ_r ξ_r² / R) ≈ 1     （对单位方差 ξ 精确成立）

于是注入写成**乘性**、和 SPW 的 σ 语义完全一致（σ 无量纲、跨配置可比）：

    z' = z ⊙ (1 + σ n)        # z 是潜变量（默认取 3 个 latent state）

`length_scale` 是**唯一**的尺度旋钮：大（如 32 token ≈ 64° 经度，波数 ~3）就是「只扰行星尺度」，
小（=1）就是「所有尺度一起扰」。这就是论文 Phase 3（SFNO 取最低十个球谐模）在没有谱域可切
的骨干上的对应做法。

## 为什么噪声走「驱动侧的额外图输入」而不是图内随机算子

图内随机算子（m3 的写法）**给不了时间相关** —— 每次前向都是新的独立抽样，没有跨步记忆。
要把 AR(1) 的递推放进图里，就得让图本身持有状态，而 ONNX 的单步图没有可靠的状态机制。
所以：**递推在驱动侧（Python）算，图只消费当前的 ξ**。

    * 图：多一个输入 `spw/xi`，形状 [batch, T, R]（T = 目标张量个数，R = rank）——
      每步只喂 3×16 = 48 个 float32，H2D 开销可忽略；
    * 驱动侧：`backends.spa.Ar1Driver` 持有 ξ，每步喂当前值再递推一步。

这带来两个结构性的好处：

1. **不用落盘成员权重**：同一份图被所有成员共享，成员之间的差别只来自 ξ 的种子（由
   `member_index` 决定），所以 51 个成员不需要 51 份 onnx —— 磁盘与加载时间都归零；
2. **和权重扰动正交、可直接叠加**：图改写是在「已经解析好的源图」上做的，所以
   `weight_perturb`（m1a 逐成员冻结 / m3 图内算子）与 `latent_forcing` 可以同时开。

代价（必须说清）：**这不再是「零改动复用现成检查点」** —— 图被改写了（多一个输入 + 十几个
节点），但**权重一个字节都没动**，也没有任何训练，所以仍然满足「事后扰动」的定位。

## 与论文的关系

论文把「沿轨迹刷新噪声」列为部分修复手段，并自承「刷新只对 SFNO 有效」，机制未解释。
我们的解释（README 2M.4-3）：SFNO 的噪声落在最粗的谱模上，而按 `p(lead, band)`，行星/天气
尺度是**唯一耐用**的通道，刷新就是反复往耐用方向注，所以有效；Aurora/AIFS 的噪声摊在含被
抑制尺度的方向上，刷新帮不上。本模块把这条解释做成**可控实验**：`length_scale` 直接决定
「注在哪一档尺度」，τ 决定「注得多持久」。

## 用法

造产物没有命令行：改本文件末尾 `if __name__ == "__main__"` 里的字面量，直接运行

    python -m xmetai_inference.weights.spa

`ACTION="build"` 生成强迫图并就地审计；`ACTION="audit"` 只审计已有产物。
`.pt2` 自动走 PT2 后端（`attach_forcing`，改内存图、不落盘）。

    # config 里（现行档是 configs/fgvp_ens_spa.py；原 fgvp_ar1.py 五臂已于 2026-10-20 删除）：
    latent_forcing={"targets": ("latent_state",), "sigma": 0.04, "tau_days": 3.0,
                    "rank": 16, "length_scale": 32.0, "seed": 0}
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


# ---------------------------------------------------------------------------
# 底座：spec 样板 + ONNX 读写/外部数据校验
#
# SPA 与 SPW（weights/spw.py）共用这套底座。为了让 SPA 的生成与图改写完全收在
# weights/ 里（不反向依赖 perturbation），SpecBase / ONNX 工具直接内联在这里，
# spw.py 从本文件 import SpecBase；口径与原实现逐字一致（尤其是 external_tensors
# 必须覆盖 Constant 节点的外置张量，否则图会静默指向源 .data，建会话时才报错）。
# ---------------------------------------------------------------------------
class SpecBase:
    """spec 类的公共样板。子类至少要提供 `budgets()`。"""

    #: `from_config` 的类型报错里用的名字。**不要加类型注解**。
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

# 驱动侧噪声通道的名字。带 `spw/` 前缀是沿用既有约定（图里的注入节点一律 spw/ 打头）。
FORCING_INPUT = "spw/xi"
NODE_PREFIX = "spw/ar1"

# 目标别名 -> 节点输出名。`latent_state` = FGVP 三个 VAE 组各自送进 decoder 的潜状态
# （512 通道、16200 个 token），即「低分辨率潜空间」那一层 —— 论文四个骨干的优胜注入点
# 都紧邻这里（README 2M.4/Q2）。`blocks` = latent_mixer 12 个 block 的残差更新出口
# （1536 通道），是「倾向项」更字面的对应物，留给后续消融。
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

    targets        注入哪些张量。内置别名 "latent_state"（默认，3 个潜状态）/ "blocks"
                   （12 个 block 残差出口）；也可以直接写节点输出名的后缀。
    sigma          相对强迫强度：`z ⊙ (1 + σn)`，`n` 的 RMS ≈ 1，所以 σ 就是「潜变量的
                   相对扰动幅度」，与 SPW 的 σ 同为无量纲量。σ=0 合法 = 空转对照
                   （gain 恒为 1，输出与源图逐位相同）。
    tau_days       AR(1) 的相关时间（天）。τ → 0 退化回「每步重抽」（= m3 的时间结构）；
                   τ → ∞ 退化成「整条轨迹冻结一个随机方向」（= m1a 的时间结构）。
                   所以这条轴把 m3 / m1a 当成两个端点夹在中间，是真正的新自由度。
    rank          低秩基的秩 R。每步只喂 T×R 个数；R 越大，空间图案越丰富、代价越高。
    length_scale   空间基的高斯低通相关长度（token 数）。大 = 只扰行星尺度。
    tokens         潜变量的 token 数；None = 自动探测（从 patch_embed 卷积推）。
    channels       潜变量的通道数；None = 自动探测（从目标张量的最后一维）。
    grid           空间基用的 (rows, cols)；None = 自动探测（从 patch_embed 推）。
    seed           驱动侧随机流的种子。实际用的是 (seed, member_index, slot)。
    hour_interval  一步多少小时（算 ρ 用）。0 = 由调用方从 config 填。
    amp_profile    时变幅度 a(t)：((step, 倍数), ...) 的分段线性折线，step = 0 起的模式步
                   （一步 6h）。None = 恒 1，与原行为**逐位相同**。两端外推取端点值。
                   **只在驱动侧乘出口 ξ，不进 budgets()** —— 与 tau_days 同理（见 budgets
                   的说明），所以换曲线复用同一份图，不触发 85 MB 改写。
                   为什么需要这个自由度：σ 是个常数，而"让集合校准所需的有效 σ"随时效
                   单调上涨（07-02 实测 6h→360h 要涨约 3.5 倍，中段还在爬），一个标量
                   给不出曲线；同时 σ 的边际收益在长时效递减（响应指数 p 从 ≈1.0 掉到
                   0.75），所以尾端的补偿必须比线性更陡。
    demean_members 跨成员去均值的成员总数。0 = 关（**与改动前逐位相同**）；= N 时驱动侧
                   在本地重放 m = 0..N-1 全体成员的 ξ 流（种子逐个与真实成员相同），
                   出口 ξ 减去全体样本均值。
                   **为什么需要**：ξ 是每个成员**独立**抽的，所以 51 个成员 ξ 的样本均值
                   不为零（sd ≈ 1/√N）。集合平均场 = 确定性场 + 这个均值踢，而 RMSE 是
                   二次量 ⇒ 踢一次赔一次：实测同一批内 360h 集合平均 RMSE 随 σ 单调变差
                   （σ=0 → 784.9、0.04 → 795.7、0.08 → 801.4）。这就是「想要离散度就得
                   赔平均场精度」那笔交易。去均值把这条耦合拆掉：集合平均场 ≡ 确定性场
                   （到一阶），RMSE 与 σ 解耦，σ 才可以放心开到校准所需的量。
                   **只在出口扣，不写回 state**（同 a(t) 的理由）。不进 budgets() ——
                   纯驱动侧，复用同一份图。
    input_name     新增的图输入名（一般不用改）。
    cache_dir      改写后的 .onnx 落盘目录。**默认必须是源图所在目录**：外部数据
                   （.data）引用要被 onnxruntime 接受，必须落在模型目录内（坑 6）。
    verify_on_load 加载时确认图里真的有 spw/xi 输入且 metadata 对得上（抓「图没被改写」）。
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
        """时变幅度 a(t)：分段线性插值，两端外推取端点值。None → 恒 1（逐位同旧）。"""
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
        """参与**图**指纹的字段。tau/hour_interval/amp_profile 只在驱动侧起作用、不改图，
        所以不进指纹 —— 换 τ 或换 a(t) 曲线复用同一份图（多个臂共图是刻意的，省一次
        85 MB 改写）。"""
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


# 图格式版本：改「图怎么改写」（输入名、节点结构、基的归一化、外部数据落位方式）就
# +1 -> 指纹变 -> 新文件。v2（2026-09-29）：跨目录落盘改为「.data 就地放进图目录 +
# location 裸文件名」（见 _localize_external_data），v1 的「相对路径指回源目录」会被
# onnxruntime 的 realpath 校验拒绝。
# v3（2026-10-12）：适配**非 gdn2 的图**。v2 的改写链有两个只在 fp32/opset17 上成立的
# 假设，换个骨干就静默产出**建不了会话**的图，而指纹不变 -> 下次直接复用坏图（第二次
# 踩到同一个坑才把这条补上）。两处：① `Split` 在 opset>=18 必须显式给 `num_outputs`
# （13-17 可靠输出个数均分）；② 链尾的 `Mul` 要求增益与目标张量同类型，fp16 图
# （iwc_gssp_im_fp16）必须在乘之前 `Cast` 到目标类型。
FORCING_FORMAT = 3


# ---------------------------------------------------------------------------
# 目标解析与布局探测
# ---------------------------------------------------------------------------
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
    """从 patch_embed 卷积推潜空间网格：kernel == stride 的那个 Conv + 静态输入的空间维。

    公式用卷积自己的口径 `out = (in - kernel) // stride + 1`（无 padding）—— FGVP 的
    721x1440 经 8x8/8 变成 90x180 = 16200 个 token，正好是 712800/8² 的那个整数。
    """
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


# ---------------------------------------------------------------------------
# 基
# ---------------------------------------------------------------------------
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
    """构造空间基 V[N,R] 与通道基 U[C,R]，并把 σ、归一化常数折叠进 U。

    返回 {"Vt": [R,N], "Ut": [R,C], "report": {...}}；图里算的是

        n[b,c,tok] = Σ_r ξ[b,r] · Ut[r,c] · Vt[r,tok]

    归一化（推导见模块头）：`‖u_r‖ = ‖v_r‖ = 1` 时 `‖n‖_F² = s² Σξ²`，取
    `s = σ·sqrt(C·N/R)` 就得到 `RMS(n) = σ·sqrt(Σξ²/R) ≈ σ`。于是「相对强迫幅度」
    恰好等于 σ，与 SPW 的 σ 同量纲。
    """
    if rank > channels:
        raise ValueError(
            f"latent_forcing.rank={rank} 不能大于通道数 {channels}"
            "（通道基最多只有 channels 个正交方向）")
    rows, cols = int(grid[0]), int(grid[1])
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed or 0), 104729, tokens, channels, int(length_scale * 1000)]))
    # 空间基：低通随机场 -> 去 DC（防「整场一致偏移」）-> 正交化。
    # 低通的相关长度一旦接近或超过网格本身（比如 8x8 的潜网格配 length_scale=32），
    # 除 DC 以外的模会被压到**精确的 0**：这时「去 DC」剩下的只是数值残差甚至恒等于 0，
    # 丢给 QR 只会得到一个**带均匀分量**的基 —— 正是要防的那个失效模式。所以逐个方向
    # 检查「去 DC 后的能量 >= 低通之前能量的 1e-8」，不达标就退回白噪声（不平滑）。
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


# ---------------------------------------------------------------------------
# 图改写
# ---------------------------------------------------------------------------
def _initializer(onnx, name: str, array: np.ndarray):
    return name, onnx.numpy_helper.from_array(np.ascontiguousarray(array), name)


def inject_forcing(onnx, model, targets, basis, spec: LatentForcingSpec) -> int:
    """把 AR(1) 强迫接到目标张量上：`z' = z ⊙ (1 + σ·U diag(ξ) Vᵀ)`，返回新增节点数。

    每个目标的算子链（`Split` 三个目标共用）：

        spw/xi [B,T,R]
          └─ Split(axis=1) -> T 份 [B,1,R]
               └─ Reshape([0,R,1])            -> [B,R,1]
                    └─ Mul(Vt [1,R,N])        -> [B,R,N]    # 先逐 r 缩放
                         └─ Transpose(0,2,1)  -> [B,N,R]    # MatMul 收缩的是**最后一个**维
                              └─ MatMul(Ut [R,C]) -> [B,N,C]  # 对 r 求和
                                   └─ Add(1) -> Mul(目标张量) -> z ⊙ (1 + n)

    注意 MatMul 的口径：A[B,N,R] x B[R,C] 收缩的是 A 的最后一维（R）—— 所以顺序必须是
    「先 MulV 再转置再 MatMul」，写成 [B,R,N] x [R,C] 会被 onnxruntime 直接判为维度不兼容
    （它按 (..., M, K) x (K, N) 解释，收缩 N 而不是 R）。

    秩 16 时每个目标只多 2 个常量（Vt 16200x16、Ut 16x512，合计约 1 MB）与 6 个节点；
    每步喂进图的只有 T x R = 48 个数，H2D 开销可忽略。

    插入位置：**每个目标自己的第一个消费者之前**（不是统一插在一处）——这样任何拓扑序
    都成立，不依赖「三个目标的产出节点彼此相邻」这个骨干特性。
    """
    graph = model.graph
    rank = int(spec.rank)
    # 目标张量的元素类型：图可能是 **fp16**（`iwc_gssp_im_fp16.onnx`），而本链的常量与
    # spw/xi 输入都是 fp32 —— 直接相乘会被 ORT 判为「T 绑定了两种类型」而拒绝建会话：
    #   Type parameter (T) of Optype (Mul) bound to different types
    #   (tensor(float16) and tensor(float)) in node (spw/ar1/t0/MulZ)
    # gdn2 是 fp32，所以在它上面一直没暴露。修法：链**全程 fp32**（数值上也更稳），
    # 只在最后往目标类型 Cast 一次；spw/xi 保持 fp32，驱动侧不用动。
    # 必须在改图**之前**推断（infer_shapes 看的是原图的类型传播）。
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
    # `Split` 的属性要求随 opset 变：13-17 可以只给输出个数，靠「均分」推 split 尺寸；
    # **18 起必须显式给 `split` 输入或 `num_outputs` 属性**，否则 ORT 建会话时直接拒收：
    #   Node (spw/ar1/Split) Op (Split) [ShapeInferenceError]
    #   Neither 'split' input nor 'num_outputs' attribute has been given
    # 而图不一定落在哪个 opset（gdn2 = 17，iwc_gssp_im_fp16 = 18+），所以按图**自己声明的**
    # 版本决定加不加 —— 无条件加会在 17 及以下变成未识别属性。
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
        # fp32 图（gdn2）不加这个节点；fp16 图（iwc_gssp_im_fp16）把增益降下来再乘。
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
    """强迫图的落盘路径：默认与源图**同目录**。

    为什么不能像 m3 那样随便换目录：图里的外部数据（.data）引用必须落在模型目录内，
    否则 onnxruntime 建会话时会以 "External data path validation failed" 拒绝（坑 6）。
    m1a 的成员图本来就在成员集目录里，所以叠加时强迫图也落在那里 —— 代价是每成员一份
    85 MB 的图（51 成员 ≈ 4.3 GB），换来的是「权重一个字节都不用重算」。
    """
    directory = (spec.cache_dir or os.environ.get("XMETAI_AR1_CACHE_DIR")
                 or osp.dirname(osp.abspath(src_path)))
    stem = osp.splitext(osp.basename(src_path))[0]
    return osp.join(osp.abspath(directory), f"{stem}.ar1-{spec.fingerprint(src_path)}.onnx")


def _localize_external_data(onnx, model, src_dir: str, out_dir: str) -> dict:
    """out_dir != src_dir 时：把外部 .data 就地放进 out_dir，location 改成裸文件名。

    为什么不是「相对路径指回源目录」（v1 的老写法，相关代码已删除）：新版
    onnxruntime 建会话会把外部数据解析成**真实路径**，再校验它必须落在图所在目录内，
    `../../..` 直接被拒（2026-09-29 A32 实踩：

        External data path: "../../../tmp/.../<stem>.data" escapes both model
        directory: ".../model_onnx" and real model directory: ...）

    符号链接同样过不了 realpath 校验（`_share_data` 的头注释，2026-09-23 实测）。
    能同时满足「换个名字、realpath 仍在图目录内」的只有**硬链接**；硬链接不可用（跨卷
    EXDEV / fs.protected_hardlinks 且源文件不属于当前用户）就整份复制**一次**——每个
    .data 在 out_dir 里只放一份，全部臂共用。

    走 external_tensors 的完整口径（initializer + Constant 节点的外置张量）：漏掉后者
    不会报错，只会让成员图静默指向源目录然后建会话失败（`external_tensors` 的注释有实测记录）。

    返回 {basename: 方式}，方式 ∈ link / copy / exist（已在位且就是同一份文件）。
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
    """要不要放宽 checker 的「外部数据定位」校验。

    两种必须放宽的布局：`.data` 不在图所在目录（我们刻意不复制 2.8 GB 权重），以及
    `.data` 是用硬链接共享源文件的（checker 会以 "multiple hard links" 拒绝，而
    硬链接正是 m1a 成员省磁盘的做法）。
    """
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
        # 跨目录：.data 必须真实落在图目录内（ORT 的 realpath 校验），link/copy 一次后
        # 所有臂共用（见 _localize_external_data 的头注释）。
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
        # 归一化自检：‖Vt‖_F = sqrt(R)（列单位正交）；‖Ut‖_F/sqrt(C*N) 就是**基本身隐含的
        # sigma**（对单位方差 ξ，n 的 RMS = 这个值），必须等于设定值 —— 这是「σ 语义没被
        # 基的缩放带偏」的唯一硬检查。
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
    """框架侧唯一入口：先按 SPW 机制解析出「该加载哪份权重」，再在其上叠加潜空间强迫。

    两步都有缓存（指纹文件名），所以重复调用只是拼一个路径。叠加是**正交**的：
    强迫图改的是图结构、SPW 改的是权重，两者互不干涉；也因此可以
    「m1a 冻结权重 + AR(1) 潜空间强迫」同时开。
    """
    # 只在「SPW 冻结权重 + SPA」叠加时用到；放在函数内，避免 spa -> spw 的顶层
    # 循环（spw.py 顶层 import 了本文件的 SpecBase）。
    from .spw import resolve_effective_path

    path = resolve_effective_path(src_path, weight_spec, member_index, verbose=verbose)
    return resolve_forcing_path(path, latent_spec, verbose=verbose)


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # 造产物入口：改下面的字面量，然后直接运行
    #
    #     python -m xmetai_inference.weights.spa
    #
    # 没有命令行参数。ACTION 选 build / audit。
    # ------------------------------------------------------------------
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
