# -*- coding: utf-8 -*-
"""权重扰动（SPW）规格：把确定性 ONNX 变成集合模型的那两个自由度。

方法出处：Adamov et al. 2026, "Stochastically Perturbed Weights: Ensembles from
Deterministic Machine-Learning Weather Models"（arXiv:2609.08412）。参数化为

    W̃ = W ⊙ (1 + σ ξ),   ξ ~ N(0, 1) 逐元素独立

逐张量、逐成员独立抽样。乘性（而非加性）的意义是每个张量的注入方差恰为
σ^2 * ΣW^2，于是 σ 是无量纲的相对强度，可跨层、跨组、跨配置配平——这正是论文用
σ_group = σ_full·√(N_total/N_group) 做组消融配平的依据。

本模块只负责「规格」：选哪组张量、多大 σ、怎么配平、缓存与指纹。真正的图改写
在 graph.py。

第三个自由度（噪声的时间结构）只有一种实现：**逐成员冻结落盘**（本仓库的 m1a）——
生成阶段按 W̃ = W ⊙ (1 + σξ) 算出每个成员的一套权重写到磁盘，成员内**整条轨迹冻结**。
这是论文的主配置，实现在 weights.py。

（曾经还有一条 M3「图内随机算子」——噪声每次 forward 重抽，等价论文的「沿轨迹刷新噪声」
变体。它 2026-09-23 被实测淘汰（离散度只有 m1a 的 1/3–1/8、CRPS 差 23%；判据见 README
2A 节），代码已于 2026-09-24 删除。）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ._specbase import SpecBase

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

    #: from_config 的类型报错里用的名字（实现在 _specbase.SpecBase）
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


__all__ = ["DEFAULT_EXCLUDE", "MEMBER_FORMAT", "NOISE_STRUCTS", "WeightPerturbSpec"]
