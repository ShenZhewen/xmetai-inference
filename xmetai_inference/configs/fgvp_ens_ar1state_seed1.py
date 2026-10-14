"""随机实现对照：与 s04 逐字相同，只把驱动侧种子 0 → 1（第 14 轮）。

---- 为什么要这一条 ----

本批（以及第 13 轮）读到的每一个效应都是「臂 X 相对 s00 / s04 的 Δ」，
但**从来没有测过"换一次随机抽签，同一个臂自己会差多少"**。这一条就是那个底。

seed 同时进两个地方（perturbation/latent.py）：

  * 图侧：``build_basis`` 的空间基 RNG 熵 = (seed, 104729, tokens, channels, iscale)
    —— 换 seed = 换一套空间扰动图案；
  * 驱动侧：``_rank_entropy`` = (seed, 7919, member_index, slot)
    —— 换 seed = 51 个成员的 ξ 序列整套换掉。

所以本臂量的**不是**纯 ε 抽签的位移，而是「整份扰动集换一次随机实现」的位移。
那正是论文要回答的问题：结论依赖我们随手挑的那次抽签吗？

⚠ seed 进 ``budgets()``（图指纹）⇒ 本臂会**新造一份 85 MB 的 .onnx**（框架按指纹
自动建子目录，不用手工干预）。这是它与 aA/aB 唯一的成本差别：那两条不进指纹、
与 s04 共用同一份图。

⚠ 写 1 而不是 0：``int(spec.seed or 0)`` 让 seed=0 与 seed=None **逐位等价**，
写 0 等于什么都没改。

判据在跑之前写死在 Job yaml 的 Q8 里：本臂与 s04 在 240/360h 的 spread 位移必须
**小于** aA/aB 相对 s04 的效应量；否则整批读到的长时效 Δ 都落在抽签噪声内。
"""
from copy import deepcopy

from xmetai_inference.configs.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.configs.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_seed1"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
# 只动这一个字段；σ=0.04 / τ=3d / length_scale=32 / targets / 51 成员 / 3 起报 / 8 卡
# 全部逐字同 s04（deepcopy 自它）。
cfg.latent_forcing["seed"] = 1
