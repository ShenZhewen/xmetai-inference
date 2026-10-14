"""Scale-axis: decoder entry, sigma=0.04, tau=3d, length_scale=4.

尺度阶梯 1 / [4] / 8 / [16] / [24] / 32 的低端补点。方括号三条是本批新增。

动机与判据见 l16（本文件的兄弟）。这里只记本点承担的份额：
l1(2°) 已经把「极小尺度 = 灌进耗散掉的尺度」这一端做透了（额外离散度 6h
最大、168h 之后塌光）。l4 = 8°、波数 ~22，仍偏小，作用是**把 1 到 8 之间
补成两点**，这样「短时效占比随 length_scale 单调下降」才是三点以上的趋势
而不是一条线段。

⚠ 与 l1 同：往小调方向上没有静默 fallback 风险（build_basis 只在
length_scale 相对网格**过大**时退回白噪声），但本臂仍应在跑完后核
改写图的 `spw.ar1.fallbacks` 计数，确认没退化。
"""
from copy import deepcopy

from xmetai_inference.configs.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.configs.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_l4"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["length_scale"] = 4.0
