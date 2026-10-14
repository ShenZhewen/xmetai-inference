"""Scale-axis far endpoint: decoder entry, sigma=0.04, tau=3d, length_scale=1.

`length_scale` = 空间基的高斯低通相关长度（token 数）。本族此前**所有**臂都被
`fgvp_ens_pert.AR1_SPEC` 钉死在 32（fgvp_ar1.py:55：32 token ~ 64 度经度、波数 ~3，
即「只扰行星尺度」），一条都没动过。这一条是设计里的 A1（fgvp_ar1.py:57：
不平滑，所有尺度一起扰）。只动 length_scale，其余与 s04 逐字相同。

动机（为什么这条轴现在最值钱）：sigma 那条路在长时效已经死了 ——
0.04 -> 0.08 只把 360h 的 SER 从 0.715 推到 0.745，短时效又已经过离散（SER > 1）。
按第 12 轮的塌缩结论，定时刻 CRPS 只是 spread 的函数，所以要再往上走**只能换轴**，
而空间尺度是唯一还没试过的、可能买到长时效 spread 的旋钮。

⚠ 方向性说明：`build_basis`（latent.py:429-441）在 length_scale 相对网格**过大**时，
低通会把除 DC 外的模压成精确的 0，于是**静默退回白噪声**（只发一条 warning）。
往小调（本臂 1.0）方向上安全；反过来要确认 32 没退化，判据是改写后图的
`spw.ar1.fallbacks`（latent.py:584-588 写进元数据）——若 s04 那条其实已经 fallback，
本臂就和它没区别了。
"""
from copy import deepcopy

from xmetai_inference.configs.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.configs.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_l1"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["length_scale"] = 1.0
