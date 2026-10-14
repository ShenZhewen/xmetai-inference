"""Scale-axis sweet spot (mechanism pick): decoder entry, sigma=0.04, tau=3d,
length_scale=16.

**这是本批三条新臂里唯一一条带「押注」性质的**，其余两条只是补阶梯。

为什么押 16 —— token 到波数的换算（latent 90 纬 x 180 经，2 度/token，
所以波数 n 对应 180/n 个 token）：

    length_scale=1  ->  2 度  ->  n ~ 180   极小尺度
    length_scale=8  -> 16 度  ->  n ~ 22
    length_scale=16 -> 32 度  ->  n ~ 11    <- 天气尺度，斜压误差增长最快处
    length_scale=32 -> 64 度  ->  n ~ 5.6   行星尺度

已经测到的两端正好各自偏在一边：
  l1（2 度）额外离散度 6h 最大、168h 之后塌到 0 —— 能量全灌进模型立刻
     耗散掉的尺度，所以长时效无痕（PAPER.md 的归一化形状表：6h 是 s04 的
     3.3 倍，288/360h 算成 0）；
  s04（64 度）额外离散度偏慢、长时效又撞上骨干自己追平（288/360h 两批
     独立地把 t005 排到 s00 之下），说明行星尺度也不是增长最快的方向。

⇒ 机制上「把扰动灌进模型真正会放大的方向」的甜点应该在天气尺度，
   即 length_scale 约 16，而 8 到 32 之间此前**一个点都没有**。

**可证伪判据**（跑完照这个对，不许事后解释）：
  P-A 6h 的额外离散度占 120h 值的比例（归一化形状）应随 length_scale
      **单调下降**：l1 > l4 > l8 > l16 > l24 > s04(32)。
  P-B l16 在 168-240h 应**仍有可观额外离散度**，不像 l1 那样塌光。
  P-C 同 spread 下 l16 的 rmse@det / acc 应优于 s08 与 l1
      （本批唯一一条「均场真的变好」的读数是 s08 在 240h 的 ACC +11.3%）。
  **推翻条件**：l16 的归一化形状落在 l8 与 s04 之间 1-8% 的噪声里、
      即三者读不出差别 ⇒ 尺度轴在 8 以上就平了，上面这套机制说法整段作废，
      《长度尺度那一节》从 PAPER.md 删掉不留残骸（原 P4 的处置方式）。

⚠ **必须核一件事**：build_basis 在 length_scale 相对网格过大时，低通会把
除 DC 外的模压成精确的 0，于是**静默退回白噪声**（只发 warning，见
fgvp_ar1.py / latent.py）。32 已实测没退化（否则 s04 会和 l1 逐位相同，
而两者 24h spread 差 30.77 vs 25.82），但 16/24/32 都在「偏大」那一侧，
跑完请核改写图的 `spw.ar1.fallbacks` 计数：
    若 l16/l24 的 fallbacks > 0 ⇒ 它其实就是 l1，本批尺度阶梯作废。
"""
from copy import deepcopy

from xmetai_inference.configs.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.configs.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_l16"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["length_scale"] = 16.0
