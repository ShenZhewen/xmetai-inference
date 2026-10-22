"""Time-axis far endpoint: decoder entry, sigma=0.04, tau -> infinity.

latent.py 的 spec docstring（`LatentForcingSpec.tau_days`）自己写着这条轴
「把 m3 / m1a 当成两个端点夹在中间，是真正的新自由度」：
tau -> 0 退化回「每步重抽」（= m3 的时间结构），tau -> infinity 退化成
「整条轨迹冻结一个随机方向」（= m1a 的时间结构）。

t005（tau=0.05）是近白噪声那一端，**这一条是另一端的端点**，此前从没跑过。

机制：rho = exp(-0.25/1e6) -> 1，decay = sqrt(1-rho^2) -> 0，递推退化成状态恒等于
__init__/reset 里抽的那个平稳初值 ⇒ 每个成员拿到一个**不随时间变的持久偏移**。
所以这一臂回答的是：AR(1) 的**动力学**到底有没有用，还是「成员各自一个固定偏移」就够。
若它的 spread 效率 >= tau=3d，那 AR(1) 的过程部分是装饰，该改的说法是「成员持久偏移」。

取 1e6 而不是 math.inf：spec 要求 tau_days > 0 且要真能算出 rho。
"""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_tinf"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["tau_days"] = 1.0e6
