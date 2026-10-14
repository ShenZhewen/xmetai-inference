"""Time-axis midpoint: decoder entry, sigma=0.04, tau=1d.

设计里的 A32T1（configs/fgvp_ar1.py:58）。只动 tau_days，其余与 s04 逐字相同。

用途是**定轴形**：s04（tau=3）/ 本臂（tau=1）/ t005（tau=0.05）/ tinf（tau->inf）
四点铺开时间轴，看 spread 效率是随 tau 单调爬、还是在 tau=3 附近就饱和。
若在 tau=3 已饱和，那 tinf 那一端不值得单独解释；若还在爬，轴就是开的。
"""
from copy import deepcopy

from xmetai_inference.configs.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.configs.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_t1"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["tau_days"] = 1.0
