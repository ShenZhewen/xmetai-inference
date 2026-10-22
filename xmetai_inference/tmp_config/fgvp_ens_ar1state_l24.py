"""Scale-axis: decoder entry, sigma=0.04, tau=3d, length_scale=24.

尺度阶梯的高端补点，唯一作用是**给 l16 的峰定位**：16 与 32 之间再插一点，
这样「甜点在天气尺度」如果成立，形状（短时效占比）应当是
l16 < l24 < l32（或 l16 ~ l24 < l32），而不是一路单调到 32。

**这条是最可裁的一条**：若机时紧，砍掉它不影响 P-A/P-B/P-C 的主判据
（那三条靠 l1/l4/l8/l16/s04 五点已能定），只少了「峰在 16 还是 24」的分辨。

⚠ 同 l16：length_scale 越大越要防 build_basis 静默退回白噪声，
跑完核 `spw.ar1.fallbacks`；> 0 就说明本臂退化成 l1 了。
"""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_l24"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["length_scale"] = 24.0
