"""Scale-axis midpoint: decoder entry, sigma=0.04, tau=3d, length_scale=8.

与 l1 同轴，只把相关长度放在两端之间（1 : 8 : 32）。用途同 t1 —— 定轴形，
看 spread 效率是随 length_scale 单调、还是到某个尺度就饱和。
"""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_l8"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["length_scale"] = 8.0
