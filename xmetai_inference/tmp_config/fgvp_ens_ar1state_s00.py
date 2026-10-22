"""Zero-forcing control: same decoder-entry injection graph, sigma=0."""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_s00"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["sigma"] = 0.0
