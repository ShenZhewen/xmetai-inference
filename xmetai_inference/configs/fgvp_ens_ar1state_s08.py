"""Higher-amplitude control: decoder entry, sigma=0.08, tau=3d."""
from copy import deepcopy

from xmetai_inference.configs.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.configs.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_s08"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["sigma"] = 0.08
