"""Position control: decoder exit / prediction-head entry, sigma=0.04, tau=3d."""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1head_s04"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["targets"] = ("add_40939", "add_46096", "add_51253")
