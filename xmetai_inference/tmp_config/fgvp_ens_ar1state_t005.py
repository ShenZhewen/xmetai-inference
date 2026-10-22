"""Near-white time control: decoder entry, sigma=0.04, tau=0.05d."""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_t005"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["tau_days"] = 0.05
