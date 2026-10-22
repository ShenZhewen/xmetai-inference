"""Far-end amplitude sentinel: decoder entry, sigma=0.50, tau=3d.

**不是精度档，是探测器。** `fgvp_ens_pert.py` 头部（第 154-155 行）自己写着：σ=0.04
与裸跑差 <1%、落在判读分辨力以下，所以要先用一条**宽跨度**的 σ 阶梯区分
「幅度太小」与「锚点没生效」。而 s00/s02/s04/s08 只有 4 倍跨度（0.04 是 A32 已验证
的配方值，当中心是对的）——4 倍是做**精度**的跨度，不是做**检测**的跨度。

本档离中心 12.5 倍，唯一职责是回答一个问题：「这个锚点到底动不动」。

判读（只看这一条）：

  * 本档与 s00 也平 ⇒ 通路在这个锚点上无效（或 ξ 没落地）⇒ s02/s04/s08 的平
    **读不出**「幅度太小」，σ 阶梯不必细读，先去查 feeds。
  * 本档明显偏离 s00 ⇒ 通路是活的 ⇒ 中间三档若平，就是幅度太小，往上加 σ 有意义。

⚠ 本档幅度大，**不进结论表**、不与 s02/s04/s08 并列做剂量-反应拟合；它只当哨兵。

2026-10-13 由 `fgvp_ens_ar1state_s50` 更名而来：第一个 Job 的 ARMS 里那条同名档
（未更名的旧名）要留在它自己的编排里不跑，本档给并行的第二个 Job 用，名字分开
就不会跟旧名混。除名字外与旧档逐字相同。
"""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_s50a"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
cfg.latent_forcing["sigma"] = 0.50
