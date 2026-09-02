# -*- coding: utf-8 -*-
"""FuXi-Ens 评测：51 成员集合 vs era5_store 实况，算 RMSE/MAE/CRPS/Spread/BSS/AROC。

对应旧 scripts/run_eval.sh。默认评 INITS 里列的几个起报（跨季节 5 天）；
把 inits 设空串即回到「扫描 fcst 下所有起报」。
"""
from configs.base import EvalConfig

cfg = EvalConfig(
    name="fuxi_ens_eval",
    fcst="/workspace/data/szw_output_fuxiens",
    spec="specs/fuxi_ens.json",
    steps=61,
    vars="z500,u200,v200,msl,tp",
    times="20250117,20250222,20250708,20251016,20251121",
    output_dir="eval_results",
)
