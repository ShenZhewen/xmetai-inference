# -*- coding: utf-8 -*-
"""iwc_fgvp_gdn2 评测：预测 vs era5_store 实况，算 RMSE/MAE。

确定性模型（members=1，forecast_type=deterministic），集合指标
（CRPS/Spread/BSS/AROC）为 NaN，只留 RMSE/MAE 有意义。
对应旧 scripts/run_eval_dzs.sh。
"""
from configs.base import EvalConfig

cfg = EvalConfig(
    name="dzs_eval",
    fcst="/workspace/data/shenzw/dzs_single_output",
    spec="specs/iwc_fgvp_gdn2.json",
    steps=60,
    vars="z500,u200,v200,msl,tp",
    times="",                      # 空 = 扫描 fcst 下所有起报；可填 "20250106,20250112"
    output_dir="eval_results_dzs",
)
