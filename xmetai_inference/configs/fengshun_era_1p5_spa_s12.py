# -*- coding: utf-8 -*-
"""fengshun_era_1p5 + SPA——decoder 入口 ``add_18518/523/528`` 注入、σ=0.12 τ=3d（幅度轴上端）。

与 ``fengshun_era_1p5_spa_s04.py`` **逐字段相同，只把 sigma 0.04→0.12**。
旧图（Tongyun 族）上 σ 在 0.09 附近就饱和；本臂用来验这张图是不是同一量级——
过冲时集合 spread 会明显大于误差本身（spread/rmse > 1）。

锚点：decoder 入口 ``add_18518`` / ``add_18523`` / ``add_18528``，各 [B, 7200, 512]
（latent_mixer 三个 ``group_projs`` 出口 + 残差；与 Tongyun add_35722/27/32 同构）。
SPA 在这里乘 ``(1+σn)``：图内噪声是**逐帧 iid** 的，SPA 叠的是**时间相关**的那部分。

实验口径：1 个起报 2024090100、51 成员 × 60 日步、只落 z500。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

S2S_ROOT = "/workspace/tmp/douzsh/models"
MODEL_PATH = f"{S2S_ROOT}/20261013/s2s_gssp_gdn2.onnx"
OPS_LIBRARY = f"{S2S_ROOT}/xmetai_onnx_plugins.ort1_24.cuda12.so"
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"
S2S_STORE = "/workspace/data/zarr_refresh_v3/s2s.19500101_20260417.c76.1p5.h24.v1"

ARM = "s12"

cfg = InferConfig(
    name="fengshun_era_1p5_spa_s12",
    model_path=MODEL_PATH,
    model_class="fengshun_era_1p5_onnx",
    ops_library=OPS_LIBRARY,
    cudnn_algo_search="EXHAUSTIVE",  # 同 Tongyun 族：HEURISTIC 下 stem 卷积建不起来

    latent_forcing={
        # decoder 入口：latent_mixer 三个 group_projs 出口 + 残差，各 [B,7200,512]。
        # 与 Tongyun fgvp_ens_spa 的 add_35722/27/32 同构；SPA 报告里这是效应最大的一档。
        "targets": ("add_18518", "add_18523", "add_18528"),
        "tokens": 7200,
        "channels": 512,
        "sigma": 0.12,
        "tau_days": 3.0,
        "hour_interval": 24.0,
        "rank": 16,
        "length_scale": 32.0,
        # 图的 token 数是 7200，detect_layout 会猜成 40×180；钉死成 60×120。
        "grid": (60, 120),
        "seed": 0,
        "verify_on_load": True,
        "cache_dir": MODEL_ONNX_DIR,
    },

    dataset={
        "type": "processed_multi_zarr",
        "paths": [S2S_STORE],
        # store 存归一化值、图内烘焙归一化，故只配 denormalize；
        # tp 在该 store 里就是物理 mm（mean=0/std=1），**不要**配 expm1_channels。
        # 不配 geometry：模型 grid 与 store 同为 lat 90→−90。
        "processors": [
            {"name": "denormalize", "stats_path": S2S_STORE},
        ],
    },
    dataloader={
        "batch_size": 6,   # 数据集项=起报；成员并批由 model_batch 管
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 各臂同批宽：逐字段可比。图的 batch 轴是符号维，6 条一前向。
    model_batch=6,

    times="2024090100",
    steps=60,
    hour_interval=24,
    members=51,
    vars="z500",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir=f"/workspace/data/shenzw/s2s_spa_exp/forecast/{ARM}",
)
