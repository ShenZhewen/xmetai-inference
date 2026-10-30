# -*- coding: utf-8 -*-
"""fengshun_era_1p5 + SPA——decoder 入口注入、a(t) 第 4 批：168h 一档突变

检验 168–360h 的"最优 σ=0.24"是不是真信号：144h 前保持 0.56，168h 直接砍到 0.28，
384h 再砍到 0.16。若比 atrack 差，说明渐变更稳、那个 0.24 是噪声；若更好，
说明降档要果断。

与 ``fengshun_era_1p5_spa_s44.py`` **同一张改写图**（base σ=0.44），只多一行 amp_profile。
折点按 **0 基 code step** 写（step 0=+24h，step s=+(s+1)×24h），线性插值：

    step     0     5     6    14    15    59
    lead   24h  144h  168h  360h  384h 1440h
    σeff  0.56  0.56  0.28  0.28  0.16  0.16

amp_profile 不进 budgets() ⇒ 与 s44 共用一张图，不重建。

实验口径：1 个起报 2024090100、51 成员 × 60 日步、只落 z500。"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

S2S_ROOT = "/workspace/tmp/douzsh/models"
MODEL_PATH = f"{S2S_ROOT}/20261013/s2s_gssp_gdn2.onnx"
OPS_LIBRARY = f"{S2S_ROOT}/xmetai_onnx_plugins.ort1_24.cuda12.so"
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"
S2S_STORE = "/workspace/data/zarr_refresh_v3/s2s.19500101_20260417.c76.1p5.h24.v1"

# base σ = 图内烘焙值；amp_profile 的因子 = σeff / BASE_SIGMA
BASE_SIGMA = 0.44

ARM = "adrop168"

cfg = InferConfig(
    name="fengshun_era_1p5_spa_adrop168",
    model_path=MODEL_PATH,
    model_class="fengshun_era_1p5_onnx",
    ops_library=OPS_LIBRARY,
    cudnn_algo_search="EXHAUSTIVE",  # 同 Tongyun 族：HEURISTIC 下 stem 卷积建不起来

    latent_forcing={
        # decoder 入口：latent_mixer 三个 group_projs 出口 + 残差，各 [B,7200,512]。
        "targets": ("add_18518", "add_18523", "add_18528"),
        "tokens": 7200,
        "channels": 512,
        "sigma": BASE_SIGMA,
        "tau_days": 3.0,
        "hour_interval": 24.0,
        "rank": 16,
        "length_scale": 32.0,
        # 图的 token 数是 7200，detect_layout 会猜成 40×180；钉死成 60×120。
        "grid": (60, 120),
        "seed": 0,
        # 有效 σ 轨迹（折点 = 0 基 step，线性插值；因子 = σeff / BASE_SIGMA）
        "amp_profile": ((0, 1.272727), (5, 1.272727), (6, 0.636364), (14, 0.636364), (15, 0.363636), (59, 0.363636)),
        "verify_on_load": True,
        "cache_dir": MODEL_ONNX_DIR,
    },

    dataset={
        "type": "processed_multi_zarr",
        "paths": [S2S_STORE],
        # store 存归一化值、图内烘焙归一化，故只配 denormalize。
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
    gpus=2,
    cuda_devices="0,1",
    output_dir=f"/workspace/data/shenzw/s2s_spa_exp/forecast/{ARM}",
)
