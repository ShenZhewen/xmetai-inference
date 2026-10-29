# -*- coding: utf-8 -*-
"""风顺 ERA 1.5°（原版，无 SPA）——51 成员集合，输入盘 s2s c76 1.5°。

源图 ``/workspace/tmp/douzsh/models/20261013/s2s_gssp_gdn2.onnx``（权重外挂 .data），
模型契约见 ``models/fengshun_era_1p5.py``：76 通道、1.5°、日步长、2 帧进 1 帧出、
物理量进/出、集合成员由图内 ``RandomNormalLike`` 自带（``perturb_std_scale``=1.0）。

本档是 SPA 实验（``fengshun_era_1p5_spa_s00`` / ``_s04`` / ``_s08`` / ``_s12`` /
``_tau10`` / ``_ls08`` / ``_r32`` / ``_pl32``）的**原版对照**：
同一张图**不改写**、同一输入盘、同一口径（1 起报 2024090100、51 成员 × 60 日步、
只落 z500），用来区分「图内 iid 先验噪声」本身和「AR(1) 时序强迫」的贡献。

数据侧：store 存归一化值、图内烘焙归一化 ⇒ 只配 ``denormalize``（stats_path=该 store）；
tp 在该 store 里就是物理 mm（mean=0/std=1），**不要**配 ``expm1_channels``；
store 与模型同为 lat 90→−90，**不要**配 ``geometry``。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

FS_ROOT = "/workspace/tmp/douzsh/models"
MODEL_PATH = f"{FS_ROOT}/20261013/s2s_gssp_gdn2.onnx"
OPS_LIBRARY = f"{FS_ROOT}/xmetai_onnx_plugins.ort1_24.cuda12.so"
S2S_STORE = "/workspace/data/zarr_refresh_v3/s2s.19500101_20260417.c76.1p5.h24.v1"

cfg = InferConfig(
    name="fengshun_era_1p5",
    model_path=MODEL_PATH,
    model_class="fengshun_era_1p5_onnx",
    ops_library=OPS_LIBRARY,
    cudnn_algo_search="EXHAUSTIVE",  # 同 Tongyun 族：HEURISTIC 下 stem 卷积建不起来

    dataset={
        "type": "processed_multi_zarr",
        "paths": [S2S_STORE],
        "processors": [
            {"name": "denormalize", "stats_path": S2S_STORE},
        ],
    },
    dataloader={
        "batch_size": 1,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 图的 batch 轴是符号维（input/pred 的 batch 维是 dim_param），6 条一前向。
    model_batch=6,

    times="2024090100",
    steps=60,
    hour_interval=24,
    members=51,
    vars="z500",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fengshun_era_1p5",
)
