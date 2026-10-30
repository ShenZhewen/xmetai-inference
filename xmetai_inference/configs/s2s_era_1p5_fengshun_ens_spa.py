# -*- coding: utf-8 -*-
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

S2S_ROOT = "/workspace/tmp/douzsh/models"
MODEL_PATH = f"{S2S_ROOT}/20261013/s2s_gssp_gdn2.onnx"
OPS_LIBRARY = f"{S2S_ROOT}/xmetai_onnx_plugins.ort1_24.cuda12.so"
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"
S2S_STORE = "/workspace/data/zarr_refresh_v3/s2s.19500101_20260417.c76.1p5.h24.v1"

BASE_SIGMA = 0.44

cfg = InferConfig(
    name="s2s_era_1p5_fengshun_ens_spa",
    model_path=MODEL_PATH,
    model_class="fengshun_era_1p5_onnx",
    ops_library=OPS_LIBRARY,
    cudnn_algo_search="EXHAUSTIVE",

    latent_forcing={
        "targets": ("add_18518", "add_18523", "add_18528"),
        "tokens": 7200,
        "channels": 512,
        "sigma": BASE_SIGMA,
        "tau_days": 3.0,
        "hour_interval": 24.0,
        "rank": 16,
        "length_scale": 32.0,
        "grid": (60, 120),
        "seed": 0,
        "amp_profile": ((0, 1.272727), (5, 1.272727), (6, 0.636364), (14, 0.636364), (15, 0.363636), (59, 0.363636)),
        "verify_on_load": True,
        "cache_dir": MODEL_ONNX_DIR,
    },

    dataset={
        "type": "processed_multi_zarr",
        "paths": [S2S_STORE],
        "processors": [
            {"name": "denormalize", "stats_path": S2S_STORE},
        ],
    },
    dataloader={
        "batch_size": 6,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    model_batch=6,

    times="2024090100",
    steps=60,
    hour_interval=24,
    members=51,
    vars="z500",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/s2s_spa_exp/forecast/adrop168",
)
