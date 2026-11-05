# -*- coding: utf-8 -*-
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig


CRA_STORE = "/workspace/data/zarr_refresh_v3/cra.19790101_20260714.c71.p25.h6.v1"
SRC_GRAPH = "/workspace/tmp/douzsh/models/20261019/tongyun_ens_cra_fp16.onnx"
OPS_LIBRARY = "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

cfg = InferConfig(
    name="weather_cra_0p25_tongyun_ens_spa",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,
    model_class="tongyun_cra_ens_onnx",
    latent_forcing={
        "targets": ("add_36086", "add_36099", "add_36112"),
        "sigma": 0.04,
        "tau_days": 3.0,
        "hour_interval": 6.0,
        "rank": 32,
        "length_scale": 32.0,
        "seed": 0,
        "amp_profile": None,
        "verify_on_load": True,
        "cache_dir": MODEL_ONNX_DIR,
    },
    perturb_graphs=0,
    model_batch=6,
    gpu_mem=0.9,

    dataset={
        "type": "processed_multi_zarr",
        "paths": [CRA_STORE],
        "processors": [
            {"name": "denormalize", "stats_path": CRA_STORE},
        ],
    },
    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    times="2026070200",
    steps=49,
    hour_interval=6,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    intra_op_threads=0,
    cudnn_algo_search="EXHAUSTIVE",
    resident_members=2,
    output_dir="/workspace/data/shenzw/tongyun_cra_spa_pl32",
)
