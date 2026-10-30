# -*- coding: utf-8 -*-
from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    CRA_CHANNELS, GRID_025_S, InferConfig, ModelProcessingConfig)


TongyunCraEnsModel = OnnxInferModel.declare(
    "TongyunCraEnsModel",
    model_name="tongyun_cra_ens_onnx",
    channels=CRA_CHANNELS,
    grid=GRID_025_S,
    forecast_type="ensemble",
    members=51,
    windowed_output=False,
)


CRA_STORE = "/workspace/data/zarr_refresh_v3/cra.19790101_20260714.c71.p25.h6.v1"
SRC_GRAPH = "/workspace/tmp/douzsh/models/20261019/tongyun_ens_cra_fp16.onnx"
OPS_LIBRARY = "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"

cfg = InferConfig(
    name="weather_cra_0p25_tongyun_ens",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,
    model_class="tongyun_cra_ens_onnx",
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
    steps=60,
    hour_interval=6,
    members=51,
    vars="z500,t2m,msl,tp",
    gpus=1,
    cuda_devices="0",
    intra_op_threads=0,
    cudnn_algo_search="EXHAUSTIVE",
    resident_members=2,
    output_dir="/workspace/data/shenzw/tongyun_cra",
)
