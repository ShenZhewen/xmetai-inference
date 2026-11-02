# -*- coding: utf-8 -*-
from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    FENGSHUN_CHANNELS, GRID_025, InferConfig, ModelProcessingConfig)


# Placeholder paths; replace before deployment.
FENGSHUN_ROOT = "/path/to/models"
FENGSHUN_STORE = "/path/to/data/zarr_refresh_v3/cra.1979_2024.c87.p25.h24.v1"
FENGSHUN_ONNX = f"{FENGSHUN_ROOT}/fengshun2.0_260901.onnx"
OPS_LIBRARY = f"{FENGSHUN_ROOT}/xmetai_onnx_plugins.ort1_24.cuda12.so"
OUTPUT_DIR = "/path/to/output/fengshun_output"


Fengshun20OnnxModel = OnnxInferModel.declare(
    "Fengshun20OnnxModel",
    model_name="fengshun20_onnx",
    channels=FENGSHUN_CHANNELS,
    grid=GRID_025,
    hour_interval=24,
    forecast_type="ensemble",
    members=11,
    windowed_output=False,
    time_features={
        "hour": {"divisor": 1.0},
        "doy": {"divisor": 1.0},
    },
)


cfg = InferConfig(
    name="s2s_cra_0p25_fengshun20_ens",
    model_path=FENGSHUN_ONNX,
    model_class="fengshun20_onnx",
    ops_library=OPS_LIBRARY,

    dataset={
        "type": "processed_multi_zarr",
        "paths": [FENGSHUN_STORE],
        "processors": [
            {
                "name": "denormalize",
                "stats_path": FENGSHUN_STORE,
                "expm1_channels": ["tp"],
            },
            {
                "name": "fill_missing",
                "rules": {"sst": {"method": "constant", "value": 287.222}},
                "unconfigured": "error",
            },
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

    times="2024090100,2024100100,2024110100",
    steps=60,
    hour_interval=24,
    members=51,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir=OUTPUT_DIR,
)
