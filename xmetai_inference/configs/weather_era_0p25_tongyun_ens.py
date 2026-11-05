# -*- coding: utf-8 -*-
from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig
from xmetai_inference.utils.config_util import FUXI_ENS_CHANNELS, GRID_025


# Placeholder paths; replace before deployment.
ERA5_ROOT = "/path/to/data/era5_foundation_store2"
TONGYUN_ENS_ONNX = "/path/to/models/20261013/tongyun_ens_era5_fp16.onnx"
OPS_LIBRARY = "/path/to/models/xmetai_onnx_plugins.ort1_24.cuda12.so"
OUTPUT_DIR = "/path/to/output/tongyun_era_ens"

TONGYUN_ENS_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]


TongyunEraEnsModel = OnnxInferModel.declare(
    "TongyunEraEnsModel",
    model_name="tongyun_era_ens_onnx",
    channels=FUXI_ENS_CHANNELS,
    grid=GRID_025,
    forecast_type="ensemble",
    members=51,
    windowed_output=False,
)


cfg = InferConfig(
    name="weather_era_0p25_tongyun_ens",
    model_path=TONGYUN_ENS_ONNX,
    ops_library=OPS_LIBRARY,
    gpu_state=False,
    model_class="tongyun_era_ens_onnx",
    model_batch=6,
    gpu_mem=0.9,

    dataset={
        "type": "processed_multi_zarr",
        "paths": TONGYUN_ENS_DYNAMIC_STORES,
        "processors": [
            {
                "name": "unit_convert",
                "scales": {
                    "q": 1000.0,
                    "ssr": 6.0 / 3600.0,
                    "ssrd": 6.0 / 3600.0,
                    "fdir": 6.0 / 3600.0,
                    "ttr": 6.0 / 3600.0,
                    "tp": 6000.0,
                },
            },
            {"name": "geometry"},
            {
                "name": "fill_missing",
                "rules": {"sst": {"method": "constant", "value": 287.066}},
                "unconfigured": "error",
            },
        ],
    },
    dataloader={
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    times="2025010200..2025121500:24",
    steps=60,
    hour_interval=6,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir=OUTPUT_DIR,
)
