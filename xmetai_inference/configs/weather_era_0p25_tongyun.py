# -*- coding: utf-8 -*-
import os

from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig
from xmetai_inference.utils.config_util import FUXI_ENS_CHANNELS, GRID_025


# Placeholder paths; replace before deployment.
ERA5_ROOT = os.environ.get(
    "ERA5_STORE_ROOT",
    "/path/to/data/era5_foundation_store2",
)
TONGYUN_ONNX = os.environ.get(
    "TONGYUN_ONNX", "/path/to/models/iwc_tongyun_gdn2_260901.onnx")
OPS_LIBRARY = os.environ.get(
    "TONGYUN_OPS_LIBRARY", "/path/to/models/xmetai_onnx_plugins.ort1_24.cuda12.so")
OUTPUT_DIR = "/path/to/output/tongyun_output"

TONGYUN_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]


IwcTongyunGdn2Model = OnnxInferModel.declare(
    "IwcTongyunGdn2Model",
    model_name="iwc_tongyun_gdn2_onnx",
    channels=FUXI_ENS_CHANNELS,
    grid=GRID_025,
    windowed_output=False,
)


cfg = InferConfig(
    name="weather_era_0p25_tongyun",
    model_path=TONGYUN_ONNX,
    ops_library=OPS_LIBRARY,
    gpu_state=os.environ.get("XMETAI_GPU_STATE", "0") == "1",
    model_class="iwc_tongyun_gdn2_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": TONGYUN_DYNAMIC_STORES,
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

    times="2025010200..2025010500:24",
    steps=60,
    hour_interval=6,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir=OUTPUT_DIR,
)
