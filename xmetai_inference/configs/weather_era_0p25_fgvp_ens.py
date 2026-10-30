# -*- coding: utf-8 -*-
from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    FUXI_ENS_CHANNELS, GRID_025, InferConfig, ModelProcessingConfig)


IwcFgvpMambaEnsModel = OnnxInferModel.declare(
    "IwcFgvpMambaEnsModel",
    model_name="iwc_fgvp_mamba_ens_onnx",
    channels=FUXI_ENS_CHANNELS,
    grid=GRID_025,
    forecast_type="ensemble",
    members=51,
    windowed_output=False,
)


ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

FGVP_ENS_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

cfg = InferConfig(
    name="weather_era_0p25_fgvp_ens",
    model_path="/workspace/tmp/douzsh/models/20261008/iwc_gssp_im.onnx",
    ops_library="/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so",
    gpu_state=False,
    model_class="iwc_fgvp_mamba_ens_onnx",
    model_batch=6,
    gpu_mem=0.9,

    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_ENS_DYNAMIC_STORES,
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
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fgvp_ens_output",
)
