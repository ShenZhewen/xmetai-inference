# -*- coding: utf-8 -*-
"""iwc_fgvp_gdn2（FuXi-Ens 的 Gated DeltaNet 骨干）确定性推理。

契约同 FuXi-Ens（78 通道、2 帧历史、物理量进/出、归一化烘焙在图内）；自定义算子
GatedDeltaNet2Fn 需 ops_library 注册，ORT 版本不符会在 ABI 报错。图只出预报帧，
gpu_state 必须 False（2 帧进 1 帧出），load() 会断言拦下。
"""
import os

from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    FUXI_ENS_CHANNELS, GRID_025, InferConfig, ModelProcessingConfig)


IwcFgvpGdn2Model = OnnxInferModel.declare(
    "IwcFgvpGdn2Model",
    model_name="iwc_fgvp_gdn2_onnx",
    channels=FUXI_ENS_CHANNELS,      # 契约同 FuXi-Ens：78 通道 + GRID_025
    grid=GRID_025,
    windowed_output=False,           # 图只出预报帧，滑窗归 Rollout
)


ERA5_ROOT = os.environ.get(
    "ERA5_STORE_ROOT",
    "/workspace/data/liujunjie/era5_foundation_store2",
)

FGVP_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

cfg = InferConfig(
    name="weather_era_0p25_fgvp",
    model_path=os.environ.get(
        "FGVP_ONNX", "/workspace/tmp/douzsh/models/iwc_fgvp_gdn2_260901.onnx"),
    ops_library=os.environ.get(
        "FGVP_OPS_LIBRARY", "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"),
    # 必须为 0：图只出 1 帧（如显式设 1）会被 load 的启动期断言拦下。
    gpu_state=os.environ.get("XMETAI_GPU_STATE", "0") == "1",
    model_class="iwc_fgvp_gdn2_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_DYNAMIC_STORES,
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
    output_dir="/workspace/data/shenzw/fgvp_output_new",
)
