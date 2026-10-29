# -*- coding: utf-8 -*-
"""FuXi-2.1 确定性推理（era5_foundation_store2 h6）。85 通道、2 帧进 1 帧出。

前处理（unit_convert / normalize / fill_missing）在 dataset.processors；诊断通道
zero_channels 与输出 denormalize 在 model_processing。
"""
import os

from xmetai_inference.backends.pt2 import Pt2InferModel
from xmetai_inference.configs.base import (
    FUXI21_CHANNELS, GRID_025, InferConfig, ModelProcessingConfig)


Fuxi21Pt2Model = Pt2InferModel.declare(
    "Fuxi21Pt2Model",
    model_name="fuxi21_pt2",
    # 输入/输出 85 通道（C5 变量 × 13 层 + 20 地面）
    channels=FUXI21_CHANNELS,
    grid=GRID_025,
)


ERA5_ROOT = os.environ.get(
    "ERA5_STORE_ROOT",
    "/workspace/data/liujunjie/era5_foundation_store2",
)

FUXI21_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_soil_2025.01-2026.07.c4.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_wave_2025.01-2026.07.c5.p25.h6.zarr",
]

cfg = InferConfig(
    name="weather_era_0p25_fuxi21",
    # 绝对路径：权重在旧仓库 xmetai-inference（无 2）；FUXI21_MODEL_PATH 可覆盖。
    model_path=os.environ.get(
        "FUXI21_MODEL_PATH",
        "/workspace/szwCode/xmetai-inference/model_artifacts/fuxi2.1/fuxi-2.1.pt2",
    ),
    model_class="fuxi21_pt2",

    dataset={
        "type": "processed_multi_zarr",
        "paths": FUXI21_DYNAMIC_STORES,
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
                "name": "normalize",
                "mean_file": "mean.nc",
                "std_file": "std.nc",
                "log1p_channels": ["tp"],
            },
            {
                "name": "fill_missing",
                "rules": {"*": {"method": "constant", "value": 0.0}},
                "unconfigured": "keep",
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

    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[
            {
                "name": "zero_channels",
                "channels": ["ssr", "ssrd", "fdir", "ttr", "tp"],
            },
        ],
        output=[
            {
                "name": "denormalize",
                "mean_file": "mean.nc",
                "std_file": "std.nc",
                "expm1_channels": ["tp"],
                "nonnegative": ["tp"],
            },
        ],
    ),

    times="2025010200..2025010500:24",
    steps=60,
    hour_interval=6,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,d2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/fuxi_single_output_new",
)
