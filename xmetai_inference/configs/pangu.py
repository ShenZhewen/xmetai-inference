# -*- coding: utf-8 -*-
"""Pangu-Weather 确定性推理（6h 步进 + 24h 跳步，实现在模型类内）。

故这里 steps 是 6h 步数。数据源为 pl + sfc，无 tp 通道；模型契约见
models/pangu_onnx.py。
License：BY-NC-SA 4.0，商用禁止。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

PANGU_ROOT = "/workspace/szwCode/xmetai-inference2/model_artifacts/Pangu-Weather-main"
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

PANGU_STORES = [
    ERA5_ROOT + "/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ERA5_ROOT + "/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]

cfg = InferConfig(
    name="pangu",
    # 6h 步进图；24h 图由模型类从同目录推导（pangu_weather_24.onnx），
    # 可用环境变量 PANGU_ONNX_24 覆盖。
    model_path=PANGU_ROOT + "/pangu_weather_6.onnx",
    model_class="pangu_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": PANGU_STORES,
        "processors": [
            {"name": "geometry"},
        ],
    },

    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[],
    ),

    times="2025010200..2025122500:24",
    steps=60,
    hour_interval=6,
    # 确定性模型必须为 1（>1 会落到基类成员路径，喂错键名）。
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl",
    gpus=1,
    cuda_devices="0",
    output_dir="/workspace/data/shenzw/pangu_output",
)