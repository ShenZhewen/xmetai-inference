# -*- coding: utf-8 -*-
"""FengQing V1.5Beta 确定性推理（tp 走 6h 累积补充 store）。"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

FENGQING_ROOT = "/workspace/szwCode/xmetai-inference2/model_artifacts/FengQing"
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
TP_ROOT = "/workspace/data/liujunjie/era5_tp_zarr"

FENGQING_STORES = [
    # 6h 累积 tp store 排最前，顶掉旧 sfc store 里 1h 的 tp。
    f"{TP_ROOT}/era5.202501-202607.c1.p25.h6.v1.lon180.zarr",
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]

cfg = InferConfig(
    name="fengqing",
    model_path=f"{FENGQING_ROOT}/onnx/fengqing_pre.onnx",
    model_class="fengqing_pre_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": FENGQING_STORES,
        "processors": [
            {"name": "unit_convert", "scales": {"tp": 1000.0}},  # 6h 累积 m → mm
            {"name": "geometry"},
            {"name": "normalize_fengqing"},
        ],
    },
    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    # 输入归一化在 dataset.processors，输出反归一化在这里。
    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[{"name": "denormalize_fengqing"}],
    ),

    # 起报数须 = gpus×batch_size；hour_interval 是模型契约（类属性同 6）不可改。
    times="2025010200..2025122800:24",
    steps=60,
    hour_interval=6,
    members=1,  # 确定性模型必须为 1
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/fengqing_output",
)