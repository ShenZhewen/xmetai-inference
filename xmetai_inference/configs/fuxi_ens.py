# -*- coding: utf-8 -*-
"""FuXi-Ens 51 成员集合推理配置（era5_foundation_store2 h6）。

输入是物理量，ONNX 图内已烘焙归一化；Dataset 端只做单位换算和几何对齐，NaN 直接
透传不填充。
"""
import os

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig, ROOT

ERA5_ROOT = os.environ.get(
    "ERA5_STORE_ROOT",
    "/workspace/data/liujunjie/era5_foundation_store2",
)

FUXI_ENS_DYNAMIC_STORES = [
    os.path.join(ERA5_ROOT, "era5_pl_2025.01-2026.07.c84.p25.h6.zarr"),
    os.path.join(ERA5_ROOT, "era5_sfc_2025.01-2026.07.c15.p25.h6.zarr"),
    os.path.join(ERA5_ROOT, "era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr"),
]

cfg = InferConfig(
    name="fuxi_ens",
    model_path=os.environ.get(
        "FUXI_ENS_ONNX",
        "/workspace/szwCode/xmetai-inference/model_artifacts/fuxiens/fuxi_ens_onnx/fuxi_ens.onnx",
    ),
    # 统一开关 XMETAI_GPU_STATE="1"/"0" 覆盖，未设时用本 config 的默认（开）。
    gpu_state=os.environ.get("XMETAI_GPU_STATE", "1") == "1",
    model_class="fuxi_ens_onnx",
    dataset={
        "type": "processed_multi_zarr",
        "paths": FUXI_ENS_DYNAMIC_STORES,
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
        recurrent=[],
        output=[],
    ),
    times=(
        "20250102..20250107:24,"
    ),
    steps=60,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/fuxi_ens_output_new",
)
