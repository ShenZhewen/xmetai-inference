# -*- coding: utf-8 -*-
"""iwc_fgvp_gdn2（FuXi-Ens 的 Gated DeltaNet 骨干变体）确定性推理。

输入是物理量，ONNX 图内已烘焙归一化；sst 等缺测用物理值填充。
必须在匹配的 onnxruntime 镜像里跑，见模型文件说明。
"""
import os

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig, ROOT

ERA5_ROOT = os.environ.get(
    "ERA5_STORE_ROOT",
    "/workspace/data/liujunjie/era5_foundation_store2",
)

FGVP_DYNAMIC_STORES = [
    os.path.join(ERA5_ROOT, "era5_pl_2025.01-2026.07.c84.p25.h6.zarr"),
    os.path.join(ERA5_ROOT, "era5_sfc_2025.01-2026.07.c15.p25.h6.zarr"),
    os.path.join(ERA5_ROOT, "era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr"),
]

cfg = InferConfig(
    name="fgvp",
    model_path=os.environ.get(
        "FGVP_ONNX", "/workspace/tmp/douzsh/models/iwc_fgvp_gdn2_260901.onnx"),
    ops_library=os.environ.get(
        "FGVP_OPS_LIBRARY", "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.so"),
    # 统一开关：XMETAI_GPU_STATE="1"/"0" 显式覆盖，未设时用本 config 的默认。
    # 本模型默认必须关（图只出 1 帧，见 models/iwc_fgvp_gdn2.py 的说明）；
    # 即使显式设 1，OnnxInferModel.load 的启动期断言也会拦下。
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
                "rules": {
                    "sst": {
                        "method": "constant",
                        "value": 287.066,
                    },
                },
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
    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[],
    ),
    times="2025010200..2025010500:24",
    steps=60,
    hour_interval=6,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/fgvp_output_new",
)
