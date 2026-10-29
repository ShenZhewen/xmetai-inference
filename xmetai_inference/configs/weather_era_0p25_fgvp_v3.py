# -*- coding: utf-8 -*-
"""FGVP 在 zarr_refresh_v3 新 ERA5 盘上的 A/B 配方：与 weather_era_0p25_fgvp 只差
``dataset`` / ``output_dir``。

前处理链：denormalize(tp 走 expm1) → 辐射 ×6 → geometry → sst 缺测填 287.066。
回算 2024 及更早时把 ``paths`` 换成 ``era5.1979_2024.c78.p25.h6.v1``，同时改 ``times``。
"""
import os

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_V3_2025 = "/workspace/data/zarr_refresh_v3/era5.202501_202607.c116.p25.h6.v3"

cfg = InferConfig(
    name="weather_era_0p25_fgvp_v3",
    model_path=os.environ.get(
        "FGVP_ONNX", "/workspace/tmp/douzsh/models/iwc_fgvp_gdn2_260901.onnx"),
    ops_library=os.environ.get(
        "FGVP_OPS_LIBRARY", "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"),
    # 必须为 0：图只出 1 帧，设 1 会被 load 的启动期断言拦下。
    gpu_state=os.environ.get("XMETAI_GPU_STATE", "0") == "1",
    model_class="iwc_fgvp_gdn2_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": [ERA5_V3_2025],
        "processors": [
            # 归一化态 → 物理量；tp: log1p(6h mm) → mm/6h
            {
                "name": "denormalize",
                "stats_path": ERA5_V3_2025,
                "expm1_channels": ["tp"],
            },
            # W/m² → 6h Wh/m²
            {
                "name": "unit_convert",
                "scales": {"ssr": 6.0, "ssrd": 6.0, "fdir": 6.0, "ttr": 6.0},
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
    output_dir="/workspace/data/shenzw/fgvp_output_v3",
)
