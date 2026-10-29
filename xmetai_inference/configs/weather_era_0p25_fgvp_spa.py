# -*- coding: utf-8 -*-
"""确定性 FGVP（gdn2）+ SPA（潜空间 AR(1) 强迫）。

三处 latent_mixer 出口；扫描时改 latent_forcing 的 sigma / tau_days，--out 分目录。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

FGVP_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

cfg = InferConfig(
    name="weather_era_0p25_fgvp_spa",
    model_path="/workspace/tmp/douzsh/models/20261013/iwc_fgvp_gdn2_260901.onnx",
    ops_library="/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so",
    gpu_state=False,  # 单帧输出图
    model_class="iwc_fgvp_gdn2_onnx",
    latent_forcing={
        "targets": (  # gdn2 三组 latent_mixer 出口
            "/latent_mixer/Add_output_0",
            "/latent_mixer/Add_1_output_0",
            "/latent_mixer/Add_2_output_0",
        ),
        "sigma": 0.04,
        "tau_days": 3.0,
        "rank": 16,
        "length_scale": 32.0,
        "seed": 0,
        "hour_interval": 6.0,
        "verify_on_load": True,
        "cache_dir": "/workspace/data/shenzw/model_onnx",
    },

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
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 默认冬季扫描窗口，正式跑用 --times 覆盖。
    times="2025121600..2025123100:24",
    steps=60,
    hour_interval=6,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/fgvp_spa",
)
