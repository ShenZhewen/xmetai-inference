# -*- coding: utf-8 -*-
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

FUXI21_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_soil_2025.01-2026.07.c4.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_wave_2025.01-2026.07.c5.p25.h6.zarr",
]

cfg = InferConfig(
    name="weather_era_0p25_fuxi_ens_spa",
    model_path="/workspace/szwCode/xmetai-inference2/model_artifacts/fuxi2.1/fuxi-2.1.pt2",
    model_class="fuxi21_pt2",
    latent_forcing={
        "targets": ("layer_stream:0,3,6",),
        "sigma": 0.04,
        "tau_days": 3.0,
        "rank": 16,
        "length_scale": 32.0,
        "grid": (120, 240),
        "seed": 0,
        "hour_interval": 6.0,
        "verify_on_load": True,
    },
    model_batch=1,

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

    times="2025070100..2025070300:24",
    steps=60,
    hour_interval=6,
    members=51,
    vars="z500",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fuxi_spa",
)
