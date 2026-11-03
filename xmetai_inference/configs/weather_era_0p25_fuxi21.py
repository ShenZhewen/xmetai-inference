# -*- coding: utf-8 -*-
import os

from xmetai_inference.backends.pt2 import Pt2InferModel
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig
from xmetai_inference.utils.config_util import (
    GRID_025, LEVELS, PL_VARS, _expand_channels)


# FuXi-2.1 surface channels: FuXi-Ens plus soil/wave-derived sfc vars (85 total = 65 upper + 20 sfc).
FUXI21_SFC_VARS = [
    "msl", "t2m", "d2m", "sst", "ws10m", "ws100m", "u10m", "v10m",
    "u100m", "v100m", "lcc", "mcc", "hcc", "tcc",
    "ssr", "ssrd", "fdir", "ttr", "tcw", "tp",
]
FUXI21_CHANNELS = _expand_channels(PL_VARS, LEVELS, FUXI21_SFC_VARS)


# Placeholder paths; replace before deployment.
ERA5_ROOT = os.environ.get(
    "ERA5_STORE_ROOT",
    "/path/to/data/era5_foundation_store2",
)
FUXI21_MODEL_PATH = os.environ.get(
    "FUXI21_MODEL_PATH",
    "/path/to/models/fuxi2.1/fuxi-2.1.pt2",
)
OUTPUT_DIR = "/path/to/output/fuxi_single_output_new"

FUXI21_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_soil_2025.01-2026.07.c4.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_wave_2025.01-2026.07.c5.p25.h6.zarr",
]


Fuxi21Pt2Model = Pt2InferModel.declare(
    "Fuxi21Pt2Model",
    model_name="fuxi21_pt2",
    channels=FUXI21_CHANNELS,
    grid=GRID_025,
)


cfg = InferConfig(
    name="weather_era_0p25_fuxi21",
    model_path=FUXI21_MODEL_PATH,
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
    output_dir=OUTPUT_DIR,
)
