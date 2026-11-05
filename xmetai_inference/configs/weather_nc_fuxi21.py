# -*- coding: utf-8 -*-
"""FuXi-2.1 single-file sample: infer straight from the upstream ``input.nc`` + ``fuxi-2.1.pt2``.

``SAMPLE_ROOT`` follows the upstream release layout: ``input.nc`` (already z-score normalized),
``mean.nc``/``std.nc`` and ``fuxi-2.1.pt2`` (3.7 GiB).

The dataset passes the normalized nc through unchanged. Diagnostic channels (ssr/ssrd/fdir/ttr/tp)
are zeroed before being fed back each step, and the output is denormalized with the mean.nc/std.nc
next to the .pt2 (tp: expm1 + non-negative). Init time is the nc's last ``time`` frame; ``times``
stays empty.
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

SAMPLE_ROOT = "/path/to/models/fuxi2.1"
FUXI21_NC = f"{SAMPLE_ROOT}/input.nc"
FUXI21_PT2 = f"{SAMPLE_ROOT}/fuxi-2.1.pt2"
OUTPUT_DIR = "/path/to/output/fuxi21_sample"


cfg = InferConfig(
    name="weather_nc_fuxi21",
    model_path=FUXI21_PT2,
    model_class="fuxi21_pt2",

    dataset={"type": "netcdf_data", "paths": [FUXI21_NC]},
    dataloader={
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
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

    # netcdf_data derives the init time from the nc itself; times is intentionally empty.
    times="",
    steps=2,
    hour_interval=6,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,d2m,msl,tp",
    gpus=1,
    cuda_devices="0",
    output_dir=OUTPUT_DIR,
)
