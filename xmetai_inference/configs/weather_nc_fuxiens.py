# -*- coding: utf-8 -*-
"""FuXi-ENS single-file sample: infer straight from the upstream ``input.nc``.

``SAMPLE_ROOT`` follows the upstream release layout: ``input.nc`` plus
``fuxi_ens_onnx/fuxi_ens.onnx`` and its external weight file ``fuxi_ens`` (≈9.3 GiB), which must
sit in the same directory.

input.nc is physical (GRID_025, north→south) in FUXI_ENS_CHANNELS order. The graph normalizes
internally and returns physical output, so no model processor or mean/std file is needed. Init
time is the nc's last ``time`` frame; ``times`` stays empty. The graph samples noise internally,
so members differ; this recipe runs 2 members × 2 steps as a smoke test (full run: 51 × 60).
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

SAMPLE_ROOT = "/path/to/models/fuxiens"
FUXI_ENS_NC = f"{SAMPLE_ROOT}/input.nc"
FUXI_ENS_ONNX = f"{SAMPLE_ROOT}/fuxi_ens_onnx/fuxi_ens.onnx"
OUTPUT_DIR = "/path/to/output/fuxi_ens_sample"


cfg = InferConfig(
    name="weather_nc_fuxiens",
    model_path=FUXI_ENS_ONNX,
    model_class="fuxi_ens_onnx",

    dataset={"type": "netcdf_data", "paths": [FUXI_ENS_NC]},
    dataloader={
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # netcdf_data derives the init time from the nc itself; times is intentionally empty.
    times="",
    steps=2,
    hour_interval=6,
    members=2,
    vars="z500,u200,v200,msl,tp",
    gpus=1,
    cuda_devices="0",
    output_dir=OUTPUT_DIR,
)
