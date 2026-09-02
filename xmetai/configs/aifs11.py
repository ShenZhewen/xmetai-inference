# -*- coding: utf-8 -*-
"""AIFS 1.1 确定性推理配置。"""
import os

from xmetai.configs.base import InferConfig
from xmetai.models.aifs11_ckpt import FIELD_MAPPING, STATIC_MAPPING


cfg = InferConfig(
    name="aifs11",
    model_path=os.environ.get(
        "AIFS_CHECKPOINT",
        "/workspace/szwCode/xmetai-inference/AIFS_single/aifs-single-mse-1.1.ckpt",
    ),
    model_class="aifs11_ckpt",
    loader="era5_store",
    pre_processors=[
        {"name": "attach_static", "fields": list(STATIC_MAPPING)},
        {"name": "geometry", "expected_shape": [721, 1440]},
        {"name": "rename_channels", "mapping": FIELD_MAPPING},
        {"name": "validate_magnitude", "profile": "aifs11"},
        {"name": "channel_order"},
        {"name": "regrid", "target": "N320"},
    ],
    recurrent_processors=[],
    output_processors=[],
    times="2025010600",
    steps=1,
    members=1,
    vars="z_500,2t,msl,tp",
    gpus=1,    cuda_devices="0",
    output_dir="/workspace/szwCode/xmetai-inference/AIFS_out",
)
