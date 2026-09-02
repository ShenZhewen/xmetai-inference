# -*- coding: utf-8 -*-
"""FuXi-Ens 51 成员集合推理配置。"""
import os

from xmetai.configs.base import InferConfig, ROOT

cfg = InferConfig(
    name="fuxi_ens",
    model_path=os.environ.get(
        "FUXI_ENS_ONNX",
        f"{ROOT}/model_artifacts/fuxiens/fuxi_ens_onnx/fuxi_ens.onnx",
    ),
    model_class="fuxi_ens_onnx",
    loader="era5_store",
    pre_processors=[
        {"name": "geometry"},
        {"name": "channel_order"},
        {"name": "unit_convert"},
    ],
    recurrent_processors=[],
    output_processors=[],
    times=(
        "20250102..20250107:24,"
        "20250316,"
        "20250428..20250519:24,"
        "20250630..20250702:24,"
        "20250922..20250924:24,"
        "20251206..20251229:24"
    ),
    steps=60,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fuxi_ens_output",
)
