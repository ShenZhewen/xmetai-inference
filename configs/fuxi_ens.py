# -*- coding: utf-8 -*-
"""FuXi-Ens 51 成员集合推理（对应旧 scripts/run_fuxi_ens.sh）。"""
import os

from configs.base import InferConfig, ROOT

cfg = InferConfig(
    name="fuxi_ens",
    model_path=os.environ.get(
        "FUXI_ENS_ONNX", f"{ROOT}/weights/fuxiens/fuxi_ens_onnx/fuxi_ens.onnx"),
    spec="specs/fuxi_ens.json",
    loader="era5_store",
    times="2025071800..2025092800:24",
    steps=60,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fuxi_ens_output",
)
