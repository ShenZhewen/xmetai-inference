# -*- coding: utf-8 -*-
"""iwc_fgvp_gdn2（FuXi-Ens 的 Gated DeltaNet 骨干变体）确定性推理。

对应旧 scripts/run_dzs_single.sh。⚠ 必须在镜像
registry.bingosoft.net/pytorch/pytorch:2.11.0-cuda12.8-mamba3-20260403 里跑：
xmetai_onnx_plugins.so 按该镜像里的 onnxruntime 编（ABI 绑定），跨环境跑要先确认
onnxruntime 版本匹配，否则 register_custom_ops_library 会报错。
"""
import os

from configs.base import InferConfig

cfg = InferConfig(
    name="dzs_single",
    model_path=os.environ.get(
        "DZS_ONNX", "/workspace/tmp/douzsh/models/iwc_fgvp_gdn2_260901.onnx"),
    ops_library=os.environ.get(
        "DZS_OPS_LIBRARY", "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.so"),
    spec="specs/iwc_fgvp_gdn2.json",
    loader="era5_store",
    times="2025010600..2025020500:24",
    steps=60,
    members=1,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/dzs_single_output",
)
