# -*- coding: utf-8 -*-
"""iwc_fgvp_gdn2 ONNX 模型（FuXi-Ens 的 Gated DeltaNet 骨干变体）。

输入契约同 FuXi-Ens（78 通道、2 帧历史 + step/hour/doy），物理量进/出，归一化已
烘焙进图。骨干用自定义算子 GatedDeltaNet2Fn，需 config 的 ops_library 注册，
onnxruntime 版本不符会在 ABI 报错。

图只输出预报那一帧，故 windowed_output=False；gpu_state 必须 False（2 帧进 1 帧出，
GPU 常驻的 buffer 复用假设不成立），load() 会断言拦下。
"""
from xmetai_inference.backends.onnx import OnnxInferModel

from . import FUXI_ENS_CHANNELS, GRID_025


class IwcFgvpGdn2Model(OnnxInferModel):
    model_name = "iwc_fgvp_gdn2_onnx"
    # 输入契约与 FuXi-Ens 完全一致：78 通道布局 + 0.25° 网格
    input_channels = FUXI_ENS_CHANNELS
    output_channels = FUXI_ENS_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 6
    forecast_type = "deterministic"
    members = 1
    # 图只出 1 帧，滑窗归 Rollout；gpu_state=False 的原因见模块 docstring。
    windowed_output = False
    gpu_state = False
