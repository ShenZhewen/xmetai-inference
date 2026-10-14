# -*- coding: utf-8 -*-
"""iwc_fgvp_gdn2 ONNX 模型（FuXi-Ens 的 Gated DeltaNet 骨干变体）。

输入契约与 FuXi-Ens 完全一致，可直接复用 fuxi_ens 的 78 通道布局
（z/t/u/v/q × 13 层 + 13 地面）：物理量直接送入 ONNX，归一化、累积通道清零、掩码和
反归一化均已烘焙进图；2 帧历史 + step/hour/doy 三个标量；0.25°×721×1440。

骨干用自定义算子 xmetai_plugins:GatedDeltaNet2Fn，建 session 前必须注册
xmetai_onnx_plugins.so（路径由 config 的 ops_library 提供）；onnxruntime 版本不匹配
时 register_custom_ops_library 会因 ABI 报错。

本模型图**只输出预报那一帧**（fuxi_ens 是图内滑窗、输出完整 [t, t+1] 窗口），故声明
windowed_output=False，滑窗归 runner.Rollout；gpu_state 必须为 False（GPU 常驻分支
的 _spare buffer 复用假设「输入输出同形」，本模型 2 帧进 1 帧出不成立），否则静默算错，
OnnxInferModel.load 的 _check_gpu_state_frames 会在加载期拦下。
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
