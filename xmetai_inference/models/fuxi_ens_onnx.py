# -*- coding: utf-8 -*-
"""FuXi-Ens ONNX 模型。

归一化已烘焙进 ONNX 图（物理量进/出，不配置额外归一化 Processor）；扰动由图内随机
算子（RandomNormalLike/RandomUniformLike）自带，51 成员 = 51 次独立前向。
"""
from xmetai_inference.backends.onnx import OnnxInferModel

from . import FUXI_ENS_CHANNELS, GRID_025


class FuxiEnsOnnxModel(OnnxInferModel):
    model_name = "fuxi_ens_onnx"
    # 输入/输出 78 通道（z/t/u/v/q × 13 层 + 13 地面）
    input_channels = FUXI_ENS_CHANNELS
    output_channels = FUXI_ENS_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 6
    forecast_type = "ensemble"
    members = 51
    # 前后处理/回填都是恒等，GPU 常驻安全；回退 numpy 版在 config 里设 gpu_state=False。
    gpu_state = True
