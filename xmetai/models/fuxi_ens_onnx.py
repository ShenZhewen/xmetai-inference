# -*- coding: utf-8 -*-
"""FuXi-Ens ONNX 模型。

归一化已烘焙进 ONNX 图，输入/输出都在物理量空间，不配置额外归一化 Processor。
扰动由图内随机算子（RandomNormalLike/RandomUniformLike）自带，51 成员 = 51 次独立
前向，框架无需加扰动。
"""
import os

from xmetai.backends.onnx import OnnxInferModel

from . import FUXI_ENS_CHANNELS, GRID_025


class FuxiEnsOnnxModel(OnnxInferModel):
    model_name = "fuxi_ens_onnx"
    # 输入/输出通道契约（78 通道）、网格、时间窗口、集合语义
    input_channels = FUXI_ENS_CHANNELS
    output_channels = FUXI_ENS_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 6
    forecast_type = "ensemble"
    members = 51
    # 前后处理/回填都是恒等，GPU 常驻安全；XMETAI_DISABLE_GPU_STATE=1 可回退 numpy 版对比
    gpu_state = os.environ.get("XMETAI_DISABLE_GPU_STATE", "0") != "1"
