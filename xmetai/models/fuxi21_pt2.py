# -*- coding: utf-8 -*-
"""FuXi-2.1 PT2 模型契约。

模型类只保留通道、网格、时间窗口和后端身份；完整 Processor 流程由
configs/fuxi21.py 声明。
"""

from xmetai.backends.pt2 import Pt2InferModel

from . import FUXI21_CHANNELS, GRID_025


class Fuxi21Pt2Model(Pt2InferModel):
    model_name = "fuxi21_pt2"
    # 输入/输出通道契约（85 通道，C85）、网格、时间窗口、确定性语义
    input_channels = FUXI21_CHANNELS
    output_channels = FUXI21_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 6
    forecast_type = "deterministic"
    members = 1
