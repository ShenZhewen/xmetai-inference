# -*- coding: utf-8 -*-
"""FuXi-2.1 PT2 模型契约；完整 Processor 流程由 configs/fuxi21.py 声明。"""

from xmetai_inference.backends.pt2 import Pt2InferModel

from . import FUXI21_CHANNELS, GRID_025


class Fuxi21Pt2Model(Pt2InferModel):
    model_name = "fuxi21_pt2"
    # 输入/输出 85 通道（C5 变量 × 13 层 + 20 地面）
    input_channels = FUXI21_CHANNELS
    output_channels = FUXI21_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 6
    forecast_type = "deterministic"
    members = 1
