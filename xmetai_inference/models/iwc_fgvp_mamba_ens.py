# -*- coding: utf-8 -*-
"""iwc_fgvp_mamba_ens ONNX 模型（FGVP 的 Mamba 集合骨干，FP32 导出）。

输入契约同 FuXi-Ens（78 通道、2 帧历史 + step/hour/doy），物理量进/出。集合语义由
图内随机算子自带，51 成员 = 51 次独立前向，输出到 member_<NNN>/ 子目录。

图只出预报那一帧，故 windowed_output=False；gpu_state 必须 False（GPU 常驻跳过
_next_state，且 buffer 复用假设输入输出同形），load() 会断言拦下。
"""
from xmetai_inference.backends.onnx import OnnxInferModel

from . import FUXI_ENS_CHANNELS, GRID_025


class IwcFgvpMambaEnsModel(OnnxInferModel):
    model_name = "iwc_fgvp_mamba_ens_onnx"
    # 输入契约与 FuXi-Ens 完全一致：78 通道布局 + 0.25° 网格
    input_channels = FUXI_ENS_CHANNELS
    output_channels = FUXI_ENS_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 6
    forecast_type = "ensemble"
    members = 51
    # 图只出 1 帧，滑窗归 Rollout；与 gpu_state=False 是绑定的，见模块 docstring。
    windowed_output = False
    gpu_state = False
