# -*- coding: utf-8 -*-
"""iwc_fgvp_mamba_ens ONNX 模型（FGVP 的 Mamba 骨干集合预报版本，FP32 导出）。

输入契约与 FuXi-Ens / iwc_fgvp_gdn2 一致，复用 fuxi_ens 的 78 通道布局
（z/t/u/v/q × 13 层 + 13 地面）：归一化、累积通道清零、掩码和反归一化均已烘焙进图，
config 侧只做单位换算与几何对齐；2 帧历史 + step/hour/doy 三个标量；0.25°×721×1440。

集合语义：扰动由图内随机算子自带，51 成员 = 51 次独立前向，框架不加扰动；落盘走
configs/fgvp_ens.py 声明的 members=51，输出到 member_<NNN>/ 子目录。

单步输出：图只出预报那一帧（实测输出帧维为 1），故 windowed_output=False，滑窗由
runner.Rollout._next_state 补成 [输入末帧, 预报]。

gpu_state 必须为 False：GPU 常驻分支（runner.run 的 forward_gpu 路径）直接
`state = result` 回填、不经 _next_state（windowed_output 失效，单帧输出会被当成完整
状态喂给需要 2 帧的图），且其 _spare buffer 复用假设「输入输出同形」也不成立。两者都
不报错、只算错，OnnxInferModel.load 的启动期断言 _check_gpu_state_frames 会拦下。

自定义算子与 iwc_fgvp_gdn2 共用 xmetai_onnx_plugins.so（路径由 config 的 ops_library
提供）；onnxruntime 版本不匹配时 register_custom_ops_library 会因 ABI 报错。

本模型导出为 FP32 而非 FP16，差异只在数值精度，不影响上述契约。
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
