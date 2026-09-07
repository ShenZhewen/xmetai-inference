# -*- coding: utf-8 -*-
"""iwc_fgvp_gdn2 ONNX 模型（FuXi-Ens 的 Gated DeltaNet 骨干变体）。

输入契约与 FuXi-Ens 完全一致，可直接复用 fuxi_ens 的 78 通道布局：
  * 物理量输入直接送入 ONNX；归一化、累积通道清零、掩码和反归一化均已烘焙进图；
  * 通道 z/t/u/v/q × 13 层 + 13 地面 = 78（顺序同 fuxi_ens）；
  * 2 帧历史 + step/hour/doy 三个标量；0.25°×721×1440。

与 fuxi_ens_onnx 的两个区别：

1. 骨干用了自定义算子 xmetai_plugins:GatedDeltaNet2Fn，建 session 前必须注册
   xmetai_onnx_plugins.so；.so 路径由 config 的 ops_library 提供，跨环境运行时请
   确认 onnxruntime 版本匹配，否则 register_custom_ops_library 会因 ABI 不匹配报错。
2. 本模型图**只输出预报那一帧**（fuxi_ens 是图内滑窗、输出完整 [t, t+1] 窗口），
   所以声明 windowed_output=False，由 runner.Rollout 负责滑成
   [输入末帧, 预报]。原先这里 override forward 手工 concat，那是在「旧契约只承认
   窗口型模型」下的迁就写法，已随契约修正一起去掉。
"""
from xmetai_inference.backends.onnx import OnnxInferModel

from . import FUXI_ENS_CHANNELS, GRID_025


class IwcFgvpGdn2Model(OnnxInferModel):
    model_name = "iwc_fgvp_gdn2_onnx"
    # 输入契约与 FuXi-Ens 完全一致，复用 fuxi_ens 的 78 通道布局 + 0.25° 网格
    input_channels = FUXI_ENS_CHANNELS
    output_channels = FUXI_ENS_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 6
    forecast_type = "deterministic"
    members = 1
    # 图只出 1 帧，滑窗归 Rollout。这也是 gpu_state 必须为 False 的原因：GPU 常驻
    # 分支的 _spare buffer 复用假设「输入输出同形」，本模型输入 2 帧、输出 1 帧，
    # 不成立。OnnxInferModel.load 有启动期断言拦这个组合
    # （_check_gpu_state_frames）。要开快路径需先实现 forward_gpu 并在其中滑窗。
    windowed_output = False
    gpu_state = False
