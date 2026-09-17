# -*- coding: utf-8 -*-
"""iwc_fgvp_mamba_ens ONNX 模型（FGVP 的 Mamba 骨干集合预报版本，FP32 导出）。

与 iwc_fgvp_gdn2 的关系：同族、同输入契约、同输出形态，区别只在骨干与集合语义。

  * 输入契约与 FuXi-Ens 完全一致，复用 fuxi_ens 的 78 通道布局：
    通道 z/t/u/v/q × 13 层 + 13 地面 = 78（顺序同 fuxi_ens）；归一化、累积通道
    清零、掩码和反归一化均已烘焙进图，config 侧只做单位换算与几何对齐。
  * 2 帧历史 + step/hour/doy 三个标量；0.25°×721×1440。
  * **集合语义**：扰动由图内随机算子自带，51 成员 = 51 次独立前向，框架不加扰动
    ——与 fuxi_ens_onnx 同一套机制，见 runner.Rollout 的 Trajectory 说明。
    落盘走 configs/fgvp_ens.py 声明的 members=51，输出到 member_<NNN>/ 子目录。
  * **单步输出**：图只出预报那一帧（实测输出帧维为 1），不是 fuxi_ens 那种图内
    滑窗的窗口型。故声明 windowed_output=False，由 runner.Rollout._next_state 补
    滑窗成 [输入末帧, 预报]。这一点与 iwc_fgvp_gdn2 相同。

gpu_state 必须为 False，两个原因叠在一起：一是 GPU 常驻分支（runner.run 的
forward_gpu 路径）直接 `state = result` 回填，完全不经过 _next_state，windowed_output
在这里不起作用，单帧输出会被当成完整状态喂给需要 2 帧的图；二是该分支的 _spare
buffer 复用假设「输入输出同形」，本模型输入 2 帧、输出 1 帧，不成立。两者都不会
报错，只会算错，所以 OnnxInferModel.load 的启动期断言（_check_gpu_state_frames）
在加载期直接拦下。要开快路径需先实现 forward_gpu 并在其中滑窗。

自定义算子与 iwc_fgvp_gdn2 共用 xmetai_onnx_plugins.so，路径由 config 的
ops_library 提供。跨环境运行时须确认 onnxruntime 版本匹配，否则
register_custom_ops_library 会因 ABI 不匹配报错。

本模型导出为 FP32 而非 FP16，差异只在数值精度，不影响上述任何契约。
"""
from xmetai_inference.backends.onnx import OnnxInferModel

from . import FUXI_ENS_CHANNELS, GRID_025


class IwcFgvpMambaEnsModel(OnnxInferModel):
    model_name = "iwc_fgvp_mamba_ens_onnx"
    # 输入契约与 FuXi-Ens 完全一致，复用 fuxi_ens 的 78 通道布局 + 0.25° 网格
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
