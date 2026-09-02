# -*- coding: utf-8 -*-
"""iwc_fgvp_gdn2 ONNX 模型（FuXi-Ens 的 Gated DeltaNet 骨干变体）。

输入契约与 FuXi-Ens 完全一致，可直接复用 fuxi_ens 的 78 通道布局：
  * 物理量输入（归一化烘焙进图，pre/post 恒等），不喂 z-score；
  * 通道 z/t/u/v/q × 13 层 + 13 地面 = 78（顺序同 fuxi_ens）；
  * 2 帧历史 + step/hour/doy 三个标量；0.25°×721×1440。

与 fuxi_ens_onnx 的唯一区别：骨干用了自定义算子 xmetai_plugins:GatedDeltaNet2Fn，
建 session 前必须注册 xmetai_onnx_plugins.so（见 ops_library）。该 .so 是按镜像
registry.bingosoft.net/pytorch/pytorch:2.11.0-cuda12.8-mamba3-20260403 里的
onnxruntime 编的，跨环境跑要先确认 onnxruntime 版本匹配，否则 register_custom_ops_library
会因 ABI 不匹配报错。
"""
import os

import numpy as np

from backends.onnx import OnnxInferModel


class IwcFgvpGdn2Model(OnnxInferModel):
    model_name = "iwc_fgvp_gdn2_onnx"
    # 自定义算子库（.so）路径，环境变量可覆盖。注册发生在 OnnxInferModel.load() 里，
    # 建 session 前调用 register_custom_ops_library。
    ops_library = os.environ.get(
        "XMETAI_OPS_LIBRARY",
        "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.so",
    )
    # 先走 numpy forward（gpu_state=False，不启用 IOBinding GPU 常驻），保证自定义
    # 算子路径正确；验证数值与 fuxi_ens 对齐后，可 XMETAI_ENABLE_GPU_STATE=1 再开快
    # 路径。fuxi_ens 默认已开，此模型保守起见默认关。
    gpu_state = os.environ.get("XMETAI_ENABLE_GPU_STATE", "0") == "1"

    def forward(self, x, step, valid_time):
        """本模型输入 2 帧、输出只有 1 帧（下一拍预报，不带回显帧）。

        框架的回填契约要求「输出帧数 = 输入帧数」（state=result 直接回填），所以这里
        把 1 帧输出拼成 2 帧滑动窗口 [输入末帧, 预报]，与 fuxi_ens 的 [t, t+1] 同构：
        末帧仍是预报（框架取 state[:, -1]），首帧是输入末帧回显（下一拍的第一历史帧）。
        """
        pred = super().forward(x, step, valid_time)              # (1, 1, C, H, W)
        return np.concatenate([x[:, -1:], pred], axis=1)         # (1, 2, C, H, W)
