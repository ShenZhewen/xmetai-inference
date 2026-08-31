# -*- coding: utf-8 -*-
"""FuXi-Ens ONNX 模型。

归一化已烘焙进 ONNX 图，输入/输出都在物理量空间，pre_process/post_process 继承恒等。
扰动由图内随机算子（RandomNormalLike/RandomUniformLike）自带，51 成员 = 51 次独立
前向，框架无需加扰动。
"""
import os

from backends.onnx import OnnxInferModel


class FuxiEnsOnnxModel(OnnxInferModel):
    model_name = "fuxi_ens_onnx"
    # 前后处理/回填都是恒等，GPU 常驻安全；XMETAI_DISABLE_GPU_STATE=1 可回退 numpy 版对比
    gpu_state = os.environ.get("XMETAI_DISABLE_GPU_STATE", "0") != "1"
