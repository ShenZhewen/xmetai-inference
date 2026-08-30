# -*- coding: utf-8 -*-
"""FuXi-Ens ONNX 模型。

归一化已烘焙进 ONNX 图，输入/输出都在物理量空间，normalize/denormalize 继承恒等。
"""
from backends.onnx import OnnxInferModel


class FuxiEnsOnnxModel(OnnxInferModel):
    model_name = "fuxi_ens_onnx"
    """FuXi ensemble ONNX：归一化已烘焙进图，normalize/denormalize 继承恒等。"""
