# -*- coding: utf-8 -*-
"""推理后端包门面：只 re-export BaseInferModel（wheel 外部消费者的对外入口）。

不要在这里 re-export onnx/pt2 引擎——onnx.py 顶层 import onnxruntime，
门面一旦带上它，PT2-only 环境导入任何 backends 符号都会被拖上 ONNX Runtime。
引擎按需从 xmetai_inference.backends.onnx / .pt2 导入。
"""
from .base import BaseInferModel

__all__ = ["BaseInferModel"]
