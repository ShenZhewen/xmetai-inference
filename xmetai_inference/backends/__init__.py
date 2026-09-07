# -*- coding: utf-8 -*-
"""推理后端包门面。

两层契约与自回归循环的实现位于 base.py（BaseInferModel + 运行时辅助函数），
具体引擎分别位于 onnx.py、pt2.py。本文件只做 re-export，保持
`from xmetai_inference.backends import BaseInferModel` 对外不变（wheel 安装
场景的外部消费者依赖这一入口，models 包的 get_model_class 也这么用）。

注意：不要在这里 re-export onnx/pt2 引擎——onnx.py 顶层 import onnxruntime，
门面一旦带上它，PT2-only 环境导入任何 backends 符号都会被拖上 ONNX Runtime。
引擎请按需从 xmetai_inference.backends.onnx / .pt2 导入（models 包就是这么做的）。
"""
from .base import BaseInferModel

__all__ = ["BaseInferModel"]
