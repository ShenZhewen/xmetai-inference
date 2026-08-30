# -*- coding: utf-8 -*-
"""模型包：基类 + 后端 + 各模型子类 + 注册表。

加新模型三步：
  1) 新建 models/<name>.py，继承对应后端（OnnxInferModel / Pt2InferModel / …），
     覆盖 normalize/denormalize（需要个性化时）或 load/forward（后端不够用时）。
  2) 在 MODEL_REGISTRY 里加一行 {模型名: 类}。
  3) config 里写 "model": "<模型名>" 即可。
"""
import os

from .base import BaseInferModel
from .onnx_backend import OnnxInferModel
from .pt2_backend import Pt2InferModel
from .fuxi_ens_onnx import FuxiEnsOnnxModel
from .fuxi21_pt2 import Fuxi21Pt2Model

# 模型名 → 类（config 里用 "model" 字段选模型）
MODEL_REGISTRY = {
    "fuxi_ens_onnx": FuxiEnsOnnxModel,
    "fuxi21_pt2": Fuxi21Pt2Model,
}

# 后端名 → 类（--backend 或按扩展名自动识别时用）
_BACKEND_REGISTRY = {
    "onnx": FuxiEnsOnnxModel,
    "pt2": Fuxi21Pt2Model,
}

# 文件扩展名 → 后端名
_EXT_MAP = {
    "onnx": "onnx",
    "pt2": "pt2",
}


def create_model(path, device_id=0, gpu_mem_fraction=0.7, backend=None):
    """按后端名（或模型名、或文件扩展名）构造推理模型。

    backend 缺省时按扩展名自动识别；backend 可以是后端名（onnx/pt2）
    也可以是模型名（fuxi_ens_onnx / fuxi21_pt2 / …）。
    """
    if backend is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        backend = _EXT_MAP.get(ext, "onnx")
    cls = MODEL_REGISTRY.get(backend) or _BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise ValueError(
            f"未知推理后端/模型 {backend!r}（可选 onnx/pt2 或 "
            f"{', '.join(MODEL_REGISTRY)}）")
    return cls(device_id=device_id, gpu_mem_fraction=gpu_mem_fraction)
