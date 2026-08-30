# -*- coding: utf-8 -*-
"""后端包：执行引擎（只懂 load + 跑，不懂模型语义）。

引擎层与模型层分离：
  * backends/  —— 执行引擎。onnx/pt2 暴露单步 forward（循环归框架），ckpt 自带
    SimpleRunner 自回归循环则整体覆盖 run()。引擎不含归一化/反归一化/诊断清零
    等模型语义，那些由 models/ 层通过钩子覆盖。
  * models/    —— 具体模型。继承某个引擎，按需覆盖 normalize/denormalize/
    zero_recurrent 钩子。

加新引擎三步：
  1) 新建 backends/<name>.py，继承 BaseInferModel（.base），实现 load + forward
     （或覆盖 run()）。
  2) 在 BACKEND_REGISTRY 里加一行 {引擎名: 类}。
  3) 需要「模型身份」时在 models/ 建一个子类并登记进 MODEL_REGISTRY。
"""
from .base import BaseInferModel
from .onnx import OnnxInferModel
from .pt2 import Pt2InferModel
from .ckpt import CkptInferModel

# 引擎层：名字 → 执行引擎类（只懂 load + 跑，不懂模型语义）
BACKEND_REGISTRY = {
    "onnx": OnnxInferModel,
    "pt2": Pt2InferModel,
    "ckpt": CkptInferModel,
}


def create_backend(backend_name, device_id=0, gpu_mem_fraction=0.7):
    """按引擎名构造裸后端（逃生舱：无模型钩子/归一化，直接跑文件）。"""
    cls = BACKEND_REGISTRY.get(backend_name)
    if cls is None:
        raise ValueError(f"未知后端 {backend_name!r}（可选 {', '.join(BACKEND_REGISTRY)}）")
    return cls(device_id=device_id, gpu_mem_fraction=gpu_mem_fraction)


__all__ = [
    "BaseInferModel",
    "OnnxInferModel",
    "Pt2InferModel",
    "CkptInferModel",
    "BACKEND_REGISTRY",
    "create_backend",
]
