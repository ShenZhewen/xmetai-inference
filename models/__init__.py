# -*- coding: utf-8 -*-
"""模型包：后端（执行引擎）+ 模型（具体网络）+ 两层注册表。

两层注册表，语义不混：
  * BACKEND_REGISTRY：引擎名 → 执行引擎类。引擎只负责 load + 跑（onnx/pt2 用
    forward 一步，ckpt 自带循环则整体覆盖 run），不含模型语义（归一化等）。
  * MODEL_REGISTRY：模型名 → 具体模型类。模型继承某个引擎，按需覆盖
    normalize/denormalize/zero_recurrent 钩子，补上「跑之前/之后」的
    模型专属处理（z-score 反归一化、诊断通道清零等）。

加新模型三步：
  1) 新建 models/<name>.py，继承对应引擎（OnnxInferModel / Pt2InferModel / …），
     按需覆盖 normalize/denormalize/zero_recurrent 钩子。
  2) 在 MODEL_REGISTRY 里加一行 {模型名: 类}。
  3) 对应 spec JSON 的 model 块写 "class": "<模型名>"。

模型选择主路径：spec 的 model.class → create_model()；--backend 只是逃生舱
（覆盖 spec，可传模型名或引擎名，引擎名走 create_backend() 裸跑、无钩子）。
"""
from .base import BaseInferModel
from .onnx_backend import OnnxInferModel
from .pt2_backend import Pt2InferModel
from .fuxi_ens_onnx import FuxiEnsOnnxModel
from .fuxi21_pt2 import Fuxi21Pt2Model
from .ckpt_backend import CkptInferModel
from .aifs11_ckpt import Aifs11CkptModel

# 引擎层：名字 → 执行引擎类（只懂 load + 跑，不懂模型语义）
BACKEND_REGISTRY = {
    "onnx": OnnxInferModel,
    "pt2": Pt2InferModel,
    "ckpt": CkptInferModel,
}

# 模型层：名字 → 具体模型类（继承某引擎 + 覆盖钩子）
MODEL_REGISTRY = {
    "fuxi_ens_onnx": FuxiEnsOnnxModel,   # backend="onnx"
    "fuxi21_pt2": Fuxi21Pt2Model,        # backend="pt2"
    "aifs11_ckpt": Aifs11CkptModel,      # backend="ckpt"
}


def create_model(model_name, device_id=0, gpu_mem_fraction=0.7):
    """按模型名（spec 的 model.class）构造推理模型。"""
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f"未知模型 {model_name!r}（可选 {', '.join(MODEL_REGISTRY)}）")
    return cls(device_id=device_id, gpu_mem_fraction=gpu_mem_fraction)


def create_backend(backend_name, device_id=0, gpu_mem_fraction=0.7):
    """按引擎名构造裸后端（逃生舱：无模型钩子/归一化，直接跑文件）。"""
    cls = BACKEND_REGISTRY.get(backend_name)
    if cls is None:
        raise ValueError(f"未知后端 {backend_name!r}（可选 {', '.join(BACKEND_REGISTRY)}）")
    return cls(device_id=device_id, gpu_mem_fraction=gpu_mem_fraction)
