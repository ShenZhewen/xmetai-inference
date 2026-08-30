# -*- coding: utf-8 -*-
"""AIFS 1.1（anemoi GNN，.ckpt）模型：继承 ckpt 后端，无额外钩子。

归一化 / 插值 / 边界全部烘焙在 checkpoint 里，SimpleRunner 内部自己处理，所以
normalize / denormalize / zero_recurrent 全部继承恒等实现。这里只是一个「模型身份」
占位：告诉框架「这是 aifs11_ckpt，跑 ckpt 后端」。
"""
from .ckpt_backend import CkptInferModel


class Aifs11CkptModel(CkptInferModel):
    """AIFS 1.1 单模型（deterministic，1 member）。"""
    model_name = "aifs11_ckpt"
