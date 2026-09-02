# -*- coding: utf-8 -*-
"""运行配置包：一份 config = 一次「跑法」（推理或评测）的运行配方。

换模型/换跑法 = 新增一个 configs/<name>.py，定义全局 `cfg`；代码主体不动。
spec JSON（模型契约：通道/单位/网格）仍独立放在 specs/，config 用 `spec=` 路径引用。
"""
from .base import EvalConfig, InferConfig, ROOT, load_config

__all__ = ["EvalConfig", "InferConfig", "ROOT", "load_config"]
