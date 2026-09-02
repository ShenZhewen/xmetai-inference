# -*- coding: utf-8 -*-
"""运行配置包：一份 config = 一次推理运行配方。

换模型/换跑法 = 新增一个 configs/<name>.py，定义全局 `cfg`；代码主体不动。
模型契约（通道/单位/网格）折叠进模型类，config 用 `model_class=` 引用。
"""
from .base import InferConfig, ROOT, load_config

__all__ = ["InferConfig", "ROOT", "load_config"]
