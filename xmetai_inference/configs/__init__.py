# -*- coding: utf-8 -*-
"""运行配置包：一份 config = 一次推理运行配方（新跑法 = 新增 configs/<name>.py 定义全局 `cfg`）。"""
from .base import InferConfig, ModelProcessingConfig, ROOT, load_config

__all__ = ["InferConfig", "ModelProcessingConfig", "ROOT", "load_config"]
