# -*- coding: utf-8 -*-
"""统一数据处理层（连接数据源、模型输入、自回归状态和模型输出）。

与数据层 data/ 分工：data/ 负责 Dataset 读取、样本选择和 DataLoader batch；
processing/ 负责模型侧转换、输入装配、回填和输出反变换。

  - pipeline.py          回填/输出 Processor 与装配流程。
  - tensor_processors.py 配置驱动的 Tensor 前处理链（dataset.processors）。
"""
