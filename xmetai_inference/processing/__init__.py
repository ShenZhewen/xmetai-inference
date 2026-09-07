# -*- coding: utf-8 -*-
"""统一数据处理层（连接数据源、模型输入、自回归状态和模型输出）。

与数据层 data/ 分工：
  * data/ —— Dataset 读取、样本选择和 DataLoader batch；
  * processing/ —— 模型侧转换、输入装配、回填和输出反变换。

    - pipeline.py          统一 State 约定、输入/回填/输出 Processor 与装配流程。
    - tensor_processors.py 配置驱动的 Tensor 前处理链（dataset.processors）。

config 的 model_processing 声明 State Processor 规则；归一化、回填和输出
反变换都由模型实例持有的 ProcessingPipeline 执行。dataset.processors 声明
输入侧 Tensor 前处理链，由数据层调用 processing.tensor_processors。
"""
