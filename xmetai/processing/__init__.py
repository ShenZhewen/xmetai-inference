# -*- coding: utf-8 -*-
"""统一数据处理层（连接数据源、模型输入、自回归状态和模型输出）。

与数据层 loaders/ 分工：
  * loaders/  —— 数据层：只读，`load(time) -> xr.Dataset` 或 `load_state(time) -> State`；
  * processing/ —— 处理层：转换 State、装配模型输入并处理回填和输出。

    - pipeline.py  统一 State 约定、输入/回填/输出 Processor 与装配流程。

config 声明 Processor 规则；具体的数据适配、归一化、回填和输出反变换都由本层执行。
"""
