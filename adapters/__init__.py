# -*- coding: utf-8 -*-
"""数据适配层（把数据层读出的 xr.Dataset 适配成模型输入）。

与数据层 loaders/ 分工：
  * loaders/  —— 数据层：只读，`load(time) -> xr.Dataset`，模型无关；
  * adapters/ —— 数据适配层：把 Dataset 变成模型要的输入表示。

    - build_input.py        FuXi 类：张量 (1, history, C, H, W)。单位推断、通道
                            选择、纬度翻转/经度滚动、装张量，全部由 spec 驱动。
    - build_input_aifs.py   AIFS 类：命名 field 字典 + N320 非结构化节点。

本层不碰归一化（z-score / log1p 是模型的事，在 models/ 的钩子里做）；只做
「物理量之间的单位 / 几何 / 布局适配」。
"""
