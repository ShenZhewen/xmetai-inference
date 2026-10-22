# -*- coding: utf-8 -*-
"""离线分析：不参与推理运行时，只读产物 / 配置 / 评测归档。

模块：

  * probe.py               单步前向的扰动量级（成员集 / 单图离散度）
  * enshcheck.py           跑完体检：成员互异、spread 随时效增长
  * spread.py              逐成员输出对比与标定曲线
  * verdict.py             注入点判据（r / k 曲线）
  * ar1_amp_audit.py       a(t) 包络离线审计（纯读 config）
  * ar1_spread_spectrum.py 逐尺度 SER（要评测仓与真实输出）
  * onnx_report.py         单图静态体检（命令行：I/O / 通道 / 自定义算子 / SPA 锚点）
  * pt2_targets.py         PT2 注入点体检（命令行：layer_stream 链路 + 候选 target 集审计）

依赖方向是单向的：analysis -> weights，analysis 不 import perturbation。
命令行仍由 ``python -m xmetai_inference.perturbation`` 统一分发，命令路径不变。
"""
