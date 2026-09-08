# -*- coding: utf-8 -*-
"""Pangu-Weather ONNX 模型（官方四档 1h/3h/6h/24h，方案B 用 6h + 24h）。

图契约（官方 `inference_gpu.py` / `inference_iterative.py` + README 已钉死）：
  * 双输入，无 batch 维：
      input           (5, 13, 721, 1440)  上层 [Z, Q, T, U, V] × 13 层（1000→50 降序）
      input_surface   (4, 721, 1440)      地面 [MSLP, U10, V10, T2M]
  * 双输出，同形：上层 + 地面 两个独立张量（不是单一 69 通道张量）。
  * 物理量进/出：归一化已烘焙进图，官方从头到尾无归一化，不需要 mean/std。
  * 单位 = ERA5 原生：z=m²/s²（geopotential，非高度）、q=kg/kg（非 g/kg，不 ×1000）、
    t/t2m=K、u/v/u10m/v10m=m/s、msl=Pa。
  * 网格 lat 90→-90、lon 0→359.75，= 框架 GRID_025；数据层 geometry 已把 lon 从
    [-180,180) 重排到 [0,360)。

方案B（对齐官方 `inference_iterative.py` 的 28 步循环）：
  每步 framework step（0-based）推 6h；当 step+1 能被 4 整除（lead 24h/48h/…），改用
  24h 模型**从锚点**直接跳 24h，锚点初始 = 起报态、每次 24h 跳步后更新；其余步用 6h
  模型从当前态推一步。

  锚点按 init_time 隔离在 ``self._anchors``（模型实例被同一 rank 的所有轨迹共享），
  init_time 由 forward 收到的 valid_time 反推（offset=0 时 = valid_time - step*6h）。
  ``Rollout.run`` 每批轨迹开始前经 ``reset_runtime_state`` 清空，避免跨批残留、也把
  常驻内存约束在当前 batch 量级。

  分层：本模型物理量进/出，不做归一化；config 里 ``dataset.processors=[geometry]``、
  ``model_processing`` 全空。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xmetai_inference.backends.onnx import OnnxInferModel

from . import GRID_025, PANGU_CHANNELS

# 上层/地面通道切分点，与 PANGU_CHANNELS = _expand_channels(5 变量 × 13 层 + 4 地面) 一致。
_N_UPPER = 5 * 13       # 65
_N_SFC = 4


class PanguOnnxModel(OnnxInferModel):
    model_name = "pangu_onnx"
    input_channels = PANGU_CHANNELS
    output_channels = PANGU_CHANNELS
    grid = GRID_025
    history_steps = 1        # 只吃当前帧，滑窗归 Rollout
    hour_interval = 6        # framework step 粒度 = 6h；24h 跳步在 forward 内处理
    forecast_type = "deterministic"
    members = 1
    # 图只出预报那一帧（双输出），不出回显窗。
    windowed_output = False
    # 无时间标量输入；offset 只在 forward 里用 valid_time 反推 init_time（锚点键）时
    # 进入，必须是 0（init = valid_time - step*interval），否则键错位。
    valid_time_step_offset = 0
    # 双会话 + 双输入图，无 forward_gpu，GPU 常驻不可用。
    gpu_state = False

    def load(self, path):
        # 6h 步进模型：沿用基类单会话加载（self.session = 6h 会话）。
        super().load(path)
        self._sess_6 = self.session
        # 24h 跳步模型：与 6h 同目录的 pangu_weather_24.onnx，环境变量可覆盖。
        env_24 = os.environ.get("PANGU_ONNX_24")
        path_24 = env_24 or str(Path(path).resolve().parent / "pangu_weather_24.onnx")
        self._sess_24 = self._make_session(path_24)
        self._anchors = {}
        return self

    def reset_runtime_state(self):
        """每批轨迹开始前清空 24h 锚点（Rollout.run 调用），防跨批残留、限内存。"""
        self._anchors.clear()

    @staticmethod
    def _split(flat):
        """(69, H, W) → upper(5, 13, H, W) + surface(4, H, W)，按 PANGU_CHANNELS 的 var-major 序。"""
        h, w = flat.shape[-2], flat.shape[-1]
        upper = np.ascontiguousarray(flat[:_N_UPPER].reshape(5, 13, h, w))
        surface = np.ascontiguousarray(flat[_N_UPPER:_N_UPPER + _N_SFC])
        return upper, surface

    def _run(self, session, upper, surface):
        """跑一次 Pangu 前向，返回 (out_upper, out_surface)；抽出来为 6h/24h 共用。"""
        outputs = session.run(None, {"input": upper, "input_surface": surface})
        if len(outputs) != 2:
            raise ValueError(
                f"Pangu 应输出 2 个张量（上层 + 地面），实际 {len(outputs)}")
        out_upper, out_surface = outputs
        if out_upper.shape[0] != 5 or out_upper.shape[1] != 13 or out_surface.shape[0] != _N_SFC:
            raise ValueError(
                f"Pangu 输出形状不符：upper={out_upper.shape} surface={out_surface.shape}")
        return out_upper, out_surface

    def forward(self, x, step, valid_time):
        """一步前向：物理量进/出，只出预报那一帧。

        x: (1, 1, 69, H, W) float32；step 0-based；valid_time = init + step*6h。
        返回 (1, 1, 69, H, W)。
        """
        state = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        flat = state.detach().cpu().numpy().astype(np.float32)[0, 0]   # (69, H, W)
        upper, surface = self._split(flat)

        # 锚点键 = init_time = valid_time - step*interval（offset=0 时成立）。
        key = pd.to_datetime(valid_time) - pd.Timedelta(hours=step * self.hour_interval)
        if key not in self._anchors:
            self._anchors[key] = (upper, surface)   # step 0 存起报态

        if (step + 1) % 4 == 0:
            # 24h 跳步：从锚点直接跳，并更新锚点，等价官方 input_24 分支。
            anchor_upper, anchor_surface = self._anchors[key]
            out_upper, out_surface = self._run(
                self._sess_24, anchor_upper, anchor_surface)
            self._anchors[key] = (out_upper, out_surface)
        else:
            out_upper, out_surface = self._run(self._sess_6, upper, surface)

        h, w = flat.shape[-2], flat.shape[-1]
        pred = np.concatenate(
            [out_upper.reshape(_N_UPPER, h, w), out_surface], axis=0)  # (69, H, W)
        return torch.as_tensor(
            pred[None, None], dtype=state.dtype, device=state.device)   # (1, 1, 69, H, W)