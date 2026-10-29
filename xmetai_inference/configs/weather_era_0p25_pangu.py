# -*- coding: utf-8 -*-
"""Pangu-Weather 确定性推理（6h 步进 + 24h 跳步，实现在模型类内）。License：BY-NC-SA 4.0 非商用。"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    GRID_025, PANGU_CHANNELS, InferConfig, ModelProcessingConfig)

# 上层/地面通道切分点（PANGU_CHANNELS = 5 变量 × 13 层 + 4 地面）。
_N_UPPER = 5 * 13       # 65
_N_SFC = 4


class PanguOnnxModel(OnnxInferModel):
    """Pangu-Weather ONNX 模型（6h 步进 + 每 24h 从锚点用 24h 图跳步）。

    双输入 input (5,13,721,1440) / input_surface (4,721,1440)，双输出同形；物理量
    进/出，单位 = ERA5 原生（z=m²/s²、q=kg/kg、msl=Pa），网格 = GRID_025。

    valid_time_step_offset 必须为 0（默认值，不要覆盖）：forward 用 valid_time 反推
    init_time 作锚点键，非 0 会键错位。其余契约走 BaseInferModel 默认值。
    """
    model_name = "pangu_onnx"
    input_channels = PANGU_CHANNELS
    output_channels = PANGU_CHANNELS
    grid = GRID_025
    history_steps = 1        # 只吃当前帧，滑窗归 Rollout
    windowed_output = False  # 图只出预报帧，不回显输入窗

    def load(self, path):
        # 6h 步进模型：基类单会话加载，self.session 即 6h 会话。
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

        x: (1, 1, 69, H, W) float32；step 0-based；valid_time = init + step*6h；
        返回 (1, 1, 69, H, W)。
        """
        state = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        flat = state.detach().cpu().numpy().astype(np.float32)[0, 0]   # (69, H, W)
        upper, surface = self._split(flat)

        # 锚点键 = init_time = valid_time - step*interval（offset=0 时成立）。
        key = pd.to_datetime(valid_time) - pd.Timedelta(hours=step * self.hour_interval)
        if key not in self._anchors:
            self._anchors[key] = (upper, surface)   # step 0：存起报态

        if (step + 1) % 4 == 0:
            # 24h 跳步：从锚点直接跳，并更新锚点。
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


PANGU_ROOT = "/workspace/szwCode/xmetai-inference2/model_artifacts/Pangu-Weather-main"
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

PANGU_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]

cfg = InferConfig(
    name="weather_era_0p25_pangu",
    # 6h 步进图；24h 图由模型类推导，可用 PANGU_ONNX_24 覆盖。
    model_path=f"{PANGU_ROOT}/pangu_weather_6.onnx",
    model_class="pangu_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": PANGU_STORES,
        "processors": [
            {"name": "geometry"},
        ],
    },
    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    times="2025010200..2025122500:24",
    steps=60,
    hour_interval=6,
    members=1,  # 确定性模型必须为 1
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl",
    gpus=1,
    cuda_devices="0",
    output_dir="/workspace/data/shenzw/pangu_output",
)
