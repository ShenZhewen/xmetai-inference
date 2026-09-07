# -*- coding: utf-8 -*-
"""FengQing V1.5Beta (precipitation ONNX) 模型契约。

图输入（4 个键，非官方 12 输入 —— 这是重导过的图）：
  upper              : [1, 130, 721, 1440] = 2 帧 × 65 通道（帧优先）
  surface            : [1,  10, 721, 1440] = 2 帧 ×  5 通道
  temporal_condition : float16 [1, 4]
  spatial_condition  : float32 [3, 721, 1440]
图输出（**归一化残差**，不是全场）：
  upper   : [1, 65, 721, 1440]
  surface : [1,  5, 721, 1440]

分层：
  * 输入归一化 → 数据层 dataset.processors ``normalize_fengqing``
  * 反归一化   → 输出阶段 model_processing.output ``denormalize_fengqing``
  * 本类只做「一步前向 + 残差反演」，全程留在归一化空间：
        next_phys = prev_phys + out·res_std
        prev_phys = prev_norm·std + mean
        ⟹ next_norm = prev_norm + out·(res_std/std)
    所以这里只需要 res_std/std 这个每通道常数（图的输出语义，属模型契约），
    不碰 mean —— 归一化本身不在模型层。

统计量文件（mean_std/）：
  upper_mean.npy (65,H,W)   upper_std.npy (65,1)   res_upper_std.npy (65,1)
  mean_pre.npy   ( 5,H,W)   std_pre.npy   ( 5,1)   res_std_pre.npy   ( 5,1)
surface 通道序为 tp, msl, u10m, v10m, t2m（与官方 readme 的
MSLP,U10,V10,T2M,TP 不同，已按 mean_pre.npy 实测确认）。
spatial_condition 来自 utils/constant_masks.npy（lat 90→-90、lon 0→359.75，
由珠峰 [247,347] 与马里亚纳 [315,569] 两点定死）。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from xmetai_inference.backends.onnx import OnnxInferModel

from . import FENGQING_CHANNELS, GRID_025


def _temporal_condition(dt):
    """与官方 FengQing timefeatures('h') 相同的 4 特征编码。

    官方 utils/timefeatures.py 的 freq='h' 给出
    [HourOfDay, MonthOfYear, DayOfMonth, DayOfYear]，注意第 2 个是月不是星期。
    """
    return np.asarray(
        [[
            dt.hour / 23.0 - 0.5,
            (dt.month - 1) / 11.0 - 0.5,
            (dt.day - 1) / 30.0 - 0.5,
            (dt.day_of_year - 1) / 365.0 - 0.5,
        ]],
        dtype=np.float16,
    )


def _find_artifact(model_path, relative_paths):
    """在模型文件及其父目录中查找工件。

    兼容两种布局：
      .../FengQing/onnx/fengqing_pre.onnx  →  .../FengQing/mean_std/
      .../models/fengqing_pre.onnx         →  .../mean_std/
    """
    bases = [model_path.parent, model_path.parents[1], model_path.parents[2]]
    for relative in relative_paths:
        for base in bases:
            candidate = base.joinpath(*relative)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"找不到 FengQing 工件 {relative_paths!r}，相对模型文件 {model_path}"
    )


class FengqingPreOnnxModel(OnnxInferModel):
    model_name = "fengqing_pre_onnx"
    input_channels = FENGQING_CHANNELS
    output_channels = FENGQING_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 6
    forecast_type = "deterministic"
    members = 1
    # 图只出预报那一帧（不带回显帧），滑窗归 runner.Rollout。
    windowed_output = False
    # 官方 inference_fengqing.py 的 temporal_condition 取**目标预报时刻**
    # （valid = init + (step+1)*interval），与 fuxi 系的「输入窗末帧时刻」不同。
    # 两者都有官方脚本背书，不要「统一」。
    valid_time_step_offset = 1
    # 无 forward_gpu 实现（图要 4 个键、输入输出不同形），GPU 常驻不可用。
    gpu_state = False

    def load(self, path):
        super().load(path)

        model_path = Path(path).resolve()
        env_mean_std = os.environ.get("FENGQING_MEAN_STD_DIR")
        if env_mean_std:
            mean_std_dir = Path(env_mean_std)
        else:
            mean_std_dir = _find_artifact(model_path, [("mean_std",)])
        env_masks = os.environ.get("FENGQING_MASKS_PATH")
        if env_masks:
            masks_path = Path(env_masks)
        else:
            masks_path = _find_artifact(
                model_path,
                [("utils", "constant_masks.npy"), ("constant_masks.npy",)],
            )

        upper_std = np.load(mean_std_dir / "upper_std.npy").astype(np.float32)
        surface_std = np.load(mean_std_dir / "std_pre.npy").astype(np.float32)
        res_upper_std = np.load(mean_std_dir / "res_upper_std.npy").astype(np.float32)
        res_std_pre = np.load(mean_std_dir / "res_std_pre.npy").astype(np.float32)
        if upper_std.shape != (65, 1) or surface_std.shape != (5, 1):
            raise ValueError(
                f"std shape 不符：upper={upper_std.shape} surface={surface_std.shape}")
        if res_upper_std.shape != (65, 1) or res_std_pre.shape != (5, 1):
            raise ValueError(
                f"res_std shape 不符：upper={res_upper_std.shape} "
                f"surface={res_std_pre.shape}")

        # 残差反演在归一化空间只需这个比值（见模块 docstring 的推导）：
        #   next_norm = prev_norm + out·(res_std/std)
        self._upper_res_ratio = (res_upper_std / upper_std).reshape(1, 65, 1, 1)
        self._surface_res_ratio = (res_std_pre / surface_std).reshape(1, 5, 1, 1)

        # tp 非负钳位：物理空间的 0 对应归一化空间的 -mean/std（FengQing 的 mean
        # 是逐像素的，所以这是一个 (H,W) 的阈值场，不是标量）。在归一化空间钳位
        # 与「反归一化→钳位→再归一化」逐位等价，但省掉一次往返。
        surface_mean = np.load(mean_std_dir / "mean_pre.npy").astype(np.float32)
        if surface_mean.shape != (5, 721, 1440):
            raise ValueError(f"mean_pre shape 不符：{surface_mean.shape}")
        tp_index = list(self.output_channels).index("tp") - 65
        self._tp_index = tp_index
        self._tp_floor_norm = (
            -surface_mean[tp_index] / surface_std[tp_index, 0]
        ).astype(np.float32)

        masks = np.load(masks_path).astype(np.float64)            # (3, 721, 1440)
        if masks.shape != (3, 721, 1440):
            raise ValueError(f"constant_masks shape 不符：{masks.shape}")
        # 与官方 utils/constant_masks.py 相同的逐掩膜标准化（var 用 ddof=0 + 1e-5）
        standardized = []
        for arr in masks:
            flat = arr.reshape(-1)
            mean = flat.mean()
            std = np.sqrt(np.var(flat, ddof=0) + 1e-5)
            standardized.append(((flat - mean) / std).reshape(721, 1440))
        self._spatial_condition = np.stack(standardized, axis=0).astype(np.float32)

        return self

    def forward(self, x, step, valid_time):
        """一步前向：归一化量进 / 归一化量出，只出预报那一帧。

        x 形状 (1, 2, 70, H, W)（数据层已归一化）。图要的是帧优先展平的
        upper[1,130,H,W] 与 surface[1,10,H,W]，所以这里按通道切开再拼帧。
        """
        state = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        arr = state.detach().cpu().numpy().astype(np.float32)     # (1,2,70,H,W)
        frames = arr[0]                                           # (2,70,H,W)
        upper = np.concatenate([frames[i, :65] for i in range(frames.shape[0])], axis=0)
        surface = np.concatenate([frames[i, 65:] for i in range(frames.shape[0])], axis=0)

        outputs = self.session.run(None, {
            "upper": upper[None],
            "surface": surface[None],
            "temporal_condition": _temporal_condition(valid_time),
            "spatial_condition": self._spatial_condition,
        })
        out_upper, out_surface = outputs
        # 输出顺序保险：图的两个输出名是 21993/21994（tracing 未命名），顺序不该
        # 反，但反了要能自愈而不是静默错位。
        if out_upper.shape[1] != 65 and out_surface.shape[1] == 65:
            out_upper, out_surface = out_surface, out_upper
        if out_upper.shape[1] != 65 or out_surface.shape[1] != 5:
            raise ValueError(
                f"FengQing 输出通道不符：upper={out_upper.shape} "
                f"surface={out_surface.shape}")

        prev_upper = frames[-1, :65][None]                        # (1,65,H,W)
        prev_surface = frames[-1, 65:][None]                      # (1, 5,H,W)
        next_upper = prev_upper + out_upper * self._upper_res_ratio
        next_surface = prev_surface + out_surface * self._surface_res_ratio
        # tp ≥ 0（在归一化空间等价钳位，见 load 里的 _tp_floor_norm）
        next_surface[:, self._tp_index] = np.maximum(
            next_surface[:, self._tp_index], self._tp_floor_norm)

        pred = np.concatenate([next_upper, next_surface], axis=1)  # (1,70,H,W)
        return torch.as_tensor(
            pred[:, None], dtype=state.dtype, device=state.device)  # (1,1,70,H,W)
