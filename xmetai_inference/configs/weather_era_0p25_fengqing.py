# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import torch

from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    FENGQING_CHANNELS, GRID_025, InferConfig, ModelProcessingConfig)


def _temporal_condition(dt):
    """temporal_condition 的 4 特征编码 = [HourOfDay, MonthOfYear, DayOfMonth,
    DayOfYear]（同官方 timefeatures('h')）；第 2 个是月不是星期。
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
    """在模型文件本身及上两级父目录里查找工件（兼容 .../onnx/ 与 .../models/ 两种布局）。"""
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
    """FengQing V1.5Beta 降水图契约。

    4 键输入：upper/surface/temporal_condition/spatial_condition；输出归一化残差，
    本类做一步前向 + 残差反演，全程归一化空间。surface 通道序 =
    tp,msl,u10m,v10m,t2m（以 mean_pre.npy 实测为准，与官方 readme 不同）。

    只写非默认契约；history_steps=2 / hour_interval=6 / deterministic / 1 成员 /
    gpu_state=False 用 BaseInferModel 默认值。无 forward_gpu。
    """
    model_name = "fengqing_pre_onnx"
    input_channels = FENGQING_CHANNELS
    output_channels = FENGQING_CHANNELS
    grid = GRID_025
    windowed_output = False
    valid_time_step_offset = 1

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

        self._upper_res_ratio = (res_upper_std / upper_std).reshape(1, 65, 1, 1)
        self._surface_res_ratio = (res_std_pre / surface_std).reshape(1, 5, 1, 1)

        surface_mean = np.load(mean_std_dir / "mean_pre.npy").astype(np.float32)
        if surface_mean.shape != (5, 721, 1440):
            raise ValueError(f"mean_pre shape 不符：{surface_mean.shape}")
        tp_index = list(self.output_channels).index("tp") - 65
        self._tp_index = tp_index
        self._tp_floor_norm = (
            -surface_mean[tp_index] / surface_std[tp_index, 0]
        ).astype(np.float32)

        masks = np.load(masks_path).astype(np.float64)
        if masks.shape != (3, 721, 1440):
            raise ValueError(f"constant_masks shape 不符：{masks.shape}")
        standardized = []
        for arr in masks:
            flat = arr.reshape(-1)
            mean = flat.mean()
            std = np.sqrt(np.var(flat, ddof=0) + 1e-5)
            standardized.append(((flat - mean) / std).reshape(721, 1440))
        self._spatial_condition = np.stack(standardized, axis=0).astype(np.float32)

        return self

    def forward(self, x, step, valid_time):
        """一步前向：归一化量进/出，只出预报帧。x = (1,2,70,H,W)。"""
        state = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        arr = state.detach().cpu().numpy().astype(np.float32)
        frames = arr[0]
        upper = np.concatenate([frames[i, :65] for i in range(frames.shape[0])], axis=0)
        surface = np.concatenate([frames[i, 65:] for i in range(frames.shape[0])], axis=0)

        outputs = self.session.run(None, {
            "upper": upper[None],
            "surface": surface[None],
            "temporal_condition": _temporal_condition(valid_time),
            "spatial_condition": self._spatial_condition,
        })
        out_upper, out_surface = outputs
        if out_upper.shape[1] != 65 and out_surface.shape[1] == 65:
            out_upper, out_surface = out_surface, out_upper
        if out_upper.shape[1] != 65 or out_surface.shape[1] != 5:
            raise ValueError(
                f"FengQing 输出通道不符：upper={out_upper.shape} "
                f"surface={out_surface.shape}")

        prev_upper = frames[-1, :65][None]
        prev_surface = frames[-1, 65:][None]
        next_upper = prev_upper + out_upper * self._upper_res_ratio
        next_surface = prev_surface + out_surface * self._surface_res_ratio
        next_surface[:, self._tp_index] = np.maximum(
            next_surface[:, self._tp_index], self._tp_floor_norm)

        pred = np.concatenate([next_upper, next_surface], axis=1)
        return torch.as_tensor(
            pred[:, None], dtype=state.dtype, device=state.device)


FENGQING_ROOT = "/workspace/szwCode/xmetai-inference2/model_artifacts/FengQing"
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
TP_ROOT = "/workspace/data/liujunjie/era5_tp_zarr"

FENGQING_STORES = [
    f"{TP_ROOT}/era5.202501-202607.c1.p25.h6.v1.lon180.zarr",
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]

cfg = InferConfig(
    name="weather_era_0p25_fengqing",
    model_path=f"{FENGQING_ROOT}/onnx/fengqing_pre.onnx",
    model_class="fengqing_pre_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": FENGQING_STORES,
        "processors": [
            {"name": "unit_convert", "scales": {"tp": 1000.0}},
            {"name": "geometry"},
            {"name": "normalize_fengqing"},
        ],
    },
    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[{"name": "denormalize_fengqing"}],
    ),

    times="2025010200..2025122800:24",
    steps=60,
    hour_interval=6,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/fengqing_output",
)
