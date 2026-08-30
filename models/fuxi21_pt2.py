# -*- coding: utf-8 -*-
"""FuXi-2.1 PT2 模型。

输入/输出都在 z-score 空间：normalize 把物理量转 z-score（tp 末通道先 log1p），
denormalize 把预测反算回物理量（×std+mean，tp 通道 expm1）。mean.nc / std.nc
从模型同目录读取。
"""
import os

import numpy as np
import xarray as xr

from .pt2_backend import Pt2InferModel

# 诊断通道（官方 README "How It Works" 第 3 步：回填前清零）。这 5 个场是
# decoder 的诊断输出（辐射通量 + 降水），不是预报量；回填下一拍输入前必须清零，
# 否则 6h 累积场会滚进下一拍。与 variables.py 的 ACCUMULATED_CHANNELS 一致。
_DIAGNOSTIC_CHANNELS = ("ssr", "ssrd", "fdir", "ttr", "tp")


class Fuxi21Pt2Model(Pt2InferModel):
    model_name = "fuxi21_pt2"

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        super().__init__(device_id, gpu_mem_fraction)
        self._mean = None
        self._std = None
        self._diag_indices = []          # 诊断通道在 C85 里的下标

    def load(self, path):
        super().load(path)
        # 输出反归一化统计量放在模型同目录（mean.nc / std.nc）
        d = os.path.dirname(path)
        mean_path, std_path = os.path.join(d, "mean.nc"), os.path.join(d, "std.nc")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean_da = xr.open_dataarray(mean_path)
            self._mean = mean_da.values.astype(np.float32)
            self._std = xr.open_dataarray(std_path).values.astype(np.float32)
            # 从 mean.nc 的 channel 坐标定位诊断通道下标（C85 顺序与模型输出一致）
            if "channel" in mean_da.coords:
                names = [str(c) for c in mean_da.coords["channel"].values]
                self._diag_indices = [i for i, n in enumerate(names)
                                      if n in _DIAGNOSTIC_CHANNELS]
        else:
            print(f"[警告] 未在 {d} 找到 mean.nc/std.nc，输出将不反归一化（仍是 z-score）")
        return self

    def zero_recurrent(self, state):
        """官方 README：辐射/降水是诊断输出，回填前在 recurrent state 里清零。"""
        if self._diag_indices:
            state[..., self._diag_indices, :, :] = 0.0
        return state

    def denormalize(self, y):
        if self._mean is None or self._std is None:
            return y
        shape = [1] * (y.ndim - 3) + [self._mean.size, 1, 1]
        out = y * self._std.reshape(shape) + self._mean.reshape(shape)
        ch = out[..., -1, :, :]           # tp 通道（最后一个通道）log1p 反变换
        np.clip(ch, None, 20, out=ch)
        np.expm1(ch, out=ch)
        np.clip(ch, 0, None, out=ch)
        return out

    def normalize(self, x):
        if self._mean is None or self._std is None:
            return x
        x = np.array(x, dtype=np.float32, copy=True)
        x[..., -1, :, :] = np.log1p(x[..., -1, :, :])   # tp: 先 log1p 再 z-score
        shape = [1] * (x.ndim - 3) + [self._mean.size, 1, 1]
        return (x - self._mean.reshape(shape)) / self._std.reshape(shape)
