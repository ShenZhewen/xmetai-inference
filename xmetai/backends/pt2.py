# -*- coding: utf-8 -*-
"""torch.export (.pt2) 后端。

通用 PT2 加载 + 一步前向。这里只做后端机制（export 加载、dtype/输入名推断、跑 module），
**不含任何模型特定归一化**——mean/std、tp 的 log1p/expm1 等全部由统一
Processor 管线处理。
"""
import os

import numpy as np

from . import BaseInferModel


class Pt2InferModel(BaseInferModel):
    """torch.export (.pt2) 后端。"""
    backend = "pt2"

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        super().__init__(device_id, gpu_mem_fraction)
        self._module = None
        self._input_names = []
        self._dtype = None
        self._device = None

    def load(self, path):
        import inspect
        from contextlib import nullcontext

        import torch

        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在：{path}")
        self._device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")

        # 模型 device 已烘焙进导出图，要在目标卡上加载（否则可能落到 cuda:0）
        ctx = torch.cuda.device(self._device) if self._device.type == "cuda" else nullcontext()
        with ctx:
            ep = torch.export.load(path)
            self._module = ep.module()
            del ep

        # 输入名：torch.export 会把参数名变成 args_0/args_1，这里按规范顺序回填
        sig_names = list(inspect.signature(self._module.forward).parameters.keys())
        if sig_names and sig_names[0].startswith("args_"):
            self._input_names = ["input", "step", "hour", "doy"][:len(sig_names)]
        else:
            self._input_names = sig_names

        for p in self._module.parameters():
            self._dtype = p.dtype
            break
        if self._dtype is None:
            self._dtype = torch.float32

        torch.cuda.empty_cache()
        return self

    def forward(self, x, step, valid_time):
        import torch

        if valid_time is None:
            raise ValueError("PT2 模型需要起报时间 init_time 来计算 hour/doy 时间条件")

        state = torch.from_numpy(np.ascontiguousarray(x)).to(self._device, self._dtype)
        # hour 用官方 FuXi-2.1 的分钟精度归一化：(hour*60+minute)/1440，与官方
        # data_util.prepare_features 的 tod 一致；doy 同理。
        tod = (valid_time.hour * 60 + valid_time.minute) / 1440.0
        feats = {
            "step": torch.tensor([step], device=self._device, dtype=self._dtype),
            "hour": torch.tensor([tod], device=self._device, dtype=self._dtype),
            "doy": torch.tensor([min(365, valid_time.day_of_year) / 365.0],
                                device=self._device, dtype=self._dtype),
        }
        args = [state]
        for name in ("step", "hour", "doy"):
            if name in self._input_names:
                args.append(feats[name])
        with torch.no_grad():
            result = self._module(*args)
        # 完整输出 (1,in_frames,C,H,W) → numpy float32；保持归一化供 run 回填。
        # 末帧是本步预报，完整 state 用于 state=result 回填（与官方一致）。
        return result.float().cpu().numpy()

    def describe(self):
        return f"pt2 (dtype={self._dtype}, inputs={self._input_names})"
