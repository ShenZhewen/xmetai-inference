# -*- coding: utf-8 -*-
"""torch.export (.pt2) backend: loading, dtype/input-name inference, single-step forward (torch in / torch out)."""
import logging
import os

import numpy as np

from .base import BaseInferModel

log = logging.getLogger(__name__)


class Pt2InferModel(BaseInferModel):
    """torch.export (.pt2) backend."""
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

        # The model device is baked into the exported graph, so it must be loaded on the target GPU
        ctx = torch.cuda.device(self._device) if self._device.type == "cuda" else nullcontext()
        with ctx:
            ep = torch.export.load(path)
            self._module = ep.module()
            del ep

        # torch.export renames parameters to args_0/args_1; restore the canonical order
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

        if isinstance(x, torch.Tensor):
            state = x
        else:
            state = torch.as_tensor(x)
        state = state.to(device=self._device, dtype=self._dtype)

        # Minute-precision normalization (matches the official prepare_features tod)
        times = [valid_time] if np.ndim(valid_time) == 0 else list(valid_time)
        tod = [(value.hour * 60 + value.minute) / 1440.0 for value in times]
        feats = {
            "step": torch.full((len(times),), step, device=self._device, dtype=self._dtype),
            "hour": torch.tensor(tod, device=self._device, dtype=self._dtype),
            "doy": torch.tensor(
                [min(365, value.day_of_year) / 365.0 for value in times],
                device=self._device,
                dtype=self._dtype,
            ),
        }
        args = [state]
        for name in ("step", "hour", "doy"):
            if name in self._input_names:
                args.append(feats[name])

        with torch.no_grad():
            result = self._module(*args)

        return result.float()    # Keep the torch tensor on device; only move D2H when writing output

    def release(self):
        """Drop the module (called when switching models or shutting down)."""
        self._module = None
        super().release()

    def describe(self):
        return f"pt2 (dtype={self._dtype}, inputs={self._input_names})"
