# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig
from xmetai_inference.utils.config_util import GRID_025, _expand_channels

# Pangu upper-air levels/vars (ascending, 1000 first) and surface channels (msl, u10m,
# v10m, t2m; no tp) + the channel list they build. Self-contained, not shared with FengQing.
PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PANGU_PL_VARS = ["z", "q", "t", "u", "v"]
PANGU_SFC_VARS = ["msl", "u10m", "v10m", "t2m"]
PANGU_CHANNELS = _expand_channels(
    PANGU_PL_VARS, PANGU_LEVELS, PANGU_SFC_VARS)

# Placeholder paths; replace before deployment.
PANGU_ROOT = "/path/to/models/Pangu-Weather-main"
ERA5_ROOT = "/path/to/data/era5_foundation_store2"
PANGU_ONNX_6 = f"{PANGU_ROOT}/pangu_weather_6.onnx"
OUTPUT_DIR = "/path/to/output/pangu_output"

PANGU_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]


_N_UPPER = 5 * 13
_N_SFC = 4


class PanguOnnxModel(OnnxInferModel):
    """Pangu-Weather ONNX model (6h stepping + a 24h jump from the anchor every 24h).

    Dual inputs input (5,13,721,1440) / input_surface (4,721,1440), dual outputs of the same shapes;
    physical quantities in and out, units = native ERA5 (z=m²/s², q=kg/kg, msl=Pa), grid = GRID_025.

    valid_time_step_offset must be 0 (the default; do not override): forward derives
    init_time from valid_time as the anchor key, and a non-zero value misaligns the key.
    All other contract entries use BaseInferModel defaults.
    """
    model_name = "pangu_onnx"
    input_channels = PANGU_CHANNELS
    output_channels = PANGU_CHANNELS
    grid = GRID_025
    history_steps = 1
    windowed_output = False

    def load(self, path):
        super().load(path)
        self._sess_6 = self.session
        env_24 = os.environ.get("PANGU_ONNX_24")
        path_24 = env_24 or str(Path(path).resolve().parent / "pangu_weather_24.onnx")
        self._sess_24 = self._make_session(path_24)
        self._anchors = {}
        return self

    def reset_runtime_state(self):
        """Clear the 24h anchors before each trajectory batch (called by Rollout.run) to
        prevent cross-batch residue and bound memory.
        """
        self._anchors.clear()

    @staticmethod
    def _split(flat):
        """(69, H, W) → upper(5, 13, H, W) + surface(4, H, W), in PANGU_CHANNELS var-major order."""
        h, w = flat.shape[-2], flat.shape[-1]
        upper = np.ascontiguousarray(flat[:_N_UPPER].reshape(5, 13, h, w))
        surface = np.ascontiguousarray(flat[_N_UPPER:_N_UPPER + _N_SFC])
        return upper, surface

    def _run(self, session, upper, surface):
        """Run one Pangu forward pass and return (out_upper, out_surface); extracted for
        shared use by the 6h/24h graphs.
        """
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
        """Single forward step: physical quantities in and out; returns only the forecast frame.

        x: (1, 1, 69, H, W) float32; step is 0-based; valid_time = init + step*6h;
        returns (1, 1, 69, H, W).
        """
        state = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        flat = state.detach().cpu().numpy().astype(np.float32)[0, 0]
        upper, surface = self._split(flat)

        key = pd.to_datetime(valid_time) - pd.Timedelta(hours=step * self.hour_interval)
        if key not in self._anchors:
            self._anchors[key] = (upper, surface)

        if (step + 1) % 4 == 0:
            anchor_upper, anchor_surface = self._anchors[key]
            out_upper, out_surface = self._run(
                self._sess_24, anchor_upper, anchor_surface)
            self._anchors[key] = (out_upper, out_surface)
        else:
            out_upper, out_surface = self._run(self._sess_6, upper, surface)

        h, w = flat.shape[-2], flat.shape[-1]
        pred = np.concatenate(
            [out_upper.reshape(_N_UPPER, h, w), out_surface], axis=0)
        return torch.as_tensor(
            pred[None, None], dtype=state.dtype, device=state.device)


cfg = InferConfig(
    name="weather_era_0p25_pangu",
    model_path=PANGU_ONNX_6,
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
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl",
    gpus=1,
    cuda_devices="0",
    output_dir=OUTPUT_DIR,
)
