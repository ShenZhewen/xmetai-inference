"""``netcdf_data``: one given ``.nc`` file = one init window, inferred directly.

    {"type": "netcdf_data", "paths": ["/abs/input.nc"]}

  * One nc = one initialization; all history frames live in the nc itself — the engine never
    looks back into a store. Init time is the last ``time`` frame's value; ``cfg.times`` is
    not consulted for this type.
  * The nc holds the model input as-is: no processor chain runs. Values are passed through
    untouched — the official FuXi-ENS ``input.nc`` is physical with normalization baked into
    the graph, the FuXi-2.1 ``input.nc`` is already z-score normalized. Only the channel axis
    is verified and rewritten to ``input_channels`` by name.
  * Expected layout: one data variable with dims (time, channel, lat, lon), ``channel`` having
    name coordinates, ``lat`` north→south or as the model expects (no geometry handling).

Samples are built eagerly at dataset build time, so this type is single-process-only and does
not depend on DataLoader workers reopening files.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

#: Axis names of the nc (the official FuXi / FengQing ``input.nc`` layout).
_TIME_DIM = "time"
_CHANNEL_DIM = "channel"


def _channel_key(name):
    """Normalize channel names: ``z_50``/``z50`` and ``T2M``/``t2m`` are the same channel."""
    return str(name).replace("_", "").strip().lower()


def _spatial_dims(sizes: Mapping[str, int]) -> list[str]:
    """Find the two spatial dims (lat/lon), left-to-right order."""
    spatial = [d for d in ("lat", "latitude", "lon", "longitude", "latitude_n", "longitude_e") if d in sizes]
    return spatial[:2]


def _data_array(ds: xr.Dataset, path) -> xr.DataArray:
    """Return the single data variable as (time, channel, lat, lon); fail loudly otherwise."""
    if not ds.data_vars:
        raise ValueError(f"netcdf_data 输入 {path} 没有数据变量")
    array = ds[list(ds.data_vars)[0]]
    spatial = _spatial_dims(ds.sizes)
    if len(spatial) != 2:
        raise ValueError(f"netcdf_data 输入 {path} 找不到两个空间维度（现有 {list(ds.sizes)}）")
    wanted = {_TIME_DIM, _CHANNEL_DIM, *spatial}
    if set(array.dims) != wanted:
        raise ValueError(
            f"netcdf_data 输入 {path} 数据维度应为 {_TIME_DIM}×{_CHANNEL_DIM}×lat×lon，"
            f"实际 {array.dims}")
    array = array.transpose(_TIME_DIM, _CHANNEL_DIM, *spatial)
    return array


def _nc_init_time(ds: xr.Dataset, path) -> pd.Timestamp:
    """Init time = the value of the last ``time`` frame (t0)."""
    if _TIME_DIM not in ds.sizes:
        raise ValueError(f"netcdf_data 输入 {path} 没有 {_TIME_DIM!r} 维度")
    return pd.to_datetime(ds[_TIME_DIM].values.ravel())[-1]


def netcdf_data_init_times(paths: Sequence[object]) -> list[pd.Timestamp]:
    """Init-time-sniffing helper for the CLI: last ``time`` frame of each nc."""
    out = []
    for path in paths:
        with xr.open_dataset(os.fspath(path)) as ds:
            out.append(_nc_init_time(ds, path))
    return out


class NetcdfData(Dataset):
    """One nc file per sample; each sample is the full model input window [T, C, H, W]."""

    def __init__(self, paths, model_names, history_steps, init_times=None):
        self.paths = tuple(os.fspath(p) for p in paths)
        if not self.paths:
            raise ValueError("netcdf_data 必须通过 dataset.paths 给出至少一个 nc")
        self.model_names = list(model_names)
        self.history_steps = int(history_steps)
        paths = list(self.paths)
        if init_times:
            # Multi-GPU workers only own a subset of init times; keep just their nc files.
            # Sniff headers first so a rank never loads the full array of another rank's nc.
            sniffed = netcdf_data_init_times(paths)
            wanted = {pd.Timestamp(t) for t in init_times}
            kept = [path for path, t in zip(paths, sniffed) if t in wanted]
            if not kept:
                raise ValueError(
                    "netcdf_data 没有 nc 的起报时间落在本次 --inits 内（nc 起报: "
                    + ", ".join(str(t) for t in sniffed) + "）")
            paths = kept
        # Build every sample eagerly so channel/geometry errors surface at build time.
        self._samples = [self._build_sample(path) for path in paths]

    def _build_sample(self, path):
        with xr.open_dataset(path) as ds:
            array = _data_array(ds, path)
            init_time = _nc_init_time(ds, path)
            tensor = torch.from_numpy(array.values.astype(np.float32, copy=False))
            available = (
                [str(v) for v in array[_CHANNEL_DIM].values]
                if _CHANNEL_DIM in array.coords else list(range(tensor.shape[1]))
            )

        tensor = _reorder_channels(tensor, available, self.model_names, path)

        if tensor.shape[0] < self.history_steps:
            raise ValueError(
                f"netcdf_data 输入 {path} 只有 {tensor.shape[0]} 帧，"
                f"模型需要 {self.history_steps} 帧历史")
        return tensor[-self.history_steps:], init_time

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        tensor, init_time = self._samples[index]
        return {
            "inputs": tensor,  # [T, C, H, W]; the batch axis is added by the DataLoader
            "times": torch.tensor(init_time.value, dtype=torch.int64),
            "sample_index": index,
        }


def _reorder_channels(values, available, model_names, path):
    """Rewrite the channel axis to ``model_names`` by name (missing / mismatch raise)."""
    nc_count = values.shape[1]
    if len(model_names) != nc_count:
        raise ValueError(
            f"netcdf_data 输入 {path} 通道数 {nc_count} 与模型输入通道数 {len(model_names)} 不一致")
    if isinstance(available[0], str):
        index = {}
        for position, name in enumerate(available):
            index.setdefault(_channel_key(name), position)
        missing = [name for name in model_names if _channel_key(name) not in index]
        if missing:
            raise ValueError(f"netcdf_data 输入 {path} 缺模型输入通道 {missing[:5]}")
        order = np.asarray([index[_channel_key(name)] for name in model_names], dtype=int)
        if np.array_equal(order, np.arange(nc_count)):
            return values
        return values[:, order, :, :]
    return values  # channel coords absent: positional, must already be in model order


def create_netcdf_data(
    spec: Mapping[str, object],
    *,
    model_cls,
    init_times: Sequence[object],
    history_steps: int,
    hour_interval: int,
    model_path: str | None = None,
):
    """Build a ``netcdf_data`` Dataset.

    Init times are read from the nc files themselves; ``init_times`` only narrows the sample list
    so each multi-GPU rank keeps the nc files it was assigned.
    """
    config = dict(spec)
    config.pop("type", None)
    paths = config.pop("paths", None)
    if not paths:
        raise ValueError("netcdf_data 必须通过 dataset.paths 给出 nc 文件")
    if config:
        raise ValueError(f"netcdf_data Dataset 未知参数: {sorted(config)}")
    return NetcdfData(
        paths,
        model_names=model_cls.input_channels,
        history_steps=history_steps,
        init_times=init_times,
    )
