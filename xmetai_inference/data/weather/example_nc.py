"""``example_nc``: one given ``.nc`` file = one init window, inferred directly.

    {"type": "example_nc", "paths": ["/abs/input_2025010600.nc"], "mode": "model_ready"}

  * One nc = one initialization; all history frames live in the nc itself — the engine never
    looks back into a store. Init time is the last ``time`` frame's value; ``cfg.times`` is
    not consulted for this type.
  * ``model_ready`` (default): the nc already holds exactly the model input — physical
    quantities in ``input_channels`` order, in the model's native grid orientation. No processor
    chain runs; only channel order is verified/rewritten. Matches the official FuXi/FengQing
    ``input.nc`` and store-extracted samples.
  * ``native``: the nc holds raw fields in its own channel order; the model's dataset processor
    chain (``unit_convert``/``geometry``/``normalize``/...) runs as in
    ``processed_multi_zarr``, then the channel axis is reordered to ``input_channels``.

Samples are built eagerly at dataset build time (fields are small relative to the model), so
this type is single-process-only and does not depend on DataLoader workers reopening files.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

from xmetai_inference.utils.dataset_processors import (
    TensorProcessorContext,
    build_tensor_processors,
)

#: Default axis names for the nc (the official FuXi / FengQing input.nc layout).
_DEFAULT_TIME_DIM = "time"
_DEFAULT_CHANNEL_DIM = "channel"


def _channel_key(name):
    """Normalize channel names: ``z_50``/``z50`` and ``T2M``/``t2m`` are the same channel."""
    return str(name).replace("_", "").strip().lower()


def _spatial_dims(sizes: Mapping[str, int]) -> list[str]:
    """Find the two spatial dims (lat/lon), left-to-right order."""
    spatial = [d for d in ("lat", "latitude", "lon", "longitude", "latitude_n", "longitude_e") if d in sizes]
    return spatial[:2]


def _data_array(ds: xr.Dataset, time_dim: str, channel_dim: str) -> xr.DataArray:
    """Return the single data variable as (time, channel, lat, lon); fail loudly otherwise."""
    if not ds.data_vars:
        raise ValueError(f"example_nc 输入 {os.fspath(ds.encoding.get('source', ''))} 没有数据变量")
    array = ds[list(ds.data_vars)[0]]
    spatial = _spatial_dims(ds.sizes)
    if len(spatial) != 2:
        raise ValueError(f"example_nc 输入找不到两个空间维度（现有 {list(ds.sizes)}）")
    wanted = {time_dim, channel_dim, *spatial}
    if set(array.dims) != wanted:
        raise ValueError(
            f"example_nc 数据维度应为 {time_dim}×{channel_dim}×lat×lon，"
            f"实际 {array.dims}")
    array = array.transpose(time_dim, channel_dim, *spatial)
    if array.ndim != len(wanted):
        raise ValueError(f"example_nc 数据应是 4 维，实际 {array.ndim} 维")
    return array


def _nc_init_time(ds: xr.Dataset, time_dim: str) -> pd.Timestamp:
    """Init time = the value of the last ``time`` frame (t0)."""
    times = pd.to_datetime(ds[time_dim].values.ravel())
    return times[-1]


def example_nc_init_times(paths, time_dim=_DEFAULT_TIME_DIM):
    """Init-time-sniffing helper for the CLI: last ``time`` frame of each nc (lazy, cheap)."""
    out = []
    for path in paths:
        with xr.open_dataset(os.fspath(path)) as ds:
            if time_dim not in ds.sizes:
                raise ValueError(f"example_nc 输入 {path} 没有 {time_dim!r} 维度")
            out.append(_nc_init_time(ds, time_dim))
    return out


class ExampleNcDataset(Dataset):
    """One nc file per sample; each sample is the full model input window [T, C, H, W]."""

    def __init__(
        self,
        paths,
        model_names,
        history_steps,
        mode="model_ready",
        processor_specs=(),
        latlon=None,
        time_dim=_DEFAULT_TIME_DIM,
        channel_dim=_DEFAULT_CHANNEL_DIM,
    ):
        self.paths = tuple(os.fspath(p) for p in paths)
        if not self.paths:
            raise ValueError("example_nc 必须通过 dataset.paths 给出至少一个 nc")
        self.model_names = list(model_names)
        self.history_steps = int(history_steps)
        self.time_dim = time_dim
        self.channel_dim = channel_dim
        if mode not in ("model_ready", "native"):
            raise ValueError(
                f"example_nc mode 只能是 model_ready/native，实际 {mode!r}")
        self.mode = mode

        if mode == "native":
            if not processor_specs:
                raise ValueError("example_nc mode=native 必须声明 dataset.processors")
            if latlon is None:
                raise ValueError("example_nc mode=native 需要 lat/lon 上下文")
            context = TensorProcessorContext(
                channel_names=self.model_names,
                latitudes=latlon[0],
                longitudes=latlon[1],
                expected_shape=latlon[2],
            )
            self.processors = build_tensor_processors(processor_specs, context)
        else:
            if processor_specs:
                raise ValueError("example_nc mode=model_ready 不应声明 dataset.processors")
            self.processors = ()

        # Build every sample eagerly so channel/geometry/processor errors surface at build time,
        # not in a DataLoader worker.
        self._samples = [self._build_sample(path) for path in self.paths]

    def _build_sample(self, path):
        with xr.open_dataset(path) as ds:
            array = _data_array(ds, self.time_dim, self.channel_dim)
            init_time = _nc_init_time(ds, self.time_dim)
            tensor = torch.from_numpy(array.values.astype(np.float32, copy=False))
            available = [
                str(v) for v in array[self.channel_dim].values
            ] if self.channel_dim in array.coords else list(range(tensor.shape[1]))

        tensor = _reorder_channels(tensor, available, self.model_names)

        if tensor.shape[0] < self.history_steps:
            raise ValueError(
                f"example_nc 输入 {path} 只有 {tensor.shape[0]} 帧，"
                f"模型需要 {self.history_steps} 帧历史")
        tensor = tensor[-self.history_steps:]

        for processor in self.processors:
            tensor = processor(tensor)
        return tensor, init_time

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        tensor, init_time = self._samples[index]
        return {
            "inputs": tensor,  # [T, C, H, W]; the batch axis is added by the DataLoader
            "times": torch.tensor(init_time.value, dtype=torch.int64),
            "sample_index": index,
        }


def _reorder_channels(values, available, model_names):
    """Rewrite the channel axis to ``model_names`` by name (missing / mismatch raise)."""
    if len(values.shape) == 4:
        nc_count = values.shape[1]
    else:
        nc_count = values.shape[0]
    if len(model_names) != nc_count:
        raise ValueError(
            f"example_nc 通道数 {nc_count} 与模型输入通道数 {len(model_names)} 不一致")
    if isinstance(available[0], str):
        index = {}
        for position, name in enumerate(available):
            index.setdefault(_channel_key(name), position)
        missing = [name for name in model_names if _channel_key(name) not in index]
        if missing:
            raise ValueError(f"example_nc 缺模型输入通道 {missing[:5]}")
        order = np.asarray(
            [index[_channel_key(name)] for name in model_names], dtype=int)
        if np.array_equal(order, np.arange(nc_count)):
            return values
        return values[:, order, :, :]
    return values  # channel coords absent: positional, must already be in model order


def _peek_latlon(path, time_dim, channel_dim):
    """Lat/lon coordinate arrays + expected shape for the native-mode processor context."""
    with xr.open_dataset(os.fspath(path)) as ds:
        array = _data_array(ds, time_dim, channel_dim)
        spatial = [d for d in array.dims if d not in (time_dim, channel_dim)]
        lat = np.asarray(array[spatial[0]].values, dtype=np.float64).ravel()
        lon = np.asarray(array[spatial[1]].values, dtype=np.float64).ravel()
    return lat, lon, (int(lat.size), int(lon.size))


def create_example_nc_dataset(
    spec: Mapping[str, object],
    *,
    model_cls,
    init_times: Sequence[object],
    history_steps: int,
    hour_interval: int,
    model_path: str | None = None,
):
    """Build a ``example_nc`` Dataset. ``init_times`` is ignored — times come from the nc files."""
    config = dict(spec)
    config.pop("type", None)
    paths = config.pop("paths", None)
    if not paths:
        raise ValueError("example_nc Dataset 必须通过 dataset.paths 给出 nc 文件")
    mode = config.pop("mode", "model_ready")
    processor_specs = config.pop("processors", ())
    time_dim = config.pop("time_dim", _DEFAULT_TIME_DIM)
    channel_dim = config.pop("channel_dim", _DEFAULT_CHANNEL_DIM)
    if config:
        raise ValueError(f"example_nc Dataset 未知参数: {sorted(config)}")

    paths = [os.fspath(p) for p in paths]
    latlon = _peek_latlon(paths[0], time_dim, channel_dim) if mode == "native" else None
    return ExampleNcDataset(
        paths,
        model_names=model_cls.input_channels,
        history_steps=history_steps,
        mode=mode,
        processor_specs=processor_specs,
        latlon=latlon,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )