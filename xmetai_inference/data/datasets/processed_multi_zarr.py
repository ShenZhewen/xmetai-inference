"""Raw-reading wrapper around xmetai-core MultiZarrDataset.

This module is responsible for opening the Zarr stores, selecting channels and
inference positions, and applying a configured tensor processor chain. The
processors themselves live in ``xmetai_inference.processing.tensor_processors``
and are selected through ``dataset.processors``.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from xmetai.data.weather.multi_zarr_dataset import MultiZarrDataset

from xmetai_inference.processing.tensor_processors import (
    TensorProcessorContext,
    aligned_coordinates,
    build_tensor_processors,
)


def _raw_channel_name(model_name: str, available: set[str]) -> str:
    """Map a model channel name (z50) to the Zarr channel name (z_50)."""
    if model_name in available:
        return model_name
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", model_name)
    candidate = f"{match.group(1)}_{match.group(2)}" if match else model_name
    if candidate in available:
        return candidate
    raise KeyError(f"Core Dataset 中找不到模型通道 {model_name!r}")


def select_init_positions(dataset, init_times: Sequence[object]) -> list[int]:
    """Map requested initialization times to positional Core Dataset indices."""
    available = {
        pd.Timestamp(dataset.id_to_time[int(raw_index)]): position
        for position, raw_index in enumerate(dataset.inds)
    }
    requested = [pd.Timestamp(value) for value in init_times]
    missing = [value for value in requested if value not in available]
    if missing:
        formatted = ", ".join(value.isoformat() for value in missing)
        raise ValueError(
            "以下起报时间在 Dataset 中不存在或缺少历史帧："
            f"{formatted}"
        )
    return [available[value] for value in requested]


class RawMultiZarrTensorDataset(Dataset):
    """Composition wrapper: read raw frames, then apply tensor processors."""

    def __init__(self, base, positions, processors=()):
        self.base = base
        self.positions = tuple(int(position) for position in positions)
        self.processors = tuple(processors)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = self.positions[index]
        raw_idx = int(self.base.inds[position])

        time_inds = np.arange(raw_idx, raw_idx + self.base.hist_frames)
        raw = self.base.ds.isel(
            time=time_inds,
            channel=self.base.select_vars,
        )

        inputs = torch.from_numpy(raw.values.astype(np.float32, copy=False))
        for processor in self.processors:
            inputs = processor(inputs)

        init_time = pd.to_datetime(self.base.id_to_time[raw_idx])
        return {
            "inputs": inputs,
            "times": torch.tensor(init_time.value, dtype=torch.int64),
            "sample_index": position,
        }


def create_processed_multi_zarr_dataset(
    spec: Mapping[str, object],
    *,
    model_cls,
    init_times: Sequence[object],
    history_steps: int,
    hour_interval: int,
    data_root: str | None = None,
    model_path: str | None = None,
):
    """Create a raw-reading processed MultiZarrDataset.

    Preprocessing is declared in ``dataset.processors`` and applied per sample in
    ``RawMultiZarrTensorDataset.__getitem__``.
    """
    config = dict(spec)
    dataset_type = config.pop("type", None)
    if dataset_type != "processed_multi_zarr":
        raise ValueError(
            f"processed_multi_zarr 只接受 type='processed_multi_zarr'，"
            f"实际为 {dataset_type!r}"
        )
    paths = config.pop("paths", None)
    processor_specs = config.pop("processors", ())
    if not paths:
        raise ValueError(
            "processed_multi_zarr Dataset 必须通过 dataset.paths 显式指定 Zarr"
        )
    if not processor_specs:
        raise ValueError(
            "processed_multi_zarr Dataset 必须通过 dataset.processors 声明前处理链"
        )
    paths = [os.fspath(path) for path in paths]

    requested_times = pd.DatetimeIndex(pd.to_datetime(list(init_times)))
    history_start = requested_times.min() - pd.Timedelta(
        hours=(history_steps - 1) * hour_interval
    )
    base = MultiZarrDataset(
        data_paths=list(paths),
        hist_frames=history_steps,
        fcst_frames=1,
        freq=hour_interval,
        interval=1,
        training=False,
        inference_only=True,
        years=(
            str(history_start),
            str(requested_times.max() + pd.Timedelta(hours=hour_interval)),
        ),
        in_names=[],
        **config,
    )

    raw_names = [str(value) for value in base.ds.channel.values]
    available = set(raw_names)
    model_names = list(model_cls.input_channels)
    selected_raw_names = [_raw_channel_name(name, available) for name in model_names]
    base.select_vars = [raw_names.index(name) for name in selected_raw_names]

    positions = select_init_positions(base, requested_times)

    expected_shape = (
        int(model_cls.grid["lat"]["size"]),
        int(model_cls.grid["lon"]["size"]),
    )
    latitudes = np.asarray(base.ds.lat.values)
    longitudes = np.asarray(base.ds.lon.values)
    context = TensorProcessorContext(
        channel_names=model_names,
        latitudes=latitudes,
        longitudes=longitudes,
        expected_shape=expected_shape,
        model_path=model_path,
    )
    processors = build_tensor_processors(processor_specs, context)

    aligned_lat, aligned_lon = aligned_coordinates(
        latitudes, longitudes, expected_shape
    )
    dataset = RawMultiZarrTensorDataset(base, positions, processors)
    # 曾经还产出 input_space / unit_scales / preprocessed 三项，但 DatasetSource
    # 收下后全项目无人读取（单位换算已由 processors 里的 unit_convert 完成），
    # 已随字段一起移除。需要换算系数的地方直接读 config，见
    # util/eval_common.defaults_from_config。
    metadata = {
        "channel_names": model_names,
        "latitudes": aligned_lat,
        "longitudes": aligned_lon,
    }
    return dataset, metadata
