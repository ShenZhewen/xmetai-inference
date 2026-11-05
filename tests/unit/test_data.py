# -*- coding: utf-8 -*-
"""Unit tests for data/ and utils/dataloader_factory.py.

Importing the package depends on xmetai_core (MultiZarrDataset); if it is not
installed locally the whole module is skipped and the tests run in the server Job.
Real Zarr reads are not tested here, only pure functions and config validation.
"""
import pandas as pd
import pytest

pytest.importorskip("xmetai.data")
pytest.importorskip("torch")

from xmetai_inference.data.weather import create_processed_multi_zarr_dataset
from xmetai_inference.data.weather.processed_multi_zarr import (
    _raw_channel_name,
    select_init_positions,
)
from xmetai_inference.utils.dataloader_factory import create_dataloader


def test_raw_channel_name_prefers_exact_then_underscore():
    available = {"z_50", "t2m", "u10m"}
    assert _raw_channel_name("z50", available) == "z_50"
    assert _raw_channel_name("t2m", available) == "t2m"
    assert _raw_channel_name("u10m", available) == "u10m"


def test_raw_channel_name_missing_raises():
    with pytest.raises(KeyError, match="u10m"):
        _raw_channel_name("u10m", {"z_50"})


class _FakeCore:
    """``select_init_positions`` reads only ``inds`` and ``id_to_time``."""

    inds = [0, 5, 10]
    id_to_time = {
        0: "2025-01-01 00:00:00",
        5: "2025-01-02 00:00:00",
        10: "2025-01-03 00:00:00",
    }


def test_select_init_positions_maps_times_to_indices():
    positions = select_init_positions(
        _FakeCore(), [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-01")])
    assert positions == [1, 0]


def test_select_init_positions_reports_missing_times():
    with pytest.raises(ValueError, match="2025-01-05"):
        select_init_positions(
            _FakeCore(), [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-05")])


def test_factory_rejects_non_mapping_spec():
    with pytest.raises(TypeError, match="配置字典"):
        create_dataloader(
            ["not-a-mapping"], model_cls=None, init_times=[],
            history_steps=1, hour_interval=6)


def test_factory_rejects_unknown_dataset_type():
    with pytest.raises(ValueError, match="未知 Dataset"):
        create_dataloader(
            {"type": "nope"}, model_cls=None, init_times=[],
            history_steps=1, hour_interval=6)


def test_dataset_requires_paths():
    with pytest.raises(ValueError, match="paths"):
        create_processed_multi_zarr_dataset(
            {"type": "processed_multi_zarr", "processors": [{"name": "unit_convert"}]},
            model_cls=None, init_times=[pd.Timestamp("2025-01-02")],
            history_steps=1, hour_interval=6)


def test_dataset_requires_processor_specs():
    with pytest.raises(ValueError, match="processors"):
        create_processed_multi_zarr_dataset(
            {"type": "processed_multi_zarr", "paths": ["dummy.zarr"]},
            model_cls=None, init_times=[pd.Timestamp("2025-01-02")],
            history_steps=1, hour_interval=6)
