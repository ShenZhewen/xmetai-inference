# -*- coding: utf-8 -*-
"""Unit tests for the time scalar encoding (time_features) in backends/onnx.py.

Locks two things:

  * the default convention (Tongyun-CRA: hour/24, doy/365 clamped to 365) is
    **bit-identical** to the pre-refactor hard-coded version;
  * a model class can change the encoding by declaring ``time_features`` alone,
    without overriding forward again (fengshun2.0 feeds raw values).

Pure function level: no graph loading, no session creation; the whole module is
skipped when onnxruntime is unavailable.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnxruntime")

from xmetai_inference.backends import onnx as onnx_backend
from xmetai_inference.utils.model_factory import get_model_class


def _legacy_feeds(valid_time, step, perturb_std_scale=1.0):
    """Verbatim implementation of the pre-refactor ``_scalar_feeds`` (comparison baseline)."""
    times = [valid_time] if np.ndim(valid_time) == 0 else list(valid_time)
    return {
        "step": np.full((len(times),), step, dtype=np.float32),
        "hour": np.asarray([value.hour / 24.0 for value in times], dtype=np.float32),
        "doy": np.asarray(
            [min(365, value.day_of_year) / 365.0 for value in times],
            dtype=np.float32,
        ),
        "perturb_std_scale": np.full(
            (len(times),), perturb_std_scale, dtype=np.float32),
    }


TIMES = [
    pd.Timestamp("2024-02-29 00:00"),   # Leap day
    pd.Timestamp("2024-12-31 06:00"),   # Last day of a leap year: doy=366, the old implementation clamps to 365
    pd.Timestamp("2025-01-01 18:00"),
]


@pytest.mark.parametrize("valid_time", [TIMES[0], TIMES], ids=["scalar", "list"])
def test_default_encoding_matches_legacy(valid_time):
    legacy = _legacy_feeds(valid_time, 7)
    feeds = onnx_backend._scalar_feeds(valid_time, 7)
    assert set(feeds) == set(legacy)
    for name, expected in legacy.items():
        assert feeds[name].dtype == np.float32
        assert np.array_equal(feeds[name], expected), name


def test_fengshun20_feeds_raw_hour_and_unclipped_doy():
    model = get_model_class("fengshun20_onnx")()
    feeds = onnx_backend._scalar_feeds(TIMES[1], 12, model._time_features)
    assert feeds["hour"].tolist() == [6.0]        # raw value, not divided by 24
    assert feeds["doy"].tolist() == [366.0]       # leap years are not clamped to 365
    assert feeds["step"].tolist() == [12.0]       # step keeps the default
    assert feeds["perturb_std_scale"].tolist() == [1.0]


def test_fengshun20_no_longer_overrides_forward():
    cls = get_model_class("fengshun20_onnx")
    assert "forward" not in vars(cls)


@pytest.mark.parametrize("features", [
    {"hourr": {"divisor": 1.0}},        # misspelled input name
    {"hour": {"divsor": 1.0}},          # misspelled field name
    {"hour": 1.0},                      # not a dict
])
def test_bad_time_features_rejected(features):
    cls = get_model_class("fengshun20_onnx")
    with pytest.raises(ValueError):
        type("BadModel", (cls,), {"time_features": features})()
