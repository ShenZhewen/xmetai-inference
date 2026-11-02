# -*- coding: utf-8 -*-
"""Unit tests for ``tests/integration/realdata_common.py``.

The frozen thresholds and tiered verdict rules are pinned as assertions -- criteria
quietly loosening means the integration suite quietly passes, the most feared failure
mode for a regression test. Small-sample cases also cover the boundary behavior of
classify / field_metrics / align_grid. Only pure functions are tested; no fixtures
are touched.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# realdata_common lives in tests/integration/ (a module shared outside the package); add it to sys.path manually.
_INTEGRATION_DIR = Path(__file__).resolve().parents[1] / "integration"
if str(_INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_DIR))

import realdata_common  # noqa: E402


@pytest.mark.parametrize("name, expected", [
    ("RTOL", 1e-5),
    ("ATOL", 1e-6),
    ("EXCEED_FRACTION_CLOSE", 1e-3),
    ("EXCEED_FRACTION_DRIFT", 0.5),
    ("DRIFT_STD_FRACTION", 1e-2),
])
def test_frozen_thresholds_unchanged(name, expected):
    """Thresholds are frozen from the old util/compare_outputs.py and shared by both
    realdata suites; read the realdata_common module docstring before changing them.
    """
    assert getattr(realdata_common, name) == expected


def test_verdict_rank_is_total_order():
    assert realdata_common.VERDICT_RANK == {
        "IDENTICAL": 0, "CLOSE": 1, "DRIFT": 2, "DIFFER": 3,
    }


def test_classify_zero_diff_is_identical():
    assert realdata_common.classify(0.0, 100.0, 0.0) == "IDENTICAL"


def test_classify_physical_resolvability_outranks_fraction():
    """max|diff| above 1% of field variability is DIFFER outright, even if the fraction of cells exceeding tolerance is tiny."""
    assert realdata_common.classify(max_diff=0.02, ref_std=1.0,
                                    exceed_fraction=1e-9) == "DIFFER"


def test_classify_ladder_by_exceed_fraction():
    """Fraction-based tiers: CLOSE < 1e-3 < DRIFT < 0.5 < DIFFER (when the amplitude is within range)."""
    assert realdata_common.classify(0.005, 1.0, 9.9e-4) == "CLOSE"
    assert realdata_common.classify(0.005, 1.0, 1.1e-3) == "DRIFT"
    assert realdata_common.classify(0.005, 1.0, 0.4999) == "DRIFT"
    assert realdata_common.classify(0.005, 1.0, 0.5) == "DIFFER"


def test_classify_survives_zero_ref_std():
    """ref_std=0 (a constant field) must not crash."""
    assert realdata_common.classify(1e-8, 0.0, 0.0) == "CLOSE"


def test_field_metrics_isolates_nan_mismatch():
    """Cells where NaN meets a value are counted separately and excluded from the tolerance statistics."""
    a = np.array([[1.0, np.nan], [1.0, 2.0]])
    b = np.array([[1.0, 1.0], [1.0, 2.0]])
    m = realdata_common.field_metrics(a, b)
    assert m["nan_mismatch"] == 1
    assert m["verdict"] == "IDENTICAL"
    assert m["exceed_frac"] == 0.0


def test_field_metrics_all_disjoint_yields_differ():
    """Two fields with no common finite grid cell must be classified DIFFER rather than dividing by zero."""
    a = np.full((2, 2), np.nan)
    b = np.ones((2, 2))
    m = realdata_common.field_metrics(a, b)
    assert m["verdict"] == "DIFFER"


def test_align_grid_flips_lat_and_sorts_lon():
    """Flip south->north latitude to north->south (the data is flipped with it) and reorder longitude ascending over 0..360."""
    values = np.array([[1, 2, 3],
                       [4, 5, 6]], dtype="f8")     # 2 latitude rows, 3 longitude columns
    lat = np.array([-10.0, 10.0])
    lon = np.array([300.0, 30.0, 100.0])
    out, lat_out, lon_out = realdata_common.align_grid(values, lat, lon)
    assert lat_out.tolist() == [10.0, -10.0]
    assert lon_out.tolist() == [30.0, 100.0, 300.0]
    assert out.tolist() == [[5, 6, 4], [2, 3, 1]]


def test_align_grid_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        realdata_common.align_grid(np.zeros((2, 3)), np.zeros(3), np.zeros(3))
