# -*- coding: utf-8 -*-
"""``tests/integration/realdata_common.py`` 的单元测试。

把冻结的阈值与分档判据钉成断言 —— 判据悄悄变松 = 集成档悄悄变绿，这是回归测试
最怕的失效方式。另给 classify / field_metrics / align_grid 的边界行为做小样本
用例。只测纯函数，不碰任何 fixture。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# realdata_common 在 tests/integration/ 下（包外共享模块），手动挂进 sys.path。
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
    """阈值从老 util/compare_outputs.py 冻结，两个 realdata 档共用；改之前先读
    realdata_common 的模块 docstring。"""
    assert getattr(realdata_common, name) == expected


def test_verdict_rank_is_total_order():
    assert realdata_common.VERDICT_RANK == {
        "IDENTICAL": 0, "CLOSE": 1, "DRIFT": 2, "DIFFER": 3,
    }


def test_classify_zero_diff_is_identical():
    assert realdata_common.classify(0.0, 100.0, 0.0) == "IDENTICAL"


def test_classify_physical_resolvability_outranks_fraction():
    """max|diff| 超场变率 1% 直接 DIFFER，哪怕超容差点占比极小。"""
    assert realdata_common.classify(max_diff=0.02, ref_std=1.0,
                                    exceed_fraction=1e-9) == "DIFFER"


def test_classify_ladder_by_exceed_fraction():
    """占比分档：CLOSE < 1e-3 < DRIFT < 0.5 < DIFFER（幅度在线内时）。"""
    assert realdata_common.classify(0.005, 1.0, 9.9e-4) == "CLOSE"
    assert realdata_common.classify(0.005, 1.0, 1.1e-3) == "DRIFT"
    assert realdata_common.classify(0.005, 1.0, 0.4999) == "DRIFT"
    assert realdata_common.classify(0.005, 1.0, 0.5) == "DIFFER"


def test_classify_survives_zero_ref_std():
    """ref_std=0（常数场）不能炸。"""
    assert realdata_common.classify(1e-8, 0.0, 0.0) == "CLOSE"


def test_field_metrics_isolates_nan_mismatch():
    """NaN↔数值的格子单独计数、不进超容差统计。"""
    a = np.array([[1.0, np.nan], [1.0, 2.0]])
    b = np.array([[1.0, 1.0], [1.0, 2.0]])
    m = realdata_common.field_metrics(a, b)
    assert m["nan_mismatch"] == 1
    assert m["verdict"] == "IDENTICAL"
    assert m["exceed_frac"] == 0.0


def test_field_metrics_all_disjoint_yields_differ():
    """两侧没有共同有限格点必须判 DIFFER，而不是除零。"""
    a = np.full((2, 2), np.nan)
    b = np.ones((2, 2))
    m = realdata_common.field_metrics(a, b)
    assert m["verdict"] == "DIFFER"


def test_align_grid_flips_lat_and_sorts_lon():
    """南→北的纬度翻成北→南（数据跟着翻），经度重排成 0..360 升序。"""
    values = np.array([[1, 2, 3],
                       [4, 5, 6]], dtype="f8")     # lat 2 行、lon 3 列
    lat = np.array([-10.0, 10.0])
    lon = np.array([300.0, 30.0, 100.0])
    out, lat_out, lon_out = realdata_common.align_grid(values, lat, lon)
    assert lat_out.tolist() == [10.0, -10.0]
    assert lon_out.tolist() == [30.0, 100.0, 300.0]
    assert out.tolist() == [[5, 6, 4], [2, 3, 1]]


def test_align_grid_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        realdata_common.align_grid(np.zeros((2, 3)), np.zeros(3), np.zeros(3))
