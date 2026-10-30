# -*- coding: utf-8 -*-
"""backends/onnx.py 时间标量编码（time_features）的单元测试。

锁两件事：

  * 默认口径（Tongyun-CRA：hour/24、doy/365 且截 365）与改造前的硬编码**逐位相同**；
  * 模型类只声明 ``time_features`` 就能改编码，不必再重写 forward（fengshun2.0 喂原值）。

纯函数级：不加载图、不建 session；没有 onnxruntime 的环境整体跳过。
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnxruntime")

from xmetai_inference.backends import onnx as onnx_backend
from xmetai_inference.utils.model_factory import get_model_class


def _legacy_feeds(valid_time, step, perturb_std_scale=1.0):
    """改造前 ``_scalar_feeds`` 的逐字实现（对拍基准）。"""
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
    pd.Timestamp("2024-02-29 00:00"),   # 闰日
    pd.Timestamp("2024-12-31 06:00"),   # 闰年最后一天：doy=366，旧实现截到 365
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
    assert feeds["hour"].tolist() == [6.0]        # 原值，不除 24
    assert feeds["doy"].tolist() == [366.0]       # 闰年不截 365
    assert feeds["step"].tolist() == [12.0]       # step 沿用默认
    assert feeds["perturb_std_scale"].tolist() == [1.0]


def test_fengshun20_no_longer_overrides_forward():
    cls = get_model_class("fengshun20_onnx")
    assert "forward" not in vars(cls)


@pytest.mark.parametrize("features", [
    {"hourr": {"divisor": 1.0}},        # 输入名拼错
    {"hour": {"divsor": 1.0}},          # 字段名拼错
    {"hour": 1.0},                      # 不是 dict
])
def test_bad_time_features_rejected(features):
    cls = get_model_class("fengshun20_onnx")
    with pytest.raises(ValueError):
        type("BadModel", (cls,), {"time_features": features})()
