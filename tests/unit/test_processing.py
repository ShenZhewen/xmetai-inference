# -*- coding: utf-8 -*-
"""processing/ 的单元测试：纯函数 + tmp_path 造的 mean.nc/std.nc，不碰真实模型和 zarr。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from xmetai_inference.processing import pipeline as pl
from xmetai_inference.processing import tensor_processors as tp


class _FakeModel:
    """Processor 只读通道契约，这里给个最小假模型类。"""

    input_channels = ["a", "b"]
    output_channels = ["a", "b"]


def _context(channels=("a", "b"), lat=None, model_path=None):
    return tp.TensorProcessorContext(
        channel_names=list(channels),
        latitudes=np.array([-10.0, 10.0]) if lat is None else lat,
        longitudes=np.array([0.0, 120.0, 240.0]),
        expected_shape=(2, 3),
        model_path=model_path,
    )


def _write_stats(directory, mean, std, channels=("a", "b")):
    """模型统计量约定：mean.nc/std.nc 放在模型文件旁边，带 channel 坐标。"""
    for name, values in (("mean.nc", mean), ("std.nc", std)):
        xr.DataArray(
            np.asarray(values, dtype="f4"),
            dims=["channel"],
            coords={"channel": list(channels)},
        ).to_netcdf(Path(directory) / name)


def test_base_name_strips_level_suffix():
    assert tp._base_name("z50") == "z"
    assert tp._base_name("q700") == "q"
    assert tp._base_name("t2m") == "t2m"
    assert tp._base_name("msl") == "msl"


def test_alignment_flags_ascending_lat_for_flip():
    flip, roll = tp._latlon_alignment(
        np.array([-10.0, 10.0]), np.array([0.0, 120.0, 240.0]), (2, 3))
    assert flip is True
    assert roll == 0


def test_alignment_leaves_northdown_0360_grid_untouched():
    lat = np.array([10.0, -10.0])
    lon = np.array([0.0, 120.0, 240.0])
    flip, roll = tp._latlon_alignment(lat, lon, (2, 3))
    assert flip is False
    assert roll == 0
    aligned_lat, aligned_lon = tp.aligned_coordinates(lat, lon, (2, 3))
    assert aligned_lat.tolist() == lat.tolist()
    assert aligned_lon.tolist() == lon.tolist()


def test_alignment_rejects_resolution_mismatch():
    with pytest.raises(ValueError, match="网格"):
        tp._latlon_alignment(np.zeros(2), np.zeros(3), (3, 3))


def test_alignment_rolls_offset_lon_to_start_at_zero():
    """偏移网格要把经度 0 挪到下标 0（roll 方向不得反）。"""
    lat = np.array([10.0, -10.0])
    lon = np.array([120.0, 240.0, 0.0])
    _, aligned_lon = tp.aligned_coordinates(lat, lon, (2, 3))
    assert aligned_lon.tolist() == [0.0, 120.0, 240.0]


def test_tensor_processor_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="未知参数"):
        tp.TensorUnitConvert(_context(), scales={}, bogus=1)


def test_unit_convert_scales_by_base_name():
    conv = tp.TensorUnitConvert(
        _context(channels=("z500", "q700")), scales={"q": 1000.0})
    out = conv(torch.ones(1, 2, 2, 2))
    assert torch.all(out[0, 0] == 1.0)
    assert torch.all(out[0, 1] == 1000.0)


def test_unit_convert_noop_without_scales():
    conv = tp.TensorUnitConvert(_context(), scales={})
    value = torch.ones(1, 2, 2, 2)
    assert conv(value) is value


def test_geometry_processor_flips_rows_for_ascending_lat():
    ctx = _context(channels=("a",), lat=np.array([-10.0, 10.0]))
    geo = tp.TensorGeometry(ctx)
    value = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])  # [1,1,2,3]
    out = geo(value)
    assert out[0, 0, 0].tolist() == [4.0, 5.0, 6.0]


def test_normalize_applies_log1p_then_zscore(tmp_path):
    _write_stats(tmp_path, mean=[1.0, 2.0], std=[2.0, 4.0])
    norm = tp.TensorNormalize(
        _context(model_path=str(tmp_path / "model.pt")), log1p_channels=["b"])
    value = torch.tensor([[[[2.0, 3.0], [4.0, 5.0]],
                          [[0.0, 7.0], [8.0, 9.0]]]])  # [1,2,2,2]
    expected_a = (value[0, 0] - 1.0) / 2.0
    expected_b = (torch.log1p(value[0, 1]) - 2.0) / 4.0  # 先算：float32 输入会被原地改写
    out = norm(value)
    assert torch.allclose(out[0, 0], expected_a)
    assert torch.allclose(out[0, 1], expected_b)


def test_normalize_rejects_negative_log1p_channel(tmp_path):
    _write_stats(tmp_path, mean=[1.0, 2.0], std=[2.0, 4.0])
    norm = tp.TensorNormalize(
        _context(model_path=str(tmp_path / "model.pt")), log1p_channels=["b"])
    with pytest.raises(ValueError, match="log1p"):
        norm(torch.ones(1, 2, 2, 2) * torch.tensor([[[[1.0]], [[-1.0]]]]))


@pytest.mark.parametrize("mean,std,channels,match", [
    ([1.0, 2.0], [2.0, 0.0], ("a", "b"), "非正标准差"),
    ([1.0, 2.0], [2.0, 4.0], ("b", "a"), "channel 顺序"),
])
def test_normalize_rejects_bad_stats(tmp_path, mean, std, channels, match):
    _write_stats(tmp_path, mean=mean, std=std, channels=channels)
    with pytest.raises(ValueError, match=match):
        tp.TensorNormalize(_context(model_path=str(tmp_path / "model.pt")))


def test_normalize_rejects_mean_std_length_mismatch(tmp_path):
    xr.DataArray(np.array([1.0, 2.0], dtype="f4"), dims=["channel"],
                 coords={"channel": ["a", "b"]}).to_netcdf(tmp_path / "mean.nc")
    xr.DataArray(np.array([2.0, 4.0, 8.0], dtype="f4"), dims=["channel"],
                 coords={"channel": ["a", "b", "c"]}).to_netcdf(tmp_path / "std.nc")
    with pytest.raises(ValueError, match="不一致"):
        tp.TensorNormalize(_context(model_path=str(tmp_path / "model.pt")))


def test_normalize_requires_model_path():
    with pytest.raises(ValueError, match="model_path"):
        tp.TensorNormalize(_context())


def test_fill_missing_fills_by_default_rule():
    fill = tp.TensorFillMissing(
        _context(), rules={"*": {"method": "constant", "value": 0.0}})
    value = torch.tensor([[[[1.0, float("nan")], [3.0, 4.0]],
                          [[float("inf"), 6.0], [7.0, 8.0]]]])  # [1,2,2,2]
    out = fill(value)
    assert torch.isfinite(out).all()
    assert out[0, 0, 0, 1] == 0.0
    assert out[0, 1, 0, 0] == 0.0


def test_fill_missing_keeps_unconfigured_channels():
    fill = tp.TensorFillMissing(
        _context(),
        rules={"a": {"method": "constant", "value": 0.0}},
        unconfigured="keep",
    )
    value = torch.tensor([[[[float("nan")], [1.0]],
                          [[float("nan")], [1.0]]]])  # [1,2,1,1]
    out = fill(value)
    assert torch.isnan(out[0, 1, 0, 0]), "keep 语义：未配置通道的缺测原样保留"


def test_fill_missing_errors_on_unconfigured_channel():
    fill = tp.TensorFillMissing(
        _context(),
        rules={"a": {"method": "constant", "value": 0.0}},
        unconfigured="error",
    )
    with pytest.raises(ValueError, match="b"):
        fill(torch.full((1, 2, 1, 1), float("nan")))


def test_fill_missing_rejects_unknown_method():
    fill = tp.TensorFillMissing(
        _context(), rules={"a": {"method": "nearest", "value": 0.0}})
    with pytest.raises(ValueError, match="不受支持"):
        fill(torch.tensor([[[[float("nan")], [1.0]], [[1.0], [1.0]]]]))


def test_fill_missing_rejects_bad_unconfigured_option():
    with pytest.raises(ValueError, match="unconfigured"):
        tp.TensorFillMissing(_context(), unconfigured="ignore")


def test_build_tensor_processors_accepts_string_and_dict_specs():
    processors = tp.build_tensor_processors(["unit_convert"], _context())
    assert len(processors) == 1


def test_build_tensor_processors_rejects_unknown_name():
    with pytest.raises(ValueError, match="未知 Tensor Processor"):
        tp.build_tensor_processors([{"name": "nope"}], _context())


def test_unit_scales_from_specs():
    specs = [
        {"name": "unit_convert", "scales": {"q": 1000.0}},
        {"name": "geometry"},
    ]
    assert tp.unit_scales_from_specs(specs) == {"q": 1000.0}
    assert tp.unit_scales_from_specs(["unit_convert"]) == {}
    assert tp.unit_scales_from_specs([]) == {}


def test_build_pipeline_rejects_input_specs():
    with pytest.raises(ValueError, match="model_processing.input"):
        pl.build_pipeline(["normalize"], _FakeModel, None)


def test_build_pipeline_rejects_unknown_processor():
    with pytest.raises(ValueError, match="未知 Processor"):
        pl.build_pipeline([], _FakeModel, None, recurrent_specs=["nope"])


def test_build_pipeline_rejects_stage_mismatch():
    with pytest.raises(ValueError, match="阶段"):
        pl.build_pipeline([], _FakeModel, None, output_specs=["zero_channels"])


def test_zero_channels_writes_literal_zero():
    pipe = pl.build_pipeline(
        [], _FakeModel, None,
        recurrent_specs=[{"name": "zero_channels", "channels": ["a"]}])
    value = torch.ones(1, 2, 2, 2)
    out = pipe.process_recurrent(value)
    assert torch.all(out[0, 0] == 0.0), "清零必须是归一化空间字面 0"
    assert torch.all(out[0, 1] == 1.0), "清单外的通道不许动"


def test_denormalize_restores_units_expm1_and_clamps(tmp_path):
    _write_stats(tmp_path, mean=[0.0, 0.0], std=[1.0, 1.0])
    pipe = pl.build_pipeline(
        [], _FakeModel, None, model_path=str(tmp_path / "model.pt"),
        output_specs=[{"name": "denormalize",
                       "expm1_channels": ["a"],
                       "nonnegative": ["b"]}])
    value = torch.tensor([[[[1.0, 25.0], [0.0, 0.0]],
                          [[-3.0, 0.5], [1.0, 2.0]]]])  # [1,2,2,2]
    out = pipe.process_output(value)
    assert torch.allclose(out[0, 0, 0, 0], torch.expm1(torch.tensor(1.0)))
    assert torch.allclose(out[0, 0, 0, 1], torch.expm1(torch.tensor(20.0))), \
        "expm1 前先 clamp 到 exp_clip_max=20"
    assert out[0, 1, 0, 0] == 0.0, "nonnegative 通道负值截 0"
    assert out[0, 1, 0, 1] == 0.5


def test_denormalize_rejects_stats_channel_mismatch(tmp_path):
    _write_stats(tmp_path, mean=[0.0, 0.0], std=[1.0, 1.0])

    class _ThreeChannelModel(_FakeModel):
        input_channels = ["a", "b", "c"]
        output_channels = ["a", "b", "c"]

    with pytest.raises(ValueError, match="统计量通道数"):
        pl.build_pipeline(
            [], _ThreeChannelModel, None, model_path=str(tmp_path / "model.pt"),
            output_specs=["denormalize"])


def test_build_dataset_batch_parses_shape_and_times():
    ts = pd.Timestamp("2025-01-02T00:00")
    batch = {
        "inputs": np.zeros((1, 1, 2, 2, 2), dtype="f4"),
        "times": np.array([ts.value], dtype="int64"),
    }
    values, init_times = pl.build_dataset_batch(batch)
    assert values.ndim == 5
    assert values.dtype == torch.float32
    assert init_times == [ts]


def test_build_dataset_batch_rejects_wrong_ndim():
    with pytest.raises(ValueError, match=r"\[B,T,C,H,W\]"):
        pl.build_dataset_batch({"inputs": np.zeros((2, 2)), "times": np.array([0])})
