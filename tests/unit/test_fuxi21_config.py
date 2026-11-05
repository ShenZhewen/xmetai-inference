# -*- coding: utf-8 -*-
"""Unit tests for ``configs/weather_era_0p25_fuxi21.py``: consistency between the
configuration and the FuXi-2.1 contract (a change detector, following
``test_tongyun_config.py``). The biggest difference from the Tongyun family: normalization
is **not** in the graph and is performed explicitly by config-side
normalize/denormalize, which makes the configuration fragile. The key invariants:
the ``log1p`` <-> ``expm1`` pairing (breaking either side makes values silently
wrong), recurrent zeroing covering all 5 diagnostic channels, and input/output
normalization using the same pair of mean/std files. Value correctness is covered
by ``test_fuxi_realdata.py``.
"""
import re
from datetime import datetime

import pytest

from xmetai_inference.configs import InferConfig, load_config, parse_times
from xmetai_inference.configs.weather_era_0p25_fuxi21 import FUXI21_CHANNELS
from xmetai_inference.utils.model_factory import get_model_class

_BASE_NAME_RE = re.compile(r"([A-Za-z]+)(\d+)")

EXPECTED_VARS = {
    "z500", "q700", "t700", "t850", "u850", "v850",
    "u10m", "v10m", "t2m", "d2m", "msl", "tp",
}

DIAGNOSTIC_CHANNELS = {"ssr", "ssrd", "fdir", "ttr", "tp"}


def _base_names(channels):
    out = set()
    for name in channels:
        match = _BASE_NAME_RE.fullmatch(name)
        out.add(match.group(1) if match else name)
    return out


@pytest.fixture(scope="module")
def cfg():
    return load_config("weather_era_0p25_fuxi21")


@pytest.fixture(scope="module")
def model_cls(cfg):
    return get_model_class(cfg.model_class)


def test_loads_as_inferconfig_named_fuxi21(cfg):
    assert isinstance(cfg, InferConfig)
    assert cfg.name == "weather_era_0p25_fuxi21"


def test_model_class_is_the_pt2_fuxi21(cfg, model_cls):
    assert cfg.model_class == "fuxi21_pt2"
    assert model_cls.__name__ == "Fuxi21Pt2Model"
    assert model_cls.backend == "pt2"


def test_deterministic_single_member(cfg, model_cls):
    """FuXi-2.1 is a deterministic model: the graph has no perturbation path, so members must be 1 on both sides."""
    assert model_cls.forecast_type == "deterministic"
    assert model_cls.members == 1
    assert cfg.members == 1


def test_channel_contract_is_85_symmetric(cfg, model_cls):
    """Input and output channels must both use the 85-channel layout and be symmetric (the torch.export graph is isomorphic on input and output)."""
    assert model_cls.input_channels == list(FUXI21_CHANNELS)
    assert model_cls.output_channels == list(FUXI21_CHANNELS)
    assert len(FUXI21_CHANNELS) == 85


def test_hour_interval_and_history_match_model(cfg, model_cls):
    assert cfg.hour_interval == 6
    assert model_cls.hour_interval == 6
    assert model_cls.history_steps == 2, "切片窗口与 golden 对拍都按 2 帧历史假设"


def test_dataset_processor_pipeline_order(cfg):
    """The order must be preserved: unit conversion -> geometry -> normalize -> fill_missing.

    normalize depends on unit conversion running first (mean/std are statistics in
    model units); fill_missing must run after normalization (filling 0 fills
    normalized values). A wrong order raises nothing and makes all values wrong.
    """
    names = [p["name"] for p in cfg.dataset["processors"]]
    assert names == ["unit_convert", "geometry", "normalize", "fill_missing"], (
        f"前处理序列变了：{names}"
    )


def test_log1p_expm1_pairing(cfg):
    """``log1p`` <-> ``expm1`` pairing invariant -- tp is log1p-transformed on the input
    side, so the output side must apply expm1 to restore it and keep it nonnegative.
    Changing only one side makes values silently wrong.
    """
    ds_processors = {p["name"]: p for p in cfg.dataset["processors"]}
    assert set(ds_processors["normalize"]["log1p_channels"]) == {"tp"}, (
        "输入侧 log1p 的通道清单变了，输出侧的 expm1 是否同步？"
    )

    _, _, output_specs = cfg.processor_specs()
    assert len(output_specs) == 1
    denorm = output_specs[0]
    assert denorm["name"] == "denormalize"
    assert set(denorm["expm1_channels"]) == {"tp"}, "expm1 必须与 log1p 严格配对"
    assert set(denorm["nonnegative"]) == {"tp"}, "log1p 通道输出必须截非负"


def test_normalize_and_denormalize_share_stats_files(cfg):
    """Input and output normalization must use the same pair of mean.nc/std.nc files."""
    ds_processors = {p["name"]: p for p in cfg.dataset["processors"]}
    normalize = ds_processors["normalize"]
    _, _, output_specs = cfg.processor_specs()
    denorm = output_specs[0]

    assert normalize["mean_file"] == denorm["mean_file"]
    assert normalize["std_file"] == denorm["std_file"]


def test_recurrent_zero_channels_covers_diagnostics(cfg, model_cls):
    """Rollout feedback must zero all 5 cumulative diagnostic channels -- otherwise the
    accumulated value from the previous step is fed into the next, invisible at short
    lead times but silently degrading at long lead times.
    """
    input_specs, recurrent_specs, _ = cfg.processor_specs()
    assert list(input_specs) == []
    assert len(recurrent_specs) == 1
    zero = recurrent_specs[0]
    assert zero["name"] == "zero_channels"
    assert set(zero["channels"]) == DIAGNOSTIC_CHANNELS, (
        f"清零清单变了：{sorted(zero['channels'])} —— 确认是有意的再改这里"
    )
    unknown = set(zero["channels"]) - _base_names(model_cls.input_channels)
    assert not unknown, f"清零清单里有对不上任何输入通道的名字：{sorted(unknown)}"


def test_unit_convert_scales_are_the_agreed_conventions(cfg, model_cls):
    """Conversion factors are pinned (same conventions as the Tongyun family, sharing the same source stores)."""
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    scales = processors["unit_convert"]["scales"]

    assert set(scales) == {"q", "ssr", "ssrd", "fdir", "ttr", "tp"}
    unknown = set(scales) - _base_names(model_cls.input_channels)
    assert not unknown, f"换算表里有对不上任何输入通道的键：{sorted(unknown)}"

    assert scales["q"] == 1000.0, "q：store kg/kg → 模型 g/kg"
    assert scales["tp"] == 6000.0, "tp：累积 m → 模型 mm"
    for field in ("ssr", "ssrd", "fdir", "ttr"):
        assert scales[field] == pytest.approx(6.0 / 3600.0), f"{field}：辐射场换算"


def test_fill_missing_is_constant_zero_after_normalization(cfg):
    """Filling with 0 happens after normalization; missing-data combinations not yet
    surveyed are kept for now (the fuxi21 source stores add soil/wave compared with
    Tongyun, unlike Tongyun's error choice).
    """
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    fill = processors["fill_missing"]
    assert set(fill["rules"]) == {"*"}
    assert fill["rules"]["*"]["method"] == "constant"
    assert fill["rules"]["*"]["value"] == 0.0
    assert fill["unconfigured"] == "keep"


def test_declared_vars_are_real_model_output_channels(cfg, model_cls):
    declared = {v.strip() for v in cfg.vars.split(",") if v.strip()}
    assert declared == EXPECTED_VARS, "落盘变量清单变了，确认是有意的再改这里"

    missing = declared - set(model_cls.output_channels)
    assert not missing, f"vars 里有模型不输出的通道：{sorted(missing)}"


def test_dynamic_stores_are_the_expected_five(cfg):
    """fuxi21 uses 5 dynamic stores (soil and wave in addition to the Tongyun family) -- the source of the 85 channels."""
    paths = cfg.dataset["paths"]
    assert cfg.dataset["type"] == "processed_multi_zarr"
    assert len(paths) == 5
    names = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in paths}
    assert names == {
        "era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
        "era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
        "era5_soil_2025.01-2026.07.c4.p25.h6.zarr",
        "era5_wave_2025.01-2026.07.c5.p25.h6.zarr",
    }, f"数据源名单变了：{sorted(names)}"
    roots = {p.replace("\\", "/").rsplit("/", 1)[0] for p in paths}
    assert len(roots) == 1, f"5 个 store 不在同一个根下：{sorted(roots)}"


def test_times_expand_to_four_daily_inits(cfg):
    inits = parse_times(cfg.times, cfg.hour_interval)
    assert inits == [
        datetime(2025, 1, 2), datetime(2025, 1, 3),
        datetime(2025, 1, 4), datetime(2025, 1, 5),
    ], f"起报列表变了：{inits}"


def test_steps_and_interval_span_the_expected_lead_time(cfg):
    assert cfg.hour_interval == 6
    assert cfg.steps == 60
    assert cfg.steps * cfg.hour_interval == 15 * 24, "预报跨度应为 15 天"
