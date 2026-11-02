# -*- coding: utf-8 -*-
"""Unit tests for ``configs/weather_era_0p25_fgvp.py``: consistency between the
configuration and the model contract.

A config is a series of declarations; most mistakes raise nothing and merely fail
silently (a misspelled unit-conversion key means no conversion happens; a misspelled
``vars`` entry means one fewer variable is written). These tests check **consistency
between declarations** and act as a change detector; whether the values themselves
are correct is verified against the reference by ``test_fgvp_realdata.py``.
"""
import re
from datetime import datetime

import pytest

from xmetai_inference.configs import InferConfig, load_config, parse_times
from xmetai_inference.configs import FUXI_ENS_CHANNELS
from xmetai_inference.utils.model_factory import get_model_class

#: Strip the level suffix from channel names (z50 -> z); trailing-letter names
#: (t2m/u10m) and pure-letter names (ssr/msl/tp) are left unchanged.
_BASE_NAME_RE = re.compile(r"([A-Za-z]+)(\d+)")

EXPECTED_VARS = {
    "z500", "q700", "t700", "t850", "u850", "v850",
    "u10m", "v10m", "t2m", "msl", "tp",
}


def _base_names(channels):
    out = set()
    for name in channels:
        match = _BASE_NAME_RE.fullmatch(name)
        out.add(match.group(1) if match else name)
    return out


@pytest.fixture(scope="module")
def cfg():
    return load_config("weather_era_0p25_fgvp")


@pytest.fixture(scope="module")
def model_cls(cfg):
    return get_model_class(cfg.model_class)


def test_loads_as_inferconfig_named_fgvp(cfg):
    assert isinstance(cfg, InferConfig)
    assert cfg.name == "weather_era_0p25_fgvp"


def test_model_class_is_the_gdn2_backbone(cfg, model_cls):
    """Must resolve to the deterministic gdn2 backbone, not the ensemble mamba_ens variant with the same channel layout."""
    assert cfg.model_class == "iwc_fgvp_gdn2_onnx"
    assert model_cls.__name__ == "IwcFgvpGdn2Model"
    assert model_cls.windowed_output is False


def test_hour_interval_matches_model(cfg, model_cls):
    assert cfg.hour_interval == 6
    assert model_cls.hour_interval == 6


def test_members_is_one_for_deterministic_model(cfg, model_cls):
    """The graph emits a single frame and has no perturbation path; members>1 only stores the same result N times."""
    assert model_cls.forecast_type == "deterministic"
    assert model_cls.members == 1
    assert cfg.members == 1


def test_gpu_state_is_off_by_default(cfg, monkeypatch):
    """GPU persistence is incompatible with this model's two-input-frame,
    one-output-frame window shape, so it must be off by default.

    Reload instead of using the fixture: gpu_state is derived from an environment
    variable at module exec time. The file loaded must be the one ``cfg`` came from
    (``cfg._source_path``), not a hard-coded "fgvp" -- otherwise the test always
    checks the packaged original config rather than the object under test.
    """
    monkeypatch.delenv("XMETAI_GPU_STATE", raising=False)
    fresh = load_config(cfg._source_path)
    assert fresh.gpu_state is False


def test_env_override_only_accepts_literal_one(cfg, monkeypatch):
    """``XMETAI_GPU_STATE`` counts as on only for the literal "1"; true/yes/on are silently treated as unset."""
    monkeypatch.setenv("XMETAI_GPU_STATE", "1")
    assert load_config(cfg._source_path).gpu_state is True

    for falsy in ("true", "yes", "on", "0"):
        monkeypatch.setenv("XMETAI_GPU_STATE", falsy)
        assert load_config(cfg._source_path).gpu_state is False, (
            f"{falsy!r} 不该被当成开"
        )


def test_declared_vars_are_real_model_output_channels(cfg, model_cls):
    """A misspelled ``vars`` entry raises nothing and simply drops that variable from the output -- often noticed only after a full year of runs."""
    declared = {v.strip() for v in cfg.vars.split(",") if v.strip()}
    assert declared == EXPECTED_VARS, "落盘变量清单变了，确认是有意的再改这里"

    available = set(model_cls.output_channels)
    missing = declared - available
    assert not missing, f"vars 里有模型不输出的通道：{sorted(missing)}"


def test_unit_convert_keys_are_real_base_variables(cfg, model_cls):
    """Conversion keys are base variable names (q rather than q700); a misspelling silently skips the conversion."""
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    scales = processors["unit_convert"]["scales"]

    assert set(scales) == {"q", "ssr", "ssrd", "fdir", "ttr", "tp"}, (
        "换算变量清单变了，确认是有意的"
    )

    unknown = set(scales) - _base_names(model_cls.input_channels)
    assert not unknown, f"换算表里有对不上任何输入通道的键：{sorted(unknown)}"


def test_unit_convert_scales_are_the_agreed_conventions(cfg):
    """The factors are measured, agreed conventions pinned against unintended changes; whether the conventions are correct is covered by the realdata suite."""
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    scales = processors["unit_convert"]["scales"]

    assert scales["q"] == 1000.0, "q：store kg/kg → 模型 g/kg"
    assert scales["tp"] == 6000.0, "tp：累积 m → 模型 mm"
    for field in ("ssr", "ssrd", "fdir", "ttr"):
        assert scales[field] == pytest.approx(6.0 / 3600.0), f"{field}：辐射场换算"


def test_fill_missing_targets_real_channel_and_is_strict(cfg, model_cls):
    """Rule variables must be real channels (a misspelling means silent no-fill, and NaN
    flows into the model); unconfigured="error" is a safety property: a channel without
    a rule should fail rather than proceed when data is missing.
    """
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    fill = processors["fill_missing"]

    assert set(fill["rules"]) == {"sst"}, "填充规则的目标变了，确认是有意的"
    unknown = set(fill["rules"]) - _base_names(model_cls.input_channels)
    assert not unknown, f"填充规则里有对不上任何输入通道的变量：{sorted(unknown)}"

    assert fill["rules"]["sst"]["method"] == "constant"
    assert fill["rules"]["sst"]["value"] == pytest.approx(287.066)
    assert fill["unconfigured"] == "error"


def test_model_processing_is_empty(cfg):
    """Normalization is baked into the graph; attaching any model-side Processor would stack on top of it and yield wrong values without raising."""
    input_specs, recurrent_specs, output_specs = cfg.processor_specs()
    assert list(input_specs) == []
    assert list(recurrent_specs) == []
    assert list(output_specs) == []


def test_cuda_devices_cover_every_rank(cfg):
    """Multi-GPU branches index cuda_devices[rank]; being one short raises IndexError only on multi-GPU startup."""
    devices = [d for d in str(cfg.cuda_devices).split(",") if d.strip()]
    assert cfg.gpus >= 1
    assert len(devices) >= cfg.gpus, (
        f"gpus={cfg.gpus} 但只给了 {len(devices)} 个卡号：{cfg.cuda_devices!r}"
    )


def test_times_expand_to_four_daily_inits(cfg):
    """The ``:24`` suffix on ``times`` is not redundant: without it the range expands at hour_interval=6."""
    inits = parse_times(cfg.times, cfg.hour_interval)
    assert inits == [
        datetime(2025, 1, 2), datetime(2025, 1, 3),
        datetime(2025, 1, 4), datetime(2025, 1, 5),
    ], f"起报列表变了：{inits}"


def test_steps_and_interval_span_the_expected_lead_time(cfg):
    assert cfg.hour_interval == 6
    assert cfg.steps == 60
    assert cfg.steps * cfg.hour_interval == 15 * 24, "预报跨度应为 15 天"


def test_dynamic_stores_are_the_expected_trio(cfg):
    """The three dynamic stores; compared as a set -- channels are selected by name, so order does not affect results."""
    paths = cfg.dataset["paths"]
    assert cfg.dataset["type"] == "processed_multi_zarr"
    assert len(paths) == 3

    names = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in paths}
    assert names == {
        "era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
        "era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
    }, f"数据源名单变了：{sorted(names)}"

    roots = {p.replace("\\", "/").rsplit("/", 1)[0] for p in paths}
    assert len(roots) == 1, f"三个 store 不在同一个根下：{sorted(roots)}"
