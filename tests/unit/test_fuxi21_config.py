# -*- coding: utf-8 -*-
"""``configs/weather_era_0p25_fuxi21.py`` 的单元测试：配置与 FuXi-2.1 契约之间的一致性（变更探测器，
思路同 ``test_fgvp_config.py``）。fuxi21 与 fgvp 族最大的不同：归一化**不在图里**，
由 config 侧的 normalize/denormalize 显式做，配置面因此危险 —— 最要紧的不变量：
``log1p`` ↔ ``expm1`` 配对（错一边数值静默全错）、recurrent 清零覆盖全部 5 个
诊断通道、进出归一化用同一对 mean/std 文件。值对不对由 ``test_fuxi_realdata.py``
对拍兜底。
"""
import re
from datetime import datetime

import pytest

from xmetai_inference.configs import InferConfig, load_config, parse_times
from xmetai_inference.configs import FUXI21_CHANNELS
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
    """FuXi-2.1 是确定性模型：图内没有扰动通路，members 必须两侧同为 1。"""
    assert model_cls.forecast_type == "deterministic"
    assert model_cls.members == 1
    assert cfg.members == 1


def test_channel_contract_is_85_symmetric(cfg, model_cls):
    """输入/输出通道都必须是 85 通道布局且对称（torch.export 图进出同构）。"""
    assert model_cls.input_channels == list(FUXI21_CHANNELS)
    assert model_cls.output_channels == list(FUXI21_CHANNELS)
    assert len(FUXI21_CHANNELS) == 85


def test_hour_interval_and_history_match_model(cfg, model_cls):
    assert cfg.hour_interval == 6
    assert model_cls.hour_interval == 6
    assert model_cls.history_steps == 2, "切片窗口与 golden 对拍都按 2 帧历史假设"


def test_dataset_processor_pipeline_order(cfg):
    """顺序必须保持：单位换算 → 几何 → 归一化 → 填缺测。

    normalize 依赖单位换算先做（mean/std 是模型单位下的统计量）；填缺测必须在
    归一化之后（填 0 填的是归一化后的值）。顺序错了不报错，数值全错。
    """
    names = [p["name"] for p in cfg.dataset["processors"]]
    assert names == ["unit_convert", "geometry", "normalize", "fill_missing"], (
        f"前处理序列变了：{names}"
    )


def test_log1p_expm1_pairing(cfg):
    """``log1p`` ↔ ``expm1`` 配对不变量 —— 输入侧对 tp 做 log1p，输出侧就必须
    expm1 恢复、还得 nonnegative。任何一侧单改，数值静默全错。"""
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
    """进/出归一化必须用同一对 mean.nc/std.nc。"""
    ds_processors = {p["name"]: p for p in cfg.dataset["processors"]}
    normalize = ds_processors["normalize"]
    _, _, output_specs = cfg.processor_specs()
    denorm = output_specs[0]

    assert normalize["mean_file"] == denorm["mean_file"]
    assert normalize["std_file"] == denorm["std_file"]


def test_recurrent_zero_channels_covers_diagnostics(cfg, model_cls):
    """rollout 回填必须清零全部 5 个累积型诊断通道 —— 不清零就是把上一步的
    累积值喂回下一步，短时效看不出、长时效静默烂掉。"""
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
    """换算系数钉住（与 fgvp 族同一套约定，共用同一个源 store）。"""
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
    """填 0 在归一化后做；未摸全的缺测组合先 keep（fuxi21 的源 store 比 fgvp 多
    soil/wave，与 fgvp 的 error 选择不同）。"""
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
    """fuxi21 用 5 个动态 store（比 fgvp 族多 soil / wave）—— 85 通道的来源。"""
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
