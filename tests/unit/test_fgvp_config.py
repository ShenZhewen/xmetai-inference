# -*- coding: utf-8 -*-
"""``configs/fgvp.py`` 的单元测试：配置与模型契约之间的一致性。

config 是一串声明，写错大多不报错、只是静默不生效（单位换算键名拼错 = 悄悄不换算；
``vars`` 拼错 = 落盘少一个变量）。这里测的是**声明之间的一致性**，作用是变更探测器；
值本身对不对要靠 ``test_fgvp_realdata.py`` 跟标准答案对拍。
"""
import re
from datetime import datetime

import pytest

from xmetai_inference.configs.base import InferConfig, load_config, parse_times
from xmetai_inference.models import FUXI_ENS_CHANNELS, get_model_class

#: 通道名去层号（z50 → z），带尾字母（t2m/u10m）和纯字母（ssr/msl/tp）原样。
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
    return load_config("fgvp")


@pytest.fixture(scope="module")
def model_cls(cfg):
    return get_model_class(cfg.model_class)


def test_loads_as_inferconfig_named_fgvp(cfg):
    assert isinstance(cfg, InferConfig)
    assert cfg.name == "fgvp"


def test_model_class_is_the_gdn2_backbone(cfg, model_cls):
    """必须解析到确定性 gdn2 那支，而不是同通道布局的集合档 mamba_ens。"""
    assert cfg.model_class == "iwc_fgvp_gdn2_onnx"
    assert model_cls.__name__ == "IwcFgvpGdn2Model"
    assert model_cls.windowed_output is False


def test_hour_interval_matches_model(cfg, model_cls):
    assert cfg.hour_interval == 6
    assert model_cls.hour_interval == 6


def test_members_is_one_for_deterministic_model(cfg, model_cls):
    """图只出一帧、没有扰动通路，members>1 只会把同一份结果存 N 遍。"""
    assert model_cls.forecast_type == "deterministic"
    assert model_cls.members == 1
    assert cfg.members == 1


def test_gpu_state_is_off_by_default(cfg, monkeypatch):
    """GPU 常驻和本模型「输入 2 帧输出 1 帧」的窗口形状不相容，默认必须关。

    重新 load 而非用 fixture：gpu_state 是模块 exec 时读环境变量算出来的。
    要 load 的必须是 ``cfg`` 那份文件（``cfg._source_path``），不是硬编码的
    "fgvp" —— 否则验的永远是包内那份原始配置而不是被测对象。
    """
    monkeypatch.delenv("XMETAI_GPU_STATE", raising=False)
    fresh = load_config(cfg._source_path)
    assert fresh.gpu_state is False


def test_env_override_only_accepts_literal_one(cfg, monkeypatch):
    """``XMETAI_GPU_STATE`` 只有字面 "1" 才算开；true/yes/on 都静默当没设。"""
    monkeypatch.setenv("XMETAI_GPU_STATE", "1")
    assert load_config(cfg._source_path).gpu_state is True

    for falsy in ("true", "yes", "on", "0"):
        monkeypatch.setenv("XMETAI_GPU_STATE", falsy)
        assert load_config(cfg._source_path).gpu_state is False, (
            f"{falsy!r} 不该被当成开"
        )


def test_declared_vars_are_real_model_output_channels(cfg, model_cls):
    """`vars` 拼错一个不报错，只是那个变量不落盘 —— 往往跑完整年才发现。"""
    declared = {v.strip() for v in cfg.vars.split(",") if v.strip()}
    assert declared == EXPECTED_VARS, "落盘变量清单变了，确认是有意的再改这里"

    available = set(model_cls.output_channels)
    missing = declared - available
    assert not missing, f"vars 里有模型不输出的通道：{sorted(missing)}"


def test_unit_convert_keys_are_real_base_variables(cfg, model_cls):
    """换算表的键是基础变量名（q 而非 q700），拼错就是静默不换算。"""
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    scales = processors["unit_convert"]["scales"]

    assert set(scales) == {"q", "ssr", "ssrd", "fdir", "ttr", "tp"}, (
        "换算变量清单变了，确认是有意的"
    )

    unknown = set(scales) - _base_names(model_cls.input_channels)
    assert not unknown, f"换算表里有对不上任何输入通道的键：{sorted(unknown)}"


def test_unit_convert_scales_are_the_agreed_conventions(cfg):
    """系数是实测对齐过的约定，钉住防无意改动；约定本身对错由 realdata 档兜底。"""
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    scales = processors["unit_convert"]["scales"]

    assert scales["q"] == 1000.0, "q：store kg/kg → 模型 g/kg"
    assert scales["tp"] == 6000.0, "tp：累积 m → 模型 mm"
    for field in ("ssr", "ssrd", "fdir", "ttr"):
        assert scales[field] == pytest.approx(6.0 / 3600.0), f"{field}：辐射场换算"


def test_fill_missing_targets_real_channel_and_is_strict(cfg, model_cls):
    """规则里的变量得是真通道（拼错 = 静默不填，NaN 一路进模型）；
    unconfigured="error" 是安全属性：没配规则的通道缺测宁可炸。"""
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    fill = processors["fill_missing"]

    assert set(fill["rules"]) == {"sst"}, "填充规则的目标变了，确认是有意的"
    unknown = set(fill["rules"]) - _base_names(model_cls.input_channels)
    assert not unknown, f"填充规则里有对不上任何输入通道的变量：{sorted(unknown)}"

    assert fill["rules"]["sst"]["method"] == "constant"
    assert fill["rules"]["sst"]["value"] == pytest.approx(287.066)
    assert fill["unconfigured"] == "error"


def test_model_processing_is_empty(cfg):
    """归一化烘焙在图里，模型侧挂任何 Processor 都会与图内处理叠加，数值不报错只是全错。"""
    input_specs, recurrent_specs, output_specs = cfg.processor_specs()
    assert list(input_specs) == []
    assert list(recurrent_specs) == []
    assert list(output_specs) == []


def test_cuda_devices_cover_every_rank(cfg):
    """多卡分支按 rank 取 cuda_devices[rank]，短一个只在多卡启动时才 IndexError。"""
    devices = [d for d in str(cfg.cuda_devices).split(",") if d.strip()]
    assert cfg.gpus >= 1
    assert len(devices) >= cfg.gpus, (
        f"gpus={cfg.gpus} 但只给了 {len(devices)} 个卡号：{cfg.cuda_devices!r}"
    )


def test_times_expand_to_four_daily_inits(cfg):
    """`times` 结尾的 ``:24`` 不是冗余：不带后缀会按 hour_interval=6 展开。"""
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
    """三个动态 store 的名单；用集合比 —— 通道按名字选，顺序不影响结果。"""
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
