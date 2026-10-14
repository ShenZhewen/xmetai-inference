# -*- coding: utf-8 -*-
"""``configs/fgvp_ens.py`` 的单元测试：配置与集合模型契约之间的一致性（变更探测器，
思路同 ``test_fgvp_config.py``）。本档最要紧的两条不变量：``members=51`` 与集合
模型类绑定（扰动是图内随机算子 + 51 次独立前向，改数字就是换语义）；
``model_processing`` 必须全空（归一化烘焙在图里，挂任何 Processor 都会叠加）。
"""
import re
from datetime import datetime

import pytest

from xmetai_inference.configs.base import InferConfig, load_config, parse_times
from xmetai_inference.models import get_model_class

_BASE_NAME_RE = re.compile(r"([A-Za-z]+)(\d+)")

EXPECTED_VARS = {"z500", "u200", "v200", "msl", "tp"}


def _base_names(channels):
    out = set()
    for name in channels:
        match = _BASE_NAME_RE.fullmatch(name)
        out.add(match.group(1) if match else name)
    return out


@pytest.fixture(scope="module")
def cfg():
    return load_config("fgvp_ens")


@pytest.fixture(scope="module")
def model_cls(cfg):
    return get_model_class(cfg.model_class)


def test_loads_as_inferconfig_named_fgvp_ens(cfg):
    assert isinstance(cfg, InferConfig)
    assert cfg.name == "fgvp_ens"


def test_model_class_is_the_ensemble_mamba_backbone(cfg, model_cls):
    """必须解析到集合档那一支，而不是同通道布局的确定性 gdn2 —— 推理语义完全不同。"""
    assert cfg.model_class == "iwc_fgvp_mamba_ens_onnx"
    assert model_cls.__name__ == "IwcFgvpMambaEnsModel"
    assert model_cls.forecast_type == "ensemble"
    assert model_cls.windowed_output is False


def test_members_is_51_on_both_sides(cfg, model_cls):
    """members 就是集合语义本身，任何一侧改数字都是换语义（噪声口径 √(2/51) 也随之失效）。"""
    assert model_cls.members == 51
    assert cfg.members == 51


def test_gpu_state_is_off(cfg):
    """GPU 常驻和单帧输出图不相容，开了只会算错不会报错。"""
    assert cfg.gpu_state is False


def test_declared_vars_are_real_model_output_channels(cfg, model_cls):
    declared = {v.strip() for v in cfg.vars.split(",") if v.strip()}
    assert declared == EXPECTED_VARS, "落盘变量清单变了，确认是有意的再改这里"

    missing = declared - set(model_cls.output_channels)
    assert not missing, f"vars 里有模型不输出的通道：{sorted(missing)}"


def test_model_processing_is_empty(cfg):
    """归一化烘焙在图里，模型侧三个 Processor 必须全空 —— 叠加一遍就是全错。"""
    input_specs, recurrent_specs, output_specs = cfg.processor_specs()
    assert list(input_specs) == []
    assert list(recurrent_specs) == []
    assert list(output_specs) == []


def test_dataset_processors_match_fgvp_conventions(cfg, model_cls):
    """数据侧前处理与 fgvp single 同一套约定（同族、同一套源 store），钉住防无意改动。"""
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    assert set(processors) == {"unit_convert", "geometry", "fill_missing"}, (
        "fgvp 族图内自带归一化，数据侧不该出现 normalize 之类的处理器"
    )

    scales = processors["unit_convert"]["scales"]
    assert set(scales) == {"q", "ssr", "ssrd", "fdir", "ttr", "tp"}
    unknown = set(scales) - _base_names(model_cls.input_channels)
    assert not unknown, f"换算表里有对不上任何输入通道的键：{sorted(unknown)}"
    assert scales["q"] == 1000.0, "q：store kg/kg → 模型 g/kg"
    assert scales["tp"] == 6000.0, "tp：累积 m → 模型 mm"
    for field in ("ssr", "ssrd", "fdir", "ttr"):
        assert scales[field] == pytest.approx(6.0 / 3600.0), f"{field}：辐射场换算"

    fill = processors["fill_missing"]
    assert set(fill["rules"]) == {"sst"}
    assert fill["rules"]["sst"]["method"] == "constant"
    assert fill["rules"]["sst"]["value"] == pytest.approx(287.066)
    assert fill["unconfigured"] == "error", "未配置通道缺测必须报错，改宽松要显式确认"


def test_dynamic_stores_are_the_expected_trio(cfg):
    """与 fgvp single 同一套源 store 三件套（集合档只是换了模型，没换数据源）。"""
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


def test_times_are_daily_midnight_inits(cfg):
    """起报全部每日 00Z、间隔均匀（``:24`` 后缀防 hour_interval=6 展开，同 fgvp 档）。"""
    inits = parse_times(cfg.times, cfg.hour_interval)
    assert inits, "times 展开为空"
    assert all(t.hour == 0 for t in inits), "出现了非 00Z 起报"
    assert all(b - a == (inits[1] - inits[0]) for a, b in zip(inits, inits[1:])), (
        "起报间隔不均匀"
    )
    assert inits[1] - inits[0] == (inits[-1] - inits[-2]), (
        f"间隔应为一天：{inits[0]} → {inits[1]}"
    )
    assert isinstance(inits[0], datetime)


def test_cuda_devices_cover_every_rank(cfg):
    devices = [d for d in str(cfg.cuda_devices).split(",") if d.strip()]
    assert cfg.gpus >= 1
    assert len(devices) >= cfg.gpus, (
        f"gpus={cfg.gpus} 但只给了 {len(devices)} 个卡号：{cfg.cuda_devices!r}"
    )


def test_steps_and_interval_span_the_expected_lead_time(cfg, model_cls):
    """steps × 有效步长 = 15 天。``hour_interval=0`` 是「读模型类」的哨兵值
    （``cli.py`` 的 ``interval = ... or cls.hour_interval``），所以断言解析后的
    有效值，不是 config 字面值。"""
    assert cfg.hour_interval in (0, 6)
    effective = cfg.hour_interval or model_cls.hour_interval
    assert effective == 6
    assert cfg.steps == 60
    assert cfg.steps * effective == 15 * 24, "预报跨度应为 15 天"
