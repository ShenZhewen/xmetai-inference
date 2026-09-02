# -*- coding: utf-8 -*-
"""统一处理管线：loader State → Processor → 模型输入、回填与输出表示。

取代旧的 build_input.py——那个把「单位换算 / 通道重排 / 网格翻转滚动 /
装张量」全塞在一个函数里的"规则引擎"。现在每个动作是一个独立的 Processor，作用在
统一的 State（普通 dict）上，`process(state) -> state`，copy-in / copy-out；
config 用 `pre_processors` / `recurrent_processors` / `output_processors` 声明完整
处理流程，可换序、可扩展（加新处理器 = 一个 @register 类）。

内置处理器：
  geometry      —— lat 南→北 翻成 北→南、lon -180:180 滚成 0:360（并校验网格分辨率）
  channel_order —— 挑出模型要的通道并按模型输入顺序重排（缺通道报错）
  unit_convert  —— 源单位 → 模型单位（scale 表来自 loader.SCALE，缺省 ×1）
  attach_static —— 从 loader 读取静态场并合并到每个动态帧
  rename_channels —— 数据源字段名映射到模型字段名
  validate_magnitude —— 对关键字段做模型专属单位量级检查
  regrid        —— 规则经纬网格插值到目标非结构化网格（当前支持 N320）

装配器由模型类的 input_assembler 声明：
  tensor       —— (1, history, C, nlat, nlon)，用于 ONNX/PT2；
  field_dict   —— {"date", "fields": {name: (history, node)}}，用于 AIFS/Anemoi。

单位换算只做 scale（无 offset）：era5_store/era 的源单位与模型规范单位之间只差一个
乘法因子（q×1000、辐射×1/3600、tp×1000），没有平移；其余通道缺省 ×1。累积窗口的
×6（1h→6h）是数据源固有约定，仍在各 loader 的 _open 里做，不在这里。
"""
import logging
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr


log = logging.getLogger(__name__)


def _parse_var(name):
    """通道名 -> 基础变量名：'z50' -> 'z'，'t2m' -> 't2m'。单位换算按基础变量查表。"""
    m = re.fullmatch(r"([a-zA-Z]+?)(\d+)", str(name).strip())
    return m.group(1).lower() if m else str(name).strip().lower()


class Processor:
    """输入前处理基类。子类实现 process(state) -> state（copy-in / copy-out）。

    model_cls / loader 由 build_pipeline 注入：处理器从中自取所需契约
    （geometry 取 model_cls.grid，channel_order 取 model_cls.input_channels，
    unit_convert 取 loader.SCALE）。
    """

    stage = "state"

    def __init__(self, model_cls=None, loader=None, model_path=None, **kwargs):
        self.model_cls = model_cls
        self.loader = loader
        self.model_path = model_path

    def process(self, state):
        raise NotImplementedError


_REGISTRY = {}


def register(name):
    """极简注册表：@register('geometry') 把类登记进 _REGISTRY，config 按名字挂载。"""

    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


@register("geometry")
class Geometry(Processor):
    """lat 翻成北→南、lon 滚成 0:360；分辨率对不上直接报错（不做插值）。"""

    def __init__(self, expected_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.expected_shape = tuple(expected_shape) if expected_shape is not None else None

    def process(self, state):
        if state.get("grid_type", "regular_latlon") != "regular_latlon":
            raise ValueError("geometry 只接受 regular_latlon State")
        lat = np.asarray(state["latitudes"], dtype=np.float64).ravel()
        lon = np.asarray(state["longitudes"], dtype=np.float64).ravel()
        expected = self.expected_shape
        if expected is None and hasattr(self.model_cls, "grid"):
            grid = self.model_cls.grid
            expected = (grid["lat"]["size"], grid["lon"]["size"])
        if expected is not None:
            nlat, nlon = expected
        else:
            nlat, nlon = lat.size, lon.size
        if lon.size != nlon or lat.size != nlat:
            raise ValueError(
                f"网格 {lat.size}x{lon.size} 不是 {nlat}x{nlon}，需要先插值到该分辨率")

        flip = bool(lat.size > 1 and lat[0] < lat[-1])   # 南→北 需要翻成 北→南
        roll = 0
        dlon = 360.0 / nlon
        if not np.allclose(lon % 360.0, (np.arange(nlon) * dlon) % 360.0, atol=1e-4):
            diff = np.abs((lon % 360.0 - 0.0 + 180.0) % 360.0 - 180.0)
            roll = int(np.argmin(diff))

        out = dict(state)
        fields = {}
        for name, arr in state["fields"].items():
            a = np.asarray(arr)
            if flip:
                a = a[::-1, :]
            if roll:
                a = np.roll(a, roll, axis=-1)
            fields[name] = a
        out["fields"] = fields
        out["latitudes"] = lat[::-1].copy() if flip else lat.copy()
        aligned_lon = np.roll(lon, roll) if roll else lon.copy()
        out["longitudes"] = aligned_lon % 360.0
        out["grid_type"] = "regular_latlon"
        return out


@register("channel_order")
class ChannelOrder(Processor):
    """按模型输入通道顺序挑字段重排；缺通道报错（与旧 build_input 的 missing 检查一致）。"""

    def process(self, state):
        order = list(getattr(self.model_cls, "input_fields",
                             getattr(self.model_cls, "input_channels", ())))
        if not order:
            raise ValueError(f"{self.model_cls.__name__} 未声明 input_fields/input_channels")
        fields = state["fields"]
        missing = [c for c in order if c not in fields]
        if missing:
            raise ValueError(f"数据里找不到 {len(missing)} 个通道: {', '.join(missing)}")
        out = dict(state)
        out["fields"] = {c: fields[c] for c in order}
        return out


@register("unit_convert")
class UnitConvert(Processor):
    """源单位 → 模型单位（scale 表来自 loader.SCALE；无 SCALE 则缺省 ×1）。"""

    def process(self, state):
        scales = getattr(self.loader, "SCALE", None) or {}
        out = dict(state)
        fields = {}
        for name, arr in state["fields"].items():
            scale = scales.get(_parse_var(name), 1.0)
            fields[name] = np.asarray(arr) * scale if scale != 1.0 else np.asarray(arr)
        out["fields"] = fields
        return out


@register("attach_static")
class AttachStatic(Processor):
    """把 loader 的静态 State 合并进当前动态帧。"""

    def __init__(self, fields=None, **kwargs):
        super().__init__(**kwargs)
        self.names = tuple(fields or ())

    def process(self, state):
        load_static = getattr(self.loader, "load_static_state", None)
        if load_static is None:
            raise TypeError(f"{type(self.loader).__name__} 不支持 load_static_state()")
        static = load_static()
        if not np.array_equal(state["latitudes"], static["latitudes"]) or \
                not np.array_equal(state["longitudes"], static["longitudes"]):
            raise ValueError("动态场与静态场经纬度坐标不一致")
        source = static["fields"]
        names = self.names or tuple(source)
        missing = [name for name in names if name not in source]
        if missing:
            raise ValueError(f"静态数据缺少字段: {', '.join(missing)}")
        out = dict(state)
        out["fields"] = dict(state["fields"])
        out["fields"].update({name: source[name] for name in names})
        out["static_fields"] = set(state.get("static_fields", ())) | set(names)
        return out


@register("rename_channels")
class RenameChannels(Processor):
    """按显式映射重命名字段，未列出的字段保留原名。"""

    def __init__(self, mapping, **kwargs):
        super().__init__(**kwargs)
        self.mapping = dict(mapping)

    def process(self, state):
        fields = {}
        for name, value in state["fields"].items():
            target = self.mapping.get(name, name)
            if target in fields:
                raise ValueError(f"字段映射后出现重复名称 {target!r}")
            fields[target] = value
        out = dict(state)
        out["fields"] = fields
        out["static_fields"] = {
            self.mapping.get(name, name) for name in state.get("static_fields", ())
        }
        return out


def _validate_aifs11_magnitude(name, arr):
    values = np.asarray(arr, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    median = float(np.median(values))
    maximum = float(np.max(values))
    if name == "z_500" and median < 20000:
        raise ValueError(
            f"[单位错误] z_500 中位数 {median:.0f}，AIFS 要位势 m²/s²，不是位势高度 m")
    if name == "z" and maximum < 20000:
        raise ValueError(
            f"[单位错误] 地形位势最大值 {maximum:.0f}，AIFS 要 m²/s²，不是高度 m")
    if name == "q_850" and median > 0.05:
        raise ValueError(
            f"[单位错误] q_850 中位数 {median:.4f}，AIFS 要 kg/kg，不是 g/kg")


@register("validate_magnitude")
class ValidateMagnitude(Processor):
    """按模型 profile 检查关键字段量级；只检查，不自动纠正单位。"""

    def __init__(self, profile, **kwargs):
        super().__init__(**kwargs)
        if profile != "aifs11":
            raise ValueError(f"未知量级检查 profile {profile!r}")
        self.validator = _validate_aifs11_magnitude

    def process(self, state):
        for name, value in state["fields"].items():
            self.validator(name, value)
        return dict(state)


@register("regrid")
class Regrid(Processor):
    """把规则经纬网格场插值到目标网格。当前只实现 AIFS 使用的 N320。"""

    def __init__(self, target, **kwargs):
        super().__init__(**kwargs)
        if str(target).upper() != "N320":
            raise ValueError(f"当前 regrid 只支持 N320，收到 {target!r}")
        self.target = "N320"
        self._static_cache = {}

    @staticmethod
    def _interpolate(arr):
        import earthkit.regrid as ekr
        return np.asarray(
            ekr.interpolate(arr, {"grid": (0.25, 0.25)}, {"grid": "N320"}),
            dtype=np.float32,
        )

    def process(self, state):
        if state.get("grid_type", "regular_latlon") != "regular_latlon":
            raise ValueError("regrid 需要 regular_latlon State")
        static_names = set(state.get("static_fields", ()))
        fields = {}
        for name, value in state["fields"].items():
            arr = np.asarray(value)
            if arr.ndim != 2:
                raise ValueError(f"字段 {name!r} 插值前应为二维，实际 shape={arr.shape}")
            if name in static_names and name in self._static_cache:
                fields[name] = self._static_cache[name]
                continue
            result = self._interpolate(arr)
            fields[name] = result
            if name in static_names:
                self._static_cache[name] = result
        out = dict(state)
        out["fields"] = fields
        out["latitudes"] = None
        out["longitudes"] = None
        out["grid_type"] = "reduced_gaussian"
        out["grid_name"] = self.target
        return out


@register("fill_missing")
class FillMissing(Processor):
    """按字段显式常数规则填充非有限值；未配置字段默认直接报错。"""

    def __init__(self, rules, unconfigured="error", **kwargs):
        super().__init__(**kwargs)
        self.rules = dict(rules)
        if unconfigured not in ("error", "keep"):
            raise ValueError("fill_missing.unconfigured 只能是 error 或 keep")
        self.unconfigured = unconfigured

    def process(self, state):
        out = dict(state)
        fields = {}
        for name, value in state["fields"].items():
            arr = np.asarray(value)
            bad = ~np.isfinite(arr)
            count = int(bad.sum())
            if not count:
                fields[name] = arr
                continue
            rule = self.rules.get(name)
            if rule is None:
                if self.unconfigured == "keep":
                    fields[name] = arr
                    continue
                raise ValueError(f"字段 {name!r} 包含 {count} 个 NaN/Inf，但没有填充规则")
            method = rule.get("method")
            if method == "constant":
                fill_value = float(rule["value"])
            else:
                raise ValueError(f"字段 {name!r} 的填充方法 {method!r} 不受支持")
            result = np.array(arr, dtype=np.float32, copy=True)
            result[bad] = fill_value
            fields[name] = result
            log.info("缺失值填充：%s 的 %d 个 NaN/Inf -> %.6g",
                     name, count, fill_value)
        out["fields"] = fields
        return out


@lru_cache(maxsize=8)
def _load_normalization_stats(model_path, mean_file, std_file):
    if not model_path:
        raise ValueError("normalize/denormalize 需要模型文件路径")
    directory = os.path.dirname(os.path.abspath(model_path))
    mean_path = os.path.join(directory, mean_file)
    std_path = os.path.join(directory, std_file)
    if not os.path.isfile(mean_path) or not os.path.isfile(std_path):
        raise FileNotFoundError(
            f"模型归一化统计量不存在：{mean_path} / {std_path}")
    mean_da = xr.open_dataarray(mean_path)
    std_da = xr.open_dataarray(std_path)
    try:
        mean = np.asarray(mean_da.values, dtype=np.float32).reshape(-1)
        std = np.asarray(std_da.values, dtype=np.float32).reshape(-1)
        names = None
        if "channel" in mean_da.coords:
            names = tuple(str(value) for value in mean_da.coords["channel"].values)
    finally:
        mean_da.close()
        std_da.close()
    if mean.shape != std.shape:
        raise ValueError(f"mean/std shape 不一致：{mean.shape} vs {std.shape}")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("mean/std 包含非有限值或非正标准差")
    return mean, std, names


class _ArrayStatsProcessor(Processor):
    stage = "input"

    def __init__(self, mean_file="mean.nc", std_file="std.nc", **kwargs):
        super().__init__(**kwargs)
        self.mean, self.std, names = _load_normalization_stats(
            self.model_path, mean_file, std_file)
        expected = tuple(getattr(self.model_cls, "input_channels", ()))
        if self.mean.size != len(expected):
            raise ValueError(
                f"统计量通道数 {self.mean.size} 与模型输入通道数 {len(expected)} 不一致")
        if names is not None and names != expected:
            raise ValueError("mean.nc 的 channel 顺序与模型 input_channels 不一致")

    def _shape(self, ndim):
        return [1] * (ndim - 3) + [self.mean.size, 1, 1]


@register("normalize")
class Normalize(_ArrayStatsProcessor):
    """物理量张量 → 模型工作空间；可对指定通道先执行 log1p。"""

    stage = "input"

    def __init__(self, log1p_channels=None, allow_nonfinite=False, **kwargs):
        super().__init__(**kwargs)
        channels = list(self.model_cls.input_channels)
        self.log1p_indices = [channels.index(name) for name in (log1p_channels or ())]
        self.allow_nonfinite = bool(allow_nonfinite)

    def process_array(self, value):
        out = np.array(value, dtype=np.float32, copy=True)
        for index in self.log1p_indices:
            channel = out[..., index, :, :]
            if np.any(channel < 0):
                raise ValueError(
                    f"log1p 通道 {self.model_cls.input_channels[index]!r} 包含负值")
            np.log1p(channel, out=channel)
        shape = self._shape(out.ndim)
        out = (out - self.mean.reshape(shape)) / self.std.reshape(shape)
        if not np.all(np.isfinite(out)):
            bad = ~np.isfinite(out)
            channel_axis = out.ndim - 3
            reduce_axes = tuple(
                axis for axis in range(out.ndim) if axis != channel_axis)
            counts = bad.sum(axis=reduce_axes)
            details = ", ".join(
                f"{name}={int(count)}"
                for name, count in zip(self.model_cls.input_channels, counts)
                if count
            )
            if self.allow_nonfinite:
                log.debug("normalize 保留 NaN/Inf 交给模型处理：%s", details)
            else:
                raise ValueError(
                    f"normalize 后模型输入仍包含 NaN/Inf：{details}")
        return out


@register("denormalize")
class Denormalize(_ArrayStatsProcessor):
    """模型工作空间 → 物理量；可对指定通道执行 expm1 和非负截断。"""

    stage = "output"

    def __init__(self, expm1_channels=None, exp_clip_max=20.0, nonnegative=None, **kwargs):
        super().__init__(**kwargs)
        channels = list(self.model_cls.output_channels)
        self.expm1_indices = [channels.index(name) for name in (expm1_channels or ())]
        self.nonnegative_indices = [channels.index(name) for name in (nonnegative or ())]
        self.exp_clip_max = float(exp_clip_max)

    def process_array(self, value):
        array = np.asarray(value, dtype=np.float32)
        shape = self._shape(array.ndim)
        out = array * self.std.reshape(shape) + self.mean.reshape(shape)
        for index in self.expm1_indices:
            channel = out[..., index, :, :]
            np.clip(channel, None, self.exp_clip_max, out=channel)
            np.expm1(channel, out=channel)
        for index in self.nonnegative_indices:
            np.clip(out[..., index, :, :], 0, None, out=out[..., index, :, :])
        return out


@register("zero_channels")
class ZeroChannels(Processor):
    """自回归回填前把指定诊断通道清零。"""

    stage = "recurrent"

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        model_channels = list(self.model_cls.output_channels)
        self.indices = [model_channels.index(name) for name in channels]

    def process_array(self, value):
        value[..., self.indices, :, :] = 0.0
        return value


class ProcessingPipeline:
    """统一管理输入 State、装配后张量、输出和回填四个处理阶段。"""

    def __init__(self, processors):
        self.state_processors = [p for p in processors if p.stage == "state"]
        self.input_processors = [p for p in processors if p.stage == "input"]
        self.output_processors = [p for p in processors if p.stage == "output"]
        self.recurrent_processors = [p for p in processors if p.stage == "recurrent"]

    def process_state(self, state):
        for processor in self.state_processors:
            state = processor.process(state)
        return state

    @staticmethod
    def _process_array(value, processors):
        for processor in processors:
            value = processor.process_array(value)
        return value

    def process_input(self, value):
        return self._process_array(value, self.input_processors)

    def process_output(self, value):
        return self._process_array(value, self.output_processors)

    def process_recurrent(self, value):
        return self._process_array(value, self.recurrent_processors)

    @property
    def has_output(self):
        return bool(self.output_processors)

    @property
    def has_recurrent(self):
        return bool(self.recurrent_processors)


def build_pipeline(specs, model_cls, loader, model_path=None,
                   recurrent_specs=None, output_specs=None):
    """config 的三个 Processor 列表 → ProcessingPipeline。

    每项是字符串（处理器名）或 {name: str, ...kwargs}。输入处理器缺省使用通用
    geometry/channel_order/unit_convert；回填和输出处理器缺省为空。
    """
    if specs is None:
        specs = ("geometry", "channel_order", "unit_convert")
    procs = []

    def add(group, allowed_stages, config_name):
        for item in group:
            name = item if isinstance(item, str) else item["name"]
            kwargs = (
                {} if isinstance(item, str)
                else {key: value for key, value in item.items() if key != "name"}
            )
            processor_cls = _REGISTRY.get(name)
            if processor_cls is None:
                raise ValueError(
                    f"未知 Processor {name!r}（可选 {', '.join(_REGISTRY)}）")
            if processor_cls.stage not in allowed_stages:
                expected = "/".join(sorted(allowed_stages))
                raise ValueError(
                    f"{config_name} 中的 {name!r} 属于 {processor_cls.stage!r} 阶段，"
                    f"此处只允许 {expected!r}")
            procs.append(processor_cls(
                model_cls=model_cls,
                loader=loader,
                model_path=model_path,
                **kwargs,
            ))

    add(specs, {"state", "input"}, "pre_processors")
    add(recurrent_specs or (), {"recurrent"}, "recurrent_processors")
    add(output_specs or (), {"output"}, "output_processors")
    return ProcessingPipeline(procs)


class TensorAssembler:
    """把多帧 State 装配成 ONNX/PT2 使用的五维张量。"""

    def assemble(self, frames, model_cls, init):
        channels = list(model_cls.input_channels)
        nlat, nlon = model_cls.grid["lat"]["size"], model_cls.grid["lon"]["size"]
        out = np.empty((len(frames), len(channels), nlat, nlon), dtype=np.float32)
        for ti, state in enumerate(frames):
            for ci, name in enumerate(channels):
                out[ti, ci] = state["fields"][name]
        return out[np.newaxis, ...]


class FieldDictAssembler:
    """把多帧 State 装配成 Anemoi 的命名 field 字典。"""

    def assemble(self, frames, model_cls, init):
        names = list(model_cls.input_fields)
        fields = {
            name: np.stack(
                [np.asarray(state["fields"][name], dtype=np.float32) for state in frames],
                axis=0,
            )
            for name in names
        }
        shapes = {value.shape for value in fields.values()}
        if len(shapes) != 1:
            raise ValueError(f"field_dict 字段 shape 不一致: {sorted(shapes)}")
        return {"date": init.to_pydatetime(), "fields": fields}


_ASSEMBLERS = {
    "tensor": TensorAssembler,
    "field_dict": FieldDictAssembler,
}


def build_model_input(init_time, model_cls, loader, processors,
                      history_steps=None, hour_interval=None, verbose=False):
    """统一加载历史帧、执行 Processor，并按模型声明装配最终输入。"""
    def _log(*items):
        if verbose:
            log.info(" ".join(str(item) for item in items))

    history = history_steps if history_steps is not None \
        else getattr(model_cls, "history_steps", 2)
    interval = hour_interval if hour_interval is not None \
        else getattr(model_cls, "hour_interval", 6)

    init = pd.to_datetime(init_time, format="%Y%m%d%H") if isinstance(init_time, str) \
        else pd.to_datetime(init_time)
    times = [init - pd.Timedelta(hours=(history - 1 - i) * interval)
             for i in range(history)]          # 时间正序 [t-6h, t0]

    frames = []
    for t in times:
        state = loader.load_state(t)
        _log(f"load({t:%Y-%m-%d %H:00}) -> {len(state['fields'])} 个通道")
        if isinstance(processors, ProcessingPipeline):
            state = processors.process_state(state)
        else:
            for p in processors:
                state = p.process(state)
        frames.append(state)

    assembler_name = getattr(model_cls, "input_assembler", "tensor")
    assembler_cls = _ASSEMBLERS.get(assembler_name)
    if assembler_cls is None:
        raise ValueError(
            f"未知 input_assembler {assembler_name!r}（可选 {', '.join(_ASSEMBLERS)}）")
    result = assembler_cls().assemble(frames, model_cls, init)
    if isinstance(processors, ProcessingPipeline):
        result = processors.process_input(result)
    if isinstance(result, np.ndarray):
        _log(f"输出 shape: {result.shape} dtype={result.dtype}")
        finite = np.isfinite(result)
        if finite.any():
            _log(f"数值范围: min={result[finite].min():.6g} max={result[finite].max():.6g} "
                 f"NaN={int((~finite).sum())}")
        else:
            _log(f"数值范围: 全部非有限，NaN={result.size}")
    else:
        shape = next(iter(result["fields"].values())).shape
        _log(f"输出 field_dict: fields={len(result['fields'])} shape={shape}")
    return result
