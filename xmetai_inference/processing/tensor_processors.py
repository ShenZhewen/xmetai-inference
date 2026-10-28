# -*- coding: utf-8 -*-
"""Config-driven tensor preprocessors for dataset inputs.

Each processor is torch.Tensor -> torch.Tensor of shape [T, C, H, W] and is
selected by name from a dataset ``processors`` list; ``build_tensor_processors``
is the only entry point configuration uses.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import xarray as xr


def _base_name(name):
    """'z50' -> 'z', 't2m' -> 't2m'."""
    match = re.fullmatch(r"([A-Za-z]+?)(\d+)", str(name).strip())
    return match.group(1).lower() if match else str(name).strip().lower()


@dataclass
class TensorProcessorContext:
    """Metadata shared by all configured tensor processors."""

    channel_names: list[str]
    latitudes: np.ndarray
    longitudes: np.ndarray
    expected_shape: tuple[int, int]
    model_path: str | None = None


class TensorProcessor:
    """Base class for config-driven tensor preprocessors.

    Subclasses receive ``context`` plus processor-specific config keys and must
    implement ``process(value)``; input/output shape is [T, C, H, W].
    """

    def __init__(self, context: TensorProcessorContext, **kwargs):
        self.context = context
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} 收到未知参数: {sorted(kwargs)}"
            )

    def process(self, value):
        raise NotImplementedError

    def __call__(self, value):
        return self.process(value)


_TENSOR_PROCESSOR_REGISTRY = {}


def register_tensor_processor(name):
    """Register a tensor processor under a configuration name."""

    def decorator(cls):
        if name in _TENSOR_PROCESSOR_REGISTRY:
            raise ValueError(f"Tensor Processor {name!r} 已注册")
        _TENSOR_PROCESSOR_REGISTRY[name] = cls
        return cls

    return decorator


def _latlon_alignment(latitudes, longitudes, expected_shape):
    """Return (flip, roll) needed for north->south lat and 0:360 lon."""
    lat = np.asarray(latitudes, dtype=np.float64).ravel()
    lon = np.asarray(longitudes, dtype=np.float64).ravel()
    nlat, nlon = expected_shape
    if lat.size != nlat or lon.size != nlon:
        raise ValueError(
            f"网格 {lat.size}x{lon.size} 不是 {nlat}x{nlon}，需要先插值到该分辨率"
        )

    flip = bool(lat.size > 1 and lat[0] < lat[-1])
    roll = 0
    dlon = 360.0 / nlon
    expected_lon = (np.arange(nlon, dtype=np.float64) * dlon) % 360.0
    if not np.allclose(lon % 360.0, expected_lon, atol=1e-4):
        diff = np.abs((lon % 360.0 + 180.0) % 360.0 - 180.0)
        roll = (-int(np.argmin(diff))) % nlon
    return flip, roll


def aligned_coordinates(latitudes, longitudes, expected_shape):
    """Coordinates after geometry has been applied."""
    flip, roll = _latlon_alignment(latitudes, longitudes, expected_shape)
    lat = np.asarray(latitudes, dtype=np.float64).ravel().copy()
    lon = np.asarray(longitudes, dtype=np.float64).ravel().copy()
    if flip:
        lat = lat[::-1]
    if roll:
        lon = np.roll(lon, roll) % 360.0
    return lat, lon


# ERA5 store 原生单位：z/t/u/v/w 无需换算；q=kg/kg（FuXi 系要 g/kg ×1000）；
# ssr/ssrd/fdir/ttr = J/m² 1h 累积（→ Wh/m² ÷3600）；tp = m 1h 累积（→ mm ×1000）。
# 累积场在 store 里是 1h 而 step 是 6h，故 config 系数 = 单位换算 × 6：
# tp → 1000×6 = 6000.0，辐射 → 6/3600。


@register_tensor_processor("unit_convert")
class TensorUnitConvert(TensorProcessor):
    """Channel-wise multiplicative unit conversion.

    ``scales`` 按通道基名匹配（``_base_name``：z500→z、t2m→t2m），未列出的通道
    系数为 1.0；系数推导见上方 ERA5 单位约定参考。
    """

    def __init__(self, context: TensorProcessorContext, scales=None, **kwargs):
        super().__init__(context=context, **kwargs)
        scales = dict(scales or {})
        self.scale = torch.tensor(
            [
                float(scales.get(_base_name(name), 1.0))
                for name in context.channel_names
            ],
            dtype=torch.float32,
        )

    def process(self, value):
        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        if not torch.any(self.scale != 1.0):
            return value
        shape = [1] * (value.ndim - 3) + [self.scale.numel(), 1, 1]
        scale = self.scale.to(device=value.device, dtype=value.dtype).reshape(shape)
        return value * scale


@register_tensor_processor("geometry")
class TensorGeometry(TensorProcessor):
    """Lat north->south and lon 0:360 alignment（在 ``process`` 里惰性计算）。"""

    def __init__(self, context: TensorProcessorContext, expected_shape=None, **kwargs):
        super().__init__(context=context, **kwargs)
        self.lat = np.asarray(context.latitudes, dtype=np.float64).ravel()
        self.lon = np.asarray(context.longitudes, dtype=np.float64).ravel()
        self.nlat, self.nlon = expected_shape or context.expected_shape
        self.flip = None
        self.roll = None

    def _ensure_alignment(self):
        if self.flip is not None:
            return
        self.flip, self.roll = _latlon_alignment(
            self.lat, self.lon, (self.nlat, self.nlon)
        )

    def process(self, value):
        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        self._ensure_alignment()
        if self.flip:
            value = torch.flip(value, dims=(-2,))
        if self.roll:
            value = torch.roll(value, shifts=self.roll, dims=-1)
        return value


@register_tensor_processor("normalize")
class TensorNormalize(TensorProcessor):
    """Normalize model inputs with model-adjacent mean.nc/std.nc.

    ``log1p_channels`` are transformed with log1p before normalization.
    """

    def __init__(
        self,
        context: TensorProcessorContext,
        mean_file="mean.nc",
        std_file="std.nc",
        log1p_channels=(),
        **kwargs,
    ):
        super().__init__(context=context, **kwargs)
        self.channel_names = list(context.channel_names)
        if not context.model_path:
            raise ValueError("normalize 需要 model_path 才能加载 mean.nc/std.nc")
        directory = os.path.dirname(os.path.abspath(os.fspath(context.model_path)))
        mean_path = os.path.join(directory, mean_file)
        std_path = os.path.join(directory, std_file)
        if not os.path.isfile(mean_path) or not os.path.isfile(std_path):
            raise FileNotFoundError(
                f"模型归一化统计量不存在：{mean_path} / {std_path}"
            )
        with xr.open_dataarray(mean_path) as mean_da, xr.open_dataarray(std_path) as std_da:
            self.mean = np.asarray(mean_da.values, dtype=np.float32).reshape(-1)
            self.std = np.asarray(std_da.values, dtype=np.float32).reshape(-1)
            names = None
            if "channel" in mean_da.coords:
                names = tuple(str(v) for v in mean_da.coords["channel"].values)

        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean/std shape 不一致：{self.mean.shape} vs {self.std.shape}"
            )
        if self.mean.size != len(self.channel_names):
            raise ValueError(
                f"统计量通道数 {self.mean.size} 与模型输入通道数 "
                f"{len(self.channel_names)} 不一致"
            )
        if (
            not np.all(np.isfinite(self.mean))
            or not np.all(np.isfinite(self.std))
            or np.any(self.std <= 0)
        ):
            raise ValueError("mean/std 包含非有限值或非正标准差")
        if names is not None and names != tuple(self.channel_names):
            raise ValueError("mean.nc 的 channel 顺序与模型 input_channels 不一致")

        self.log1p_indices = [
            self.channel_names.index(name) for name in log1p_channels
        ]

    def process(self, value):
        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        out = value.to(torch.float32)
        for index in self.log1p_indices:
            channel = out[..., index, :, :]
            if torch.any(channel < 0):
                raise ValueError(
                    f"log1p 通道 {self.channel_names[index]!r} 包含负值"
                )
            torch.log1p(channel, out=channel)
        shape = [1] * (out.ndim - 3) + [len(self.channel_names), 1, 1]
        mean = torch.as_tensor(self.mean, dtype=out.dtype, device=out.device).reshape(shape)
        std = torch.as_tensor(self.std, dtype=out.dtype, device=out.device).reshape(shape)
        return (out - mean) / std


def _channel_key(name):
    """通道名归一化：``z_50``/``z50``、``T2M``/``t2m`` 视为同一个通道。"""
    return str(name).replace("_", "").strip().lower()


def _open_stats_zarr(path):
    """打开统计量 store：优先 consolidated，缺元数据合并的盘退回普通模式。

    zarr v2 的合并元数据在 ``.zmetadata``、v3 在根 ``zarr.json`` 里，缺了不一定是
    坏盘（zarr_refresh_v3 的盘就没有），所以读不到就退回逐节点模式再试一次；
    真坏盘在第二次打开时照样报错。
    """
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:  # noqa: BLE001 —— 退回非 consolidated 后仍有错会原样抛出
        return xr.open_zarr(path, consolidated=False)


@register_tensor_processor("denormalize")
class TensorDenormalize(TensorProcessor):
    """Recover model input space from an already-normalized store: ``x*std + mean``.

    与 ``normalize`` 对称。用于「store 里存的就是归一化值、而图自己会归一化」的数据
    源（CRA c71 store、zarr_refresh_v3 的 ERA5 盘）：图头是 ``Div(Sub(x, mean), std)``，
    所以喂进去的必须是反归一化后的值。

    统计量从 ``stats_path`` 指向的 zarr store 的 ``mean``/``std`` 数组读取（与数据
    同一个 store 时，正是训练用的那一套；图中烘焙的常量须与它一致）。

    通道按**名字**对齐（``z_50`` 与 ``z50`` 视为同一个），不要求 store 的通道数与
    顺序等于模型：zarr_refresh_v3 的 ERA5 盘是 116/78 通道的超集，模型只取自己
    要的那几十个，按名字挑统计量。

    注意经 log1p 再归一化的通道：反归一化得到的是 ``log1p(物理值)``。CRA 的 tp
    正是图期望的输入表示，**不要** expm1；zarr_refresh_v3 的 tp 则要用
    ``expm1_channels=["tp"]`` 还原成物理量（与 CRA 相反）。
    """

    def __init__(
        self,
        context: TensorProcessorContext,
        stats_path="",
        mean_name="mean",
        std_name="std",
        expm1_channels=None,
        exp_clip_max=20.0,
        **kwargs,
    ):
        super().__init__(context=context, **kwargs)
        self.channel_names = list(context.channel_names)
        if not stats_path:
            raise ValueError(
                "denormalize 需要 stats_path（含 mean/std 数组的 zarr store）")
        stats_path = os.fspath(stats_path)
        if not os.path.isdir(stats_path):
            raise FileNotFoundError(f"归一化统计量所在 store 不存在：{stats_path}")
        with _open_stats_zarr(stats_path) as stats:
            for name in (mean_name, std_name):
                if name not in stats:
                    raise ValueError(
                        f"store {stats_path} 里没有 {name!r} 数组"
                        f"（有 {list(stats.data_vars)}）")
            mean = np.asarray(
                stats[mean_name].values, dtype=np.float32).reshape(-1)
            std = np.asarray(
                stats[std_name].values, dtype=np.float32).reshape(-1)
            names = None
            if "channel" in stats.coords:
                names = tuple(str(v) for v in stats.channel.values)

        if mean.shape != std.shape:
            raise ValueError(
                f"mean/std shape 不一致：{mean.shape} vs {std.shape}")
        if names is None:
            # 没有 channel 坐标只能按下标对齐（旧行为）。
            if mean.size != len(self.channel_names):
                raise ValueError(
                    f"统计量通道数 {mean.size} 与模型输入通道数 "
                    f"{len(self.channel_names)} 不一致")
            self.mean, self.std = mean, std
        else:
            index = {}
            for position, name in enumerate(names):
                index.setdefault(_channel_key(name), position)
            missing = [
                name for name in self.channel_names
                if _channel_key(name) not in index
            ]
            if missing:
                raise ValueError(
                    f"store {stats_path} 缺模型输入通道 {missing[:5]}"
                    f"（模型要 {len(self.channel_names)} 个，store 有 {len(names)} 个）")
            order = np.asarray(
                [index[_channel_key(name)] for name in self.channel_names])
            self.mean, self.std = mean[order], std[order]

        if (
            not np.all(np.isfinite(self.mean))
            or not np.all(np.isfinite(self.std))
            or np.any(self.std <= 0)
        ):
            raise ValueError("mean/std 包含非有限值或非正标准差")

        wanted = {_channel_key(name) for name in (expm1_channels or ())}
        unknown = wanted - {_channel_key(name) for name in self.channel_names}
        if unknown:
            raise ValueError(
                f"expm1_channels 不在模型输入通道里：{sorted(unknown)}")
        self.expm1_indices = [
            position for position, name in enumerate(self.channel_names)
            if _channel_key(name) in wanted
        ]
        self.exp_clip_max = float(exp_clip_max)

    def process(self, value):
        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        out = value.to(torch.float32)
        shape = [1] * (out.ndim - 3) + [len(self.channel_names), 1, 1]
        mean = torch.as_tensor(
            self.mean, dtype=out.dtype, device=out.device).reshape(shape)
        std = torch.as_tensor(
            self.std, dtype=out.dtype, device=out.device).reshape(shape)
        out = out * std + mean
        for position in self.expm1_indices:
            channel = out[..., position, :, :]
            torch.clamp(channel, max=self.exp_clip_max, out=channel)
            torch.expm1(channel, out=channel)
        return out


def _resolve_fengqing_mean_std_dir(model_path, mean_std_dir=None):
    """解析 FengQing mean/std 目录。

    优先级：显式 ``mean_std_dir`` > 环境变量 ``FENGQING_MEAN_STD_DIR`` >
    模型文件上两级的 ``mean_std/``。
    """
    if mean_std_dir:
        return os.fspath(mean_std_dir)
    env = os.environ.get("FENGQING_MEAN_STD_DIR")
    if env:
        return env
    if not model_path:
        raise ValueError(
            "normalize_fengqing 需要 model_path（或 FENGQING_MEAN_STD_DIR）"
            "才能定位 FengQing mean/std 统计量")
    model_path = Path(model_path)
    for base in (model_path.parent, model_path.parents[1], model_path.parents[2]):
        candidate = base / "mean_std"
        if candidate.is_dir():
            return os.fspath(candidate)
    raise FileNotFoundError(
        f"找不到 FengQing mean_std 目录，相对模型文件 {model_path}")


@lru_cache(maxsize=4)
def _load_fengqing_stats(model_path, mean_std_dir=None):
    """Load FengQing per-pixel normalization stats -> (mean, std)。

    mean shape (70, H, W)、std shape (70, 1)：前 65 通道(upper air) 来自
    upper_mean/upper_std，后 5 通道(surface) 来自 mean_pre/std_pre。
    """
    directory = Path(_resolve_fengqing_mean_std_dir(model_path, mean_std_dir))
    upper_mean = np.load(directory / "upper_mean.npy").astype(np.float32)
    upper_std = np.load(directory / "upper_std.npy").astype(np.float32)
    surface_mean = np.load(directory / "mean_pre.npy").astype(np.float32)
    surface_std = np.load(directory / "std_pre.npy").astype(np.float32)
    if upper_mean.ndim != 3 or upper_std.ndim != 2:
        raise ValueError(
            f"FengQing upper 统计量 shape 异常："
            f"upper_mean={upper_mean.shape}, upper_std={upper_std.shape}")
    if surface_mean.ndim != 3 or surface_std.ndim != 2:
        raise ValueError(
            f"FengQing surface 统计量 shape 异常："
            f"surface_mean={surface_mean.shape}, surface_std={surface_std.shape}")
    if upper_std.shape != (upper_mean.shape[0], 1) or \
            surface_std.shape != (surface_mean.shape[0], 1):
        raise ValueError(
            f"FengQing std 应为 (C, 1)：upper={upper_std.shape}, "
            f"surface={surface_std.shape}")
    mean = np.concatenate([upper_mean, surface_mean], axis=0)  # (70, H, W)
    std = np.concatenate([upper_std, surface_std], axis=0)     # (70, 1)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) \
            or np.any(std <= 0):
        raise ValueError("FengQing mean/std 包含非有限值或非正标准差")
    return mean, std


@register_tensor_processor("normalize_fengqing")
class TensorNormalizeFengqing(TensorProcessor):
    """FengQing per-pixel normalization: ``(x - mean) / std`` over ``[T,70,H,W]``.

    前 65 通道(upper air) 用 upper_mean(65,H,W)/upper_std(65,1)，后 5 通道
    (surface) 用 mean_pre(5,H,W)/std_pre(5,1)；mean 逐像素、std 逐通道。
    非有限结果置 0。
    """

    def __init__(self, context, mean_std_dir=None, **kwargs):
        super().__init__(context=context, **kwargs)
        channels = list(context.channel_names)
        if len(channels) != 70:
            raise ValueError(
                f"normalize_fengqing 期望 70 通道（65 upper + 5 surface），"
                f"实际 {len(channels)}")
        mean, std = _load_fengqing_stats(context.model_path, mean_std_dir)
        expected = (70, *context.expected_shape)
        if mean.shape != expected:
            raise ValueError(
                f"FengQing mean shape {mean.shape} 与模型契约 {expected} 不一致")
        if std.shape != (70, 1):
            raise ValueError(
                f"FengQing std shape {std.shape} 应为 (70, 1)")
        self.mean = mean
        self.std = std

    def process(self, value):
        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        out = value.to(torch.float32)
        lead = [1] * (out.ndim - 3)
        mean = torch.as_tensor(
            self.mean, dtype=out.dtype, device=out.device
        ).reshape(lead + list(self.mean.shape))
        std = torch.as_tensor(
            self.std, dtype=out.dtype, device=out.device
        ).reshape(lead + [self.std.shape[0], 1, 1])
        out = (out - mean) / std
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@register_tensor_processor("fill_missing")
class TensorFillMissing(TensorProcessor):
    """Per-channel constant fill for non-finite values."""

    def __init__(self, context: TensorProcessorContext, rules=None, unconfigured="keep", **kwargs):
        super().__init__(context=context, **kwargs)
        self.channel_names = list(context.channel_names)
        if unconfigured not in ("error", "keep"):
            raise ValueError("fill_missing.unconfigured 只能是 error 或 keep")
        self.unconfigured = unconfigured

        rules = dict(rules or {})
        self.default_rule = rules.get("*", rules.get("default"))
        self.rules = {
            name: rule
            for name, rule in rules.items()
            if name not in ("*", "default")
        }

    def process(self, value):
        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        out = value
        for index, name in enumerate(self.channel_names):
            bad = ~torch.isfinite(out[..., index, :, :])
            if not torch.any(bad):
                continue
            rule = self.rules.get(name, self.default_rule)
            if rule is None:
                if self.unconfigured == "keep":
                    continue
                raise ValueError(
                    f"字段 {name!r} 包含 {int(torch.count_nonzero(bad))} 个 "
                    "NaN/Inf，但没有填充规则"
                )
            method = rule.get("method")
            if method != "constant":
                raise ValueError(
                    f"通道 {name!r} 的填充方法 {method!r} 不受支持"
                )
            if out is value:
                out = value.clone()
            out[..., index, :, :][bad] = float(rule["value"])
        return out


def build_tensor_processors(specs, context: TensorProcessorContext):
    """Build the configured processor chain."""
    processors = []
    for item in specs:
        if isinstance(item, str):
            name = item
            kwargs = {}
        else:
            name = item.get("name")
            kwargs = {key: value for key, value in item.items() if key != "name"}
        processor_cls = _TENSOR_PROCESSOR_REGISTRY.get(name)
        if processor_cls is None:
            raise ValueError(
                f"未知 Tensor Processor {name!r}（可选 "
                f"{', '.join(sorted(_TENSOR_PROCESSOR_REGISTRY))}）"
            )
        processors.append(processor_cls(context=context, **kwargs))
    return processors


def unit_scales_from_specs(specs):
    """Extract unit_convert scales from a processor specification list."""
    for item in specs:
        if not isinstance(item, str) and item.get("name") == "unit_convert":
            return dict(item.get("scales", {}))
    return {}
