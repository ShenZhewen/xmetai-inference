# -*- coding: utf-8 -*-
"""Config-driven tensor preprocessors for dataset inputs.

Each processor is torch.Tensor -> torch.Tensor and is selected by name from a
dataset ``processors`` list. ``build_tensor_processors`` is the only entry point
that configuration uses; no branching by input-space strings is needed here.
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
    implement ``process(value)``. Input/output are torch.Tensor with shape
    [T, C, H, W], which keeps the configured chain composable.
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
        roll = int(np.argmin(diff))
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


# ---------------------------------------------------------------------------
# ERA5 单位约定参考（写 config 的 unit_convert.scales 时据此推导）
# ---------------------------------------------------------------------------
# era5_foundation_store2 的原生单位与采样约定（这是 ERA5 数据的固有性质，与模型
# 无关，原先散落在已删除的 loaders/ 包里）：
#
#   通道                     store 原生单位        备注
#   z/t/u/v/w                m²/s², K, m/s        无需换算
#   q                        kg/kg                FuXi 系要 g/kg（×1000）；
#                                                 FengQing 要 kg/kg（不换算）
#   ssr/ssrd/fdir/ttr        J/m²，**1h 累积**     → Wh/m² 需 ÷3600
#   tp                       m，**1h 累积**        → mm 需 ×1000
#
# 关键点：这几个累积场在 store 里是「1h 累积」，而 time step 是 6h。所以模型要
# 「每步(6h)累积」时，除物理单位换算外还要再 ×6（累积窗口换算）。两者相乘才是
# config 里该写的系数：
#
#   tp   → 6h 累积 mm ：1000 (m→mm)   × 6 = 6000.0
#   辐射 → 6h Wh/m²   ：1/3600 (J→Wh) × 6 = 6/3600
#
# 这就是为什么 configs/fengqing.py 写 tp: 6000.0、configs/fuxi21.py 写
# ssr: 6.0/3600.0 —— 不是随手凑的数，是「物理单位 × 累积窗口」两步的乘积。
# 各 config 显式写出最终系数（不在这里做隐式二次换算），这张表只作推导依据。


@register_tensor_processor("unit_convert")
class TensorUnitConvert(TensorProcessor):
    """Channel-wise multiplicative unit conversion.

    ``scales`` 按通道基名匹配（``_base_name``：z500→z、t2m→t2m），未列出的通道
    系数为 1.0。系数怎么推导见上方 ERA5 单位约定参考。
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
    """Lat north->south and lon 0:360 alignment.

    Alignment is computed lazily in ``process``; the constructor only keeps the
    coordinates and target shape.
    """

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


def _resolve_fengqing_mean_std_dir(model_path, mean_std_dir=None):
    """Parse the FengQing mean/std directory.

    Priority: explicit ``mean_std_dir`` > env ``FENGQING_MEAN_STD_DIR`` >
    ``mean_std`` directory relative to the model file (searched up two parents,
    mirroring the model layer's ``_find_artifact``).
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
    """Load FengQing per-pixel normalization stats -> (mean, std).

    Returns mean of shape (70, H, W) and std of shape (70, 1): the first 65
    channels (upper air) come from upper_mean/upper_std, the last 5 (surface)
    from mean_pre/std_pre. Cached so the data-layer normalize and any output-
    side processor share one in-process copy.
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

    The first 65 channels (upper air) use upper_mean(65,H,W)/upper_std(65,1);
    the last 5 (surface) use mean_pre(5,H,W)/std_pre(5,1). mean is per-pixel,
    std is per-channel, matching the FengQing pre-ONNX contract. Non-finite
    results are zeroed, mirroring the model's former ``_normalize_initial``.
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
