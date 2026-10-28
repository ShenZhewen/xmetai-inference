# -*- coding: utf-8 -*-
"""模型侧（主进程里跑）：自回归回填（recurrent）与输出反变换（output）两个阶段。

热路径全程 torch tensor，numpy 只用于统计量加载和最终落盘。
数据侧（store → 模型输入，DataLoader worker）在 ``dataset_processors.py``。
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr


log = logging.getLogger(__name__)


def _as_torch(value, dtype=None, device=None):
    """尽可能零拷贝地转成 torch.Tensor；已是 tensor 时只按需 cast/搬运。"""
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != torch.device(device):
        tensor = tensor.to(device=device)
    return tensor


class Processor:
    """数组阶段处理基类。

    子类必须声明 stage（"recurrent" / "output"）并实现
    process_array(value) -> value（copy-in / copy-out）。

    model_cls / model_path / data_source 由 build_pipeline 注入。
    """

    def __init__(self, model_cls=None, data_source=None, model_path=None, **kwargs):
        self.model_cls = model_cls
        self.data_source = data_source
        self.model_path = model_path

    def process_array(self, value):
        raise NotImplementedError


_REGISTRY = {}


def register(name):
    """极简注册表：@register('denormalize') 把类登记进 _REGISTRY，config 按名字挂载。"""

    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


@lru_cache(maxsize=8)
def _load_normalization_stats(model_path, mean_file, std_file):
    if not model_path:
        raise ValueError("加载归一化统计量需要模型文件路径")
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

    def _stats(self, stats, value, shape):
        return torch.as_tensor(
            stats, dtype=value.dtype, device=value.device
        ).reshape(shape)


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
        value = _as_torch(value, dtype=torch.float32)
        shape = self._shape(value.ndim)
        mean = self._stats(self.mean, value, shape)
        std = self._stats(self.std, value, shape)
        out = value * std + mean
        for index in self.expm1_indices:
            channel = out[..., index, :, :]
            torch.clamp(channel, max=self.exp_clip_max, out=channel)
            torch.expm1(channel, out=channel)
        for index in self.nonnegative_indices:
            channel = out[..., index, :, :]
            torch.clamp(channel, min=0, out=channel)
        return out


@register("denormalize_fengqing")
class DenormalizeFengqing(Processor):
    """FengQing 归一化空间 → 物理量（与数据层 normalize_fengqing 互逆）。

    mean 逐像素、std 逐通道，upper/surface 分两套文件；只做 x·std + mean，
    tp 的非负钳位在模型 forward 里完成。
    """

    stage = "output"

    def __init__(self, mean_std_dir=None, **kwargs):
        super().__init__(**kwargs)
        directory = self._resolve_dir(mean_std_dir)
        upper_mean = np.load(directory / "upper_mean.npy").astype(np.float32)
        upper_std = np.load(directory / "upper_std.npy").astype(np.float32)
        surface_mean = np.load(directory / "mean_pre.npy").astype(np.float32)
        surface_std = np.load(directory / "std_pre.npy").astype(np.float32)
        if upper_mean.shape[0] != 65 or surface_mean.shape[0] != 5:
            raise ValueError(
                f"FengQing 统计量通道数不符：upper={upper_mean.shape} "
                f"surface={surface_mean.shape}")
        expected = len(self.model_cls.output_channels)
        if upper_mean.shape[0] + surface_mean.shape[0] != expected:
            raise ValueError(
                f"统计量合计 {upper_mean.shape[0] + surface_mean.shape[0]} 通道，"
                f"与模型 output_channels {expected} 不一致")
        self.mean = np.concatenate([upper_mean, surface_mean], axis=0)   # (70,H,W)
        self.std = np.concatenate(
            [upper_std.reshape(-1), surface_std.reshape(-1)], axis=0)    # (70,)

    def _resolve_dir(self, mean_std_dir):
        if mean_std_dir:
            return Path(mean_std_dir)
        env = os.environ.get("FENGQING_MEAN_STD_DIR")
        if env:
            return Path(env)
        if not self.model_path:
            raise ValueError(
                "denormalize_fengqing 需要 model_path（或 FENGQING_MEAN_STD_DIR）"
                "才能定位 mean_std/")
        model_path = Path(self.model_path).resolve()
        for base in (model_path.parent, model_path.parents[1], model_path.parents[2]):
            candidate = base / "mean_std"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"找不到 FengQing mean_std/，相对模型文件 {model_path}")

    def process_array(self, value):
        value = _as_torch(value, dtype=torch.float32)
        lead = [1] * (value.ndim - 3)
        mean = torch.as_tensor(
            self.mean, dtype=value.dtype, device=value.device
        ).reshape(lead + list(self.mean.shape))
        std = torch.as_tensor(
            self.std, dtype=value.dtype, device=value.device
        ).reshape(lead + [self.std.shape[0], 1, 1])
        return value * std + mean


@register("zero_channels")
class ZeroChannels(Processor):
    """自回归回填前把指定诊断通道置成**归一化空间的 0**（不是物理量清零）。

    对齐官方预归一化 input.nc（ssr/ssrd/fdir/ttr/tp 在归一化空间精确为 0），
    必须写字面 0.0，改成「物理清零」会偏离官方行为。
    """

    stage = "recurrent"

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        model_channels = list(self.model_cls.output_channels)
        self.indices = [model_channels.index(name) for name in channels]

    def process_array(self, value):
        value = _as_torch(value)
        value[..., self.indices, :, :] = 0.0
        return value


class ProcessingPipeline:
    """统一管理自回归回填（recurrent）与输出（output）两个处理阶段。"""

    def __init__(self, processors):
        self.recurrent_processors = [p for p in processors if p.stage == "recurrent"]
        self.output_processors = [p for p in processors if p.stage == "output"]

    @staticmethod
    def _process_array(value, processors):
        for processor in processors:
            value = processor.process_array(value)
        return value

    def process_output(self, value):
        return self._process_array(value, self.output_processors)

    def process_recurrent(self, value):
        return self._process_array(value, self.recurrent_processors)


def build_pipeline(input_specs, model_cls, data_source, model_path=None,
                   recurrent_specs=None, output_specs=None):
    """config 的 Processor 列表 → ProcessingPipeline（recurrent + output 两阶段）。

    input_specs（model_processing.input / 旧 pre_processors）必须为空：输入侧前处理
    由 dataset.processors 在 DataLoader worker 里完成，该参数只为误配时报错保留。
    """
    if input_specs:
        raise ValueError(
            "model_processing.input（旧 pre_processors）必须为空：输入侧前处理"
            "（单位/几何/归一化/填缺测）请声明在 dataset.processors，"
            "由 DataLoader worker 执行")
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
                data_source=data_source,
                model_path=model_path,
                **kwargs,
            ))

    add(recurrent_specs or (), {"recurrent"}, "recurrent_processors")
    add(output_specs or (), {"output"}, "output_processors")
    return ProcessingPipeline(procs)


def build_dataset_batch(batch, verbose=False):
    """DataLoader batch → (inputs, init_times)。

    只做类型/形状校验和起报时间解析；inputs 保持 [B,T,C,H,W] tensor 不进 numpy。
    """
    values = _as_torch(batch["inputs"], dtype=torch.float32)
    if values.ndim != 5:
        raise ValueError(f"Dataset inputs 应为 [B,T,C,H,W]，实际 shape={values.shape}")
    raw_times = batch["times"]
    if isinstance(raw_times, torch.Tensor):
        raw_times = raw_times.detach().cpu().numpy()
    init_times = pd.to_datetime(np.asarray(raw_times, dtype=np.int64))
    if verbose:
        log.info(
            "Dataset batch: inputs=%s init=%s..%s",
            values.shape,
            init_times[0],
            init_times[-1],
        )
    return values, list(init_times)
