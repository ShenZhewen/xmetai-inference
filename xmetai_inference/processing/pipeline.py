# -*- coding: utf-8 -*-
"""模型侧处理管线：Dataset batch 透传、自回归回填与输出反变换。

取代旧的 build_input.py——那个把「单位换算 / 通道重排 / 网格翻转滚动 /
装张量」全塞在一个函数里的"规则引擎"。输入侧前处理已于 2026-09 全部迁往
dataset.processors（DataLoader worker 里经 tensor_processors.py 执行），旧的
State 装配路径（geometry/channel_order/unit_convert/attach_static 等 state
Processor、Normalize 输入阶段与 TensorAssembler）随之移除；模型侧只保留两个阶段：

  recurrent（回填）—— 自回归循环 state=result 回填前执行（如诊断通道清零）
  output（输出）  —— 落盘前把模型工作空间还原成物理量（denormalize）

热路径全程 torch tensor：denormalize/zero_channels 不做 numpy 往返；numpy 只
保留在归一化统计量加载和最终 xarray/netCDF 落盘等边界处。
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

    model_cls / model_path 由 build_pipeline 注入：处理器从中自取所需契约
    （zero_channels 取 model_cls.output_channels，denormalize 取模型旁的
    mean.nc/std.nc）。data_source 预留给需要数据源元数据的未来处理器。
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
    """FengQing 归一化空间 → 物理量（与数据层 ``normalize_fengqing`` 互逆）。

    FengQing 的统计量与 ``Denormalize`` 用的 mean.nc/std.nc 不同构，所以单独一个
    处理器而不是复用：
      * mean 是**逐像素**的（65,H,W）/（5,H,W），不是每通道标量；
      * std 是每通道（65,1）/（5,1）；
      * upper 与 surface 分两套文件（upper_mean/upper_std、mean_pre/std_pre）。

    只做 ``x·std + mean``。tp 的非负钳位已在模型 forward 里于归一化空间完成
    （见 FengqingPreOnnxModel.load 的 _tp_floor_norm），这里不重复；
    BaseInferModel.to_dataset 落盘前还有一道 tp clamp 兜底。
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
    """自回归回填前把指定诊断通道置成**归一化空间的 0**。

    注意「清零」在这里指的是归一化空间取 0，不是物理量清零——两者只在个别
    通道上恰好重合。这是有意的：对齐官方预归一化 input.nc（该文件里
    ssr/ssrd/fdir/ttr/tp 五个通道在归一化空间精确为 0）。

    以 fuxi21 为例（configs/fuxi21.py 的 normalize + 本处理器 + denormalize 三处联动）：
      * tp   —— mean=0/std=1（identity z-score）且输入侧做过 log1p，
                而 log1p(0)=0 恰好精确，所以归一化空间的 0 就是物理 0，两种
                解释重合；
      * ssr/ssrd/fdir/ttr —— mean/std 量级 600~1400（6h Wh/m²），归一化空间
                的 0 对应 physical = mean，即**气候态均值，不是零**。

    因此这里必须写字面 0.0。若改成「物理清零」（先反归一化、置 0、再归一化）
    会偏离官方行为；若 tp 的 mean/std 不再是 0/1，字面 0.0 对 tp 的物理含义
    也会随之改变。改动本处理器或那套 mean/std 前请先确认这两个前提。
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

    ``input_specs``（``model_processing.input``，旧配置的 ``pre_processors``）
    **必须为空**：输入侧前处理由 dataset.processors 在 DataLoader worker 里完成，
    模型侧不再有输入阶段。这个参数保留只为在误配时给出明确报错 —— 它是配置面的
    兼容检查，不是可用的扩展点。recurrent/output 每项是处理器名（字符串）或
    ``{name: str, ...kwargs}``，缺省为空。
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

    dataset.processors 已在 DataLoader worker 里完成全部输入前处理，这里只做
    类型/形状校验和起报时间解析；inputs 保持 [B,T,C,H,W] tensor 不进 numpy。
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
