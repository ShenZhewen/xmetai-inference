# -*- coding: utf-8 -*-
"""推理后端共享运行契约：单步 forward + 常驻 session 池；引擎在 onnx.py / pt2.py。"""
import collections
import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch

from .spa import LatentStepStream

log = logging.getLogger(__name__)


def _fmt_dur(sec):
    """秒 -> 人类可读时长（'45s' / '12m34s' / '1h02m'）。"""
    sec = int(round(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _default_gpu_mem_limit(device_id, fraction):
    """用 torch 查真实显存，返回可分配给后端的字节数；查不到返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device_id)
    except (AssertionError, RuntimeError):
        return None
    return max(int(props.total_memory * fraction), 0)


def _to_numpy(value):
    """torch → numpy；只用于 ONNX/earthkit/xarray/netCDF 等边界。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class BaseInferModel(ABC):
    """所有推理后端的公共基类，子类实现 load + forward。

    forward：x 为 (1, in_frames, C, H, W)，返回同形状完整输出（末帧为本步预报）。
    """
    backend = "base"
    supports_weight_perturb = False      # 权重扰动（SPW）
    supports_latent_forcing = False      # 潜空间 AR(1) 强迫（SPA）
    latent_forcing = None                # 由 create_model 构造后赋值，None = 不加

    intra_op_threads = 0                 # 0 = 后端自定（cli 按 worker 数折算）
    cudnn_algo_search = ""               # "" = 默认（EXHAUSTIVE）

    # 落盘前兜底的非负钳位；真正的钳位在各自输出阶段。漏配反归一化时这里会把
    # 负的归一化值削成 0，输出看着合理却不报错。
    nonnegative_channels = ("tp",)

    # 模型契约默认值；落成真属性后 declare 生成的类可直接自省（单测会读）。
    history_steps = 2
    hour_interval = 6
    forecast_type = "deterministic"
    members = 1
    windowed_output = True
    valid_time_step_offset = 0

    @classmethod
    def declare(cls, class_name, *, model_name, channels, grid,
                output_channels=None, history_steps=None, hour_interval=None,
                forecast_type=None, members=None, windowed_output=None,
                gpu_state=None, valid_time_step_offset=None,
                time_features=None, **extra):
        """按参数生成模型类；只写入显式给出的属性，其余沿用基类/后端默认值。

        ``class_name`` 同时作为 ``__name__``。需要重写 load / forward 的模型
        （FengQing、Pangu）直接写 class。
        """
        values = {
            "model_name": model_name,
            "input_channels": channels,
            "output_channels": output_channels or channels,
            "grid": grid,
            "history_steps": history_steps,
            "hour_interval": hour_interval,
            "forecast_type": forecast_type,
            "members": members,
            "windowed_output": windowed_output,
            "gpu_state": gpu_state,
            "valid_time_step_offset": valid_time_step_offset,
            "time_features": time_features,
        }
        attrs = {key: value for key, value in values.items() if value is not None}
        attrs.update(extra)
        attrs.setdefault("__doc__", f"{model_name}（{cls.backend} 后端）的声明式契约。")
        return type(class_name, (cls,), attrs)

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        self.device_id = device_id
        self.gpu_mem_fraction = gpu_mem_fraction
        self.processing = None
        self.data_source = None
        self.weight_perturb = None           # config 的 weight_perturb，None = 不加
        self.member_index = 0                # 当前成员，决定加载哪份成员模型
        self._latent_n_targets = 0           # 注入目标数，0 = 未启用
        self._latent_stream = LatentStepStream()

    def configure_processing(self, processing, data_source):
        """绑定 config 里的 processing 配方。"""
        self.processing = processing
        self.data_source = data_source
        return self

    def prepare_dataset_batch(self, batch, verbose=False):
        """DataLoader batch → (inputs[B,T,C,H,W], init_times)。"""
        if self.processing is None or self.data_source is None:
            raise RuntimeError("模型尚未绑定 processing 和 data_source")
        from xmetai_inference.utils.model_processors import build_dataset_batch

        return build_dataset_batch(batch, verbose=verbose)

    def _latent_row(self, step, width=1):
        """本步本块要喂的 ξ，形状 (width, T, R)；None = 未启用强迫。

        ``width`` = 本次 forward 堆了几条轨迹，块内每条轨迹各吃一行独立 ξ。
        """
        if not self.latent_forcing or self._latent_n_targets <= 0:
            return None
        index = self._latent_stream(
            self.latent_forcing, step,
            n_targets=self._latent_n_targets,
            member_index=self.member_index,
            hour_interval=getattr(self, "hour_interval", None),
            width=width)
        return self._latent_stream.ar1.row(index, width)

    def release(self):
        """后端资源释放的公共部分（潜空间驱动状态）；子类覆盖时调 super()。"""
        if self._latent_stream is not None:
            self._latent_stream.release()

    @abstractmethod
    def load(self, path):
        """加载模型到 self.device_id。"""

    @abstractmethod
    def forward(self, x, step, valid_time):
        """一步前向，返回完整输出窗；末帧即本步预报。"""

    def describe(self):
        """返回后端/设备描述（加载后打印用）。子类可覆盖。"""
        return self.backend

    def reset_runtime_state(self):
        """每批轨迹开始前清空跨步内部状态（缺省无操作）。"""
        return None

    # 置 True 并实现 to_gpu / forward_gpu / to_numpy 后，state 全程留在 GPU；
    # GPU state 不支持 numpy recurrent transform，需要回填的模型保持 False。
    gpu_state = False

    # 图的 batch 轴是否为符号维，load 时实测置位；False 时 Rollout 退回逐条。
    batch_axis_dynamic = False

    def to_gpu(self, x):
        """numpy 初始 state -> GPU 载体（gpu_state=True 时实现）。"""
        raise NotImplementedError

    def forward_gpu(self, state, step, valid_time):
        """GPU -> GPU 一步前向，返回新 state 的 GPU 载体（gpu_state=True 时实现）。"""
        raise NotImplementedError

    def to_numpy(self, state):
        """GPU 载体 -> numpy（落盘/取预测帧用，gpu_state=True 时实现）。"""
        raise NotImplementedError

    def to_dataset(self, step_state, save_names=None, lat=None, lon=None):
        """一步 × 一成员的输出 (C,H,W) → xarray Dataset；save_names=None 表示全部。"""
        import xarray as xr

        channels = list(self.output_channels)
        if save_names is None:
            save_indices = list(range(len(channels)))
        else:
            name2idx = {str(c).lower(): i for i, c in enumerate(channels)}
            save_indices = [name2idx[str(n).lower()] for n in save_names]
        data_vars = {}
        for ci in save_indices:
            channel = channels[ci]
            arr = _to_numpy(step_state[ci])                # (H, W)，北->南
            if channel in self.nonnegative_channels:
                arr = np.maximum(arr, 0.0)
            data_vars[channel.upper()] = (("lat", "lon"), arr)
        return xr.Dataset(data_vars, coords={"lat": lat, "lon": lon})

# 常驻 session 池（config 的 resident_members）：1 = 一换一，N = 最多 N 份，
# -1 = 全常驻。键是解析后的路径而不是成员号——一个 session 可服务多个成员。
def normalize_budget(value):
    """resident_members -> 池子额度：0 / None / 非法值都当 1（缺省口径），负数 = 不限。"""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 1
    if value < 0:
        return -1
    return max(1, value)


class SessionPool:
    """路径 -> session 的 LRU 池。额度 1 时退化成「用一个换一个」，与改造前等价。"""

    def __init__(self, budget=1, logger=None):
        self.budget = normalize_budget(budget)
        self.log = logger or log
        self.builds = 0
        self.hits = 0
        self.evictions = 0
        self._items = collections.OrderedDict()

    def acquire(self, key, build):
        """取 session：命中则复用（hit=True），否则调用 ``build()`` 新建并入池。"""
        key = os.path.abspath(key)
        cached = self._items.pop(key, None)
        if cached is not None:
            self._items[key] = cached          # 重新入尾 = LRU 更新
            self.hits += 1
            return cached, True
        session = build()
        self.builds += 1
        self._items[key] = session
        self._evict()
        return session, False

    def _evict(self):
        """按额度从 LRU 一端换出；额度 -1 = 不换。"""
        if self.budget < 0:
            return
        dropped = []
        while len(self._items) > self.budget:
            dropped.append(self._items.popitem(last=False)[0])
        if dropped:
            self.evictions += len(dropped)
            self.log.info("常驻 session 池已满（额度 %d），换出 %d 份：%s",
                          self.budget, len(dropped),
                          "、".join(os.path.basename(item) for item in dropped))

    def purge(self):
        """清空池子（驻留的 session 全部放开，显存随之归还）。返回清掉的份数。"""
        count = len(self._items)
        self._items.clear()
        return count

    def __len__(self):
        return len(self._items)

    def stats(self):
        return {
            "budget": self.budget,
            "size": len(self._items),
            "builds": self.builds,
            "hits": self.hits,
            "evictions": self.evictions,
        }
