# -*- coding: utf-8 -*-
"""推理后端共享运行契约：单步 forward；引擎在 onnx.py / pt2.py。"""
import logging
from abc import ABC, abstractmethod

import numpy as np
import torch

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

    def release(self):
        """后端资源释放的公共部分；子类覆盖时调 super()。"""

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
