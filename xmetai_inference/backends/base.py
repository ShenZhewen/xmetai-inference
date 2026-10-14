# -*- coding: utf-8 -*-
"""推理后端共享运行契约。

换后端（ONNX / PT2）只需实现 load + forward；数据适配、输出反变换与回填规则由统一
Processor 管线注入，后端和模型类不实现处理钩子。热路径支持 torch.Tensor，numpy 只留在
ONNX Runtime、earthkit 或 netCDF 落盘等必须 numpy 的边界。
"""
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


def _copy_state(value):
    """numpy 用 copy()，torch 用 clone()。"""
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value.copy()


def _stack_outputs(outputs, axis):
    """torch 输出用 torch.stack，numpy 输出用 np.stack。"""
    if outputs and isinstance(outputs[0], torch.Tensor):
        return torch.stack(outputs, dim=axis)
    return np.stack(outputs, axis)


class BaseInferModel(ABC):
    """所有推理后端的公共基类：子类实现 load + forward。

    forward 契约：输入 x 为 (1, in_frames, C, H, W) 的 float32 数组/tensor，返回同形状的
    完整模型输出（末帧是本步预报）。PT2/ONNX 后端可保持 torch 全程待在设备上，框架只在
    落盘时搬到 CPU；ONNX Runtime 的 legacy 路径仍可返回 numpy。
    """
    backend = "base"

    # 落盘前做非负钳位的通道，各模型可通过类属性自行声明。这是兜底：模型漏配
    # 反归一化时 tp 仍在归一化空间，钳位会把负值削平、输出看着合理而不报错。
    nonnegative_channels = ("tp",)

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        self.device_id = device_id
        self.gpu_mem_fraction = gpu_mem_fraction
        self.processing = None
        self.data_source = None

    def configure_processing(self, processing, data_source):
        """Bind the configured model processing recipe to this model."""
        self.processing = processing
        self.data_source = data_source
        return self

    def prepare_dataset_batch(self, batch, verbose=False):
        """Convert a DataLoader batch into this model's input representation."""
        if self.processing is None or self.data_source is None:
            raise RuntimeError("模型尚未绑定 processing 和 data_source")
        from xmetai_inference.processing.pipeline import build_dataset_batch

        return build_dataset_batch(batch, verbose=verbose)


    def _processing_transform(self, stage):
        if self.processing is None:
            return None
        processors = getattr(self.processing, f"{stage}_processors")
        if not processors:
            return None
        return getattr(self.processing, f"process_{stage}")

    @abstractmethod
    def load(self, path):
        """加载模型到 self.device_id。"""

    @abstractmethod
    def forward(self, x, step, valid_time):
        """一步前向。x: (1,in_frames,C,H,W) float32 → (1,in_frames,C,H,W) float32（完整输出，末帧=预报）。"""

    def describe(self):
        """返回后端/设备描述（加载后打印用）。子类可覆盖。"""
        return self.backend

    def reset_runtime_state(self):
        """自回归轨迹间重置跨步内部状态（缺省无操作），Rollout 每批轨迹开始前调用一次。
        """
        return None

    # ------------------------------------------------------------------
    # GPU 常驻（state 全程待在 GPU，只在落盘时搬回 CPU）
    # ------------------------------------------------------------------
    # 置 gpu_state=True 须实现 to_gpu / forward_gpu / to_numpy。GPU state 不支持
    # numpy recurrent transform，需要回填处理的模型必须保持 gpu_state=False。
    gpu_state = False

    # 一次 forward 能否堆多条轨迹：图输入/输出的 batch 轴都须是符号维。onnx 后端
    # load 时实测置位；缺省 False = Rollout 打 WARNING 把 model_batch 退回 1。
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
        """一步 × 一成员的输出 → xarray Dataset（落盘前最后一步，后端无关入口）。

        step_state 形状 (C,H,W)，用 self.output_channels 解码变量名、套 lat/lon 坐标；
        save_names 为 None 表示全部通道。此处做唯一一次 tensor→numpy（xarray 需要 numpy）。
        """
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

    # 自回归 rollout 循环见 xmetai_inference/runner.py 的 Rollout。
