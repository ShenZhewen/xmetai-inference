# -*- coding: utf-8 -*-
"""推理后端共享运行契约。

两层契约：单步前向 forward（默认循环走它）+ 自回归 run 循环（推理主入口）。
换后端（ONNX / PT2）只需实现 load + forward 两个方法，框架用默认
run 循环把它们串起来。数据进入模型前的适配、输出反变换和回填规则由统一 Processor 管线注入，
后端和模型类不再实现处理钩子。

具体引擎分别位于同目录 onnx.py、pt2.py。引擎不能挂到包根门面 re-export——
onnx.py 顶层 import onnxruntime，一旦挂上，PT2-only 环境导入任何 backends
符号都会被拖上 ONNX Runtime；包根 __init__.py 只 re-export BaseInferModel，
保持 `from xmetai_inference.backends import BaseInferModel` 对外不变。

自 2026-09 起，PT2/ONNX 的 tensor 后端热路径支持 torch.Tensor；numpy 只保留在
ONNX Runtime、earthkit 或 xarray/netCDF 落盘等必须 numpy 的边界。
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
    """所有推理后端的公共基类。

    子类只需实现两个方法：
      load(path)                    —— 加载模型到 self.device_id
      forward(x, step, valid_time)  —— 一步前向

    forward 的约定：输入 x 为 (1, in_frames, C, H, W) 的 float32 数组/tensor，
    返回同样 (1, in_frames, C, H, W) 的完整模型输出（末帧是本步预报）。
    PT2/ONNX 后端可保持 torch 全程待在设备上，框架只在落盘时搬到 CPU；ONNX
    Runtime 的 legacy 路径仍可返回 numpy，run 循环按实际类型分别处理。
    这条约定定了，run 就完全不用知道后端是 ONNX 还是 PyTorch；回填用
    完整输出（state = result），与官方 inference.py 的 `new_input = model.run(...)` 一致。
    """
    backend = "base"

    # 落盘前做非负钳位的通道。原先 to_dataset 里硬编码 `if channel == "tp"`，
    # 那是模型语义写进了后端基类。做成类属性后各模型可自己声明；缺省 ("tp",)
    # 是因为四个模型都有 tp 且都需要钳位，行为与原来一致。
    #
    # 注意这只是落盘前的兜底 —— 真正的钳位应在各自的输出阶段完成
    # （fuxi21 的 denormalize.nonnegative、FengQing 在 forward 的归一化空间里）。
    # 这道兜底的隐患是**掩错**：若某模型漏配反归一化，tp 仍处归一化空间，
    # np.maximum(arr, 0) 会把负的归一化值削平，输出看着「合理」而不报错。
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
        """自回归轨迹间重置模型持有的跨步内部状态（缺省无操作）。

        Rollout.run 每批轨迹开始前调用一次。多会话/跨步有状态模型（如 Pangu
        方案B 的 24h 锚点）覆盖此方法清空上一批残留状态，避免状态在轨迹间泄漏，
        也把常驻内存约束在当前 batch 量级。
        """
        return None

    # ------------------------------------------------------------------
    # GPU 常驻（state 全程待在 GPU，只在落盘时搬回 CPU，省每步 H2D/D2H）
    # ------------------------------------------------------------------
    # 子类置 gpu_state=True 并实现 to_gpu / forward_gpu / to_numpy 后，run 循环
    # 就走 GPU 常驻分支。当前 GPU state 不支持 numpy recurrent transform；需要
    # 回填处理的模型应保持 gpu_state=False。
    gpu_state = False

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

        默认实现按 onnx/pt2 通道张量语义：step_state 形状 (C,H,W)，用 self.output_channels
        （模型类的输出通道契约）解码变量名、套 lat/lon 坐标、挑 save_names（None=全部
        通道）。step_state 可以是 torch.Tensor，也可以沿用 ONNX legacy 的 numpy；
        这里做唯一一次 tensor→numpy，xarray 需要 numpy 数据。
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

    # 自回归 rollout 循环已移至 xmetai_inference/runner.py 的 Rollout。
    #
    # 原先这里有 run（按 member 循环，124 行）和 run_batch（按 batch 循环，61 行）
    # 两份平行实现：循环体逐字相同，差别只在「轨迹从哪来」和「返回值形状」，且已
    # 出现单边演进（帧数校验严格度不同、防别名一边显式 clone 一边靠隐式拷贝、
    # 进度/ETA 只有 run 有）。Rollout 用统一的「轨迹」概念覆盖两者：集合的成员和
    # 批量的起报都是一条轨迹。
    #
    # 后端/模型类现在只需实现 load + forward（GPU 常驻另需 to_gpu/forward_gpu/
    # to_numpy），循环、多轨迹、回填、进度归编排层。
