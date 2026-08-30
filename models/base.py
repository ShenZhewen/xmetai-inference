# -*- coding: utf-8 -*-
"""推理后端基类。

抽象出「加载 + 一步前向」两个接口，其余（自回归 rollout、归一化钩子）全部复用。
换后端（ONNX / PT2 / checkpoint）或换模型（FuXi-Ens / FuXi-2.1 / FengShun）只需实现
load + forward 两个方法，需要个性化处理（如 z-score 归一化、扰动）时覆盖
normalize / denormalize 钩子即可。
"""
import gc
from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
import pandas as pd

_BAR_WIDTH = 24  # 进度条宽度（字符数）


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


class BaseInferModel(ABC):
    """所有推理后端的公共基类。

    子类只需实现两个方法：
      load(path)                    —— 加载模型到 self.device_id
      forward(x, step, valid_time)  —— 一步前向

    forward 的约定：输入 x 为 (1, in_frames, C, H, W) 的 numpy float32，
    返回同样 (1, in_frames, C, H, W) 的完整模型输出（末帧是本步预报）。
    这条约定定了，rollout 就完全不用知道后端是 ONNX 还是 PyTorch；回填用
    完整输出（state = result），与官方 inference.py 的 `new_input = model.run(...)` 一致。
    """
    backend = "base"

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        self.device_id = device_id
        self.gpu_mem_fraction = gpu_mem_fraction

    @abstractmethod
    def load(self, path):
        """加载模型到 self.device_id。"""

    @abstractmethod
    def forward(self, x, step, valid_time):
        """一步前向。x: (1,in_frames,C,H,W) float32 → (1,in_frames,C,H,W) float32（完整输出，末帧=预报）。"""

    def describe(self):
        """返回后端/设备描述（加载后打印用）。子类可覆盖。"""
        return self.backend

    def denormalize(self, y):
        """输出反归一化（默认恒等）。子类如 Fuxi21Pt2Model 覆盖为 ×std+mean。

        注意：只在输出/落盘前调用，不参与自回归回填——回填必须保持模型自身的
        工作空间（归一化），denormalize 只把"要给别人看的预测"变回物理量。
        """
        return y

    def normalize(self, x):
        """输入归一化（默认恒等）。子类如 Fuxi21Pt2Model 覆盖为 (x-mean)/std。

        与 denormalize 对称：build_input 产出物理量，模型进入工作空间前归一化一次。
        不需要归一化的模型（如 ONNX 已融合进图）直接继承恒等实现，零成本。
        """
        return x

    def zero_recurrent(self, state):
        """回填前对 recurrent state 做通道清零（默认恒等）。

        有些模型（如 FuXi-2.1）把辐射通量 / 降水当作**诊断输出**：它们照常出现在
        预测里给用户看，但不参与自回归反馈——回填下一拍输入前必须清零，否则累积场
        会带着上一拍的值滚下去，污染后续预报。是否清零、清哪些通道是模型属性，
        官方没说清的模型（如 FuXi-Ens ONNX）默认不清，继承本恒等实现即可。
        """
        return state

    def rollout(self, base_input, steps, members=1, hour_interval=6,
                init_time=None, member_start=0, member_stride=1, on_step=None,
                log_step=False, progress=True, progress_label=""):
        """自回归 rollout（step 外层、member 内层），与后端无关。

        base_input: (1, in_frames, C, H, W) 的初始状态。
        member_start / member_stride 用于多卡拆成员：rank r 传 (r, world_size)，
        各卡只跑自己那份成员。

        on_step(step_index, step_buf) 每算完一个 step 的全体成员回调一次，step_buf
        形状 (n_my_members, C, H, W)。传 on_step 就走流式（边算边落盘，内存不随
        steps 增长）；不传则把 (n_my_members, steps, C, H, W) 全攒内存里返回。

        返回 (outs, member_indices)。流式模式下 outs 为 None。
        """
        init = pd.to_datetime(init_time, format="%Y%m%d%H") if isinstance(init_time, str) \
            else (pd.to_datetime(init_time) if init_time is not None else None)

        # 进入模型工作空间：build_input 产物理量，这里归一化一次；之后全程在该空间
        # 自回归，输出侧再用 denormalize 变回物理量。恒等实现则等价于物理量直进直出。
        base_input = self.normalize(base_input)

        in_frames = base_input.shape[1]
        member_indices = list(range(member_start, members, member_stride))
        n_my = len(member_indices)
        C, H, W = base_input.shape[2:]

        streaming = on_step is not None
        outs = None if streaming else np.empty((n_my, steps, C, H, W), dtype=np.float32)

        member_inputs = [base_input.copy() for _ in member_indices]
        step_times = []
        for s in range(steps):
            t0 = perf_counter()
            step_buf = np.empty((n_my, C, H, W), dtype=np.float32)
            # valid = 起报 + s 步 = 输入窗口最新帧的时刻（不是 s+1 步的预报时刻）。
            # hour/doy 标注的是"当前输入状态"的时间，官方 inference.py 与参考
            # onnx_infer_dfens.py 都用 t*interval，这里对齐；step 标量仍用 s（0-based）。
            valid = init + pd.Timedelta(hours=s * hour_interval) if init is not None else None
            for mi in range(n_my):
                buf = member_inputs[mi]                       # 完整状态 (1,in_frames,C,H,W)
                state = self.forward(buf, s, valid)           # (1,in_frames,C,H,W) 完整输出
                if state.shape[1] != in_frames:
                    raise ValueError(
                        f"模型输出 {state.shape[1]} 帧，与输入 {in_frames} 帧不一致，"
                        f"无法做 state=result 回填")
                pred = state[:, -1]                           # 末帧 = 本步预报 (1,C,H,W)
                step_buf[mi] = pred[0]
                if not streaming:
                    outs[mi, s] = pred[0]
                # state = result：把模型完整输出（含第 0 帧回显）原样回填成下一拍
                # 输入，与官方 inference.py 的 `new_input = model.run(...)` 一致。
                # 滚动窗口 [上一末帧, 新预测] 只在第 0 帧是直通时才等价；回显帧更忠实。
                # 回填前先做诊断通道清零（FuXi-2.1 的辐射/降水不反馈；恒等实现则不清）。
                member_inputs[mi] = self.zero_recurrent(state)
            dt = perf_counter() - t0
            step_times.append(dt)
            if log_step:
                elapsed = sum(step_times)
                eta = elapsed / (s + 1) * (steps - s - 1)
                print(f"  [step {s + 1}/{steps}] {n_my} members 耗时 {dt:.3f}s "
                      f"(累计 {elapsed:.1f}s, 预计剩余 {eta:.1f}s)")
            elif progress:
                # 默认进度条：单行刷新（\r），不刷屏。多卡时各 rank 的进度条会各自
                # 刷新自己那一行，靠 progress_label 前缀区分。
                done = s + 1
                ratio = done / steps
                fill = int(_BAR_WIDTH * ratio)
                bar = "█" * fill + "░" * (_BAR_WIDTH - fill)
                eta = sum(step_times) / done * (steps - done)
                label = f"{progress_label} " if progress_label else ""
                print(f"\r{label}[{bar}] {done}/{steps} 步 ({ratio:5.1%}) "
                      f"平均{sum(step_times)/done:4.2f}s/步 ETA {eta:6.1f}s",
                      end="", flush=True)
            if streaming:
                on_step(s, self.denormalize(step_buf))

        if progress and not log_step:
            # 进度条结束后补个换行，避免下一行日志接在同一行上
            print(flush=True)
        if log_step and step_times:
            total = sum(step_times)
            print(f"  推理循环: {steps} steps x {n_my} members 合计 {total:.1f}s, "
                  f"单步 {min(step_times):.3f}~{max(step_times):.3f}s "
                  f"(均值 {total / len(step_times):.3f}s)")

        del member_inputs
        gc.collect()
        if not streaming:
            outs = self.denormalize(outs)
        return outs, member_indices
