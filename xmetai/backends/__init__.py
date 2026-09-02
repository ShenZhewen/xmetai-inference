# -*- coding: utf-8 -*-
"""推理后端共享运行契约。

两层契约：单步前向 forward（默认循环走它）+ 自回归 run 循环（推理主入口）。
换后端（ONNX / PT2 / checkpoint）只需实现 load + forward 两个方法，框架用默认
run 循环把它们串起来；后端若自带循环（如 ckpt/anemoi 的 SimpleRunner）可整体
覆盖 run()。数据进入模型前的适配、输出反变换和回填规则由统一 Processor 管线注入，
后端和模型类不再实现处理钩子。

具体引擎分别位于 onnx.py、pt2.py 和 ckpt.py。本模块只保留它们共同依赖的
BaseInferModel、自回归循环和运行时辅助函数，不再维护未使用的 backend registry。
"""
import gc
import logging
from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
import pandas as pd

_PROGRESS_INTERVAL = 5  # 默认进度：每隔多少步打印一条完整行（多卡时逐步 \r 会互相覆盖）
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


class BaseInferModel(ABC):
    """所有推理后端的公共基类。

    子类只需实现两个方法：
      load(path)                    —— 加载模型到 self.device_id
      forward(x, step, valid_time)  —— 一步前向

    forward 的约定：输入 x 为 (1, in_frames, C, H, W) 的 numpy float32，
    返回同样 (1, in_frames, C, H, W) 的完整模型输出（末帧是本步预报）。
    这条约定定了，run 就完全不用知道后端是 ONNX 还是 PyTorch；回填用
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
        通道）。ckpt 后端覆盖为 field dict → N320 节点 Dataset。落盘统一走这里，
        inference.py
        不再关心状态表示。
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
            arr = step_state[ci]                        # (H, W)，北->南
            if channel == "tp":
                # tp 模型规范单位即 mm，输出只做非负 clamp。
                arr = np.maximum(arr, 0.0)
            data_vars[channel.upper()] = (("lat", "lon"), arr)
        return xr.Dataset(data_vars, coords={"lat": lat, "lon": lon})

    def run(self, base_input, steps, members=1, hour_interval=6,
            init_time=None, member_start=0, member_stride=1, on_step=None,
            log_step=False, progress=True, progress_label="",
            recurrent_transform=None, output_transform=None):
        """自回归推理循环（step 外层、member 内层），与后端无关——推理主契约。

        默认实现走「单步 forward + state=result 回填」，适配 onnx/pt2 这类「后端只
        暴露单步、循环归框架」的引擎；后端若自带循环（如 ckpt/anemoi 的
        SimpleRunner），可整体覆盖 run()，完全不碰 forward。

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

        in_frames = base_input.shape[1]
        member_indices = list(range(member_start, members, member_stride))
        n_my = len(member_indices)
        C, H, W = base_input.shape[2:]

        streaming = on_step is not None
        outs = None if streaming else np.empty((n_my, steps, C, H, W), dtype=np.float32)

        use_gpu = self.gpu_state
        if use_gpu and recurrent_transform is not None:
            raise ValueError("GPU 常驻模式暂不支持 recurrent Processor")
        if use_gpu:
            # 每个成员一份独立的 GPU state（初始都是同一份 base_input，扰动在图内随机）
            member_inputs = [self.to_gpu(base_input) for _ in member_indices]
        else:
            member_inputs = [base_input.copy() for _ in member_indices]
        step_times = []
        for s in range(steps):
            t0 = perf_counter()
            step_buf = np.empty((n_my, C, H, W), dtype=np.float32)
            # valid = 起报 + s 步 = 输入窗口最新帧的时刻（不是 s+1 步的预报时刻）。
            # hour/doy 标注的是"当前输入状态"的时间，fuxiens inference.py 与参考
            # onnx_infer_dfens.py 都用 t*interval，这里对齐；step 标量仍用 s（0-based）。
            valid = init + pd.Timedelta(hours=s * hour_interval) if init is not None else None
            for mi in range(n_my):
                if use_gpu:
                    # GPU 常驻：state 全程待在 GPU，forward GPU→GPU；落盘前才
                    # to_numpy 只在落盘前取预测帧；GPU 常驻模型当前不配置回填 Processor。
                    state = self.forward_gpu(member_inputs[mi], s, valid)
                    member_inputs[mi] = state
                    step_buf[mi] = self.to_numpy(state)[0, -1]
                    if not streaming:
                        outs[mi, s] = step_buf[mi]
                else:
                    buf = member_inputs[mi]                   # 完整状态 (1,in_frames,C,H,W)
                    state = self.forward(buf, s, valid)       # (1,in_frames,C,H,W) 完整输出
                    if state.shape[1] != in_frames:
                        raise ValueError(
                            f"模型输出 {state.shape[1]} 帧，与输入 {in_frames} 帧不一致，"
                            f"无法做 state=result 回填")
                    pred = state[:, -1]                       # 末帧 = 本步预报 (1,C,H,W)
                    step_buf[mi] = pred[0]
                    if not streaming:
                        outs[mi, s] = pred[0]
                    # state = result：把模型完整输出（含第 0 帧回显）原样回填成下一拍
                    # 输入，与官方 inference.py 的 `new_input = model.run(...)` 一致。
                    # 滚动窗口 [上一末帧, 新预测] 只在第 0 帧是直通时才等价；回显帧更忠实。
                    # 回填前执行统一 recurrent Processor（如诊断通道清零）。
                    member_inputs[mi] = recurrent_transform(state) \
                        if recurrent_transform is not None else state
            dt = perf_counter() - t0
            step_times.append(dt)
            if log_step:
                elapsed = sum(step_times)
                eta = elapsed / (s + 1) * (steps - s - 1)
                log.debug("step %d/%d members=%d 耗时=%.3fs 累计=%.1fs 预计剩余=%.1fs",
                          s + 1, steps, n_my, dt, elapsed, eta)
            elif progress:
                # 默认进度：每隔 _PROGRESS_INTERVAL 步打一条**完整行**（带换行）。
                # 逐步 \r 进度条在四卡并发时会互相覆盖刷屏，改成稀疏的完整行就不会乱。
                # 行内只放关键信息：已预报几天 / 剩余几天 / 预计还要多久。
                done = s + 1
                if done == steps or done % _PROGRESS_INTERVAL == 0:
                    avg = sum(step_times) / done
                    eta = avg * (steps - done)
                    fdays = done * hour_interval / 24.0
                    tdays = steps * hour_interval / 24.0
                    rdays = (steps - done) * hour_interval / 24.0
                    label = f"{progress_label} " if progress_label else ""
                    log.info("%s预报 %.1f/%.1f 天，剩 %.1f 天，预计还需 %s",
                             label, fdays, tdays, rdays, _fmt_dur(eta))
            if streaming:
                output = output_transform(step_buf) \
                    if output_transform is not None else step_buf
                on_step(s, output)

        if log_step and step_times:
            total = sum(step_times)
            log.debug("推理循环 steps=%d members=%d 合计=%.1fs 单步=%.3f~%.3fs 均值=%.3fs",
                      steps, n_my, total, min(step_times), max(step_times),
                      total / len(step_times))

        del member_inputs
        gc.collect()
        if not streaming:
            if output_transform is not None:
                outs = output_transform(outs)
        return outs, member_indices
