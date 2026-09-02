# -*- coding: utf-8 -*-
"""推理后端基类。

两层契约：单步前向 forward（默认循环走它）+ 自回归 run 循环（推理主入口）。
换后端（ONNX / PT2 / checkpoint）只需实现 load + forward 两个方法，框架用默认
run 循环把它们串起来；后端若自带循环（如 ckpt/anemoi 的 SimpleRunner）可整体
覆盖 run()。模型层需要个性化处理（如 z-score 归一化、扰动）时覆盖
pre_process / post_process / zero_recurrent 钩子即可。
"""
import gc
from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
import pandas as pd

_PROGRESS_INTERVAL = 5  # 默认进度：每隔多少步打印一条完整行（多卡时逐步 \r 会互相覆盖）


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


def _out_name(spec, channel):
    """输出变量名（大写）：Z50/T50/.../TP。"""
    var, level = spec["_parse"][channel]
    if level is not None:
        return f"{var.upper()}{level}"
    return var.upper()


def _transform(var, data, spec=None):
    """tp: 模型规范单位 -> mm（clamp>=0）；q 保持 g/kg（官方输出即 g/kg，不再转 kg/kg）。"""
    if var == "tp":
        unit = spec["variables"]["tp"]["unit"] if spec is not None else "m"
        scale = 1000.0 if unit == "m" else 1.0
        return np.maximum(data, 0.0) * scale
    return data


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

    def post_process(self, y):
        """输出后处理（默认恒等）。子类如 Fuxi21Pt2Model 覆盖为「×std+mean + expm1」。

        与 pre_process 对称：模型工作空间 -> 物理量。只在输出/落盘前调用，不参与
        自回归回填——回填必须保持模型自身的工作空间，post_process 只把"要给别人
        看的预测"变回物理量。tp→mm 这类单位换算留在 to_dataset 的
        _transform（spec 驱动、所有模型统一），不在这里做；q 官方输出即 g/kg，
        不做二次换算。
        """
        return y

    def pre_process(self, x):
        """输入前处理（默认恒等）。子类如 Fuxi21Pt2Model 覆盖为 z-score（tp 先 log1p）。

        与 post_process 对称：build_input 产出物理量，模型进入工作空间前处理一次。
        不需要前处理的模型（如 ONNX 已融合进图）直接继承恒等实现，零成本。
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

    # ------------------------------------------------------------------
    # GPU 常驻（state 全程待在 GPU，只在落盘时搬回 CPU，省每步 H2D/D2H）
    # ------------------------------------------------------------------
    # 子类置 gpu_state=True 并实现 to_gpu / forward_gpu / to_numpy 后，run 循环
    # 就走 GPU 常驻分支。zero_recurrent_gpu 是回填清零的 GPU 版（默认恒等）。只有
    # 前后处理/回填都是恒等（或已实现 GPU 版）的模型才安全启用，否则别开。
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

    def zero_recurrent_gpu(self, state):
        """GPU 载体上的回填清零钩子（默认恒等）。"""
        return state

    def to_dataset(self, step_state, spec, save_names=None, lat=None, lon=None):
        """一步 × 一成员的输出 → xarray Dataset（落盘前最后一步，后端无关入口）。

        默认实现按 onnx/pt2 通道张量语义：step_state 形状 (C,H,W)，用 spec 的
        _channels/_parse 解码变量名、套 lat/lon 坐标、挑 save_names（None=全部通道）。
        ckpt 后端覆盖为 field dict → N320 节点 Dataset。落盘统一走这里，runner.py
        不再关心状态表示（Step 4 接入）。
        """
        import xarray as xr

        channels = spec["_channels"]
        parse = spec["_parse"]
        if save_names is None:
            save_indices = list(range(len(channels)))
        else:
            name2idx = {str(c).lower(): i for i, c in enumerate(channels)}
            save_indices = [name2idx[str(n).lower()] for n in save_names]
        data_vars = {}
        for ci in save_indices:
            channel = channels[ci]
            var = parse[channel][0]
            arr = step_state[ci]                        # (H, W)，北->南
            arr = _transform(var, arr, spec)
            data_vars[_out_name(spec, channel)] = (("lat", "lon"), arr)
        return xr.Dataset(data_vars, coords={"lat": lat, "lon": lon})

    def run(self, base_input, steps, members=1, hour_interval=6,
            init_time=None, member_start=0, member_stride=1, on_step=None,
            log_step=False, progress=True, progress_label=""):
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

        # 进入模型工作空间：build_input 产物理量，这里前处理一次；之后全程在该空间
        # 自回归，输出侧再用 post_process 变回物理量。恒等实现则等价于物理量直进直出。
        base_input = self.pre_process(base_input)

        in_frames = base_input.shape[1]
        member_indices = list(range(member_start, members, member_stride))
        n_my = len(member_indices)
        C, H, W = base_input.shape[2:]

        streaming = on_step is not None
        outs = None if streaming else np.empty((n_my, steps, C, H, W), dtype=np.float32)

        use_gpu = self.gpu_state
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
                    # to_numpy 取预测帧。zero_recurrent_gpu 是回填清零的 GPU 版（恒等）。
                    state = self.forward_gpu(member_inputs[mi], s, valid)
                    member_inputs[mi] = self.zero_recurrent_gpu(state)
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
                    print(f"{label}预报 {fdays:.1f}/{tdays:.1f} 天 · "
                          f"剩 {rdays:.1f} 天 · 还要 {_fmt_dur(eta)}",
                          flush=True)
            if streaming:
                on_step(s, self.post_process(step_buf))

        if log_step and step_times:
            total = sum(step_times)
            print(f"  推理循环: {steps} steps x {n_my} members 合计 {total:.1f}s, "
                  f"单步 {min(step_times):.3f}~{max(step_times):.3f}s "
                  f"(均值 {total / len(step_times):.3f}s)")

        del member_inputs
        gc.collect()
        if not streaming:
            outs = self.post_process(outs)
        return outs, member_indices
