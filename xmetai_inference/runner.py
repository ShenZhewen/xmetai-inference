# -*- coding: utf-8 -*-
"""自回归 rollout 编排层。

模型层只负责「一步前向」（load + forward），循环、多轨迹、回填、进度、GPU 快
路径全部归这里。这样换模型不用碰循环，换循环策略不用碰模型。

**轨迹（Trajectory）** 是本模块唯一的并行概念，取代了原先 base.py 里
``run``（按 member 循环）和 ``run_batch``（按 batch 循环）两份平行实现：

  * 集合预报：N 个成员 = N 条轨迹，共享同一初始态（扰动来自图内随机算子，
    框架不加扰动），init_time 相同；
  * 批量确定性：N 个起报 = N 条轨迹，各自的初始态与 init_time。

两者的循环体逐字相同，差别只在「轨迹从哪来」和「输出落到哪」——前者由调用方
构造 Trajectory 列表决定，后者由 on_step 回调决定。
"""
import gc
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from .backends.base import _fmt_dur

log = logging.getLogger(__name__)

_PROGRESS_INTERVAL = 5   # 每隔多少步打一条完整进度行（多卡时 \r 会互相覆盖）


def _copy_state(value):
    """numpy 用 copy()，torch 用 clone()。"""
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value.copy()


def _stack(values, axis=0):
    """torch 用 torch.stack，numpy 用 np.stack。"""
    if values and isinstance(values[0], torch.Tensor):
        return torch.stack(values, dim=axis)
    return np.stack(values, axis)


@dataclass
class Trajectory:
    """一条独立的自回归轨迹。

    state     初始状态 (1, in_frames, C, H, W)，torch.Tensor 或 numpy
    init_time 该轨迹的起报时刻；rollout 中 valid_time 由它推出
    label     仅用于日志（如 "member_007" / "2025010200"）
    """

    state: Any
    init_time: pd.Timestamp
    label: str = ""


class Rollout:
    """把模型的单步 forward 串成多轨迹自回归循环。

    model      需实现 forward(x, step, valid_time)；gpu_state=True 时还需
               to_gpu / forward_gpu / to_numpy
    processing 提供 recurrent / output 两个阶段的变换；None = 都不做
    """

    def __init__(self, model, processing=None):
        self.model = model
        self.processing = processing

    def _transform(self, stage):
        if self.processing is None:
            return None
        if not getattr(self.processing, f"{stage}_processors", None):
            return None
        return getattr(self.processing, f"process_{stage}")

    def run(self, trajectories, steps, hour_interval=6, on_step=None,
            progress=True, progress_label=""):
        """跑 steps 步，每步对全部轨迹各前向一次。

        on_step(step_index, step_state) 每步回调一次，step_state 形状
        (n_trajectories, C, H, W)，顺序与传入的 trajectories 一致。传了 on_step
        就走流式（内存不随 steps 增长）；不传则把 (n_traj, steps, C, H, W) 全攒
        在内存里返回。

        返回流式时为 None，否则为堆叠结果。
        """
        if not trajectories:
            raise ValueError("trajectories 为空，没有可跑的轨迹")

        recurrent_transform = self._transform("recurrent")
        output_transform = self._transform("output")
        use_gpu = getattr(self.model, "gpu_state", False)
        if use_gpu and recurrent_transform is not None:
            raise ValueError("GPU 常驻模式暂不支持 recurrent Processor")

        # 每条轨迹一份独立缓冲：forward 可能原地改，且 state=result 回填会替换它。
        if use_gpu:
            states = [self.model.to_gpu(t.state) for t in trajectories]
        else:
            states = [_copy_state(t.state) for t in trajectories]
        in_frames = trajectories[0].state.shape[1]
        init_times = [pd.to_datetime(t.init_time) for t in trajectories]

        # 时间条件的口径按模型声明，不要「统一」—— 两家官方脚本确实不同：
        #   fuxi 系  valid = init + step*interval     （输入窗末帧时刻）
        #            官方 inference.py 喂 prepare_features 的是 forecast_time+t*dt
        #   FengQing valid = init + (step+1)*interval （目标预报时刻）
        #            官方 inference_fengqing.py 的 temporal_condition 取目标时刻
        # 模型类用 valid_time_step_offset 声明（缺省 0 = fuxi 系口径）。
        offset = getattr(self.model, "valid_time_step_offset", 0)
        windowed = getattr(self.model, "windowed_output", True)

        streaming = on_step is not None
        collected = None if streaming else []
        step_times = []

        for step in range(steps):
            t0 = perf_counter()
            parts = []
            for index, init_time in enumerate(init_times):
                valid_time = init_time + pd.Timedelta(
                    hours=(step + offset) * hour_interval)

                if use_gpu:
                    result = self.model.forward_gpu(
                        states[index], step, valid_time)
                    states[index] = result
                    parts.append(self.model.to_numpy(result)[0, -1])
                    continue

                result = self.model.forward(states[index], step, valid_time)
                next_state = self._next_state(
                    states[index], result, in_frames, windowed)
                # 末帧 = 本步预报。先取出副本再回填 —— recurrent Processor
                # （如 zero_channels）是原地改，不复制会把已取出的预报也改掉。
                pred = next_state[0, -1]
                parts.append(
                    pred.clone() if isinstance(pred, torch.Tensor) else pred.copy())
                states[index] = (
                    recurrent_transform(next_state)
                    if recurrent_transform is not None else next_state)

            step_state = _stack(parts, 0)
            if output_transform is not None:
                step_state = output_transform(step_state)
            if streaming:
                on_step(step, step_state)
            else:
                collected.append(step_state)

            step_times.append(perf_counter() - t0)
            if progress:
                self._log_progress(
                    step, steps, hour_interval, step_times, progress_label)

        del states
        gc.collect()
        if streaming:
            return None
        return _stack(collected, 1)

    @staticmethod
    def _next_state(prev, result, in_frames, windowed):
        """把 forward 的输出整理成下一拍的完整输入窗。

        模型有两种输出形状，本方法是它们唯一的汇合点：

        **窗口型**（``windowed_output=True``，缺省）：图自己做滑窗，喂
        ``[t-1, t]`` 出 ``[t, t+1]``，输出帧数 == 输入帧数。直接整体回填，与官方
        ``inference.py`` 的 ``new_input = model.run(...)`` 一致（第 0 帧是图内
        回显帧，比手工滑窗更忠实）。fuxi_ens / fuxi21 属此类。

        **单步型**（``windowed_output=False``）：图只出预报那一帧，由这里滑窗成
        ``[输入末帧, 预报]``。iwc_fgvp_gdn2 / FengQing 属此类 —— 它们原先各自在
        ``forward`` 里手工 concat（fgvp）或干脆拒绝实现 ``forward``（FengQing），
        就是因为旧契约只承认窗口型，逼单步型假装自己是窗口型。
        """
        if windowed:
            if result.shape[1] != in_frames:
                raise ValueError(
                    f"模型声明 windowed_output=True，但输出 {result.shape[1]} 帧、"
                    f"输入 {in_frames} 帧，不一致，无法做 state=result 回填。"
                    "若该模型只输出预报帧，请在模型类声明 windowed_output=False。")
            return result
        # 单步型：输出可以是 (1,1,C,H,W) 或 (1,C,H,W)，都归一成前者再滑窗。
        if result.ndim == prev.ndim - 1:
            result = result[:, None]
        if result.shape[1] != 1:
            raise ValueError(
                f"模型声明 windowed_output=False，应只输出 1 帧，"
                f"实际 {result.shape[1]} 帧")
        last = prev[:, -in_frames + 1:] if in_frames > 1 else prev[:, :0]
        if isinstance(result, torch.Tensor):
            last = last if isinstance(last, torch.Tensor) else torch.as_tensor(last)
            return torch.cat([last, result], dim=1)
        return np.concatenate([np.asarray(last), np.asarray(result)], axis=1)

    def _log_progress(self, step, steps, hour_interval, step_times, label):
        """每隔 _PROGRESS_INTERVAL 步打一条完整行（带换行）。

        逐步 \\r 进度条在多卡并发时会互相覆盖刷屏，改成稀疏完整行就不会乱。
        """
        done = step + 1
        if done != steps and done % _PROGRESS_INTERVAL:
            return
        elapsed = sum(step_times)
        eta = elapsed / done * (steps - done)
        prefix = f"{label} " if label else ""
        log.info(
            "%s预报 %.1f/%.1f 天，剩 %.1f 天，预计还需 %s",
            prefix,
            done * hour_interval / 24.0,
            steps * hour_interval / 24.0,
            (steps - done) * hour_interval / 24.0,
            _fmt_dur(eta),
        )
