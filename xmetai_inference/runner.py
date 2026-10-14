# -*- coding: utf-8 -*-
"""自回归 rollout 编排层：把模型的单步 forward 串成多轨迹自回归循环。

一条 Trajectory = 一条独立轨迹。集合预报（members>1）是 N 条轨迹共享同一初始
态、init_time 相同（扰动来自图内随机算子，框架不加扰动）；批量确定性是 N 个起
报各一条轨迹，各有自己的初始态与 init_time。
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

_PROGRESS_INTERVAL = 5   # 每隔多少步打一条完整进度行

#: model_batch 生效 / 退回的日志标记，集成档 import 它们判「批量化有没有进图」。
BATCH_ACTIVE_MARK = "前向批生效"
BATCH_FALLBACK_MARK = "模型未声明 batch_axis_dynamic"

#: 已经打过的 model_batch 提示，按 key 在进程内去重（集合路径下每个起报都会新建
#: 一个 Rollout，不去重会被打「起报数」遍）。
_batch_notices = set()


def _notice_once(key, level, message, *args):
    """同一条 model_batch 提示在一个进程里只打一次。"""
    if key in _batch_notices:
        return
    _batch_notices.add(key)
    log.log(level, message, *args)


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


def _cat(values, axis=0):
    """torch 用 torch.cat，numpy 用 np.concatenate。"""
    if values and isinstance(values[0], torch.Tensor):
        return torch.cat(values, dim=axis)
    return np.concatenate([np.asarray(v) for v in values], axis=axis)


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
            progress=True, progress_label="", model_batch=1):
        """跑 steps 步，每步对全部轨迹各前向一次。

        on_step(step_index, step_state) 每步回调一次，step_state 形状
        (n_trajectories, C, H, W)，顺序与 trajectories 一致。传了 on_step 就走流式
        （内存不随 steps 增长），否则把 (n_traj, steps, C, H, W) 攒在内存里返回；
        流式时返回 None。

        ``model_batch`` 是一次 forward 最多堆几条轨迹（缺省 1 = 逐条）。前提：图的
        输入/输出 batch 轴都是符号维（模型层用 ``batch_axis_dynamic`` 声明，没声明
        就退回 1）；不能与 gpu_state=True 同用（互斥，直接报错）；各轨迹时刻可以
        不同 —— step/hour/doy 按 (k,) 向量喂。forward override 里写死 batch=1 的
        模型类（fengqing / pangu）不支持，拿到 k>1 会炸形状。

        显存按 k 倍涨：0.25° 78 通道两帧的轨迹状态约 648 MB，k 照着可用显存给。
        生效/退回各留一行日志（``BATCH_ACTIVE_MARK`` / ``BATCH_FALLBACK_MARK``）。
        """
        if not trajectories:
            raise ValueError("trajectories 为空，没有可跑的轨迹")

        # 跨步有状态模型（Pangu 的 24h 锚点）在每批轨迹开始前清空；
        # BaseInferModel 缺省 no-op。
        reset = getattr(self.model, "reset_runtime_state", None)
        if reset is not None:
            reset()

        recurrent_transform = self._transform("recurrent")
        output_transform = self._transform("output")
        use_gpu = getattr(self.model, "gpu_state", False)
        if use_gpu and recurrent_transform is not None:
            raise ValueError("GPU 常驻模式暂不支持 recurrent Processor")

        if model_batch < 1:
            raise ValueError(f"model_batch 必须 >= 1，实际 {model_batch}")
        if model_batch > 1:
            if use_gpu:
                raise ValueError(
                    "model_batch>1 与 gpu_state=True 互斥：GPU 常驻分支"
                    "（forward_gpu）的 _spare buffer 复用假设固定 shape，"
                    "动态 batch 轴建不出 buffer。")
            if not getattr(self.model, "batch_axis_dynamic", False):
                # 图是静态 batch 时喂 k>1 会被 ORT 拒，退回 1 至少让这一趟跑完。
                _notice_once(
                    ("fallback", type(self.model).__name__),
                    logging.WARNING,
                    BATCH_FALLBACK_MARK +
                    "（导出时 batch 轴不是符号维），model_batch=%d 退回 1；"
                    "要真多 batch 推理得重导动态 batch 的图。",
                    model_batch)
                model_batch = 1
            else:
                _notice_once(
                    ("active", type(self.model).__name__, model_batch),
                    logging.INFO,
                    BATCH_ACTIVE_MARK +
                    "：一次 forward 最多 %d 条轨迹（图的输入/输出 batch 轴都是符号维）",
                    model_batch)

        chunks = [list(range(start, min(start + model_batch, len(trajectories))))
                  for start in range(0, len(trajectories), model_batch)]

        # 每块一份独立缓冲（forward 可能原地改）；多条用 _cat 不用 _stack ——
        # state 已带 batch 维 (1,F,C,H,W)，stack 会多出一维。
        if use_gpu:
            # 上面已挡掉 model_batch>1，块必是单条，下标与轨迹一一对应
            states = [self.model.to_gpu(t.state) for t in trajectories]
        else:
            states = [
                _copy_state(trajectories[chunk[0]].state) if len(chunk) == 1
                else _cat([_copy_state(trajectories[i].state) for i in chunk], 0)
                for chunk in chunks
            ]
        in_frames = trajectories[0].state.shape[1]
        init_times = [pd.to_datetime(t.init_time) for t in trajectories]

        # 时间条件口径按模型声明：0 = fuxi 系（valid = init + step*interval）；
        # FengQing 用 +1（目标预报时刻）。模型类用 valid_time_step_offset 声明。
        offset = getattr(self.model, "valid_time_step_offset", 0)
        windowed = getattr(self.model, "windowed_output", True)

        streaming = on_step is not None
        collected = None if streaming else []
        step_times = []

        for step in range(steps):
            t0 = perf_counter()
            parts = []
            for chunk_index, chunk in enumerate(chunks):
                valid_times = [
                    init_times[index] + pd.Timedelta(
                        hours=(step + offset) * hour_interval)
                    for index in chunk
                ]

                if use_gpu:
                    result = self.model.forward_gpu(
                        states[chunk_index], step, valid_times[0])
                    states[chunk_index] = result
                    parts.append(self.model.to_numpy(result)[0, -1])
                    continue

                prev = states[chunk_index]
                # 单条块喂标量时刻；多条块喂时刻列表，_scalar_feeds 把后者拼成
                # (k,) 的 step/hour/doy，正是动态 batch 图要的形状。
                result = self.model.forward(
                    prev, step,
                    valid_times[0] if len(chunk) == 1 else valid_times)
                next_state = self._next_state(prev, result, in_frames, windowed)
                # 先取副本再回填：recurrent Processor（如 zero_channels）是原地改，
                # 不复制会把已取出的预报也改掉。
                preds = next_state[:, -1]
                for row in range(preds.shape[0]):
                    frame = preds[row]
                    parts.append(
                        frame.clone() if isinstance(frame, torch.Tensor)
                        else frame.copy())
                states[chunk_index] = (
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

        ``windowed_output=True``（缺省）：图自己做滑窗，输出帧数 == 输入帧数，整体
        回填。``windowed_output=False``：图只出预报那一帧，由这里滑窗成
        ``[输入末帧, 预报]``。
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
        """每隔 _PROGRESS_INTERVAL 步打一条完整行（多卡并发时进度条会互相覆盖）。"""
        done = step + 1
        # 首步必打：cudnn 算法穷举搜索可能在第一个前向耗几分钟。
        first = done == 1
        if done != steps and not first and done % _PROGRESS_INTERVAL:
            return
        elapsed = sum(step_times)
        eta = elapsed / done * (steps - done)
        prefix = f"{label} " if label else ""
        note = "（首步含 cudnn 算法穷举搜索，偏慢属正常；ETA 按首步外推会偏大）" if first else ""
        log.info(
            "%s预报 %.1f/%.1f 天，剩 %.1f 天，预计还需 %s%s",
            prefix,
            done * hour_interval / 24.0,
            steps * hour_interval / 24.0,
            (steps - done) * hour_interval / 24.0,
            _fmt_dur(eta),
            note,
        )
