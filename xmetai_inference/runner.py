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

#: model_batch 生效 / 退回的日志标记。集成档（tests/integration 的 fgvp_ens 档）
#: import 这两个常量来判「批量化到底有没有进图」—— 图是静态 batch 轴时 runner 只打
#: 一条 WARNING 就退回 1，这一趟照跑完，其余用例照样绿，却跟批量化无关。
BATCH_ACTIVE_MARK = "前向批生效"
BATCH_FALLBACK_MARK = "模型未声明 batch_axis_dynamic"

#: 已经打过的 model_batch 提示，按 key 在进程内去重。集合路径下每个起报新建一个
#: Rollout 实例（cli.py 里 Rollout(...) 在起报循环内），不去重的话同一条提示会被
#: 打「起报数」遍（348 个起报 = 348 行噪音，正好盖住真正要看的那行）。
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
        (n_trajectories, C, H, W)，顺序与传入的 trajectories 一致。传了 on_step
        就走流式（内存不随 steps 增长）；不传则把 (n_traj, steps, C, H, W) 全攒
        在内存里返回。

        ``model_batch`` 是一次 forward 最多堆几条轨迹（缺省 1 = 逐条前向，与历史
        行为逐位一致）。>1 时把轨迹按块 _cat 成 (k, F, C, H, W) 一次喂给模型，
        GPU 算力占用明显不同（Mamba 这类顺序扫描骨干尤其吃这个）。三个前提：

          * 图导出时输入/输出的 batch 轴都是符号维（动态）—— 模型层用
            ``batch_axis_dynamic`` 声明，没声明就自动退回 1（见下）；
          * 不能与 gpu_state=True 同用（配置互斥，直接报错，不静默退回）；
          * 各轨迹的时刻可以不同 —— step/hour/doy 按 (k,) 向量喂，同一批里
            多成员、多起报都成立。但 forward override 里写死 batch=1 的模型类
            （fengqing / pangu）不支持：它们拿到 k>1 的状态会在前向里炸形状/类型
            （不会静默算错，但报错信息里看不出是 model_batch 惹的），别给它们开。

        显存按 k 倍涨：一条轨迹的状态是 (F, C, H, W) 浮点，0.25° 78 通道两帧约
        648 MB，k 要照着可用显存给。

        生效还是退回，进程内各留一行日志（``BATCH_ACTIVE_MARK`` /
        ``BATCH_FALLBACK_MARK``），只打一次 —— 集合路径下每个起报都会新起一个
        Rollout，打多了是噪音。

        返回流式时为 None，否则为堆叠结果。
        """
        if not trajectories:
            raise ValueError("trajectories 为空，没有可跑的轨迹")

        # 轨迹间状态重置：跨步有状态模型（Pangu 的 24h 锚点）在每批轨迹开始前清空，
        # 避免上一批残留、限制常驻内存；BaseInferModel 缺省 no-op。
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
                # 不静默退回：这是配置互斥，悄悄跑成逐条会让人以为批量化生效了。
                raise ValueError(
                    "model_batch>1 与 gpu_state=True 互斥：GPU 常驻分支"
                    "（forward_gpu）的 _spare buffer 复用假设固定 shape，"
                    "动态 batch 轴建不出 buffer。")
            if not getattr(self.model, "batch_axis_dynamic", False):
                # 静默退回会让人以为批量化生效了，所以先喊一声。图是静态 batch 时
                # 喂 k>1 会被 ORT 拒，退回 1 至少让这一趟跑完。
                _notice_once(
                    ("fallback", type(self.model).__name__),
                    logging.WARNING,
                    BATCH_FALLBACK_MARK +
                    "（导出时 batch 轴不是符号维），model_batch=%d 退回 1；"
                    "要真多 batch 推理得重导动态 batch 的图。",
                    model_batch)
                model_batch = 1
            else:
                # 生效也留一行：批量化是「静默加速」，日志里没有痕迹的话，跑完只
                # 能靠墙钟时间猜它有没有生效（集成档就是靠这行判的）。
                _notice_once(
                    ("active", type(self.model).__name__, model_batch),
                    logging.INFO,
                    BATCH_ACTIVE_MARK +
                    "：一次 forward 最多 %d 条轨迹（图的输入/输出 batch 轴都是符号维）",
                    model_batch)

        # 轨迹按 model_batch 分块，每块一次前向。model_batch=1 时每块一条，
        # 与逐轨迹路径逐位等价。
        chunks = [list(range(start, min(start + model_batch, len(trajectories))))
                  for start in range(0, len(trajectories), model_batch)]

        # 每块一份独立缓冲：forward 可能原地改，且 state=result 回填会替换它。
        # 多条用 _cat 不用 _stack：state 已带 batch 维 (1,F,C,H,W)，要拼成
        # (k,F,C,H,W)，stack 会多出一维。
        if use_gpu:
            # model_batch>1 已在上面挡掉，块必是单条，下标与轨迹一一对应
            states = [self.model.to_gpu(t.state) for t in trajectories]
        else:
            states = [
                _copy_state(trajectories[chunk[0]].state) if len(chunk) == 1
                else _cat([_copy_state(trajectories[i].state) for i in chunk], 0)
                for chunk in chunks
            ]
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
                # 单条块喂标量时刻（与逐条路径逐位一致）；多条块喂时刻列表 ——
                # _scalar_feeds 把后者拼成 (k,) 的 step/hour/doy，正是动态
                # batch 图要的形状（同一批里各轨迹的时刻可以不同，所以逐行给）。
                result = self.model.forward(
                    prev, step,
                    valid_times[0] if len(chunk) == 1 else valid_times)
                next_state = self._next_state(prev, result, in_frames, windowed)
                # 末帧 = 本步预报，逐轨迹切一份，下标语义与 (n_traj, C, H, W)
                # 数组一致。先取副本再回填 —— recurrent Processor（如
                # zero_channels）是原地改，不复制会把已取出的预报也改掉。
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
        # 首步必打：cudnn 算法穷举搜索发生在第一个前向，可能几分钟没有动静，
        # 让用户知道在等什么；之后的步才按 _PROGRESS_INTERVAL 稀疏输出。
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
