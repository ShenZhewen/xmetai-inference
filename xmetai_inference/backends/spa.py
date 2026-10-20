# -*- coding: utf-8 -*-
"""潜空间 AR(1) 的推理驱动：``Ar1Driver`` 持 ξ_t 并递推，``LatentStepStream``
决定每次 forward 用哪几行 ξ、何时 reset / advance。

图手术在 ``weights/spa.py`` / ``weights/spa_pt2.py``；ONNX 与 PT2 后端共用本实现。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # 只用于类型标注；运行时避免 weights -> backends 的反向依赖。
    from xmetai_inference.weights.spa import LatentForcingSpec


def _rank_offset() -> int:
    """本进程的 rank；必须计入随机熵，否则多卡各 rank 的 ξ 流逐位相同。"""
    for name in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                continue            # 脏值当没有，退化成单进程
    return 0


def _rank_entropy(spec, member_index, slot):
    """驱动侧随机流的熵：种子 + 成员号 + 槽位；各成员必须互不相同。"""
    return [int(spec.seed or 0), 7919, int(member_index or 0), int(slot or 0)]


def _world_size() -> int:
    """卡数（读 cli 设置的 WORLD_SIZE）；只在 ``demean_members`` 打开时用到。

    slot 取 rank（cli 按 ``range(rank, members, world_size)`` 切成员），卡数不对
    就重放不出真实成员的 ξ 流。
    """
    try:
        return int(os.environ.get("WORLD_SIZE", "1") or 1)
    except (TypeError, ValueError):
        return 1


class Ar1Driver:
    """持有 ξ_t 并按 ``ξ_{t+1} = ρξ_t + sqrt(1-ρ²)ε_t`` 递推（平稳、单位方差）。

    状态放在驱动侧是因为 ONNX 单步图没有可靠的跨步状态；成员差异只来自随机熵。
    """

    def __init__(self, spec: LatentForcingSpec, n_targets: int, member_index: int = 0,
                 slot: int = 0, batch: int = 1, hour_interval: float | None = None,
                 world_size: int = 1, _build_cohort: bool = True):
        self.spec = spec
        self.n_targets = int(n_targets)
        self.member_index = int(member_index)
        self.slot = int(slot)
        self.rho = spec.rho(hour_interval)
        self.rng = np.random.default_rng(
            np.random.SeedSequence(_rank_entropy(spec, member_index, slot)))
        self.state = self._draw(int(batch))
        self.steps = 0
        # 时变幅度 a(t)，由 LatentStepStream 按 step 设置；1.0 = 不缩放。
        self.amp = 1.0
        # 跨成员去均值：用真实成员的种子重放其他成员的 ξ 流来算均值。
        # `_build_cohort=False` 供 cohort 内部使用，防无限递归。
        self.demean_members = int(getattr(spec, "demean_members", 0) or 0)
        self.cohort: list = []
        if _build_cohort and self.demean_members > 1:
            world = max(1, int(world_size or 1))
            self.cohort = [
                Ar1Driver(spec, n_targets, member_index=other, slot=other % world,
                          batch=batch, hour_interval=hour_interval,
                          world_size=world, _build_cohort=False)
                for other in range(self.demean_members) if other != self.member_index
            ]

    def _draw(self, batch: int) -> np.ndarray:
        return self.rng.standard_normal(
            (int(batch), self.n_targets, int(self.spec.rank))).astype(np.float32)

    def _scaled(self, block: np.ndarray) -> np.ndarray:
        """给出口 ξ 乘 a(t)；amp == 1 时原样返回。

        只作用于出口，绝不写回 ``self.state``：递推必须保持平稳单位方差，否则 τ
        不再是相关时间。
        """
        if self.amp == 1.0:
            return block
        return (block * np.float32(self.amp)).astype(np.float32)

    def _slice_raw(self, index: int, width: int = 1) -> np.ndarray:
        """第 index..index+width 行的原始 ξ；行数不够自动加宽。

        cohort 成员必须同步加宽，否则行号对不上。
        """
        need = int(index) + int(width)
        if need > self.state.shape[0]:
            self.state = np.concatenate(
                [self.state, self._draw(need - self.state.shape[0])], axis=0)
        return self.state[int(index):need]

    def _export(self, index: int, width: int = 1) -> np.ndarray:
        """本次要喂进图的 ξ：先跨成员去均值（若开），再乘 a(t)（若开）。"""
        block = self._slice_raw(index, width)
        if self.cohort:
            total = block.astype(np.float64)
            for driver in self.cohort:
                total += driver._slice_raw(index, width)
            block = (block - (total / self.demean_members).astype(np.float32)).astype(np.float32)
        return self._scaled(block)

    def value(self, batch: int) -> np.ndarray:
        """本步要喂进图的 ξ（形状 [batch, T, R]）。"""
        if int(batch) != self.state.shape[0]:
            self.reset(int(batch))
        if not self.cohort:
            return self._scaled(self.state)          # 不复制，保持原有行为
        return self._export(0, self.state.shape[0])

    def row(self, index: int, width: int = 1) -> np.ndarray:
        """第 index 行起的 ξ（``[width, T, R]``）；行 = 一条独立的 AR(1) 流。

        行号由 ``LatentStepStream`` 分配，这里只取行；行数不够自动加宽。
        """
        return self._export(index, width)

    def advance(self) -> None:
        noise = self.rng.standard_normal(self.state.shape).astype(np.float32)
        decay = float(np.sqrt(max(0.0, 1.0 - self.rho * self.rho)))
        self.state = (self.rho * self.state + decay * noise).astype(np.float32)
        self.steps += 1
        for driver in self.cohort:
            driver.advance()

    def reset(self, batch: int | None = None) -> None:
        """重新抽 ξ_0（每条轨迹开始时调用）。种子流不重置 —— 重抽即新轨迹。"""
        if batch is not None:
            self.batch = int(batch)
        width = int(getattr(self, "batch", self.state.shape[0]))
        self.state = self._draw(width)
        self.steps = 0
        for driver in self.cohort:
            driver.reset(width)

    def describe(self) -> str:
        amp = "" if self.amp == 1.0 else " a(t)=%.3f" % self.amp
        demean = "" if not self.cohort else " 去均值/%d成员" % self.demean_members
        return ("latent-ar1 member=%d slot=%d ρ=%.4f τ=%g 天 sigma=%g rank=%d 步数=%d%s%s"
                % (self.member_index, self.slot, self.rho, self.spec.tau_days,
                   self.spec.sigma, self.spec.rank, self.steps, amp, demean))


class LatentStepStream:
    """潜空间 AR(1) 的驱动状态机；ONNX / PT2 后端各持一份。

    step 变化时整批推进一次并归零槽位；step == 0 表示新轨迹批，全部行重抽。
    同一步内按调用次序发号，一次可消费 ``width`` 行（``model_batch > 1`` 时一次
    forward 堆 k 条轨迹，块起点 = 前几块宽度之和），行数不够自动加宽。
    返回起始行号，None = 未启用强迫。

    ``width=1`` 与逐轨迹调用逐位相同。一次只吃 1 行却喂 k 条轨迹会让块内成员
    几乎重复、集合静默欠散。
    """

    def __init__(self):
        self.ar1 = None
        self.key = None
        self.step = None
        self.slot = 0

    def __call__(self, forcing, step, *, n_targets, member_index=0,
                 hour_interval=None, width=1):
        """返回本次 forward 的起始行号，None = 未启用强迫。

        width = 本次 forward 堆了几条轨迹；槽位前进 width，调用方用
        ``ar1.row(index, width)`` 取行。
        """
        if not forcing or int(n_targets) <= 0:
            return None
        width = int(width)
        if width < 1:
            raise ValueError(f"width 必须 >= 1（一次 forward 至少一条轨迹），实际 {width}")
        # rank 必须计进熵：无权重扰动时成员身份来自轨迹行号，不同 rank 若用同一
        # 随机流，会静默产生重复成员。
        rank = int(_rank_offset())
        key = (int(member_index or 0), rank)
        if self.ar1 is None or self.key != key:
            self.ar1 = Ar1Driver(forcing, int(n_targets),
                                 member_index=key[0], slot=rank, batch=1,
                                 hour_interval=hour_interval,
                                 world_size=_world_size())
            self.key = key
            self.step = None
            self.slot = 0
        if int(step) != self.step:
            if int(step) == 0:
                # 新轨迹批：全部行重抽
                self.ar1.reset()
            else:
                # 新模式步：整批推进一次；step 0 喂 ξ_0，step t 喂推进 t 次的 ξ_t
                self.ar1.advance()
            self.step = int(step)
            self.slot = 0
            # a(t) 同一步内只定一次（可能有多块 width 进来）；拿不到就恒为 1
            amp_at = getattr(forcing, "amp_at", None)
            self.ar1.amp = float(amp_at(self.step)) if amp_at else 1.0
        index = self.slot
        # 本块吃掉 width 行；下一块从 index + width 起
        self.slot += width
        return index

    def release(self):
        """丢掉驱动状态（后端 release()/release_all() 时调，成员之间不共享 ξ）。"""
        self.ar1 = None
        self.key = None
        self.step = None
        self.slot = 0


def lag1_autocorr(values) -> float:
    """诊断用：一串标量序列的 lag-1 自相关（应 ≈ ρ）。"""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size < 3 or series.std() == 0:
        return float("nan")
    left, right = series[:-1], series[1:]
    return float(np.corrcoef(left, right)[0, 1])


__all__ = ["Ar1Driver", "LatentStepStream", "lag1_autocorr"]
