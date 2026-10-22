# -*- coding: utf-8 -*-
"""潜空间 AR(1) 的推理驱动（SPA 的运行时半边）。

图手术（把 ``spw/xi`` 输入与强迫链接进 ONNX / PT2 图）在
``weights/spa.py`` / ``weights/spa_pt2.py``；本模块只持有**推理时**的状态机：

  * ``Ar1Driver``：持 ξ_t，按 ``ξ_{t+1} = ρ ξ_t + sqrt(1-ρ²) ε_t`` 递推；
  * ``LatentStepStream``：在 runner 逐轨迹 / 批前向的编排下，决定本次 forward
    该拿哪几行 ξ，以及何时 reset / advance。

ONNX（``backends/onnx.py``）与 PT2（``backends/pt2.py``）共用这一份实现——
这段逻辑出过 A32 接线事故，两份拷贝比一次受控共享危险。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # 只用于类型标注；运行时避免 weights -> backends 的反向依赖。
    from xmetai_inference.weights.spa import LatentForcingSpec


def _rank_offset() -> int:
    """本进程的 rank（掺进驱动侧随机流的熵，防多卡各 rank 的噪声流逐位相同）。"""
    for name in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                continue            # 脏值当没有，宁可退化成单进程也别炸
    return 0


def _rank_entropy(spec, member_index, slot):
    """驱动侧随机流的熵：种子 + 成员号（+ 轨迹槽）。成员之间必须互不相同。"""
    return [int(spec.seed or 0), 7919, int(member_index or 0), int(slot or 0)]


def _world_size() -> int:
    """卡数。读法与 cli 一致（cli 给每个 worker 都设了 WORLD_SIZE），退回 1。

    只在 `demean_members` 打开时用到：本地重放全体成员时要复现**真实**成员的流，
    而真实成员的流 = `(seed, 7919, m, m % world_size)`（cli 按 `range(rank, members,
    world_size)` 切成员，`slot` 取的是 rank）。卡数不对，重放出来的就不是那 51 条流，
    均值踢扣不干净。
    """
    try:
        return int(os.environ.get("WORLD_SIZE", "1") or 1)
    except (TypeError, ValueError):
        return 1


class Ar1Driver:
    """持有 ξ_t 的递推状态：`ξ_{t+1} = ρ ξ_t + sqrt(1-ρ²) ε_t`（平稳、单位方差）。

    为什么状态放在驱动侧而不是图里：ONNX 的单步图没有可靠的跨步状态机制，而 AR(1) 恰恰
    必须跨步记忆。代价是每步要把 ξ 喂进图（T x R = 48 个数），收益是「同一份图被所有成员
    共享」——成员之间的差别只来自 ξ 的种子。

    成员独立性：熵 = (spec.seed, member_index, slot)。m1a 的成员循环会给每成员设
    member_index，所以「同一份图 + 不同 member_index」就得到互不相同、且可复现的成员。
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
        # 时变幅度 a(t)：由 LatentStepStream 在每次 step 变化时设。1.0 = 不缩放。
        self.amp = 1.0
        # 跨成员去均值（spec.demean_members）：本地重放**其他**成员的 ξ 流，仅为算均值。
        # 种子逐个与真实成员相同（member_index=m、slot=m % world_size，见 cli 的成员切分
        # `range(rank, members, world_size)`），所以自己不动、别人不动时，扣掉的正是真实
        # 的均值踢。`_build_cohort=False` 是给 cohort 内部那一层的（防无限递归）。
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
        """给出口 ξ 乘 a(t)。amp == 1 时**原样返回**（同一对象、不复制）——没给
        amp_profile 的臂因此与改动前逐位相同。

        缩放只作用于**出口**，绝不写回 self.state：ξ 的递推必须保持平稳单位方差
        （`ξ_{t+1} = ρξ_t + √(1-ρ²)ε`），把 a(t) 乘进状态会让它反馈进下一步、
        变成一条被自身包络驱动的非平稳过程（τ 就不再是相关时间了）。
        """
        if self.amp == 1.0:
            return block
        return (block * np.float32(self.amp)).astype(np.float32)

    def _slice_raw(self, index: int, width: int = 1) -> np.ndarray:
        """第 index..index+width 行的**原始** ξ（未去均值、未缩放）。行数不够自动加宽。

        加宽与去均值都要在这里发生：cohort 里的成员必须和本成员**同步加宽**，否则
        行号对不上，扣的就不是同一批轨迹的均值。
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
            return self._scaled(self.state)          # 与改动前逐位相同（同一对象）
        return self._export(0, self.state.shape[0])

    def row(self, index: int, width: int = 1) -> np.ndarray:
        """第 index 条轨迹的 ξ（[width, T, R]）。行 = 一条**独立**的 AR(1) 流，行间
        独立（同一 rng 流的先后抽样）；行数不够自动加宽，新行独立新抽、不影响已有行
        的递推。供 runner **逐轨迹**调 forward 的编排用（x.shape[0] 恒 1，batch 维
        到不了驱动侧）—— 发号（本步第几条轨迹拿第几行）由 onnx._latent_step_stream
        负责，这里只管取行。
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
    """潜空间 AR(1) 的驱动状态机（后端共享；onnx / pt2 各持一份实例）。

    职责：在「runner **逐轨迹**调 forward（x.shape[0] 恒 1）」的编排下，决定本次
    forward 调用该拿第几行 ξ、以及何时 reset / advance。返回行号，None = 不强迫。

    语义（2026-09-29 修正版；A32 首跑暴露的原接线错误见类尾）:

      * 一行 = 一条轨迹**独立**的 AR(1) 流（行间独立，来自同一 rng 流的先后抽样）；
      * step 变化 = 新的模式步：整批推进一次（advance 对 state 的 batch 维是向量化的，
        等价 batched forward 里的一步递推，ρ 按每步 6h 折算），槽位归零；
      * step == 0 = 新的轨迹批（新起报 / 新成员）：全部行重抽 —— 即原 reset 语义
        （重抽不重置随机流，「重抽 = 新轨迹」，整个任务的抽样序列仍可复现）；
      * 同一步内按调用次序轮流发第 0,1,2... 行（Ar1Driver.row），行数不够自动加宽；
      * **一次调用可以吃掉 k 行**（`width=k`，见下），槽位前进 k。

    ---- width：一次 forward 堆了 k 条轨迹（rollout_batch > 1）----

    runner 的 `batch_size=rollout_batch` 会把轨迹切成**连续**区间、每块一次 forward
    （`chunks = range(0, n, rb)`）。块内 k 条轨迹要的是**同一 index 起、横跨 k 行**的
    ξ，所以后端得把块宽告诉这里：`width=k` → 返回 index，`slot += k`，后端再
    `ar1.row(index, k)` 取 `[k, T, R]` 一次喂进去。块起点 = 上一块起点 + 上块宽度，
    与「rb=1 时第 j 条轨迹拿第 j 行」是同一个映射的推广。

    **不传 width（=1）时与旧行为逐位相同**，所以已经在跑的批次不受影响。
    width>1 的路径与 rb=1 逐位同流：行是 `Ar1Driver.row` 按需从同一条 rng 流里顺序补出来的，
    `_draw` 顺序填充（无跨调用缓存），所以「一次抽 k 行」与「抽 k 次每次一行」逐位相同
    ⇒ **rollout_batch 对集合中性**（回放测试 tmp/ar1_row_replay.py 第 4 条）。

    原接线错误是**相反**方向：width 恒 1 却按 k 条轨迹喂块 ⇒ 块内 k 条轨迹拿到
    row, row+1, ... 相邻行而不是同一 index 的 k 行，块内成员几乎重复、集合静默欠散。

    为什么不覆盖 reset_runtime_state：那是「换 session 之后状态还在不在」的契约，
    而 xi 的递推本来就在驱动侧、与 session 无关。用 step 判新轨迹，语义等价且不牵动
    那条契约。

    ---- 原接线错误（为什么不能直接 value()/advance()）----

    runner 是逐轨迹调 forward 的（`for index, init_time in enumerate(...)` 里
    `self.model.forward(states[index], ...)`），「一个 batch 多条轨迹、batch 维区分成员」
    的假设在这条编排下不成立。直接共用单行驱动（batch=1）时，同一 Rollout 的 n 条轨迹
    会穿插共用一条流：step 0 各自 reset 还各自独立，step >= 1 起全体消费同一条流的相邻
    相位 —— 成员间强迫相关 rho（tau=3 天时 rho=0.92，成员几乎重复、集合严重欠散），
    每成员自身的有效相关时间被压成 tau/n。成员间仍然互不相同，所以 check 不会报错 ——
    这正是它危险的地方。
    """

    def __init__(self):
        self.ar1 = None
        self.key = None
        self.step = None
        self.slot = 0

    def __call__(self, forcing, step, *, n_targets, member_index=0,
                 hour_interval=None, width=1):
        """返回本次 forward 该拿的**起始**行号（int），None = 未启用强迫。

        forcing / n_targets 由后端给（onnx：图 metadata 里的目标数；pt2：注入时
        记下的目标数）。后端自己先判 None 早退也行 —— 这里再判一次只是保险。

        width = 本次 forward 堆了几条轨迹（rollout_batch），槽位前进 width、由调用方
        用 ``ar1.row(index, width)`` 取行。缺省 1 = 逐轨迹调用，与旧行为逐位相同。
        """
        if not forcing or int(n_targets) <= 0:
            return None
        width = int(width)
        if width < 1:
            raise ValueError(f"width 必须 >= 1（一次 forward 至少一条轨迹），实际 {width}")
        # 进程 rank 必须计进熵：只开 latent_forcing（没有 m1a 权重）时，集合走的是
        # 「一个 session + 一批轨迹」那条路，member_index 恒为 0，成员身份 = 轨迹批里
        # 的行号。多卡按成员切时每个 rank 跑一部分成员，不把 rank 计进熵的话各 rank 的
        # xi 序列**逐位相同** —— 静默地变成重复成员。
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
                # 新轨迹批（新的起报 / 新的成员）：全部行重抽
                self.ar1.reset()
            else:
                # 新的模式步：整批（每行）推进一次。首个 step==0 的行由创建时的初始
                # 抽样给出，不推进 —— 所以这里恰好做到「step 0 喂 xi_0，step t 喂
                # 推进 t 次的 xi_t」。
                self.ar1.advance()
            self.step = int(step)
            self.slot = 0
            # 时变幅度 a(t)：只在**出口**缩放 ξ，不进图指纹（同 tau_days）。
            # 设在这里而不是每次 __call__，是因为同一步内会有多个 width 块反复进来，
            # 倍数按 step 定一次就够。forcing 理论上必是 spec（上面 rho 已经用了），
            # 但仍按属性取 —— 拿不到就恒 1，绝不因为一个新字段把老路径弄挂。
            amp_at = getattr(forcing, "amp_at", None)
            self.ar1.amp = float(amp_at(self.step)) if amp_at else 1.0
        index = self.slot
        # 本次调用吃掉了 width 条轨迹的行；同一步内的下一块从 index+width 起。
        # 这就是「rb=1 时第 j 条轨迹拿第 j 行」的推广（块起点 = 前几块宽度之和）。
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
