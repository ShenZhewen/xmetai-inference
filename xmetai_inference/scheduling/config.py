# -*- coding: utf-8 -*-
"""调度超参数：config 里的 ``scheduling={...}`` 解析成 :class:`SchedulingConfig`。

这些量放在 config 而不是命令行，理由和 times / steps / members 一样：它们属于
「这份配方怎么跑」，要跟着配方进版本控制、要能被 --config 的外部配方一起带走。
命令行只留三个日常要改的覆盖项：``--procs-per-gpu`` / ``--schedule`` / ``--resume``。

字段含义（都是可调超参数，都有缺省）：

  mode                 claim = 任务级动态认领（默认）；static = 旧静态分片（回退开关）
  procs_per_gpu        每张卡起几个 worker 进程（>1 时显存按份数分，见 gpu_mem）
  members_per_task     一个任务装几个集合成员；0 = 自动（1 个成员一个任务）
  claim_order          认领偏好：auto = m1a 按成员优先（省 session 换入）、其余按起报
                       优先（省输入重读）；也可写死 member / init
  resume               断点续跑口径：outputs = 产物齐全即跳过（默认）；marker = 只认完成
                       标记；off = 整份重算（只认**本次运行**写的标记，旧标记不算数，
                       也不需要手删 {output_dir}/.xmetai_plan 就能重跑一遍）
  task_retries         单个任务最多重跑几次（超过判失败、不再重派）
  max_passes           主控最多重派几轮（只有某轮零进展才重派）
  heartbeat_interval   worker 心跳写盘间隔（秒）
  heartbeat_timeout    心跳超过这么久判该 worker 已死、它的任务可被回收（秒）
  task_ttl_seconds     认领的硬租约上限（秒）；0 = 关，只靠心跳判死
  prefetch_tasks       预取队列深度（任务数）：认领与读帧在后台线程做，GPU 不等 IO
  input_cache          输入帧缓存条数（同一起报的多个成员任务复用同一份输入；0 = 关）
  poll_interval        无任务可认领时的轮询间隔（秒）
  monitor_interval     主控进度汇报间隔（秒）
  atomic_output        落盘先写隐藏临时名再 os.replace，下游读不到半成品
  plan_dir             计划/认领目录；空 = {output_dir}/.xmetai_plan
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields


MODES = ("claim", "static")
RESUME_MODES = ("outputs", "marker", "off")
CLAIM_ORDERS = ("auto", "member", "init")


@dataclass
class SchedulingConfig:
    """一次运行的调度配方（由 config 的 ``scheduling`` dict 解析而来）。"""

    mode: str = "claim"
    procs_per_gpu: int = 1
    members_per_task: int = 0
    claim_order: str = "auto"
    resume: str = "outputs"
    task_retries: int = 1
    max_passes: int = 3
    heartbeat_interval: float = 15.0
    heartbeat_timeout: float = 120.0
    task_ttl_seconds: float = 0.0
    prefetch_tasks: int = 2
    input_cache: int = 2
    poll_interval: float = 5.0
    monitor_interval: float = 30.0
    atomic_output: bool = True
    plan_dir: str = ""

    @classmethod
    def from_config(cls, value):
        """config 的 ``scheduling`` 字段 -> SchedulingConfig（None = 全默认）。

        键名写错必须报错而不是被静默忽略：这里的开关有好几个属于「不生效就等于
        没升级」（比如把 resume 写成 resume_mode），静默忽略会让人以为改过了。
        """
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError(
                f"config 的 scheduling 必须是 dict，实际是 {type(value).__name__}")
        data = dict(value)
        unknown = sorted(set(data) - set(FIELD_TYPES))
        if unknown:
            raise ValueError(
                f"config 的 scheduling 里有不认识的键：{', '.join(unknown)}；"
                f"可选：{', '.join(sorted(FIELD_TYPES))}")
        coerced = {}
        for key, raw in data.items():
            if key == "atomic_output":
                coerced[key] = raw if isinstance(raw, bool) else _as_bool(raw)
                continue
            caster = FIELD_TYPES[key]
            try:
                coerced[key] = caster(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"config 的 scheduling.{key} 值非法（{raw!r}）：{error}") from None
        config = cls(**coerced)
        config.validate()
        return config

    def validate(self):
        if self.mode not in MODES:
            raise ValueError(f"scheduling.mode 必须是 {MODES}，实际 {self.mode!r}")
        if self.resume not in RESUME_MODES:
            raise ValueError(
                f"scheduling.resume 必须是 {RESUME_MODES}，实际 {self.resume!r}")
        if self.claim_order not in CLAIM_ORDERS:
            raise ValueError(
                f"scheduling.claim_order 必须是 {CLAIM_ORDERS}，"
                f"实际 {self.claim_order!r}")
        if self.procs_per_gpu < 1:
            raise ValueError("scheduling.procs_per_gpu 必须 >= 1")
        if self.members_per_task < 0:
            raise ValueError("scheduling.members_per_task 不能为负（0 = 自动）")
        if self.task_retries < 0:
            raise ValueError("scheduling.task_retries 不能为负")
        if self.max_passes < 1:
            raise ValueError("scheduling.max_passes 必须 >= 1")
        if self.heartbeat_interval <= 0:
            raise ValueError("scheduling.heartbeat_interval 必须 > 0")
        if self.heartbeat_timeout <= self.heartbeat_interval:
            raise ValueError(
                "scheduling.heartbeat_timeout 必须大于 heartbeat_interval"
                f"（{self.heartbeat_timeout} <= {self.heartbeat_interval}），"
                "否则活着的 worker 会被判死、任务被重复认领")
        if self.task_ttl_seconds < 0:
            raise ValueError("scheduling.task_ttl_seconds 不能为负（0 = 关）")
        if self.prefetch_tasks < 1:
            raise ValueError("scheduling.prefetch_tasks 必须 >= 1")
        if self.input_cache < 0:
            raise ValueError("scheduling.input_cache 不能为负（0 = 关）")
        if self.poll_interval <= 0:
            raise ValueError("scheduling.poll_interval 必须 > 0")
        if self.monitor_interval <= 0:
            raise ValueError("scheduling.monitor_interval 必须 > 0")
        return self

    def members_per_task_for(self, members):
        """本任务装几个成员：显式配了就用它（不超过总数），否则 1。"""
        if self.members_per_task > 0:
            return max(1, min(int(self.members_per_task), int(members)))
        return 1

    def claim_order_for(self, perturbed):
        """认领偏好：扰动档的成员各是一份冻结权重，连跑同一成员能省一次 session 换入。"""
        if self.claim_order != "auto":
            return self.claim_order
        return "member" if perturbed else "init"


FIELD_TYPES = {
    "mode": str,
    "procs_per_gpu": int,
    "members_per_task": int,
    "claim_order": str,
    "resume": str,
    "task_retries": int,
    "max_passes": int,
    "heartbeat_interval": float,
    "heartbeat_timeout": float,
    "task_ttl_seconds": float,
    "prefetch_tasks": int,
    "input_cache": int,
    "poll_interval": float,
    "monitor_interval": float,
    "atomic_output": bool,
    "plan_dir": str,
}


def resolve_intra_op_threads(override, workers):
    """ORT 的 intra_op 线程数：显式配了就用，否则 = 核数 / worker 数（至少 1）。

    不显式设置时 ORT 默认吃满物理核；本框架同时起 workers 个进程，于是 CPU 侧
    线程被超订 workers 倍（zarr 解码、CPU 算子、拷贝都在抢这些线程）。
    """
    if override and int(override) > 0:
        return int(override)
    import os

    cores = os.cpu_count() or 1
    return max(1, int(cores) // max(1, int(workers)))


def _as_bool(value):
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"不是布尔值：{value!r}")


_DECLARED = {field.name for field in fields(SchedulingConfig)}
_MISSING = _DECLARED - set(FIELD_TYPES)
if _MISSING:  # pragma: no cover - 防呆：加了字段忘了登记类型
    raise RuntimeError(f"SchedulingConfig 的字段没有登记类型：{sorted(_MISSING)}")


__all__ = [
    "CLAIM_ORDERS",
    "MODES",
    "RESUME_MODES",
    "SchedulingConfig",
    "resolve_intra_op_threads",
]
