# -*- coding: utf-8 -*-
"""多进程 / 多卡调度：任务级动态认领。

把一次推理拆成可独立认领、独立落盘的 ``(起报, 成员块)`` 任务，worker 自己去共享盘
上认领。相对旧的静态分片，换来三件事：

  * **没有拖尾**：慢的 worker 少拿、快的多拿；谁先干完谁接着认领，不再空等。
  * **断了能续**：完成标记写在共享盘上，重启后已完成的任务直接跳过；崩掉的 worker
    占着的任务会被别人回收（判据见 :mod:`xmetai_inference.scheduling.store`）。
  * **进程数不再等于卡数**：``procs_per_gpu`` 可以让两张卡跑三个进程，也可以让一张卡
    跑两个进程（显存按份数分）。

超参数全部在 config 里配（见 :class:`SchedulingConfig`）：
``InferConfig(scheduling={...})``。命令行只留三个日常覆盖项。

  * 任务表与指纹   plan.TaskPlan / plan.run_payload
  * 共享盘账本     store.ClaimStore / store.LocalIndex
  * 认领 + 预取    feed.TaskFeed
  * worker 心跳    heartbeat.Heartbeat
  * 自检           selftest.run_selftest
"""
from .config import SchedulingConfig, resolve_intra_op_threads
from .feed import TaskFeed
from .heartbeat import Heartbeat
from .plan import (
    LAYOUT_VERSION,
    Task,
    TaskPlan,
    fingerprint,
    init_dir_name,
    member_blocks,
    model_identity,
    run_payload,
)
from .store import ClaimStore, LocalIndex, new_launch_id

__all__ = [
    "ClaimStore",
    "Heartbeat",
    "LAYOUT_VERSION",
    "LocalIndex",
    "SchedulingConfig",
    "Task",
    "TaskFeed",
    "TaskPlan",
    "fingerprint",
    "init_dir_name",
    "member_blocks",
    "model_identity",
    "new_launch_id",
    "resolve_intra_op_threads",
    "run_payload",
]
