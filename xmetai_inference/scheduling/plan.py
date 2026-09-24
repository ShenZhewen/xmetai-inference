# -*- coding: utf-8 -*-
"""任务表：把一次推理切成可独立认领、独立落盘的 (起报, 成员块) 任务。

为什么需要它：旧的多卡调度在启动时就把「起报 × 成员」静态分给各 rank，分完不再变。
一个 rank 慢下来（IO 抖动、成员权重换入、被别的进程抢 CPU），其它 rank 只能空等；
起报数少于卡数时还有 rank 分不到任务直接退出。任务表把同样的工作量变成一堆可动态
认领的小任务：慢的少拿、快的多拿，中断过的任务也能被重新认领。

任务的输出仍落在原有布局上，下游（util/eval_*、perturbation/enscheck）无需改动：

  确定性（members == 1）：{out}/{起报}/{步号}.nc
  集合（members > 1）  ：{out}/{起报}/member_{成员3位}/{步号}.nc

``LAYOUT_VERSION`` 与模型身份一起进指纹：落盘布局或权重一变，旧任务表与旧完成标记
就整体失效（与 m1a 成员集目录名带落盘格式版本是同一个思路）。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import pandas as pd


# 落盘布局版本。改输出路径的任何一部分（目录名、文件名的位数、成员目录的语义）都要 +1。
LAYOUT_VERSION = 1


def init_dir_name(init):
    """起报时间 -> 输出目录名。

    00 UTC 用 YYYYMMDD，其他起报时次保留 YYYYMMDDHH，避免同日多时次互相覆盖。
    起报间隔混用（如既有 00Z 又有 12Z）会产生 8 位与 10 位并存的目录名，下游按目录名
    解析时要能同时吃两种——util/eval_single_util.py 的 _parse_init_dir 已兼容。
    """
    init = pd.Timestamp(init)
    return init.strftime("%Y%m%d" if init.hour == 0 and init.minute == 0 else "%Y%m%d%H")


@dataclass(frozen=True)
class Task:
    """一个可独立认领、独立落盘的工作单元：一个起报 × 一段连续成员。"""

    init: pd.Timestamp
    member_start: int
    member_count: int

    @property
    def key(self):
        """稳定的任务键（同时用作共享盘上的目录/文件名）。"""
        return f"{self.init:%Y%m%d%H}m{self.member_start:03d}x{self.member_count:03d}"

    def members(self):
        return list(range(self.member_start, self.member_start + self.member_count))

    def to_list(self):
        return [self.init.isoformat(), self.member_start, self.member_count]


def fingerprint(payload):
    """运行身份指纹：同样的输入必然得到同样的 16 位十六进制串。"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def model_identity(path):
    """权重文件身份（路径 + 大小 + mtime）：换了权重就该换指纹、旧产物不再复用。"""
    identity = {"path": os.path.abspath(path) if path else ""}
    try:
        stat = os.stat(identity["path"])
        identity["size"] = stat.st_size
        identity["mtime_ns"] = stat.st_mtime_ns
    except OSError:
        identity["size"] = -1
        identity["mtime_ns"] = -1
    return identity


def run_payload(*, model_path, steps, hour_interval, members, variables, inits,
                members_per_task):
    """参与指纹的运行参数。这里是「产物等价」的判据：改了其中任何一项都必须重算。"""
    return {
        "layout": LAYOUT_VERSION,
        "model": model_identity(model_path),
        "steps": int(steps or 0),
        "hour_interval": int(hour_interval or 0),
        "members": int(members),
        "members_per_task": int(members_per_task),
        "vars": sorted(str(item).strip().lower() for item in (variables or [])),
        "inits": [pd.Timestamp(item).isoformat() for item in inits],
    }


def member_blocks(members, per_task):
    """把 members 个成员切成 ceil(members/per_task) 段连续区间。"""
    members = int(members)
    per_task = max(1, int(per_task))
    return [
        (start, min(per_task, members - start))
        for start in range(0, members, per_task)
    ]


class TaskPlan:
    """任务表 + 落盘布局。成员块按「起报优先、成员次之」排序，与旧路径的遍历顺序一致。"""

    def __init__(self, tasks, *, members, steps, hour_interval, fingerprint_value):
        self.tasks = list(tasks)
        self.members = int(members)
        self.steps = int(steps or 0)
        self.hour_interval = int(hour_interval or 0)
        self.fingerprint = fingerprint_value
        self.by_key = {}
        for task in self.tasks:
            if task.key in self.by_key:
                raise ValueError(f"任务表里有重复的任务键：{task.key}")
            self.by_key[task.key] = task
        if self.steps <= 0:
            raise ValueError("任务表要求 steps >= 1（没有步数就没有产物）")

    # ------------------------------------------------------------------
    # 构造 / 序列化
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, *, inits, members, steps, hour_interval, members_per_task,
              payload):
        tasks = [
            Task(pd.Timestamp(init), start, count)
            for init in inits
            for start, count in member_blocks(members, members_per_task)
        ]
        return cls(tasks, members=members, steps=steps, hour_interval=hour_interval,
                   fingerprint_value=fingerprint(payload))

    def to_dict(self):
        return {
            "layout": LAYOUT_VERSION,
            "fingerprint": self.fingerprint,
            "members": self.members,
            "steps": self.steps,
            "hour_interval": self.hour_interval,
            "tasks": [task.to_list() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data):
        tasks = [
            Task(pd.Timestamp(init), int(start), int(count))
            for init, start, count in data["tasks"]
        ]
        return cls(tasks, members=data["members"], steps=data["steps"],
                   hour_interval=data.get("hour_interval", 0),
                   fingerprint_value=data["fingerprint"])

    # ------------------------------------------------------------------
    # 布局
    # ------------------------------------------------------------------
    def describe_task(self, task):
        if self.members <= 1:
            return f"{task.init:%Y%m%d%H}"
        if task.member_count <= 1:
            return f"{task.init:%Y%m%d%H}/member_{task.member_start:03d}"
        return (f"{task.init:%Y%m%d%H}/member_{task.member_start:03d}"
                f"-{task.member_start + task.member_count - 1:03d}")

    def output_relpaths(self, task):
        """该任务应有的全部产物（相对 output_dir）。顺序 = 步号升序。"""
        init_dir = init_dir_name(task.init)
        if self.members <= 1:
            return [f"{init_dir}/{step:03d}.nc" for step in range(1, self.steps + 1)]
        return [
            f"{init_dir}/member_{member:03d}/{step:03d}.nc"
            for member in task.members()
            for step in range(1, self.steps + 1)
        ]

    def complete_on_disk(self, task, output_dir):
        """产物是否齐全（resume=outputs 的判据）。

        查最后一步的文件就够：写盘是逐步串行的，最后一步在盘上意味着前面那些步都写过
        （写失败会走 .npy 兜底并报错，不会留下一个看着完整的目录）。
        """
        relpaths = self.output_relpaths(task)
        last = relpaths[-1]
        return os.path.isfile(os.path.join(output_dir, last))

    def summary(self):
        return (f"{len(self.tasks)} 个任务（{len({t.init for t in self.tasks})} 起报 × "
                f"{len(member_blocks(self.members, max(1, self.tasks[0].member_count)))} 成员块）"
                if self.tasks else "0 个任务")


__all__ = [
    "LAYOUT_VERSION",
    "Task",
    "TaskPlan",
    "fingerprint",
    "init_dir_name",
    "member_blocks",
    "model_identity",
    "run_payload",
]
