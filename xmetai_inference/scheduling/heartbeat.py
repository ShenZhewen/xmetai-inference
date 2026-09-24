# -*- coding: utf-8 -*-
"""worker 心跳：后台线程按固定间隔把「我还活着、在跑哪个任务、跑完几个」写进共享盘。

为什么必须是**独立线程**：一条 60 步的轨迹是分钟级的，前向之间没有可插入写盘的时机，
而主控判「worker 是不是卡死」只看心跳的新鲜度（cli._monitor_pass 与 ClaimStore.stale）。
把心跳写在主循环里，就等于把判死阈值绑在单步耗时上 —— 一个慢但正常的 worker 会被误杀。

写进盘的内容刻意保持最少：task（当前任务键）、done（本进程完成的任务数）、外加
ClaimStore.heartbeat 自己补的 pid / launch_id / hostname / ts。主控要的是「这口气还在
不在」，进度明细交给 done 计数；stale() 则靠 pid 与 ts 区分「同一个 worker_id 的上一世代」
与「心跳过期」。
"""
from __future__ import annotations

import logging
import threading


class Heartbeat:
    """一个后台写盘的心跳；start() 之后由它自己按 interval 续命。"""

    def __init__(self, store, interval=15.0, *, logger=None):
        self.store = store
        # 下限 0.05s 只是防住 0 或负数（写盘比 interval 还快时没有意义）；自检会把它压到 0.05。
        self.interval = max(0.05, float(interval))
        self.log = logger or logging.getLogger(__name__)
        self.task = None
        self.done = 0
        self.beats = 0
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        self.beat()          # 先落一条：主控不必等到第一个间隔才知道这个 worker 起来了
        self._thread = threading.Thread(
            target=self._loop, name="heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self, clear=True):
        """停掉线程。``clear=True`` 时删掉心跳文件。

        干净退出必须删：foreign_launches() 靠心跳判断「还有另一次运行活着」，留下一个
        永远不会更新的心跳文件会让下一次运行白等一个 heartbeat_timeout。
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if clear:
            self.store.clear_heartbeat()
        return self

    # ------------------------------------------------------------------
    # 写盘
    # ------------------------------------------------------------------
    def beat(self, **info):
        payload = {"task": self.task, "done": self.done}
        payload.update(info)
        try:
            self.store.heartbeat(**payload)
        except OSError as error:
            # 共享盘抖动不该带走一个正在算的 worker：下个间隔会重写一次。
            self.log.warning("心跳写盘失败（%s），下个间隔再试", error)
            return self
        self.beats += 1
        return self

    def set_task(self, key):
        """切到新任务：立刻写一条，主控的进度里就不会一直挂着上一个任务的名字。"""
        self.task = key
        return self.beat()

    def bump_done(self, count=1):
        self.done += int(count)
        return self.beat()

    # ------------------------------------------------------------------
    # 线程体
    # ------------------------------------------------------------------
    def _loop(self):
        while not self._stop.wait(self.interval):
            self.beat()


__all__ = ["Heartbeat"]
