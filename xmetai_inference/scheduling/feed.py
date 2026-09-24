# -*- coding: utf-8 -*-
"""worker 侧的认领 + 预取：后台线程负责「认领任务、把输入帧读进来」。

为什么要有预取线程：单个任务 = 一个起报的一条轨迹，跑之前要从 zarr 读历史帧
（0.25° 78 通道两帧约 650 MB）。串行做的话每个任务开头 GPU 都要干等一次 IO。
把认领与读帧放到后台线程、GPU 只在前台做前向，两者就重叠了。

顺带一个结构上的后果：claim 模式下 worker 不走 DataLoader 的多进程预取，于是
总进程数 = 1 主控 + G 个 worker，不再乘 (1 + num_workers)。
"""
from __future__ import annotations

import collections
import logging
import queue
import threading

from .store import LocalIndex


_END = object()


class TaskFeed:
    """按认领顺序产出 ``(task, sample)``；sample 由调用方的 ``read_sample`` 给出。"""

    def __init__(self, store, plan, read_sample, *, order="init", depth=2,
                 poll_interval=5.0, cache_size=0, skip_done=True, logger=None):
        self.store = store
        self.plan = plan
        self.read_sample = read_sample
        self.order = order
        self.poll_interval = float(poll_interval)
        self.cache_size = int(cache_size or 0)
        self.skip_done = bool(skip_done)
        self.log = logger or logging.getLogger(__name__)
        self.index = LocalIndex(store, plan, ignore_done=not self.skip_done)
        self.stats = {"claimed": 0, "read": 0, "cache_hit": 0, "failed": 0}
        self._hint = None
        self._depth = max(1, int(depth))
        self._queue = queue.Queue(maxsize=self._depth)
        self._stop = threading.Event()
        self._error = None
        self._cache = collections.OrderedDict()
        self._thread = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(
            target=self._produce, name="task-feed", daemon=True)
        self._thread.start()
        return self

    def close(self, release_claims=False):
        """停掉预取线程。``release_claims=True`` 时把自己认领但没跑完的任务放回池子。"""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # 生产者可能正卡在 put 上（队列满），排空队列让它有机会看到停止标志。
            deadline = 5.0
            while thread.is_alive() and deadline > 0:
                try:
                    self._queue.get(timeout=0.1)
                except queue.Empty:
                    pass
                deadline -= 0.1
            thread.join(timeout=5)
        if release_claims:
            self.release_mine()

    def release_mine(self):
        """把自己认领但没跑完的任务交还（崩溃/被中断时用，让它被别的 worker 捡走）。"""
        released = 0
        for key in list(self.index.mine):
            self.store.release_claim(key)
            self.index.mark(key, LocalIndex.FREE)
            released += 1
        return released

    def finish(self, key):
        """消费方跑完一个任务后调用：把它从「我手头」摘掉，否则预取线程会一直等自己。"""
        self.index.mark(key, LocalIndex.DONE)
        return self

    # ------------------------------------------------------------------
    # 消费
    # ------------------------------------------------------------------
    def __iter__(self):
        return self._consume()

    def _consume(self):
        while True:
            item = self._queue.get()
            if item is _END:
                return
            if isinstance(item, _FeedError):
                raise item.error
            yield item

    # ------------------------------------------------------------------
    # 生产
    # ------------------------------------------------------------------
    def _produce(self):
        try:
            while not self._stop.is_set():
                key = self._next_key()
                if key is None:
                    break
                task = self.plan.by_key[key]
                try:
                    sample = self._read(task)
                except Exception as error:  # noqa: BLE001 - 单个任务读失败不该带走整个 worker
                    attempt = self.store.fail(
                        key, f"{type(error).__name__}: {error}",
                        extra={"stage": "read_sample"})
                    self.stats["failed"] += 1
                    give_up = attempt > self.store.task_retries
                    self.index.mark(
                        key, LocalIndex.FAILED if give_up else LocalIndex.FREE)
                    self.log.error(
                        "任务 %s 读输入失败（第 %d 次，%s）：%s",
                        key, attempt, "判失败" if give_up else "待重派", error)
                    continue
                self._hint = (task.member_start if self.order == "member"
                              else task.init)
                self.stats["claimed"] += 1
                self._queue.put((task, sample))
        except Exception as error:  # noqa: BLE001 - 生产线程里的异常要带回主线程
            self._error = error
            self.log.exception("任务预取线程异常退出")
            self._queue.put(_FeedError(error))
            return
        self._queue.put(_END)

    def _next_key(self):
        """下一个认领对象；返回 None 表示「本进程这边没活可干了」。"""
        while not self._stop.is_set():
            free = self.index.free_keys()
            if free:
                for key in self._ordered(free):
                    if self._stop.is_set():
                        return None
                    if self.store.claim(key):
                        self.index.mark(key, LocalIndex.HELD)   # HELD = 我已认领
                        return key
                    # 被别人抢先：记成 TAKEN（不是我的），等 rescan 再判它有没有变僵尸。
                    self.index.mark(key, LocalIndex.TAKEN)
                # 有可认领但一个都没抢到 -> 让出 CPU 再看一次，别打成忙等。
                self._stop.wait(self.poll_interval)
                self.index.rescan()
                continue
            if not self.index.mine:
                return None
            # 我手头还有没跑完的；等一会儿看看有没有新放出来的任务。
            self._stop.wait(self.poll_interval)
            self.index.rescan()
        return None

    def _ordered(self, keys):
        """把「和上次同一个成员/起报」的任务排到前面。

        order="member"（m1a）：连续跑同一个成员能省一次 session 换入（FGVP 上是一份
        几 GB 的 onnx）。order="init"（同一 session 服务全部成员）：连续跑同一个起报
        能命中输入缓存，省一次 650 MB 的读。
        """
        if self._hint is None:
            return keys
        hint = self._hint
        preferred, rest = [], []
        for key in keys:
            (preferred if self._hint_of(key) == hint else rest).append(key)
        return preferred + rest

    def _hint_of(self, key):
        task = self.plan.by_key[key]
        return task.member_start if self.order == "member" else task.init

    def _read(self, task):
        if not self.cache_size:
            self.stats["read"] += 1
            return self.read_sample(task)
        cached = self._cache.get(task.init)
        if cached is not None:
            self._cache.move_to_end(task.init)
            self.stats["cache_hit"] += 1
            return cached
        sample = self.read_sample(task)
        self.stats["read"] += 1
        self._cache[task.init] = sample
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return sample


class _FeedError:
    """把生产线程里的异常搬到消费线程里原样抛出。"""

    __slots__ = ("error",)

    def __init__(self, error):
        self.error = error


__all__ = ["TaskFeed"]
