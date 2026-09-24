# -*- coding: utf-8 -*-
"""共享盘上的任务账本：认领 / 完成 / 失败 / 心跳。

设计取向是**没有 broker**：主控不派活，worker 自己去共享盘上认领。于是断点续跑
（重启后 done 的任务直接跳过）、崩溃重派（没人认领的任务自然会被别人拿走）和多机
（同一块共享盘）都是免费拿到的；代价只是每次认领一次 ``mkdir`` 系统调用，相对
分钟级的单任务推理可以忽略。

目录（root = config 的 ``scheduling.plan_dir``，缺省 ``{output_dir}/.xmetai_plan``）::

  {root}/{fingerprint}/plan.json               任务表 + 指纹
  {root}/{fingerprint}/claims/{key}/            认领（mkdir 原子；owner.json 记归属）
  {root}/{fingerprint}/done/{key}               完成标记（产物 flush 之后才写）
  {root}/{fingerprint}/failed/{key}/{n}.json    第 n 次失败的记录
  {root}/{fingerprint}/failed/{key}/giveup      超过重试上限，不再重派
  {root}/{fingerprint}/workers/{wid}.json       worker 心跳（当前任务 / 完成数 / pid）

用 ``mkdir`` 而不是 ``O_CREAT|O_EXCL``：mkdir 在 NFS/Lustre 这类共享盘上的原子性比
open-exclusive 更可靠，而共享盘正是这套设计的目标场景。

判「认领是不是僵尸」的规则（顺序即优先级，全在 :meth:`ClaimStore.stale`）：别的
launch 且它还有活心跳 -> 不动；该 worker 的 pid 变了 -> 上一世代遗留；心跳过期 ->
worker 死了（OOM SIGKILL 的典型形态）；认领年龄超过 ``task_ttl_seconds`` -> 兜底。

完成标记（``done/``）默认**跨运行**一律采信：它记的是「这个任务的产物已经 flush 到盘上」，
所以断点续跑与「重跑一份跑了一半的输出目录」都靠它。唯一的例外是
``scheduling.resume="off"``（整份重算）：那一次运行只认**自己这个 launch 写的**标记
（``done_scope="launch"``）—— 旧标记不算数，但同一次运行里别的 worker 刚跑完的任务仍然
不许重跑。少了后半条就是自检里那个「24 个任务跑了 79 遍」：认领目录是跑完就删的，
本地 FREE 视图会把别人刚跑完的任务又看成空闲的。
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import time
import uuid


DONE_SCOPES = ("all", "launch")


def _write_json(path, payload):
    """原子写 JSON：先写同目录的临时名再 os.replace，读者永远看不到半个文件。"""
    tmp = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def new_launch_id():
    """一次运行的标识：主控生成、全部 worker 共用（用来区分不同次运行）。"""
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class ClaimStore:
    """任务账本。主控与 worker 共用同一个类，只是用法不同（见各方法注释）。"""

    def __init__(self, root, fingerprint, *, launch_id="", worker_id="",
                 heartbeat_timeout=120.0, task_ttl_seconds=0.0, task_retries=1,
                 skip_done=True, done_scope="all", hostname=None):
        self.root = os.path.abspath(root)
        self.fingerprint = str(fingerprint)
        self.launch_id = str(launch_id)
        self.worker_id = str(worker_id)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.task_ttl_seconds = float(task_ttl_seconds or 0.0)
        self.task_retries = int(task_retries)
        # 认领时是否拒绝已完成的键。必须为 True（resume=off 才关）：本地 FREE 视图是
        # 缓存，一个刚被别的 worker 跑完并释放认领的任务，在这里看起来仍然是空着的。
        self.skip_done = bool(skip_done)
        # 完成标记的采信范围：all = 谁写的都算（默认）；launch = 只认本次运行写的。
        # resume=off 用的是后者，见模块文档。
        if done_scope not in DONE_SCOPES:
            raise ValueError(f"done_scope 必须是 {DONE_SCOPES}，实际 {done_scope!r}")
        self.done_scope = str(done_scope)
        self.hostname = hostname or socket.gethostname()

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------
    @property
    def dir(self):
        return os.path.join(self.root, self.fingerprint)

    @property
    def claims_dir(self):
        return os.path.join(self.dir, "claims")

    @property
    def done_dir(self):
        return os.path.join(self.dir, "done")

    @property
    def failed_dir(self):
        return os.path.join(self.dir, "failed")

    @property
    def workers_dir(self):
        return os.path.join(self.dir, "workers")

    @property
    def trash_dir(self):
        return os.path.join(self.dir, "trash")

    @property
    def plan_path(self):
        return os.path.join(self.dir, "plan.json")

    def claim_dir(self, key):
        return os.path.join(self.claims_dir, key)

    def done_path(self, key):
        return os.path.join(self.done_dir, key)

    def failed_key_dir(self, key):
        return os.path.join(self.failed_dir, key)

    def worker_path(self, worker_id):
        return os.path.join(self.workers_dir, f"{worker_id}.json")

    def ensure_dirs(self):
        for directory in (self.dir, self.claims_dir, self.done_dir,
                          self.failed_dir, self.workers_dir):
            os.makedirs(directory, exist_ok=True)
        return self

    # ------------------------------------------------------------------
    # 计划文件
    # ------------------------------------------------------------------
    def write_plan(self, plan):
        self.ensure_dirs()
        _write_json(self.plan_path, plan.to_dict())
        return self.plan_path

    def read_plan(self):
        from .plan import TaskPlan

        data = _read_json(self.plan_path)
        return TaskPlan.from_dict(data) if data else None

    # ------------------------------------------------------------------
    # 完成 / 失败
    # ------------------------------------------------------------------
    def done_keys(self):
        """有完成标记的任务键（采信范围见 ``done_scope``）。"""
        try:
            names = os.listdir(self.done_dir)
        except OSError:
            return set()
        if self.done_scope == "all":
            return set(names)
        return {name for name in names if self._marker_in_launch(name)}

    def _marker_in_launch(self, key):
        """完成标记是不是本次运行写的（标记不存在 -> False）。"""
        payload = _read_json(self.done_path(key))
        return bool(payload) and payload.get("launch_id") == self.launch_id

    def failed_keys(self):
        """判失败（超过重试上限）的任务键。"""
        result = set()
        try:
            entries = os.listdir(self.failed_dir)
        except OSError:
            return result
        for key in entries:
            if os.path.exists(os.path.join(self.failed_dir, key, "giveup")):
                result.add(key)
        return result

    def attempts(self, key):
        try:
            names = os.listdir(self.failed_key_dir(key))
        except OSError:
            return 0
        return len([name for name in names if name.endswith(".json")])

    def fail(self, key, reason, extra=None):
        """记一次失败并释放认领；返回累计失败次数。"""
        directory = self.failed_key_dir(key)
        os.makedirs(directory, exist_ok=True)
        attempt = self.attempts(key) + 1
        payload = {
            "key": key, "attempt": attempt, "reason": str(reason)[:2000],
            "worker_id": self.worker_id, "launch_id": self.launch_id,
            "hostname": self.hostname, "ts": time.time(),
        }
        if extra:
            payload.update(extra)
        _write_json(os.path.join(directory, f"{attempt:02d}.json"), payload)
        self.release_claim(key)
        if attempt > self.task_retries:
            # 超过上限：留 giveup 标记，本地视图与主控都不再重派这个任务。
            _write_json(os.path.join(directory, "giveup"), payload)
        return attempt

    def mark_done(self, key):
        """写完成标记。必须在产物 flush 之后调用——它是 resume=marker 的唯一判据。"""
        os.makedirs(self.done_dir, exist_ok=True)
        _write_json(self.done_path(key), {
            "key": key, "worker_id": self.worker_id, "launch_id": self.launch_id,
            "hostname": self.hostname, "ts": time.time(),
        })
        self.release_claim(key)

    def is_done(self, key):
        if self.done_scope == "all":
            return os.path.exists(self.done_path(key))
        return self._marker_in_launch(key)

    def release_claim(self, key):
        shutil.rmtree(self.claim_dir(key), ignore_errors=True)

    def force_release(self, key):
        """主控用：把认领丢掉（只在确认没有任何 worker 在跑之后）。"""
        shutil.rmtree(self.claim_dir(key), ignore_errors=True)

    # ------------------------------------------------------------------
    # 认领
    # ------------------------------------------------------------------
    def claimed(self, key):
        return os.path.isdir(self.claim_dir(key))

    def claim(self, key):
        """认领任务：成功返回 True。同一任务全集群只有一个进程能成功。

        先看完成标记：认领目录是**跑完就删**的，所以「目录不存在」既可能是没人跑，
        也可能是刚有人跑完删掉的。少了这一步，一个 worker 的本地 FREE 视图还没刷新时
        就会把别人刚跑完的任务再跑一遍（自检里实测到 24 个任务跑了 79 遍）。
        """
        if self.skip_done and self.is_done(key):
            return False
        if self._try_mkdir(key):
            return True
        if not self.stale(key):
            return False
        if not self._take_over(key):
            return False
        return self._try_mkdir(key)

    def _try_mkdir(self, key):
        directory = self.claim_dir(key)
        try:
            os.makedirs(self.claims_dir, exist_ok=True)
            os.mkdir(directory)
        except FileExistsError:
            return False
        except OSError:
            return False
        _write_json(os.path.join(directory, "owner.json"), {
            "key": key, "worker_id": self.worker_id, "launch_id": self.launch_id,
            "pid": os.getpid(), "hostname": self.hostname, "ts": time.time(),
        })
        return True

    def _take_over(self, key):
        """把僵尸认领目录改名挪走：os.rename 原子，两个回收者只有一个能成功。"""
        source = self.claim_dir(key)
        try:
            os.makedirs(self.trash_dir, exist_ok=True)
            target = os.path.join(self.trash_dir, f"{key}.{uuid.uuid4().hex[:8]}")
            os.rename(source, target)
        except OSError:
            return False
        shutil.rmtree(target, ignore_errors=True)
        return True

    def stale(self, key):
        """该认领是不是僵尸（可以安全回收）。规则见模块文档。"""
        directory = self.claim_dir(key)
        try:
            age = time.time() - os.path.getmtime(directory)
        except OSError:
            return False
        owner = _read_json(os.path.join(directory, "owner.json"))
        now = time.time()
        if owner is None:
            # mkdir 成功但 owner.json 还没落盘（或写失败）：窗口极短，用保守下限兜底，
            # 免得抢走一个刚刚开始的合法认领。
            return age > max(self.heartbeat_timeout, 60.0)
        if owner.get("launch_id") != self.launch_id:
            # 别的运行留下的：只有那一次运行确定不在了才能回收。
            return not self.launch_alive(owner.get("launch_id"), now)
        heartbeat = _read_json(self.worker_path(owner.get("worker_id") or ""))
        if not heartbeat:
            # 心跳文件还没落盘（worker 认领得比第一次心跳还早）：窗口很短，用保守下限
            # 兜底。宁可多等，也不能抢走一个刚认领、正在读输入的合法任务。
            return age > max(self.heartbeat_timeout, 60.0)
        if heartbeat.get("launch_id") != owner.get("launch_id"):
            return True
        if heartbeat.get("pid") != owner.get("pid"):
            # 同一个 worker_id 换了进程：这是它上一世代留下的认领。
            return True
        if now - float(heartbeat.get("ts") or 0) > self.heartbeat_timeout:
            return True
        if self.task_ttl_seconds and age > self.task_ttl_seconds:
            return True
        return False

    def launch_alive(self, launch_id, now=None):
        if not launch_id:
            return False
        now = time.time() if now is None else now
        for status in self.read_status().values():
            if status.get("launch_id") != launch_id:
                continue
            if now - float(status.get("ts") or 0) <= self.heartbeat_timeout:
                return True
        return False

    # ------------------------------------------------------------------
    # 心跳 / 状态
    # ------------------------------------------------------------------
    def heartbeat(self, **info):
        """worker 用：把「我还活着、在跑哪个任务、跑完几个」写进共享盘。"""
        os.makedirs(self.workers_dir, exist_ok=True)
        payload = {
            "worker_id": self.worker_id, "launch_id": self.launch_id,
            "pid": os.getpid(), "hostname": self.hostname, "ts": time.time(),
        }
        payload.update(info)
        _write_json(self.worker_path(self.worker_id), payload)
        return payload

    def clear_heartbeat(self):
        try:
            os.remove(self.worker_path(self.worker_id))
        except OSError:
            pass

    def read_status(self):
        result = {}
        try:
            names = os.listdir(self.workers_dir)
        except OSError:
            return result
        for name in names:
            if not name.endswith(".json"):
                continue
            status = _read_json(os.path.join(self.workers_dir, name))
            if status:
                result[name[:-5]] = status
        return result

    def foreign_launches(self):
        """还有活心跳的其它 launch：两份运行同时在写同一个输出目录时必须拦下来。"""
        now = time.time()
        found = {}
        for worker_id, status in self.read_status().items():
            launch_id = status.get("launch_id")
            if not launch_id or launch_id == self.launch_id:
                continue
            if now - float(status.get("ts") or 0) <= self.heartbeat_timeout:
                found[launch_id] = status
        return found


class LocalIndex:
    """worker 侧的本地任务状态视图，避免每次认领都做 O(N) 次 stat。

    只在「本进程手头没有可认领的任务」时才 :meth:`rescan` —— 那时才去问盘上最新的
    状态（含其它 worker 的进度、以及可能已经变僵尸的认领）。这样认领热路径是纯内存的。
    """

    FREE = "free"
    DONE = "done"
    HELD = "held"        # 本进程已认领
    TAKEN = "taken"      # 别人认领走了（与 HELD 分开：它不属于 mine）
    FAILED = "failed"

    def __init__(self, store, plan, *, ignore_done=False):
        self.store = store
        self.plan = plan
        # resume=off 时完成标记不该影响认领（否则「重算一遍」会一个任务都跑不了）。
        self.ignore_done = bool(ignore_done)
        self.keys = [task.key for task in plan.tasks]
        self.state = {}
        self.mine = set()
        self.rescans = 0
        self.rescan()

    def rescan(self):
        self.rescans += 1
        done = set() if self.ignore_done else self.store.done_keys()
        failed = self.store.failed_keys()
        for key in self.keys:
            if key in self.mine:
                continue
            if key in done:
                self.state[key] = self.DONE
            elif key in failed:
                self.state[key] = self.FAILED
            elif self.store.claimed(key):
                # 僵尸认领要当空闲处理，否则崩过的任务永远没人捡。
                self.state[key] = self.FREE if self.store.stale(key) else self.HELD
            else:
                self.state[key] = self.FREE
        return self

    def free_keys(self):
        return [key for key in self.keys if self.state.get(key) == self.FREE]

    def mark(self, key, value):
        """更新本地视图。只有 HELD 才进 ``mine``——认领失败（TAKEN）不是我的任务，

        否则 ``mine`` 永不为空，worker 就永远退不出认领循环（这条踩过）。
        """
        self.state[key] = value
        if value == self.HELD:
            self.mine.add(key)
        else:
            self.mine.discard(key)

    def counts(self):
        summary = {self.FREE: 0, self.DONE: 0, self.HELD: 0, self.TAKEN: 0,
                   self.FAILED: 0}
        for value in self.state.values():
            summary[value] = summary.get(value, 0) + 1
        return summary


__all__ = ["ClaimStore", "LocalIndex", "new_launch_id"]
