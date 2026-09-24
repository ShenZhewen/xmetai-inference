#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一推理入口：配置加载、多卡调度、自回归推理与 NetCDF 输出。

推理后端（BaseInferModel）拆到 xmetai_inference.backends 包、各模型子类拆到
xmetai_inference.models 包，数据源拆到 xmetai_inference.data 包；本文件负责
读取配置、启动多卡 worker，并完成输出选择、异步写盘和主流程编排。

用法：
    xmetai-infer --model fuxi_ens
    xmetai-infer --model fgvp --times 2025010700..2025020500:24
    python -m xmetai_inference --model fgvp --out /tmp/output --gpus 1

输出目录（集合 members>1）：{out}/{起报目录}/member_{成员3位}/{预测步序号3位}.nc
输出目录（确定性 members=1）：{out}/{起报目录}/{预测步序号3位}.nc
00 UTC 起报目录使用 YYYYMMDD；其他时次使用 YYYYMMDDHH，避免同日多时次互相覆盖。

多卡时，主进程使用同一个 inference.py 的内部 --worker 模式启动独立子进程，并通过
CUDA_VISIBLE_DEVICES 隔离单卡。ONNX Runtime 单个 session 只使用一张卡；加速来自
把集合成员或起报次数分配给不同进程，不是让单次 forward 跨卡执行。

推理是 step 外层、member 内层：每算完一个 step 的全体成员就异步丢给后台线程
落盘，GPU 不等磁盘写；netCDF4/HDF5 非线程安全，所以只用一个 writer 线程串行写。
"""
import argparse
import gc
import importlib
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time

# Allow `python xmetai_inference/cli.py --config ...` without installing the package.
if __package__ in (None, ""):
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
from time import perf_counter
from dataclasses import asdict

import numpy as np
import pandas as pd
import xarray as xr

from xmetai_inference.configs.base import load_config, parse_times
from xmetai_inference.logging_util import configure_logging


# 推理后端和具体模型在 worker 启动后按需导入，配置帮助不依赖可选模型运行环境。
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 输出（后端无关）
# ---------------------------------------------------------------------------
def _output_init_dir(init):
    """00 UTC 用 YYYYMMDD，其他起报时次保留 YYYYMMDDHH。

    按天起报（每天 00Z）时目录就是干净的 YYYYMMDD；同日多时次才带上小时，
    避免互相覆盖。注意：起报间隔混用（如既有 00Z 又有 12Z）会产生 8 位与
    10 位并存的目录名，下游按目录名解析时要能同时吃两种——
    util/eval_single_util.py 的 _parse_init_dir 已兼容。
    """
    # 唯一实现搬到了 scheduling.plan.init_dir_name（任务表也要用它算产物路径，
    # 两边必须是同一个口径）；这里保留同名包装，调用方不用改。
    from xmetai_inference.scheduling import init_dir_name

    return init_dir_name(init)


def _resolve_output_indices(channels, requested):
    """把要保存的变量名（z500/u200/v200/msl/tp）映射成通道下标。

    requested=None 时保存全部通道；名字大小写不敏感，不存在的跳过并告警。
    """
    if requested is None:
        return list(range(len(channels)))
    name2idx = {str(c).lower(): i for i, c in enumerate(channels)}
    idxs = []
    for name in requested:
        key = str(name).strip().lower()
        if key in name2idx:
            idxs.append(name2idx[key])
        else:
            log.warning("输出变量 %r 不在模型通道里，已跳过", name)
    if not idxs:
        raise SystemExit("没有有效的输出变量可保存（检查 --vars 是否写对）")
    return idxs


def _select_netcdf_engine():
    """挑选可用的最快 NetCDF 写引擎：netcdf4 > h5netcdf > scipy（兜底）。

    避免 scipy/NETCDF3——它写大数组极慢且有 2GB/变量上限。注意 xarray 的引擎名
    是小写，但 Python 导入名大小写不同（netCDF4 大写 F、h5netcdf 全小写），
    探测要按导入名来。返回 (engine, netcdf_format)。
    """
    for eng, import_name in (("netcdf4", "netCDF4"), ("h5netcdf", "h5netcdf")):
        try:
            importlib.import_module(import_name)
            return eng, "NETCDF4"
        except ImportError:
            continue
    log.warning("netcdf4/h5netcdf 未安装，退回 scipy 写 NETCDF3_64BIT（大数组会很慢）")
    return "scipy", "NETCDF3_64BIT"


class _AsyncWriter:
    """后台单线程写 NetCDF；写失败则把该 step 原始数组兜底存 .npy。"""

    def __init__(self, save_dir, atomic=True):
        self.save_dir = save_dir
        # 先写隐藏临时名再 os.replace 原子改名：下游（评测、enscheck、别的 worker）
        # 永远读不到半个文件。前导点让 `*.nc` 通配不会把半成品当产物。
        self.atomic = bool(atomic)
        self.q = queue.Queue(maxsize=2)
        self.errors = []
        self.engine, self.netcdf_format = _select_netcdf_engine()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            fname, ds, raw = item
            path = os.path.join(self.save_dir, fname)
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # 不压缩（与参考实现最终结论一致）：zlib 压缩 281MB/步耗时数秒，
                # 裸写 ~500MB/s，推理不等磁盘；引擎用 netcdf4 避免 scipy/NETCDF3 慢路径。
                target = path
                if self.atomic:
                    target = os.path.join(
                        os.path.dirname(path),
                        f".{os.path.basename(path)}.tmp-{os.getpid()}")
                ds.to_netcdf(target, engine=self.engine, format=self.netcdf_format)
                if self.atomic:
                    os.replace(target, path)
            except Exception as e:  # noqa: BLE001
                self.errors.append(e)
                log.exception("写入 NetCDF 失败：%s", path)
                if raw is not None:
                    step_buf, s, init = raw
                    npy = os.path.join(self.save_dir, f"{init:%Y%m%d%H}_raw_step_{s:03d}.npy")
                    try:
                        step_buf_np = (step_buf.detach().cpu().numpy()
                                       if hasattr(step_buf, "detach") else np.asarray(step_buf))
                        np.save(npy, step_buf_np.astype(np.float32))
                        log.error("step %d 原始数组已兜底保存：%s", s + 1, npy)
                    except Exception as fallback_error:  # noqa: BLE001
                        self.errors.append(fallback_error)
                        log.exception("step %d 的 .npy 兜底保存也失败：%s", s + 1, npy)
            finally:
                self.q.task_done()

    def flush(self):
        """阻塞直到队列里所有待写文件都落盘完成。

        关键：netCDF4/HDF5 非线程安全。写线程在写输出 NetCDF 时，若主线程同时
        读下一轮起报的输入 NetCDF，两个线程并发访问 HDF5 会触发 C 层段错误。
        所以进入下一个 build_input 前必须先 drain 写队列，让读写彻底串行。
        """
        self.q.join()

    def put(self, fname, ds, raw=None):
        self.q.put((fname, ds, raw))

    def close(self):
        self.q.put(None)
        self.t.join()
        return self.errors


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def _init_times(args, interval):
    """把 --inits（列表）/ --start/--end/--freq（区间）/ --time（单次）展开成起报时间列表。"""
    if args.inits is not None:
        toks = [t for t in re.split(r"[,\s]+", args.inits) if t]
        if not toks:
            raise SystemExit("--inits 为空")
        return [pd.to_datetime(t, format="%Y%m%d" if len(t) == 8 else "%Y%m%d%H")
                for t in toks]
    freq = args.freq if args.freq is not None else interval
    if freq <= 0:
        raise SystemExit("--freq 必须 > 0")
    if args.start is not None:
        start = pd.to_datetime(args.start, format="%Y%m%d%H")
        end = pd.to_datetime(args.end, format="%Y%m%d%H") if args.end else start
        times = []
        t = start
        while t <= end:
            times.append(t)
            t += pd.Timedelta(hours=freq)
        if not times:
            raise SystemExit("--start 晚于 --end，起报时间列表为空")
        return times
    if args.time is not None:
        return [pd.to_datetime(args.time, format="%Y%m%d%H")]
    raise SystemExit("必须提供 --time / --start / --inits 之一")


def _pick(overrides, key, default):
    """覆盖参数优先，没给（None）则回落 config 默认值。"""
    value = overrides.get(key)
    return default if value is None else value


def _run_scope(cfg, overrides):
    """本次运行的范围（模型 / 起报 / 步数 / 成员 / 变量 / 输出 / 步长）。

    worker 参数与任务表都从这一份取：两处各算一遍最容易「members 算得不一样」，
    那样任务表与实际产物就对不上了。
    """
    model_path = _pick(overrides, "model_path", cfg.model_path)
    inits = parse_times(_pick(overrides, "times", cfg.times))
    if not inits:
        raise SystemExit(
            f"配置 {cfg.name}：推理必须指定 times（如 \"2025010600..2025020500:24\"）")
    return {
        "model_path": model_path,
        "model_abs": cfg.model_path_abs(model_path),
        "inits": inits,
        "steps": _pick(overrides, "steps", cfg.steps),
        "members": _pick(overrides, "members", cfg.members),
        "vars": _pick(overrides, "vars", cfg.vars),
        "output_dir": _pick(overrides, "out", cfg.output_dir),
        "interval": getattr(cfg, "hour_interval", 0),
    }


def _build_run_plan(cfg, scope, sched):
    """建任务表（claim 模式）。members / interval 按模型类的缺省补齐后再入表。"""
    from xmetai_inference.models import get_model_class
    from xmetai_inference.scheduling import TaskPlan, run_payload

    try:
        cls = get_model_class(cfg.model_class)
    except (TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    members = int(scope["members"] or getattr(cls, "members", 1) or 1)
    interval = int(scope["interval"] or getattr(cls, "hour_interval", 6))
    per_task = sched.members_per_task_for(members)
    variables = [item for item in str(scope["vars"] or "").split(",") if item.strip()]
    payload = run_payload(
        model_path=scope["model_abs"], steps=scope["steps"], hour_interval=interval,
        members=members, variables=variables, inits=scope["inits"],
        members_per_task=per_task)
    plan = TaskPlan.build(
        inits=scope["inits"], members=members, steps=scope["steps"],
        hour_interval=interval, members_per_task=per_task, payload=payload)
    plan_dir = sched.plan_dir or os.path.join(scope["output_dir"], ".xmetai_plan")
    log.info("任务表：%d 个任务（%d 起报 x %d 成员，每任务装 %d 个成员），指纹 %s",
             len(plan.tasks), len(scope["inits"]), members, per_task,
             plan.fingerprint)
    return plan, os.path.abspath(plan_dir)


def _gpu_mem_per_proc(gpu_mem, sched, managed):
    """config 的 gpu_mem 是**每张卡**的预算，不是每个进程的。

    claim 模式下一张卡可以起 procs_per_gpu 个进程（见 _worker_slots），显存就得按份数
    分：不然每个进程都按整卡的 gpu_mem 去申请 ORT 的显存上限，两个进程加起来就是成倍
    超订，症状是跑到中途才 OOM（启动时看不出来）。静态模式每卡只起一个进程（_run_infer
    的静态分支），不除。
    """
    if not managed:
        return float(gpu_mem)
    return float(gpu_mem) / max(1, int(sched.procs_per_gpu))


def _build_infer_argv(cfg, overrides, *, scope=None, sched=None, plan=None,
                      plan_dir=None, intra_op_threads=0, gpu_mem=None):
    """将运行范围覆盖转为 worker 参数；组件由 worker 重新加载 config 获得。"""
    scope = scope or _run_scope(cfg, overrides)
    sched = sched or cfg.scheduling_config(overrides)
    argv = [
        "--config-source", cfg._source_path,
        "--model", scope["model_abs"],
        # 取值由调用方按「每卡几个进程」算好传进来（见 _gpu_mem_per_proc）：主控日志里
        # 报的「每份约 X」与实际传给 worker 的必须是同一个数。
        "--gpu-mem", str(cfg.gpu_mem if gpu_mem is None else gpu_mem),
        "--log-level", str(_pick(overrides, "log_level", cfg.log_level)),
        "--intra-op-threads", str(int(intra_op_threads or 0)),
        "--cudnn-algo-search", str(cfg.cudnn_algo_search or ""),
        # 常驻 session 池的额度（resident_members）。它和 intra_op_threads 同类：属于
        # 「这份配方怎么跑」，所以只有 config 能配，命令行不开放覆盖。
        "--resident-members", str(int(cfg.resident_members or 1)),
        # 调度配置整份传过去：worker 会重新加载 config，但命令行的三个覆盖
        # （--schedule / --procs-per-gpu / --resume）只在主进程手里，用一份 JSON
        # 保证父子进程看到的是同一套调度参数。
        "--scheduling-json", json.dumps(asdict(sched), sort_keys=True),
    ]
    if plan is not None:
        argv += ["--schedule", "claim",
                 "--plan-dir", plan_dir,
                 "--plan-fingerprint", plan.fingerprint]
    interval = plan.hour_interval if plan is not None else scope["interval"]
    if interval:
        argv += ["--interval", str(interval)]

    inits = scope["inits"]
    if len(inits) == 1:
        argv += ["--time", inits[0].strftime("%Y%m%d%H")]
    else:
        argv += ["--inits", ",".join(value.strftime("%Y%m%d%H") for value in inits)]

    steps = scope["steps"]
    if steps:
        argv += ["--steps", str(steps)]
    members = plan.members if plan is not None else scope["members"]
    if members:
        argv += ["--members", str(members)]
    variables = scope["vars"]
    if variables:
        argv += ["--vars", variables]
    if scope["output_dir"]:
        argv += ["--out", scope["output_dir"]]
    batch_size = overrides.get("batch_size")
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    num_workers = overrides.get("num_workers")
    if num_workers is not None:
        argv += ["--num-workers", str(num_workers)]
    return argv


def _run_infer(cfg, overrides):
    """使用同一个包模块启动一个或多个隔离 worker。"""
    from xmetai_inference.scheduling import resolve_intra_op_threads

    try:
        sched = cfg.scheduling_config(overrides)
    except (TypeError, ValueError) as error:
        # 键名写错、枚举写错、阈值矛盾都在这里拦：这些错的后果往往是「不生效」而不是
        # 「跑不起来」，静默放过会让人以为升级生效了。
        raise SystemExit(
            f"调度配置有误（config 的 scheduling 或命令行覆盖）：{error}") from error
    scope = _run_scope(cfg, overrides)
    gpus = _pick(overrides, "gpus", cfg.gpus)
    cuda_devices = _pick(
        overrides, "cuda_devices", cfg.cuda_devices).split(",")
    output_dir = scope["output_dir"]
    # 进程总数 = 卡数 x 每卡进程数；intra_op 线程按它均分，避免 N 个进程各自吃满物理核。
    workers = max(1, int(gpus)) * sched.procs_per_gpu
    plan = plan_dir = None
    managed = sched.mode == "claim" and bool(output_dir)
    if managed:
        plan, plan_dir = _build_run_plan(cfg, scope, sched)
    argv = _build_infer_argv(
        cfg, overrides, scope=scope, sched=sched, plan=plan, plan_dir=plan_dir,
        intra_op_threads=resolve_intra_op_threads(cfg.intra_op_threads, workers),
        gpu_mem=_gpu_mem_per_proc(cfg.gpu_mem, sched, managed))

    env = dict(os.environ)

    command = [
        sys.executable,
        "-u",
        "-m",
        "xmetai_inference",
        "--worker",
    ] + argv
    if managed:
        return _run_managed(cfg, sched, scope, plan, plan_dir, command, gpus,
                            cuda_devices)
    if gpus <= 1:
        if output_dir:
            command += ["--log-file", os.path.join(output_dir, "rank_0.log")]
        return subprocess.call(command, env=env)

    if not output_dir:
        raise SystemExit(
            f"配置 {cfg.name}：多卡（gpus={gpus}）需要 output_dir，才能落 rank 日志")
    os.makedirs(output_dir, exist_ok=True)
    processes = []
    log_files = []
    for rank in range(gpus):
        gpu = cuda_devices[rank] if rank < len(cuda_devices) else str(rank)
        rank_env = dict(env)
        rank_env["CUDA_VISIBLE_DEVICES"] = gpu
        rank_env["LOCAL_RANK"] = str(rank)
        rank_env["WORLD_SIZE"] = str(gpus)
        log_path = os.path.join(output_dir, f"rank_{rank}.log")
        log_file = open(log_path, "a", encoding="utf-8")
        rank_command = command + [
            "--log-file", log_path,
            "--no-console-log",
        ]
        log.info(
            "启动 rank %d/%d（GPU %s），日志 %s",
            rank, gpus, gpu, log_path,
        )
        processes.append(subprocess.Popen(
            rank_command,
            env=rank_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        ))
        log_files.append(log_file)

    status = 0
    # 主进程心跳：rank 的日志全在各自文件里（--no-console-log），没有心跳的话
    # 控制台在 spawn 完到全部退出之间一行不出，看起来像冻住。每 30 秒报一次
    # 还有哪些 rank 活着、已运行多久。
    from time import sleep as _sleep

    from xmetai_inference.backends.base import _fmt_dur as _fmt_dur_main

    t_heartbeat = perf_counter()
    reported = set()
    while True:
        _sleep(30)
        statuses = [
            (rank, process.poll()) for rank, process in enumerate(processes)
        ]
        # rank 一死立刻报告（含退出码与日志路径），不等全部结束——OOM kill 是
        # SIGKILL，rank 自己的日志戛然而止，主进程这里是唯一的告警点。
        for rank, code in statuses:
            if code is None or rank in reported:
                continue
            reported.add(rank)
            if code == 0:
                log.info("rank %d 已完成", rank)
            else:
                log.error(
                    "rank %d 已退出（code=%s），日志 %s",
                    rank, code, os.path.join(output_dir, f"rank_{rank}.log"))
        alive = [rank for rank, code in statuses if code is None]
        if not alive:
            break
        log.info(
            "推理进行中（已 %s）：rank %s 仍在跑，明细看 %s",
            _fmt_dur_main(perf_counter() - t_heartbeat),
            alive,
            os.path.join(output_dir, "rank_*.log"),
        )
    for rank, (process, log_file) in enumerate(zip(processes, log_files)):
        if process.wait() != 0:
            status = 1
            log.error(
                "rank %d 失败，日志 %s",
                rank, os.path.join(output_dir, f"rank_{rank}.log"),
            )
        else:
            log.info("rank %d 完成", rank)
        log_file.close()
    if status:
        raise SystemExit("有 rank 失败，退出非零")
    return 0


def _worker_slots(gpus, cuda_devices, procs_per_gpu):
    """worker 槽位表 [(槽位号, 卡号)]：每张卡 procs_per_gpu 个 worker。

    卡号按 cuda_devices 顺序取，超出列表长度时退回「第几组就是第几张卡」——
    与旧的静态路径同一套兜底（写错的卡号会在 CUDA 初始化时报 failure 100）。
    """
    slots = []
    for index in range(max(1, int(gpus)) * max(1, int(procs_per_gpu))):
        position = index // max(1, int(procs_per_gpu))
        gpu = (cuda_devices[position] if position < len(cuda_devices)
               else str(position))
        slots.append((index, gpu))
    return slots


def _pending_keys(store, plan):
    """还没完成、也没判失败的任务键（判失败的不再重派，只在报告里出现）。"""
    failed = store.failed_keys()
    return [task.key for task in plan.tasks
            if task.key not in failed and not store.is_done(task.key)]


def _monitor_pass(sched, store, plan, processes, log_paths, started):
    """看着本轮 worker：报进度、终止卡死的进程（心跳断但进程还活着）。"""
    from xmetai_inference.backends.base import _fmt_dur

    total = len(plan.tasks)
    keys = set(plan.by_key)
    # 启动宽限：worker 要先 import torch/onnx、建数据集、加载权重，才写第一条心跳。
    grace = max(600.0, 5.0 * sched.heartbeat_timeout)
    while True:
        alive = [index for index, process in enumerate(processes)
                 if process.poll() is None]
        if not alive:
            return
        elapsed = perf_counter() - started
        done = len(store.done_keys() & keys)
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else None
        statuses = store.read_status()
        detail = []
        for index in alive:
            worker_id = f"w{index}"
            status = statuses.get(worker_id)
            age = (time.time() - float(status.get("ts") or 0)) if status else None
            if status is None and elapsed < grace:
                detail.append(f"{worker_id}:启动中")
                continue
            if age is None or age > sched.heartbeat_timeout:
                # 心跳断了但进程还在：卡死的 worker（例如某个不返回的 kernel）。
                # 终止它，让它认领的任务被别人回收 —— 否则整轮都会卡在这里。
                log.error("worker %s 心跳停了 %s 但进程还活着，判定卡死并终止；"
                          "它手里的任务会被其它 worker 回收",
                          worker_id, "从未上报" if age is None else _fmt_dur(age))
                processes[index].terminate()
                continue
            # task 为空时回落到 state：worker 在建数据集 / 造成员集的那段时间还没有
            # 任务可报，一律显示 "-" 就分不清「在启动」和「卡住」了。
            detail.append(f"{worker_id}:{status.get('task') or status.get('state') or '-'}"
                          f"(完成 {status.get('done', 0)}，心跳 {_fmt_dur(age)}前)")
        log.info("进度 %d/%d（%.1f%%）%s | 已跑 %s%s", done, total,
                 100.0 * done / total if total else 100.0,
                 "".join(f" | {item}" for item in detail),
                 _fmt_dur(elapsed),
                 f" | 预计还需 {_fmt_dur(eta)}" if eta else "")
        time.sleep(sched.monitor_interval)


def _run_managed(cfg, sched, scope, plan, plan_dir, base_command, gpus,
                 cuda_devices):
    """任务级动态认领的主控：起 worker + 看心跳 + 无进展时清僵尸认领再跑一轮。

    主控**不派活**（派活是 worker 在共享盘上自己认领），它只管三件事：写任务表、
    看好进程、本轮结束后把没主的认领清掉再给一轮机会。断点续跑与崩溃重派因此都落在
    文件系统上，不需要任何常驻的调度服务。
    """
    from xmetai_inference.scheduling import ClaimStore, new_launch_id
    from xmetai_inference.backends.base import _fmt_dur

    t_run = perf_counter()
    output_dir = scope["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    slots = _worker_slots(gpus, cuda_devices, sched.procs_per_gpu)
    log_paths = [os.path.join(output_dir, f"worker_w{index}_gpu{gpu}.log")
                 for index, gpu in slots]
    log.info("调度：任务级动态认领，%d 个任务 / %d 卡 x %d 进程 = %d 个 worker",
             len(plan.tasks), max(1, int(gpus)), sched.procs_per_gpu, len(slots))
    log.info("      计划目录 %s（指纹 %s），续跑口径 resume=%s",
             plan_dir, plan.fingerprint, sched.resume)
    if sched.resume == "off":
        log.warning("      resume=off：整份重算。完成标记只认本次运行写的"
                    "（旧的产物会被覆盖），失败记录仍然算数")
    if sched.procs_per_gpu > 1:
        # 同一个数也传给了 worker（见 _run_infer 的 gpu_mem 参数）：日志与实际必须一致。
        share = _gpu_mem_per_proc(cfg.gpu_mem, sched, True)
        log.info("      每卡 %d 个进程：显存按份数分，gpu_mem=%.2f -> 每份约 %.2f",
                 sched.procs_per_gpu, cfg.gpu_mem, share)
        if share > 0.9:
            log.warning("      每份显存 %.2f 偏大（>0.9），容易 OOM："
                        "请把 config 的 gpu_mem 调成 <= %.2f",
                        share, 0.9 * sched.procs_per_gpu)

    attempt = 0
    store = None
    while True:
        attempt += 1
        launch_id = new_launch_id()
        store = ClaimStore(
            plan_dir, plan.fingerprint, launch_id=launch_id,
            heartbeat_timeout=sched.heartbeat_timeout,
            task_ttl_seconds=sched.task_ttl_seconds,
            task_retries=sched.task_retries,
            # resume=off = 整份重算：只认本次运行写的完成标记，否则第一次跑就会被上一次
            # 的 100% 完成度挡回去（一轮都不跑，直接「没有待跑任务」）。
            done_scope="launch" if sched.resume == "off" else "all")
        store.write_plan(plan)
        if attempt == 1:
            foreign = store.foreign_launches()
            if foreign:
                who = next(iter(foreign.values()))
                raise SystemExit(
                    "输出目录里还有另一次运行活着（worker_id=%s host=%s）。"
                    "同一份输出目录不能同时跑两次：先停掉它，或者换个 --out。"
                    % (who.get("worker_id"), who.get("hostname")))
        pending = _pending_keys(store, plan)
        if not pending:
            log.info("没有待跑任务（%d 个任务全部已有完成标记或产物），跳过第 %d 轮",
                     len(plan.tasks), attempt)
            break

        processes, handles = [], []
        started = perf_counter()
        for index, gpu in slots:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["LOCAL_RANK"] = str(index)
            env["WORLD_SIZE"] = str(len(slots))
            command = base_command + [
                "--launch-id", launch_id,
                "--worker-id", f"w{index}",
                "--gpu-id", gpu,
                "--log-file", log_paths[index],
                "--no-console-log",
            ]
            handle = open(log_paths[index], "a", encoding="utf-8")
            log.info("启动 worker w%d（GPU %s，本轮 %d 个待跑任务），日志 %s",
                     index, gpu, len(pending), log_paths[index])
            processes.append(subprocess.Popen(
                command, env=env, stdout=handle, stderr=subprocess.STDOUT))
            handles.append(handle)

        _monitor_pass(sched, store, plan, processes, log_paths, started)
        for index, process in enumerate(processes):
            code = process.wait()
            if code != 0:
                log.warning("worker w%d 退出码 %s（判定以任务表为准，见 %s）",
                            index, code, log_paths[index])
        for handle in handles:
            handle.close()

        remaining = _pending_keys(store, plan)
        if not remaining:
            break
        if attempt >= sched.max_passes:
            log.error("跑了 %d 轮，仍有 %d 个任务没完成：%s", attempt,
                      len(remaining), ", ".join(remaining[:8]))
            raise SystemExit(
                f"有 {len(remaining)} 个任务没能完成（细节看 "
                f"{os.path.join(output_dir, 'worker_w*_gpu*.log')}）")
        log.warning("本轮结束仍有 %d 个任务没完成（%s）；清掉无主认领后再跑一轮"
                    "（第 %d/%d 轮）", len(remaining), ", ".join(remaining[:6]),
                    attempt + 1, sched.max_passes)
        for key in remaining:
            store.force_release(key)

    done = len(store.done_keys() & set(plan.by_key))
    failed = store.failed_keys()
    log.info("任务表收尾：%d/%d 完成、%d 判失败，用时 %s",
             done, len(plan.tasks), len(failed), _fmt_dur(perf_counter() - t_run))
    if failed:
        log.error("有 %d 个任务判失败（超过 task_retries=%d）：%s", len(failed),
                  sched.task_retries, ", ".join(sorted(failed)[:8]))
        return 1
    return 0


def _config_main(argv=None):
    parser = argparse.ArgumentParser(
        description="统一推理入口：读取内置配置名或外部配置文件并执行推理")
    parser.add_argument(
        "config_path",
        nargs="?",
        help="外部 config.py；等价于 --config config.py",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model",
        help="注册模型配方名，如 fuxi21、fuxi_ens、fgvp、fgvp_spw_ab",
    )
    source.add_argument(
        "--config",
        help="高级用法：内置配置名或外部配置文件路径",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="覆盖模型配方中的权重文件路径",
    )
    parser.add_argument(
        "--times", default=None,
        help="覆盖：起报时间（统一格式，见 configs/base.py::parse_times）")
    parser.add_argument("--members", type=int, default=None, help="覆盖：成员数")
    parser.add_argument("--steps", type=int, default=None, help="覆盖：预报步数")
    parser.add_argument("--vars", default=None, help="覆盖：输出变量，逗号分隔")
    parser.add_argument("--out", default=None, help="覆盖：输出目录")
    parser.add_argument("--gpus", type=int, default=None, help="覆盖：卡数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖：Dataset batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="覆盖：DataLoader worker 数")
    parser.add_argument(
        "--cuda-devices", default=None, help="覆盖：物理卡号，逗号分隔")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="覆盖：控制台日志级别",
    )
    # 调度的三个日常覆盖项：其余调度超参数一律在 config 的 scheduling={...} 里配
    # （它们属于「这份配方怎么跑」，要和 times / steps / members 一起进版本控制）。
    parser.add_argument(
        "--schedule",
        dest="mode",
        default=None,
        choices=["claim", "static"],
        help="覆盖：调度方式（claim=任务级动态认领，static=旧的启动期静态分片）",
    )
    parser.add_argument(
        "--procs-per-gpu",
        type=int,
        default=None,
        help="覆盖：每张卡起几个 worker 进程（显存按份数分，注意 gpu_mem）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        choices=["outputs", "marker", "off"],
        help="覆盖：断点续跑口径（outputs=产物齐全即跳过 / marker=只认完成标记 / off=整份重算）",
    )
    args = parser.parse_args(argv)

    selected = [
        value for value in (args.config_path, args.config, args.model)
        if value is not None
    ]
    if len(selected) != 1:
        parser.error(
            "必须且只能指定一个 config.py、--config 或 --model")
    recipe = selected[0]
    cfg = load_config(recipe)
    log_level = args.log_level or cfg.log_level
    configure_logging(level=log_level)
    overrides = {
        key: getattr(args, key)
        for key in (
            "times", "members", "steps", "vars",
            "out", "gpus", "cuda_devices", "model_path",
            "log_level", "batch_size", "num_workers",
            "mode", "procs_per_gpu", "resume",
        )
    }
    log.info(
        "加载推理配方 %s：model=%s dataset=%s",
        cfg.name,
        args.model_path or cfg.model_path,
        (
            (cfg.dataset or {}).get("type")
            if isinstance(cfg.dataset, dict)
            else cfg.dataset
        ),
    )
    return _run_infer(cfg, overrides)


def _fmt_index_ranges(indices):
    """[0, 4, 5, 6, 7] -> "0,4-7"；日志里 13 个成员号不必逐个列。"""
    if not indices:
        return "-"
    parts, start, previous = [], indices[0], indices[0]
    for value in list(indices[1:]) + [None]:
        if value is not None and value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return ",".join(parts)


def _guard_segment_model(cls, plan):
    """分段换权重的前提：模型在 session 内**没有跨步状态**。

    段边界是「换 session 接着跑」实现的，Rollout 自己持有的 state 不受影响，所以对
    「状态全在输入窗里」的图（FGVP 这类单步型、FuXi 这类窗口型）逐位等价。但跨步有状态的
    模型（如 Pangu 方案B 的 24h 锚点，靠 reset_runtime_state 清空）会在换 session 时静默
    丢掉那份状态 —— 输出照样有，只是物理上错了。静默算错比报错糟，所以这里直接拦。
    """
    from xmetai_inference.backends.base import BaseInferModel

    # getattr 而不是 cls.reset_runtime_state：没有这个属性的类（测试桩 / 非 BaseInferModel
    # 出来的类）也要走到同一条明确的报错，而不是 AttributeError。
    if getattr(cls, "reset_runtime_state", None) is not BaseInferModel.reset_runtime_state:
        raise SystemExit(
            f"{cls.__name__} 覆盖了 reset_runtime_state（session 内有跨步状态），"
            "分段 rollout 会在段边界换 session、从而静默丢掉那份状态，因此不支持。"
            "要跑分段请换「状态全部在输入窗里」的模型。")
    if getattr(cls, "gpu_state", False):
        raise SystemExit(
            f"{cls.__name__} 开了 gpu_state（state 常驻显存），分段 rollout 尚未在该路径上"
            "验证过（换 session 会重置输出 buffer 复用池）。请先用 gpu_state=False 跑分段。")


def _prepare_member_set(src, spec, count, member_indices, local_rank, world_size):
    """m1a：生成本 rank 那一份成员，然后等齐本 rank 要跑的那些成员。

    分片（``range(rank, count, world_size)``）必须互不相交：`perturbed.mNNN.data` 是直接以
    "wb" 打开的，两个 rank 同时造同一个成员会写出交错的字节（不报错，只是权重错）。
    cli 对集合强制「members >= 卡数时按成员切」，正常情况下本 rank 要跑的就是本 rank
    造的那一份，不会有等待；只有成员比卡还少时才会等别的 rank 把剩下的造完。
    """
    # 惰性导入：backends.base 顶层 import torch，而 --help / 配置清单这条路不该被拖上 torch。
    from xmetai_inference.backends.base import _fmt_dur
    from xmetai_inference.perturbation import build_member_set, member_exists

    shard = list(range(local_rank, count, world_size))
    # 只造本 rank 分片里**还不在盘上**的那些。build_member_set 是确定性的（同名即同内容），
    # 但它不做存在性检查就直接重写 perturbed.mNNN.data——所以这里必须自己过滤，否则「先造 5 个
    # 再补到 51」会把已有的 5 个白算一遍（FGVP 上每组几十 GB 的读写）。
    todo = [i for i in shard if not member_exists(src, spec, i)]
    if todo:
        t0 = perf_counter()
        report = build_member_set(src, spec, count, indices=todo)
        log.info("M1a 成员集：本 rank 生成 %d/%d 个成员（分片 %s，用时 %s），目录 %s",
                 len(todo), count, _fmt_index_ranges(todo),
                 _fmt_dur(perf_counter() - t0), report["set_dir"])
        log.info("  被扰动元素 %.2fM/成员，sigma: %s",
                 report["perturbed_elements"] / 1e6,
                 ", ".join(f"{k}={v:.6g}" for k, v in report["sigmas"].items()))
        for warning in report["warnings"]:
            log.warning("[SPW] %s", warning)
    else:
        log.info("M1a 成员集：本 rank 要用的 %d 个成员都已在盘上，跳过生成",
                 len(member_indices))

    missing = [i for i in member_indices if not member_exists(src, spec, i)]
    if missing:
        deadline = time.monotonic() + 3600
        log.info("等待其它 rank 生成剩余成员（%s）...", missing[:8])
        while missing and time.monotonic() < deadline:
            time.sleep(5)
            missing = [i for i in member_indices if not member_exists(src, spec, i)]
        if missing:
            raise RuntimeError(
                f"等了 1 小时，成员 {missing[:8]} 仍未生成。请检查是否有 rank 提前退出"
                "（某个 rank 挂了，它负责的那片成员就永远造不出来）。")


def _pool_summary(model):
    """常驻 session 池的计数摘要（收尾日志用）。

    只有 ONNX 后端的 m1a 会真的往池子里放第二份 session（一份 session 服务全部成员时
    没有第二份可驻留），别的后端返回空串，让调用方把日志拼成一句。
    """
    stats = getattr(model, "pool_stats", None)
    if stats is None:
        return ""
    info = stats()
    if not info.get("builds"):
        return ""
    return "；session 池：新建 %d 份、命中 %d 次、换出 %d 份（额度 %s）" % (
        info["builds"], info["hits"], info["evictions"],
        "不限" if info["budget"] < 0 else info["budget"])


def _drop_sessions(model):
    """把模型的 session 与常驻池一起放开。

    任务半路失败后要求「下个任务重 load = 干净重建」；池子留着的话重 load 会直接命中
    那份可能已经坏掉的 session，把这条前提悄悄破掉。没有 release_all 的后端退回
    release()（额度 1 时它连池子一起清），两者都没有就什么都不做。
    """
    release_all = getattr(model, "release_all", None)
    if release_all is not None:
        release_all()
        return
    release = getattr(model, "release", None)
    if release is not None:
        release()


def _run_member_outer(model, processors, dataset_source, writer, args, init_times,
                      member_indices, save_names, lat, lon, interval, local_rank,
                      world_size, segments=None):
    """m1a 的集合编排：成员在外层、数据集在内层（每个成员把数据集走一遍）。

    为什么成员要在外层：m1a 的成员不是「同一 session 里的另一条轨迹」，而是另一份
    onnx 权重。51 个 session 同时驻留会按 ~3 GB/份吃光显存，所以缺省一次只挂一个成员
    （config 的 resident_members=1，跑完立即释放）；显存够大就把 resident_members 调大，
    池子按 LRU 留 N 份，换回来时不重建 session、不重预热 cuDNN。代价是把数据集迭代
    len(member_indices) 遍——相对于一遍的算力，每遍的装载 IO 很小；而且 Rollout 本来
    就是逐轨迹 batch=1 前向，「成员内层」并不会因为批起来而更快，所以这个换法不损失
    吞吐。

    成员内的循环体与非集合的确定性路径逐字相同（同样是一次 Rollout 跑完整个 batch 的
    起报），区别只在输出落到 member_<NNN>/ 子目录——所以下游的产物体检（enscheck）和
    评测脚本不需要区分 m3 / m1a。
    """
    from xmetai_inference.backends.base import _fmt_dur

    total_t0 = perf_counter()
    _run_member_block(
        model, processors, dataset_source, writer, args, init_times, member_indices,
        save_names, lat, lon, interval, local_rank, world_size, segments=segments)
    errors = writer.close()
    if errors:
        log.error("写盘阶段发生 %d 个错误，任务失败", len(errors))
        return 1
    log.info("全部完成：成员=%d steps=%d output=%s 总耗时=%.1fs%s",
             len(member_indices), args.steps, args.out,
             perf_counter() - total_t0, _pool_summary(model))
    return 0


def _run_member_block(model, processors, dataset_source, writer, args, init_times,
                      member_indices, save_names, lat, lon, interval, local_rank,
                      world_size, segments=None, loaded_member=None):
    """成员外层的一段编舞 = 上面那个「成员循环」本身（不含收尾的 close 与总结）。

    拆出来的理由在 claim 模式：那里一个任务就是「一个起报 × 一段成员」，跑完一个任务
    要 flush + 写完成标记，而 writer 与 model 都要活到整个 worker 退出，不能每个任务
    关一次。

    ``loaded_member`` 是「上一个任务跑完之后仍挂在 model 上的成员号」。认领顺序按成员
    优先时（m1a 的 claim_order="auto"）相邻任务常常是同一个成员，于是这一次 onnx 装载
    （FGVP 上 1.5 GB + cuDNN 预热）就白省下来了。返回本次跑完时挂着的成员号；分段
    rollout 不能复用（见下），返回 None。
    """
    from xmetai_inference.backends.base import _fmt_dur
    from xmetai_inference.runner import Rollout, Trajectory

    release = getattr(model, "release", None)
    # 分段 rollout（E3）：第一段必须用 segments 里的规格，而不是 config 里那份 base 规格 ——
    # 两者的成员集目录（指纹）不同，用错规格会去找一个根本没生成的成员集。
    switch_after = {}
    first_spec = None
    if segments is not None:
        first_spec = segments.segments[0].spec
        model.weight_perturb = first_spec
        switch_after = segments.switch_after()
        log.info("分段 rollout：%s（%d 段，段边界在跑完第 %s 步之后；末段之后不再换）",
                 segments.describe(), len(segments.segments),
                 "、".join(str(step) for step in sorted(switch_after)))
    # 分段时不许跨任务复用：一个成员跑完最后一段后，驻留的是**末段**的权重，而下一个
    # 成员必须从第 0 段开始 —— 拿一个改过权重的 session 接着跑，前几步的权重是错的，
    # 而且不报错（只有 e2e 那条「边界之前逐位相同」能抓到）。
    reuse = None if segments is not None else loaded_member
    for order, member_id in enumerate(member_indices, 1):
        model.member_index = member_id
        if first_spec is not None:
            # 每个成员都必须从**第 0 段**开始：上一个成员跑到段边界时把 model.weight_perturb
            # 换成了后一段，不重置的话这个成员的头几步会带着上一段的权重跑 —— 安静地错，
            # 只有 e2e 那条「边界之前逐位相同」能抓到它。
            model.weight_perturb = first_spec
        t_member = perf_counter()
        if reuse == member_id:
            log.info("[%d/%d] member_%03d 与上一个任务同一成员，沿用已装载的权重",
                     order, len(member_indices), member_id)
        else:
            model.load(args.model)
            log.info("[%d/%d] member_%03d 权重就绪，加载耗时 %s",
                     order, len(member_indices), member_id,
                     _fmt_dur(perf_counter() - t_member))
        completed = 0
        for data_batch_index, batch in enumerate(dataset_source):
            t_fetch = perf_counter()
            state, batch_inits = model.prepare_dataset_batch(batch)
            if data_batch_index == 0:
                log.info("  member_%03d 首个 batch 就绪：state 形状 %s，装载+前处理耗时 %s",
                         member_id, tuple(state.shape), _fmt_dur(perf_counter() - t_fetch))

            def on_step(step, step_state, batch_inits=batch_inits, member_id=member_id):
                step_idx = step + 1
                for batch_index, init in enumerate(batch_inits):
                    member_ds = model.to_dataset(
                        step_state[batch_index],
                        save_names=save_names,
                        lat=lat,
                        lon=lon,
                    )
                    writer.put(
                        f"{_output_init_dir(init)}/member_{member_id:03d}/{step_idx:03d}.nc",
                        member_ds,
                        raw=(step_state[batch_index], step, init),
                    )

                # 段边界：本步跑完就换成下一段的冻结权重，下一步的前向用它。Rollout 的
                # state 是它自己持有的数组（不在 session 里），所以换 session 对它是透明的：
                # 轨迹连续、帧号连续，只有注入的权重变了。放在写盘之后 —— 这一步的预报
                # 属于**上一段**的权重。成员号不变（同一成员号在每段各有自己那份权重）。
                next_segment = switch_after.get(step_idx)
                if next_segment is not None:
                    segment = segments.segments[next_segment]
                    model.weight_perturb = segment.spec
                    t_switch = perf_counter()
                    model.load(args.model)
                    log.info("  member_%03d 第 %d 段权重就绪（%s，换段耗时 %s）",
                             member_id, next_segment, segment.amplitude(),
                             _fmt_dur(perf_counter() - t_switch))

            Rollout(model, processors).run(
                [
                    Trajectory(
                        state=state[b:b + 1],
                        init_time=init,
                        label=f"member_{member_id:03d}",
                    )
                    for b, init in enumerate(batch_inits)
                ],
                steps=args.steps,
                hour_interval=interval,
                on_step=on_step,
                progress=False,
                progress_label=f"[rank {local_rank}/{world_size}] m{member_id:03d}",
            )
            writer.flush()
            completed += len(batch_inits)
            log.info("  member_%03d 起报进度：%d/%d，本批耗时 %s",
                     member_id, completed, len(init_times),
                     _fmt_dur(perf_counter() - t_fetch))
            del state
            gc.collect()
        # 最后一个成员**不放手**：认领模式下相邻任务常常还是这个成员（下一个任务靠
        # loaded_member 命中复用），放手了下次就得重建 session 并重预热 cuDNN。
        # 额度 1 的池子只剩这一份，显存口径与「跑完立即释放」逐位一样。
        if release is not None and order < len(member_indices):
            release()
            gc.collect()
        log.info("member_%03d 完成，总耗时 %s",
                 member_id, _fmt_dur(perf_counter() - t_member))
    if not member_indices or segments is not None:
        return None
    return int(member_indices[-1])


def _run_plain_block(model, processors, dataset_source, writer, args, member_indices,
                     save_names, lat, lon, interval, local_rank, world_size, members,
                     total_inits, num_workers=0):
    """确定性 / 同 session 集合的一段编舞：数据集在内层遍历，逐 batch 落盘并 flush。

    与 m1a 的成员外层相反：这里的「成员」是同一次 Rollout 里的多条轨迹（m3 的图内随机
    算子、或 latent_forcing 的潜空间强迫），一份 session 服务全部成员。members==1 时
    每个起报一条轨迹、输出落 ``{起报}/{步号}.nc``；members>1 时每个成员一条轨迹、输出落
    ``{起报}/member_{成员3位}/{步号}.nc``。返回跑完的起报条数。

    拆出来同样是为了 claim 模式：那里一个任务 = 一个起报，``dataset_source`` 就是只装
    那一个样本的单元素列表，``member_indices`` 就是该任务分到的成员。
    """
    from xmetai_inference.backends.base import _fmt_dur
    from xmetai_inference.runner import Rollout, Trajectory

    completed = 0
    log.info("开始迭代数据集，等待首个 batch 装载（workers=%s，"
             "每个 worker 要重新打开 zarr 并读满一个 batch，可能较慢）...", num_workers)
    t_fetch = perf_counter()
    # 注意：循环体内已有局部 batch_index（按 batch_inits 枚举），这里用
    # data_batch_index 区分，避免遮蔽/改绑。
    for data_batch_index, batch in enumerate(dataset_source):
        fetch_dur = perf_counter() - t_fetch
        state, batch_inits = model.prepare_dataset_batch(batch)
        if data_batch_index == 0:
            log.info("首个 batch 就绪：state 形状 %s，装载+前处理耗时 %s",
                     tuple(state.shape), _fmt_dur(fetch_dur))

        # 确定性（members==1）：每个起报一条轨迹，一次 Rollout 跑完整个 batch。
        # 集合（members>1）：每个起报单独一次 Rollout，成员即轨迹。
        # 两者用的是同一个循环，区别只在「轨迹怎么构造」和「输出落到哪」。
        if members == 1:
            def on_batch_step(step, step_state, batch_inits=batch_inits):
                step_idx = step + 1
                for batch_index, init in enumerate(batch_inits):
                    init_dir = _output_init_dir(init)
                    ds = model.to_dataset(
                        step_state[batch_index],
                        save_names=save_names,
                        lat=lat,
                        lon=lon,
                    )
                    writer.put(
                        f"{init_dir}/{step_idx:03d}.nc",
                        ds,
                        raw=(step_state[batch_index], step, init),
                    )

            Rollout(model, processors).run(
                [
                    Trajectory(
                        state=state[b:b + 1],
                        init_time=init,
                        label=_output_init_dir(init),
                    )
                    for b, init in enumerate(batch_inits)
                ],
                steps=args.steps,
                hour_interval=interval,
                on_step=on_batch_step,
                progress=False,
            )
        else:
            for batch_index, init in enumerate(batch_inits):
                init_dir = _output_init_dir(init)
                base_input = state[batch_index:batch_index + 1]

                def on_step(step, step_state, init=init, init_dir=init_dir):
                    step_idx = step + 1
                    for local_index, member_id in enumerate(member_indices):
                        member_ds = model.to_dataset(
                            step_state[local_index],
                            save_names=save_names,
                            lat=lat,
                            lon=lon,
                        )
                        writer.put(
                            f"{init_dir}/member_{member_id:03d}/{step_idx:03d}.nc",
                            member_ds,
                            raw=(step_state[local_index], step, init),
                        )

                # 成员即轨迹：全部共享同一初始态（扰动来自图内随机算子），
                # init_time 相同。多卡按成员切时 member_indices 已是本 rank
                # 自己那份，所以这里只造本 rank 的轨迹。
                Rollout(model, processors).run(
                    [
                        Trajectory(
                            state=base_input,
                            init_time=init,
                            label=f"member_{member_id:03d}",
                        )
                        for member_id in member_indices
                    ],
                    steps=args.steps,
                    hour_interval=interval,
                on_step=on_step,
                # 这一个 Rollout 要跑「51 成员 x 60 步」约 3000 次前向（几十分钟），
                # 关掉进度行等于全程零日志 —— 连首步那条「含 cudnn 穷举搜索、偏慢属正常」
                # 的提示也一起没了（这条正是为长任务写的，别处成员外层有逐成员日志兜着）。
                # 逐轨迹的进度行本身是稀疏的（每 5 步一行 + 首步必打），不会刷屏。
                progress=True,
                progress_label=f"[rank {local_rank}/{world_size}] {init:%m%d%H}",
                )

        writer.flush()
        completed += len(batch_inits)
        log.info(
            "Dataset batch 完成：%d/%d，batch=%d（装载 %s，本批总耗时 %s）",
            completed,
            total_inits,
            len(batch_inits),
            _fmt_dur(fetch_dur),
            _fmt_dur(perf_counter() - t_fetch),
        )
        t_fetch = perf_counter()
        del state
        gc.collect()
    return completed


def _worker_scheduling(args, worker_cfg):
    """worker 侧的调度配置：优先用主进程算好的那份 JSON，其次回落到 config。

    命令行的三个调度覆盖（--schedule / --procs-per-gpu / --resume）只在主进程手里，
    主进程把它们并进 config 之后整份 JSON 传下来；worker 自己再读一遍 config 会丢掉
    那些覆盖 —— 「主进程按 claim 起了 worker，worker 却按 static 切片」这种错很难看出来。
    """
    from xmetai_inference.scheduling import SchedulingConfig

    if args.scheduling_json:
        try:
            return SchedulingConfig.from_config(json.loads(args.scheduling_json))
        except (TypeError, ValueError) as error:
            raise SystemExit(f"--scheduling-json 解析失败：{error}") from error
    try:
        return worker_cfg.scheduling_config()
    except (TypeError, ValueError) as error:
        raise SystemExit(f"config 的 scheduling 有误：{error}") from error


def _claim_setup(args, sched):
    """claim 模式的启动段：建账本 + 读任务表，并校验指纹。

    指纹对不上必须停下来：任务表决定产物路径与完成标记的内容，拿着一张别人的任务表跑，
    写出来的目录会「标记齐全、产物错位」。计划目录被两次运行覆盖是唯一成因（同一份
    output_dir 同时跑两次由主控的 foreign_launches 挡住）。
    """
    from xmetai_inference.scheduling import ClaimStore

    store = ClaimStore(
        args.plan_dir,
        args.plan_fingerprint,
        launch_id=args.launch_id or "",
        worker_id=args.worker_id or f"pid{os.getpid()}",
        heartbeat_timeout=sched.heartbeat_timeout,
        task_ttl_seconds=sched.task_ttl_seconds,
        task_retries=sched.task_retries,
        # resume=off = 整份重算：完成标记只认本次运行写的（见 scheduling.store 模块文档）。
        done_scope="launch" if sched.resume == "off" else "all",
    )
    store.ensure_dirs()
    plan = store.read_plan()
    if plan is None:
        raise SystemExit(
            f"claim 模式需要任务表，但 {store.dir} 下没有 plan.json。"
            "两种可能：主控还没写它（别直接调 --worker），或者 --plan-dir / 指纹"
            "与主控给的不一致。")
    if plan.fingerprint != args.plan_fingerprint:
        raise SystemExit(
            f"任务表指纹对不上：{store.dir}/plan.json 里是 {plan.fingerprint}，"
            f"本次运行要的是 {args.plan_fingerprint}。多半是另一次运行覆盖了同一个计划"
            "目录（plan_dir）；换个输出目录，或等它跑完。")
    return plan, store


def _run_task(model, processors, task, batch, writer, args, save_names, lat, lon,
              interval, local_rank, world_size, members, member_outer, segments,
              loaded_member, member_specs):
    """跑一个任务（一个起报 × 一段成员），返回跑完之后还挂在 model 上的成员号。

    m1a 走成员外层（一个成员一份权重、逐成员装载），其余（确定性 / m3 / latent）走
    「一份 session + 一批轨迹」。两条路的循环体与静态模式是同一份实现（同一个
    _run_member_block / _run_plain_block），所以产物布局逐位一致。
    """
    if not member_outer:
        _run_plain_block(
            model, processors, [batch], writer, args, task.members(), save_names,
            lat, lon, interval, local_rank, world_size, members, total_inits=1)
        return loaded_member
    from xmetai_inference.perturbation import member_exists

    for spec in member_specs:
        missing = [index for index in task.members()
                   if not member_exists(args.model, spec, index)]
        if missing:
            # 本 worker 启动时已经把自己那一整片造好了（见 _worker_main），所以这里等的
            # 是别人的份额，有上界；等不到会带明确报错抛出来，被主控按任务失败记下。
            log.info("任务 %s 要用的成员 %s 还没落盘，等其它 worker 造完",
                     task.key, _fmt_index_ranges(missing))
            _prepare_member_set(args.model, spec, members, task.members(),
                                local_rank, world_size)
    return _run_member_block(
        model, processors, [batch], writer, args, [task.init], task.members(),
        save_names, lat, lon, interval, local_rank, world_size, segments=segments,
        loaded_member=loaded_member)


def _run_claimed(model, processors, dataset_source, writer, args, sched, plan, store,
                 beater, init_times, save_names, lat, lon, interval, member_outer,
                 segments, member_specs, perturb_spec, local_rank, world_size):
    """认领模式的主循环：认领一个任务 -> 跑 -> flush 落盘 -> 写完成标记。

    这里刻意没有「派活」这一步：任务由 feed 在后台线程上从共享盘自己认领，所以
    「谁跑哪个任务」是运行期决定的 —— 慢的少拿、快的多拿，中断过的不必从头再来，
    worker 崩掉后它手里的任务会被别人捡走（判僵尸的规则见 scheduling.store）。

    先 flush 再写完成标记，顺序不能反：标记只说一件事 ——「产物已经在盘上」。反过来写
    就会把半成品标成完成，续跑时直接跳过它。

    读输入的是 feed 线程、写 NetCDF 的是 writer 线程，两者不会同时摸 HDF5：输入是
    zarr（xarray 走 zarr 解码器，不经过 HDF5），HDF5 只有 writer 一个使用者。这也是
    「读输入可以放到后台线程」的前提。
    """
    from xmetai_inference.backends.base import _fmt_dur
    from xmetai_inference.scheduling import LocalIndex, TaskFeed

    positions = {pd.Timestamp(init): index for index, init in enumerate(init_times)}

    def read_sample(task):
        index = positions.get(pd.Timestamp(task.init))
        if index is None:
            raise KeyError(
                f"任务 {task.key} 的起报不在本 worker 的数据集里：{task.init}")
        item = dataset_source.dataset[index]
        # 单个样本补回 batch 维：形状与 DataLoader 出来的 batch 逐位一致（[1,T,C,H,W]），
        # 于是 model.prepare_dataset_batch 那条路一个字都不用改。
        return {key: (value.unsqueeze(0) if hasattr(value, "unsqueeze") else value)
                for key, value in item.items()}

    order = sched.claim_order_for(perturb_spec is not None)
    # 输入缓存只在「按起报优先」时有复用机会（相邻任务是同一起报的不同成员）。按成员
    # 优先时相邻任务的起报每次都在换，缓存只会白占一份 650 MB 的主机内存。
    cache_size = sched.input_cache if order == "init" else 0
    log.info("认领模式：order=%s，预取 %d 个任务，输入缓存 %d 条，心跳 %.0fs/超时 %.0fs，"
             "resume=%s", order, sched.prefetch_tasks, cache_size,
             sched.heartbeat_interval, sched.heartbeat_timeout, sched.resume)

    feed = TaskFeed(store, plan, read_sample, order=order, depth=sched.prefetch_tasks,
                    poll_interval=sched.poll_interval, cache_size=cache_size,
                    logger=log).start()
    beater.beat(state="running")
    stats = {"done": 0, "skipped": 0, "failed": 0}
    loaded_member = None
    t_all = perf_counter()
    try:
        for task, batch in feed:
            beater.set_task(task.key)
            if sched.resume == "outputs" and plan.complete_on_disk(task, args.out):
                # 产物齐全（上一次跑到一半，或主控重派）：补一个完成标记再跳过。补标记
                # 是为了让别的 worker 也不必再算它 —— 它们看到的状态同样是「已完成」。
                store.mark_done(task.key)
                feed.finish(task.key)
                stats["skipped"] += 1
                beater.bump_done()
                log.info("任务 %s（%s）的产物已在盘上，跳过（resume=outputs）",
                         task.key, plan.describe_task(task))
                continue
            t_task = perf_counter()
            try:
                loaded_member = _run_task(
                    model, processors, task, batch, writer, args, save_names, lat,
                    lon, interval, local_rank, world_size, plan.members, member_outer,
                    segments, loaded_member, member_specs)
            except Exception as error:  # noqa: BLE001 - 一个任务失败不该带走整个 worker
                attempt = store.fail(task.key, f"{type(error).__name__}: {error}")
                give_up = attempt > sched.task_retries
                # 无论判失败还是等重派，都要把它从「我手头」摘掉：预取线程靠 mine 为空
                # 判断收工（见 scheduling.feed），留着它就永远等不到 END，worker 会挂死。
                # 写回 FREE 时会自己重认领一次 —— 那就是重试。
                feed.index.mark(task.key,
                                LocalIndex.FAILED if give_up else LocalIndex.FREE)
                stats["failed"] += 1
                # 半路抛出来的失败，模型状态不可信（可能跑了几步才炸），下个任务重 load。
                loaded_member = None
                # 常驻池也要一起放手：池子留着的话重 load 会直接命中那份可能已经坏掉的
                # session，把「重 load = 干净重建」这条前提悄悄破掉。
                _drop_sessions(model)
                log.exception("任务 %s（%s）失败：第 %d 次，%s", task.key,
                              plan.describe_task(task), attempt,
                              "判失败、不再重派" if give_up else "待重派")
                continue
            writer.flush()
            store.mark_done(task.key)
            feed.finish(task.key)
            stats["done"] += 1
            beater.bump_done()
            log.info("任务 %s 完成（%s，本任务 %s，累计 %d/%d）", task.key,
                     plan.describe_task(task), _fmt_dur(perf_counter() - t_task),
                     stats["done"], len(plan.tasks))
    finally:
        feed.close(release_claims=True)
        beater.stop()
    errors = writer.close()
    log.info("worker 收尾：完成 %d、跳过 %d、失败 %d，用时 %s（读输入 %d 次、命中缓存 %d 次）%s",
             stats["done"], stats["skipped"], stats["failed"],
             _fmt_dur(perf_counter() - t_all), feed.stats.get("read", 0),
             feed.stats.get("cache_hit", 0), _pool_summary(model))
    if errors:
        log.error("写盘阶段发生 %d 个错误", len(errors))
        return 1
    return 0

def _worker_main(argv=None):
    from xmetai_inference.backends.base import _fmt_dur
    from xmetai_inference.backends.session_pool import normalize_budget
    from xmetai_inference.data import create_dataset_source
    from xmetai_inference.models import create_model, get_model_class, grid_coords
    from xmetai_inference.processing.pipeline import build_pipeline
    from xmetai_inference.runner import Rollout, Trajectory

    p = argparse.ArgumentParser(description="气象模型自回归推理（后端可插拔）")
    p.add_argument(
        "--config-source",
        required=True,
        help="主进程解析后的 config 文件绝对路径",
    )
    p.add_argument("--model", required=True, help="模型文件路径（.onnx/.pt2）")
    p.add_argument("--time", default=None, help="单次起报时间 YYYYMMDDHH（与 --start/--end 二选一）")
    p.add_argument("--start", default=None, help="起始起报时间 YYYYMMDDHH")
    p.add_argument("--end", default=None, help="结束起报时间 YYYYMMDDHH（含，默认=--start）")
    p.add_argument("--freq", type=int, default=None, help="相邻起报间隔小时（默认=步长 interval）")
    p.add_argument("--inits", default=None,
                   help="起报时间列表，逗号/空格分隔（YYYYMMDD 或 YYYYMMDDHH）；"
                        "只跑这几个起报，与 --time/--start 互斥")
    p.add_argument("--steps", type=int, default=10, help="预报步数")
    p.add_argument("--members", type=int, default=None,
                   help="集合成员总数（缺省读模型类的 members；确定性=1）")
    p.add_argument("--history", type=int, default=None, help="输入历史帧数（默认用模型类）")
    p.add_argument("--interval", type=int, default=None, help="时间步长小时（默认用模型类）")
    p.add_argument("--device", type=int, default=None, help="GPU 设备号（默认 0；多卡时由 CUDA_VISIBLE_DEVICES 隔离）")
    p.add_argument("--world-size", type=int, default=None, help="卡数（默认读 WORLD_SIZE）")
    p.add_argument("--gpu-mem", type=float, default=0.7, help="显存占用比例")
    p.add_argument("--batch-size", type=int, default=None, help="Dataset batch size")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader worker 数")
    p.add_argument("--out", default=None, help="输出目录；不写则只做输入校验")
    p.add_argument("--vars", default=None,
                   help="要保存的输出变量，逗号分隔（如 z500,u200,v200,msl,tp）；不传则保存全部通道")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="控制台日志级别")
    p.add_argument("--log-file", default=None, help="完整 DEBUG 日志文件")
    p.add_argument("--no-console-log", action="store_true", help="关闭控制台日志（多卡子进程使用）")
    # 调度相关：主进程算好整份配置传下来（--scheduling-json），worker 不再自己拼 ——
    # 命令行的 --schedule/--procs-per-gpu/--resume 只在主进程手里，父子必须看到同一套。
    p.add_argument("--schedule", default=None, choices=["claim", "static"],
                   help="调度方式；claim 还需要 --plan-dir 与 --plan-fingerprint")
    p.add_argument("--plan-dir", default=None, help="计划/认领目录（claim 模式）")
    p.add_argument("--plan-fingerprint", default=None, help="任务表指纹（claim 模式）")
    p.add_argument("--launch-id", default=None, help="本次运行标识（主控生成，全部 worker 共用）")
    p.add_argument("--worker-id", default=None, help="worker 标识（主控分配，如 w0）")
    p.add_argument("--gpu-id", default=None, help="物理卡号（只用于日志；隔离靠 CUDA_VISIBLE_DEVICES）")
    p.add_argument("--scheduling-json", default=None, help="整份调度配置 JSON（主进程算好）")
    p.add_argument("--intra-op-threads", type=int, default=0,
                   help="ORT intra_op 线程数（0 = 自动：核数 / worker 数）")
    p.add_argument("--cudnn-algo-search", default="",
                   help="cuDNN 卷积算法搜索（HEURISTIC 可缩短每卡多进程的启动预热）")
    p.add_argument("--resident-members", type=int, default=None,
                   help="常驻 session 池额度（1=一换一放 / N=驻留 N 份 / -1=全常驻）；"
                        "缺省时以 config 的 resident_members 为准")
    args = p.parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = args.world_size if args.world_size is not None \
        else int(os.environ.get("WORLD_SIZE", "1"))
    configure_logging(
        level=args.log_level,
        log_file=args.log_file,
        console=not args.no_console_log,
        rank=f"{local_rank}/{world_size}",
    )
    # 多卡时 inference.py 已用 CUDA_VISIBLE_DEVICES 隔离每个进程，device 恒为 0；
    # 单进程想指定别的卡仍可用 --device 覆盖。
    device_id = args.device if args.device is not None else 0

    # 每个 worker 重新加载同一 config，确保外部 Model/Dataset 在多进程中可用。
    worker_cfg = load_config(args.config_source)
    # 常驻 session 池的额度（只对 m1a 有意义）。优先级：worker 参数（主进程从 config
    # 传下来）> config > 1（一换一放）。0 / 缺省在 normalize_budget 里归一到 1，负数
    # = 不限 —— 显存够大就让成员 session 全常驻，换回来时不重建、不重预热 cuDNN。
    resident_members = normalize_budget(
        args.resident_members if args.resident_members is not None
        else getattr(worker_cfg, "resident_members", 1))
    sched = _worker_scheduling(args, worker_cfg)
    # claim 模式：读任务表 + 建账本，并立刻开始心跳 —— 接下来要建数据集、加载 1.5 GB
    # 的图、可能还要造自己那一片成员（几十分钟），主控判卡死只看心跳。
    claim_mode = args.schedule == "claim" and bool(args.out)
    plan = store = beater = None
    if claim_mode:
        plan, store = _claim_setup(args, sched)
        from xmetai_inference.scheduling import Heartbeat

        beater = Heartbeat(store, sched.heartbeat_interval, logger=log).start()
        beater.beat(state="setup")
    elif args.schedule == "claim":
        log.warning("--schedule claim 但没给 --out（产物落哪都不知道），退回静态分片"
                    "（与旧行为一致：只做输入校验）")
    model_spec = worker_cfg.model_class
    try:
        cls = get_model_class(model_spec)
    except (TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    model_name = (
        model_spec if isinstance(model_spec, str) else
        f"{model_spec.__module__}.{model_spec.__name__}"
    )

    interval = args.interval or getattr(cls, "hour_interval", 6)
    history = args.history or getattr(cls, "history_steps", 2)
    # 集合/确定性显式化：members 缺省从模型类读；类型与成员数不一致时告警（不报错）
    forecast_type = getattr(cls, "forecast_type", "deterministic")
    perturb_spec = None
    if worker_cfg.weight_perturb:
        from xmetai_inference.perturbation import WeightPerturbSpec

        perturb_spec = WeightPerturbSpec.from_config(worker_cfg.weight_perturb)
    if perturb_spec is not None:
        # 权重扰动把确定性模型变成集合，这里按集合语义检查成员数。两种机制的「成员」
        # 含义不同：m3 = 同一 session 里同一初始态的 N 条轨迹（扰动来自图内随机算子）；
        # m1a = N 份冻结权重，每个成员本身就是一次普通的确定性推理。
        forecast_type = "ensemble"
    if worker_cfg.latent_forcing:
        # 潜空间 AR(1) 强迫同样把确定性模型变成集合：成员之间的差别来自每步喂进去的 xi
        # （种子由 member_index 决定），所以同一份图服务全部成员，不必落盘 51 份权重。
        # 这里只声明语义；真正的校验（图里必须有 spw/xi）在 create_model / load 里做。
        forecast_type = "ensemble"
    members = args.members if args.members is not None else getattr(cls, "members", 1)
    if forecast_type == "deterministic" and members > 1:
        log.warning("模型类 %s 声明确定性，但 members=%d；确定性模型应跑 1 个成员",
                    cls.__name__, members)
    elif forecast_type == "ensemble" and members <= 1:
        log.warning("模型类 %s 声明集合，但 members=%d；集合成员数应大于 1",
                    cls.__name__, members)
    init_times = _init_times(args, interval)

    if claim_mode:
        # claim 模式没有「本 rank 那一份」：任务是运行期从共享盘上认领的，所以这里
        # 不切片。起报 / 成员 / 步数一律以任务表为准 —— 产物路径与完成标记都由它算，
        # 两边不一致会写出「标记说完成了、产物却在别的步数上」这种对不上账的目录。
        plan_inits = sorted({task.init for task in plan.tasks})
        if plan_inits != init_times:
            log.warning("命令行给了 %d 个起报、任务表里有 %d 个；以任务表为准",
                        len(init_times), len(plan_inits))
            init_times = plan_inits
        if int(args.steps) != int(plan.steps):
            log.warning("命令行 steps=%s 与任务表 steps=%d 不一致；以任务表为准",
                        args.steps, plan.steps)
            args.steps = int(plan.steps)
        if int(members) != int(plan.members):
            log.warning("命令行 members=%s 与任务表 members=%d 不一致；以任务表为准",
                        members, plan.members)
            members = int(plan.members)
        # 占位：真正跑哪些成员由每个任务决定（任务表里的成员块），这里的「全部成员」
        # 只用于「成员比卡少时会有卡分不到活」那类静态判断以及成员集的生成分片。
        member_indices = list(range(members))
        member_start, member_stride = 0, 1
    else:
    # 多卡分工：把「起报时间 × 成员」的任务摊给各卡，有两种切法——
    #   按起报切：每个 rank 跑一部分起报、全部成员；
    #   按成员切：每个 rank 跑一部分成员、全部起报。
    # 均衡上两种切法通常只差几个百分点，但内存上差异巨大：Rollout 给每条轨迹
    # 持有独立 state 拷贝（~645MB/条），按起报切每卡要挂全部 members 条
    # （51 成员 ≈ 33GB/卡），按成员切只挂 ceil(members/world_size) 条
    # （4 卡 51 成员 ≈ 8GB/卡）。实测按起报切曾把 pod 的 cgroup 内存限额顶穿、
    # 两个 rank 被 OOM kill（SIGKILL，rank 日志无 traceback 直接断）。
    # 故集合（members>1 且成员数 ≥ 卡数）优先按成员切；members < 卡数时按成员
    # 切会有卡分不到任务，退回均衡比较。确定性（members==1）本就只能按起报切。
        n_init = len(init_times)
        per_rank_init = (n_init + world_size - 1) // world_size
        per_rank_member = (members + world_size - 1) // world_size
        cost_by_init = per_rank_init * members
        cost_by_member = n_init * per_rank_member
        if members > 1 and members >= world_size:
            split_by_member = True
        else:
            split_by_member = cost_by_init > cost_by_member
        if not split_by_member:
            base, rem = divmod(n_init, world_size)
            start_i = local_rank * base + min(local_rank, rem)
            span = base + (1 if local_rank < rem else 0)
            init_times = init_times[start_i:start_i + span]
            member_indices = list(range(members))
            member_start, member_stride = 0, 1
        else:
            member_indices = list(range(local_rank, members, world_size))
            member_start, member_stride = local_rank, world_size

    # 扰动档的「成员」是各自一份冻结权重（= 各自的 onnx session），所以成员必须在外层
    # 循环、数据集在内层；这样任何时刻只有 1 个 session 驻留。详见 _run_member_outer。
    member_outer = perturb_spec is not None

    # 分段 rollout（E3，config 的 spw_segments）：在同一条轨迹里按段更换冻结权重。
    # 它不动「成员在外层」的结构，只在段边界多挂一次 session（见 _run_member_outer 的
    # on_step），所以帧号、输出布局、下游体检都与非分段路径完全一致。
    segments = None
    if worker_cfg.spw_segments:
        from xmetai_inference.perturbation.segments import parse_segment_plan

        try:
            segments = parse_segment_plan(worker_cfg.spw_segments, perturb_spec, args.steps)
        except ValueError as error:
            raise SystemExit(f"config 的 spw_segments 无法使用：{error}") from error
        _guard_segment_model(cls, segments)
        log.info("分段 rollout 已启用：%d 段 / 共 %d 步：%s",
                 len(segments.segments), segments.total_steps, segments.describe())

    if not init_times:
        log.info("没有分配到起报时间，退出")
        return 0
    if not member_indices:
        log.info("没有分配到成员，退出")
        return 0
    log.info("任务分配：起报=%d（%s..%s）成员=%d",
             len(init_times), init_times[0].strftime("%Y%m%d%H"),
             init_times[-1].strftime("%Y%m%d%H"), len(member_indices))

    if worker_cfg.dataset is None:
        raise SystemExit("配置必须使用 dataset 路径（旧 loader 已移除）")
    dataloader_config = dict(worker_cfg.dataloader or {})
    if args.batch_size is not None:
        dataloader_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dataloader_config["num_workers"] = args.num_workers
    dataset_spec = worker_cfg.dataset
    t_dataset = perf_counter()
    log.info("正在构建数据集（打开 zarr 存储、装配时间窗），起报 %d 个...",
             len(init_times))
    dataset_source = create_dataset_source(
        dataset_spec,
        model_cls=cls,
        init_times=init_times,
        history_steps=history,
        hour_interval=interval,
        model_path=args.model,
        dataloader=dataloader_config,
    )
    data_source = dataset_source
    log.info("数据集就绪：%d 个 batch（batch_size=%s），构建耗时 %s",
             len(dataset_source),
             dataloader_config.get("batch_size"),
             _fmt_dur(perf_counter() - t_dataset))
    if claim_mode:
        # claim 模式不看 DataLoader：任务是一个一个从共享盘上认领的，读哪个起报由任务
        # 决定（feed 线程直接下标取样本，见 _run_claimed）。于是 num_workers 与
        # batch_size 都不参与，进程数也不再是「1 + num_workers」倍。
        log.info("认领模式：绕过 DataLoader（num_workers=%s / batch_size=%s 不生效），"
                 "输入由预取线程按任务直接取",
                 dataloader_config.get("num_workers"), dataloader_config.get("batch_size"))

    input_specs, recurrent_specs, output_specs = worker_cfg.processor_specs()
    t_pipeline = perf_counter()
    processors = build_pipeline(
        input_specs,
        cls,
        data_source,
        model_path=args.model,
        recurrent_specs=recurrent_specs,
        output_specs=output_specs,
    )
    log.info("前处理管线就绪（input=%d recurrent=%d output=%d），耗时 %s",
             len(input_specs), len(recurrent_specs), len(output_specs),
             _fmt_dur(perf_counter() - t_pipeline))
    model = create_model(
        model_spec,
        device_id=device_id,
        gpu_mem_fraction=args.gpu_mem,
        ops_library=worker_cfg.ops_library_abs(),
        gpu_state=worker_cfg.gpu_state,
        weight_perturb=worker_cfg.weight_perturb,
        latent_forcing=worker_cfg.latent_forcing,
        intra_op_threads=args.intra_op_threads,
        cudnn_algo_search=args.cudnn_algo_search,
        resident_members=resident_members,
    )
    model.configure_processing(processors, data_source)

    if args.out is None:
        model.prepare_dataset_batch(next(iter(dataset_source)), verbose=True)
        log.info("未指定 --out，仅完成首个起报时间的输入构建检查")
        return 0

    os.makedirs(args.out, exist_ok=True)
    member_specs = (list(segments.specs()) if segments is not None
                    else ([perturb_spec] if perturb_spec is not None else []))
    if member_outer:
        # 分段时每一段都有自己的成员集（sigma / 种子不同 -> 指纹不同 -> 各自一个目录），
        # 必须在开跑前全部造好：段边界是在推理中途到的，那里既不能等生成、也不该写盘。
        if claim_mode:
            # 认领模式下 worker 事先不知道自己会摊到哪些成员，所以**先把自己那一整片
            # 造出来**（与静态分片同一个口径：按 LOCAL_RANK 均分），再开始认领。
            # 少了这一步就会互等：每个 worker 都在等一个「只有别人认领到才会被生成」的
            # 成员，谁也不敢先动手。跑完这一片之后，任务里要用的成员最多只需等别的
            # worker 造完它自己那片。
            shard = list(range(local_rank, members, world_size))
            for spec_k in member_specs:
                _prepare_member_set(args.model, spec_k, members, shard,
                                    local_rank, world_size)
        else:
            for spec_k in member_specs:
                _prepare_member_set(args.model, spec_k, members, member_indices,
                                    local_rank, world_size)
    log.info("加载模型：class=%s backend=%s device=%d CUDA_VISIBLE_DEVICES=%s path=%s",
             model_name, model.backend, device_id,
             os.environ.get("CUDA_VISIBLE_DEVICES"), args.model)
    if member_outer:
        # m1a：每个成员一套权重，真正的 load 在下面成员循环里逐个做。
        # resident_members 决定同时驻留几份 session：1 = 一换一放（小显存口径，与改造前
        # 逐位相同的资源行为）；N > 1 = 池子留 N 份，换回来不再重建、不再重新预热 cuDNN。
        pool_desc = ("1 份（一换一放）" if resident_members == 1
                     else ("不限（全常驻）" if resident_members < 0
                           else "%d 份" % resident_members))
        if claim_mode:
            log.info("M1a：成员由任务决定（每个任务的成员块见任务表），逐成员加载；"
                     "常驻 session 池 = %s；同一个成员的连续任务会沿用已装载的权重",
                     pool_desc)
        else:
            log.info("M1a：本 rank 负责 %d 个成员，逐成员加载；常驻 session 池 = %s",
                     len(member_indices), pool_desc)
    else:
        t_load_model = perf_counter()
        model.load(args.model)
        log.info("模型加载完成：%s，耗时 %s",
                 model.describe(), _fmt_dur(perf_counter() - t_load_model))

    # 输出变量选择：--vars 指定则用，否则保存全部。
    requested = [v for v in args.vars.split(",") if v.strip()] if args.vars else None
    channels = list(cls.output_channels)
    save_indices = _resolve_output_indices(channels, requested)
    save_names = [channels[ci] for ci in save_indices]
    names_desc = ", ".join(save_names)
    log.info("输出变量：%s", names_desc)

    lat, lon = grid_coords(cls.grid)
    writer = _AsyncWriter(args.out, atomic=sched.atomic_output)

    if claim_mode:
        return _run_claimed(
            model, processors, dataset_source, writer, args, sched, plan, store,
            beater, init_times, save_names, lat, lon, interval, member_outer,
            segments, member_specs, perturb_spec, local_rank, world_size)

    if member_outer:
        return _run_member_outer(
            model, processors, dataset_source, writer, args, init_times,
            member_indices, save_names, lat, lon, interval, local_rank, world_size,
            segments=segments)

    if dataset_source is not None:
        total_t0 = perf_counter()
        _run_plain_block(
            model, processors, dataset_source, writer, args, member_indices,
            save_names, lat, lon, interval, local_rank, world_size, members,
            total_inits=len(init_times),
            num_workers=dataloader_config.get("num_workers", 0))
        errors = writer.close()
        if errors:
            log.error("写盘阶段发生 %d 个错误，任务失败", len(errors))
            return 1
        log.info(
            "全部完成：起报=%d steps=%d batch_size=%s output=%s 总耗时=%.1fs",
            len(init_times),
            args.steps,
            getattr(dataset_source.dataloader, "batch_size", None),
            args.out,
            perf_counter() - total_t0,
        )
        return 0

def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--worker":
        return _worker_main(args[1:])
    return _config_main(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        log.exception("推理任务发生未处理异常")
        raise
