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
    init = pd.Timestamp(init)
    if init.hour == 0 and init.minute == 0:
        return init.strftime("%Y%m%d")
    return init.strftime("%Y%m%d%H")


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


def _build_infer_argv(cfg, overrides, *, scope=None, intra_op_threads=0):
    """将运行范围覆盖转为 worker 参数；组件由 worker 重新加载 config 获得。"""
    scope = scope or _run_scope(cfg, overrides)
    argv = [
        "--config-source", cfg._source_path,
        "--model", scope["model_abs"],
        "--gpu-mem", str(cfg.gpu_mem),
        "--log-level", str(_pick(overrides, "log_level", cfg.log_level)),
        "--intra-op-threads", str(int(intra_op_threads or 0)),
        "--cudnn-algo-search", str(cfg.cudnn_algo_search or ""),
        # 常驻 session 池的额度（resident_members）。它和 intra_op_threads 同类：属于
        # 「这份配方怎么跑」，所以只有 config 能配，命令行不开放覆盖。
        "--resident-members", str(int(cfg.resident_members or 1)),
    ]
    interval = scope["interval"]
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
    members = scope["members"]
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
    scope = _run_scope(cfg, overrides)
    gpus = _pick(overrides, "gpus", cfg.gpus)
    cuda_devices = _pick(
        overrides, "cuda_devices", cfg.cuda_devices).split(",")
    output_dir = scope["output_dir"]
    argv = _build_infer_argv(
        cfg, overrides, scope=scope,
        intra_op_threads=cfg.intra_op_threads)

    env = dict(os.environ)

    command = [
        sys.executable,
        "-u",
        "-m",
        "xmetai_inference",
        "--worker",
    ] + argv
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


def _graph_member_split(members, graphs):
    """把 members 个成员摊到 graphs 份扰动图上：均分 + 余数给前面的图。

    集合模型 + 权重扰动（config 的 perturb_graphs）用：成员 m = 第 g 份冻结权重 x
    该图 session 内的第 j 次前向。返回 [(graph_id, [全局成员号...]), ...]，长度 =
    graphs、每份非空。例：members=51, graphs=5 -> 11 + 10 + 10 + 10 + 10。
    """
    if graphs < 1:
        raise ValueError(f"graphs 必须 >= 1，拿到 {graphs}")
    if members < graphs:
        raise ValueError(f"members({members}) < graphs({graphs})：会有图分不到成员")
    base, rem = divmod(members, graphs)
    out, start = [], 0
    for graph_id in range(graphs):
        size = base + (1 if graph_id < rem else 0)
        out.append((graph_id, list(range(start, start + size))))
        start += size
    return out


def _prepare_member_set(src, spec, count, member_indices, local_rank, world_size):
    """m1a：生成本 rank 那一份成员，然后等齐本 rank 要跑的那些成员。

    分片（``range(rank, count, world_size)``）必须互不相交：`perturbed.mNNN.data` 是直接以
    "wb" 打开的，两个 rank 同时造同一个成员会写出交错的字节（不报错，只是权重错）。
    cli 对集合强制「members >= 卡数时按成员切」，正常情况下本 rank 要跑的就是本 rank
    造的那一份，不会有等待；只有成员比卡还少时才会等别的 rank 把剩下的造完。
    """
    # 惰性导入：backends.base 顶层 import torch，而 --help / 配置清单这条路不该被拖上 torch。
    from xmetai_inference.backends.base import _fmt_dur
    from xmetai_inference.weights.spw import build_member_set, member_exists

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


def _run_member_outer(model, processors, dataset_source, writer, args, init_times,
                      member_indices, save_names, lat, lon, interval, local_rank,
                      world_size):
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
        save_names, lat, lon, interval, local_rank, world_size)
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
                      world_size, rollout_batch=1):
    """成员外层的一段编舞 = 上面那个「成员循环」本身（不含收尾的 close 与总结）。

    逐成员跑完 ``member_indices``（每个成员一份冻结权重，进来前先 load），逐帧写盘并
    flush。调用方负责开跑前的准备与跑完的收尾。
    """
    from xmetai_inference.backends.base import _fmt_dur
    from xmetai_inference.runner import Rollout, Trajectory

    release = getattr(model, "release", None)
    for order, member_id in enumerate(member_indices, 1):
        model.member_index = member_id
        t_member = perf_counter()
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
                batch_size=rollout_batch,
            )
            writer.flush()
            completed += len(batch_inits)
            log.info("  member_%03d 起报进度：%d/%d，本批耗时 %s",
                     member_id, completed, len(init_times),
                     _fmt_dur(perf_counter() - t_fetch))
            del state
            gc.collect()
        # 最后一个成员**不放手**：跑完就退出去了，这一放只是让池子里那份 session 提前
        # 销毁，对显存没有意义（额度 1 时池子里本来就只剩它）。
        if release is not None and order < len(member_indices):
            release()
            gc.collect()
        log.info("member_%03d 完成，总耗时 %s",
                 member_id, _fmt_dur(perf_counter() - t_member))


def _run_graph_outer(model, processors, dataset_source, writer, args, init_times,
                     graph_members, save_names, lat, lon, interval, local_rank,
                     world_size, num_workers=0, rollout_batch=1):
    """集合模型 + 权重扰动（G 图 x J 成员）的编舞：扰动图在外层，图内成员在内层。

    与 m1a 的成员外层（每成员一份权重、一次一条轨迹）不同：这里的成员来自集合图
    **图内随机算子**（同一 session 里的一次前向 = 一个成员），权重扰动只提供 G 条
    独立轴（每图一份冻结权重）。每份扰动图：load -> 复用 _run_plain_block 的
    members>1 分支（该图的 J 个成员 = 一次 Rollout 的 J 条轨迹，输出落
    member_<全局号>）-> release。多卡按图切（每个 rank 领 range(rank, G, 卡数)），
    所以各图的成员永远在同一进程的同一 session 里，不跨进程。

    model.member_index 在此路径下是**图号**（m1a 的 spw.member 自检用的就是它），
    与全局成员号无关 —— 全局成员号只决定输出目录。
    """
    from xmetai_inference.backends.base import _fmt_dur

    total_t0 = perf_counter()
    release = getattr(model, "release", None)
    for order, (graph_id, member_ids) in enumerate(graph_members, 1):
        model.member_index = graph_id
        t_graph = perf_counter()
        model.load(args.model)
        log.info("[%d/%d] 扰动图 m%03d 权重就绪（服务成员 %03d..%03d 共 %d 个），"
                 "加载耗时 %s",
                 order, len(graph_members), graph_id,
                 member_ids[0], member_ids[-1], len(member_ids),
                 _fmt_dur(perf_counter() - t_graph))
        _run_plain_block(
            model, processors, dataset_source, writer, args, member_ids,
            save_names, lat, lon, interval, local_rank, world_size,
            members=len(member_ids), total_inits=len(init_times),
            num_workers=num_workers, rollout_batch=rollout_batch)
        # 最后一份图不放手：跑完就退出，放了只会让池子提前销毁（与 m1a 同款理由）。
        if release is not None and order < len(graph_members):
            release()
            gc.collect()
        log.info("扰动图 m%03d 完成（%d 个成员），总耗时 %s",
                 graph_id, len(member_ids), _fmt_dur(perf_counter() - t_graph))
    errors = writer.close()
    if errors:
        log.error("写盘阶段发生 %d 个错误，任务失败", len(errors))
        return 1
    total_members = sum(len(ids) for _, ids in graph_members)
    log.info("全部完成：扰动图=%d 成员=%d steps=%d output=%s 总耗时=%.1fs%s",
             len(graph_members), total_members, args.steps, args.out,
             perf_counter() - total_t0, _pool_summary(model))
    return 0


def _run_plain_block(model, processors, dataset_source, writer, args, member_indices,
                     save_names, lat, lon, interval, local_rank, world_size, members,
                     total_inits, num_workers=0, rollout_batch=1):
    """确定性 / 同 session 集合的一段编舞：数据集在内层遍历，逐 batch 落盘并 flush。

    与 m1a 的成员外层相反：这里的「成员」是同一次 Rollout 里的多条轨迹（m3 的图内随机
    算子、或 latent_forcing 的潜空间强迫），一份 session 服务全部成员。members==1 时
    每个起报一条轨迹、输出落 ``{起报}/{步号}.nc``；members>1 时每个成员一条轨迹、输出落
    ``{起报}/member_{成员3位}/{步号}.nc``。返回跑完的起报条数。
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
                batch_size=rollout_batch,
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
                    batch_size=rollout_batch,
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


def _worker_main(argv=None):
    from xmetai_inference.backends.base import _fmt_dur, normalize_budget
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
    p.add_argument("--intra-op-threads", type=int, default=0,
                   help="ORT intra_op 线程数（0 = 不设置，用 ORT 默认）")
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
        from xmetai_inference.weights.spw import WeightPerturbSpec

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
    # 集合模型 + 权重扰动（perturb_graphs）：成员 = 扰动图 x 图内随机，走图外层编舞
    # （_run_graph_outer）。只对图内随机的集合模型类有意义 —— 确定性模型的成员
    # 只能来自权重本身（m1a 的成员外层），配了这里直接拦下。
    n_graphs = int(getattr(worker_cfg, "perturb_graphs", 0) or 0)
    graph_outer = perturb_spec is not None and n_graphs > 0
    if graph_outer:
        if getattr(cls, "forecast_type", "deterministic") != "ensemble":
            raise SystemExit(
                f"perturb_graphs={n_graphs} 只适用于集合模型类（图内随机算子出成员），"
                f"但 {cls.__name__} 声明 forecast_type="
                f"{getattr(cls, 'forecast_type', 'deterministic')!r}。"
                "确定性模型的权重扰动走 m1a（去掉 perturb_graphs 即可）。")
        if n_graphs > members:
            raise SystemExit(
                f"perturb_graphs({n_graphs}) > members({members})：会有图分不到成员。"
                "要么调小 perturb_graphs，要么调大 members。")
        if members // n_graphs < 2:
            raise SystemExit(
                f"perturb_graphs={n_graphs} / members={members}：每图分不到 2 个成员。"
                "图外层模式复用 _run_plain_block 的 members>1 分支（成员 = 一次 Rollout "
                "的多条轨迹），单成员图会落进确定性分支、丢掉 member_NNN/ 子目录"
                "——要单成员就别用这个模式（走 m1a）。")
    if forecast_type == "deterministic" and members > 1:
        log.warning("模型类 %s 声明确定性，但 members=%d；确定性模型应跑 1 个成员",
                    cls.__name__, members)
    elif forecast_type == "ensemble" and members <= 1:
        log.warning("模型类 %s 声明集合，但 members=%d；集合成员数应大于 1",
                    cls.__name__, members)
    init_times = _init_times(args, interval)

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
    graph_members = None
    if graph_outer:
        # 图外层：按**图**切（每个 rank 领 range(rank, G, 卡数) 那几份图、跑全部起报）。
        # 不走按成员切 —— 一份图的 J 个成员必须在同一进程的同一 session 里连跑，
        # 拆到不同进程会把「图内第 j 次前向」这个成员身份打散（fgvp_ens 头部的
        # 待验怀疑：不同 session 的随机序列逐位相同 -> 拆散后成员会重复）。
        graph_members = [entry for entry in _graph_member_split(members, n_graphs)
                         if entry[0] % world_size == local_rank]
        if not graph_members:
            log.info("本 rank 没有分到扰动图（图数 %d、卡数 %d），退出",
                     n_graphs, world_size)
            return 0
        member_indices = list(range(members))
        member_start, member_stride = 0, 1
    else:
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
    # 图外层（集合模型 + perturb_graphs）不在此列：它的外层是扰动图不是成员，走
    # _run_graph_outer，复用 _run_plain_block 的「一 session x 多轨迹」。
    member_outer = perturb_spec is not None and not graph_outer

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
    if member_outer:
        _prepare_member_set(args.model, perturb_spec, members, member_indices,
                            local_rank, world_size)
    elif graph_outer:
        # 图外层：成员集只有 n_graphs 份（每图一份冻结权重），「成员号」= 图号。
        # 造完本 rank 分到的那几份，缺的等其它 rank（与 m1a 同一套分片协议）。
        _prepare_member_set(args.model, perturb_spec, n_graphs,
                            [g for g, _ in graph_members], local_rank, world_size)
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
        log.info("M1a：本 rank 负责 %d 个成员，逐成员加载；常驻 session 池 = %s",
                 len(member_indices), pool_desc)
    elif graph_outer:
        log.info("图外层（集合+SPW）：本 rank 负责 %d 份扰动图（%s），逐图加载；"
                 "每图 %s 个成员由图内随机算子出",
                 len(graph_members), [g for g, _ in graph_members],
                 "/".join(str(len(ids)) for _, ids in graph_members))
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
    writer = _AsyncWriter(args.out, atomic=True)

    if graph_outer:
        return _run_graph_outer(
            model, processors, dataset_source, writer, args, init_times,
            graph_members, save_names, lat, lon, interval, local_rank, world_size,
            num_workers=dataloader_config.get("num_workers", 0),
            rollout_batch=worker_cfg.rollout_batch)

    if member_outer:
        return _run_member_outer(
            model, processors, dataset_source, writer, args, init_times,
            member_indices, save_names, lat, lon, interval, local_rank, world_size)

    if dataset_source is not None:
        total_t0 = perf_counter()
        _run_plain_block(
            model, processors, dataset_source, writer, args, member_indices,
            save_names, lat, lon, interval, local_rank, world_size, members,
            total_inits=len(init_times),
            num_workers=dataloader_config.get("num_workers", 0),
            rollout_batch=worker_cfg.rollout_batch)
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
