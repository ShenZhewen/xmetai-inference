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
每个 .nc 存单个 data 变量，五维 (time=1, lead_time=1, channel, lat, lon)：
channel 为小写变量名维度，time=起报时刻、lead_time=预报时效（小时）。
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
    return init.strftime("%Y%m%d" if init.hour == 0 and init.minute == 0 else "%Y%m%d%H")


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

    def __init__(self, save_dir):
        self.save_dir = save_dir
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
                ds.to_netcdf(path, engine=self.engine, format=self.netcdf_format)
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


def _build_infer_argv(cfg, overrides):
    """将运行范围覆盖转为 worker 参数；组件由 worker 重新加载 config 获得。"""
    model_path = _pick(overrides, "model_path", cfg.model_path)
    argv = [
        "--config-source", cfg._source_path,
        "--model", cfg.model_path_abs(model_path),
    ]
    argv += ["--gpu-mem", str(cfg.gpu_mem)]
    argv += ["--log-level", str(_pick(overrides, "log_level", cfg.log_level))]
    interval = getattr(cfg, "hour_interval", 0)
    if interval:
        argv += ["--interval", str(interval)]

    inits = parse_times(_pick(overrides, "times", cfg.times))
    if not inits:
        raise SystemExit(
            f"配置 {cfg.name}：推理必须指定 times（如 \"2025010600..2025020500:24\"）")
    if len(inits) == 1:
        argv += ["--time", inits[0].strftime("%Y%m%d%H")]
    else:
        argv += ["--inits", ",".join(value.strftime("%Y%m%d%H") for value in inits)]

    steps = _pick(overrides, "steps", cfg.steps)
    if steps:
        argv += ["--steps", str(steps)]
    members = _pick(overrides, "members", cfg.members)
    if members:
        argv += ["--members", str(members)]
    variables = _pick(overrides, "vars", cfg.vars)
    if variables:
        argv += ["--vars", variables]
    output_dir = _pick(overrides, "out", cfg.output_dir)
    if output_dir:
        argv += ["--out", output_dir]
    batch_size = overrides.get("batch_size")
    if batch_size is not None:
        argv += ["--batch-size", str(batch_size)]
    num_workers = overrides.get("num_workers")
    if num_workers is not None:
        argv += ["--num-workers", str(num_workers)]
    return argv


def _run_infer(cfg, overrides):
    """使用同一个包模块启动一个或多个隔离 worker。"""
    argv = _build_infer_argv(cfg, overrides)
    gpus = _pick(overrides, "gpus", cfg.gpus)
    cuda_devices = _pick(
        overrides, "cuda_devices", cfg.cuda_devices).split(",")
    output_dir = _pick(overrides, "out", cfg.output_dir)

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
        help="注册模型配方名，如 fuxi21、fuxi_ens、fgvp",
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


def _worker_main(argv=None):
    from xmetai_inference.backends.base import _fmt_dur
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
    members = args.members if args.members is not None else getattr(cls, "members", 1)
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
    # 选单卡最大任务数更小的那一种（更均衡）。集合成员多、起报少时按成员切更均衡
    # （如 7 起报 × 51 成员：按起报切最忙卡 102 任务 vs 按成员切 91）；确定性
    # 1 成员、起报多时按起报切更均衡。
    n_init = len(init_times)
    per_rank_init = (n_init + world_size - 1) // world_size
    per_rank_member = (members + world_size - 1) // world_size
    cost_by_init = per_rank_init * members
    cost_by_member = n_init * per_rank_member
    if cost_by_init <= cost_by_member:
        base, rem = divmod(n_init, world_size)
        start_i = local_rank * base + min(local_rank, rem)
        span = base + (1 if local_rank < rem else 0)
        init_times = init_times[start_i:start_i + span]
        member_indices = list(range(members))
        member_start, member_stride = 0, 1
    else:
        member_indices = list(range(local_rank, members, world_size))
        member_start, member_stride = local_rank, world_size

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

    input_specs, recurrent_specs, output_specs = worker_cfg.processor_specs()
    processors = build_pipeline(
        input_specs,
        cls,
        data_source,
        model_path=args.model,
        recurrent_specs=recurrent_specs,
        output_specs=output_specs,
    )
    model = create_model(
        model_spec,
        device_id=device_id,
        gpu_mem_fraction=args.gpu_mem,
        ops_library=worker_cfg.ops_library_abs(),
        gpu_state=worker_cfg.gpu_state,
    )
    model.configure_processing(processors, data_source)

    if args.out is None:
        model.prepare_dataset_batch(next(iter(dataset_source)), verbose=True)
        log.info("未指定 --out，仅完成首个起报时间的输入构建检查")
        return 0

    os.makedirs(args.out, exist_ok=True)
    log.info("加载模型：class=%s backend=%s device=%d CUDA_VISIBLE_DEVICES=%s path=%s",
             model_name, model.backend, device_id,
             os.environ.get("CUDA_VISIBLE_DEVICES"), args.model)
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
    writer = _AsyncWriter(args.out)

    if dataset_source is not None:
        total_t0 = perf_counter()
        completed = 0
        for batch in dataset_source:
            state, batch_inits = model.prepare_dataset_batch(batch)

            # 确定性（members==1）：每个起报一条轨迹，一次 Rollout 跑完整个 batch。
            # 集合（members>1）：每个起报单独一次 Rollout，成员即轨迹。
            # 两者用的是同一个循环，区别只在「轨迹怎么构造」和「输出落到哪」。
            if members == 1:
                def on_batch_step(step, step_state, batch_inits=batch_inits):
                    step_idx = step + 1
                    lead_hour = step_idx * interval
                    for batch_index, init in enumerate(batch_inits):
                        init_dir = _output_init_dir(init)
                        ds = model.to_dataset(
                            step_state[batch_index],
                            save_names=save_names,
                            lat=lat,
                            lon=lon,
                            init_time=init,
                            lead_hour=lead_hour,
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
                        lead_hour = step_idx * interval
                        for local_index, member_id in enumerate(member_indices):
                            member_ds = model.to_dataset(
                                step_state[local_index],
                                save_names=save_names,
                                lat=lat,
                                lon=lon,
                                init_time=init,
                                lead_hour=lead_hour,
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
                        progress=False,
                        progress_label=f"[rank {local_rank}/{world_size}] {init:%m%d%H}",
                    )

            writer.flush()
            completed += len(batch_inits)
            log.info(
                "Dataset batch 完成：%d/%d，batch=%d",
                completed,
                len(init_times),
                len(batch_inits),
            )
            del state
            gc.collect()

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
