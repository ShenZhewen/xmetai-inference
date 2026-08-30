#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第②③步：自回归推理 + NetCDF 输出（合并版）。

推理后端（BaseInferModel）拆到 backends/ 包、各模型子类拆到 models/ 包，数据源
拆到 loaders/ 包；本文件只保留「输出选择/单位换算/异步写盘/多卡切分」和主流程
编排。加新模型、新数据源只需在对应注册表登记，无需改动本文件。

用法（单次起报）：
    python infer.py --model fuxi.onnx --time 2024010200 --steps 10 --out ./output

用法（一段时期，逐个起报）：
    python infer.py --model fuxi.onnx --start 2024010200 --end 2024010500 \
        --freq 6 --steps 10 --out ./output

输出目录（集合 members>1）：{out}/{起报日 yyyymmdd}/member_{成员3位}/{预测步序号3位}.nc
输出目录（确定性 members=1）：{out}/{起报日 yyyymmdd}/{预测步序号3位}.nc

不写 --out 时只做一次输入构建校验（不跑模型）。

多卡（数据并行，按成员拆分）：每张卡一个进程，用环境变量选卡、拆成员：
    LOCAL_RANK=0 WORLD_SIZE=4 python infer.py ... --members 21
    ...
或者干脆用 CUDA_VISIBLE_DEVICES 隔离 + LOCAL_RANK 标 rank。ONNX Runtime 单个
session 只用一张卡（device_id），没有跨卡并行；多卡的加速来自把集合成员 /
起报次数分到不同卡上，不是单次 run 变快。

推理是 step 外层、member 内层：每算完一个 step 的全体成员就异步丢给后台线程
落盘，GPU 不等磁盘写；netCDF4/HDF5 非线程安全，所以只用一个 writer 线程串行写。
"""
import argparse
import gc
import importlib
import json
import os
import queue
import threading
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from build_input import build_input, load_spec, grid_coords
from loaders import create_loader
from backends import BACKEND_REGISTRY, create_backend
from models import MODEL_REGISTRY, create_model


# 推理后端（BaseInferModel / Onnx / Pt2 / Ckpt）在 backends/，各模型子类在 models/，见顶部 import。


# ---------------------------------------------------------------------------
# 输出（后端无关）
# ---------------------------------------------------------------------------
def _resolve_output_indices(spec, requested):
    """把要保存的变量名（z500/u200/v200/msl/tp）映射成通道下标。

    requested=None 时保存全部通道；名字大小写不敏感，不存在的跳过并告警。
    """
    channels = spec["_channels"]
    if requested is None:
        return list(range(len(channels)))
    name2idx = {str(c).lower(): i for i, c in enumerate(channels)}
    idxs = []
    for name in requested:
        key = str(name).strip().lower()
        if key in name2idx:
            idxs.append(name2idx[key])
        else:
            print(f"[警告] 输出变量 {name!r} 不在模型通道里，已跳过")
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
    print("[警告] netcdf4/h5netcdf 未安装，退回 scipy 写 NETCDF3_64BIT（大数组会很慢）")
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
                if raw is not None:
                    step_buf, s, init = raw
                    npy = os.path.join(self.save_dir, f"{init:%Y%m%d%H}_raw_step_{s:03d}.npy")
                    np.save(npy, step_buf.astype(np.float32))
                    self.errors.append(f"step {s} 已兜底存 {npy}")
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
def _print_input_summary(x, spec):
    print(f"输入 shape: {x.shape}  dtype: {x.dtype}")
    finite = np.isfinite(x)
    print(f"数值范围: min={x[finite].min():.6g}  max={x[finite].max():.6g}  "
          f"NaN={int((~finite).sum())}")
    for name in ("t2m", "z500", "q850", "u10m"):
        if name not in spec["_channels"]:
            continue
        ci = spec["_channels"].index(name)
        ch = x[0, :, ci]
        for ti in range(x.shape[1]):
            f = ch[ti][np.isfinite(ch[ti])]
            if f.size:
                print(f"  {name} 帧{ti}: min={f.min():.6g} max={f.max():.6g}")


def _init_times(args, interval):
    """把 --start/--end/--freq（或 --time）展开成起报时间列表。"""
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
    raise SystemExit("必须提供 --time 或 --start")


def _create_loader(args, spec):
    """按 --loader 选择输入数据源。各 loader 只实现 load(time) -> xr.Dataset，
    build_input 对其余逻辑（单位/层级/网格适配）一视同仁。"""
    return create_loader(args.loader, spec=spec, path=args.zarr)


def main():
    p = argparse.ArgumentParser(description="气象模型自回归推理（后端可插拔）")
    p.add_argument("--model", required=True, help="模型文件路径（.onnx/.pt2/.ckpt；后端由 spec 的 model.class 决定）")
    p.add_argument("--backend", default=None,
                   help="逃生舱：覆盖 spec 的 model.class，可传模型名(fuxi_ens_onnx/fuxi21_pt2/"
                        "aifs11_ckpt)或引擎名(onnx/pt2/ckpt，裸后端无模型钩子)")
    p.add_argument("--time", default=None, help="单次起报时间 YYYYMMDDHH（与 --start/--end 二选一）")
    p.add_argument("--start", default=None, help="起始起报时间 YYYYMMDDHH")
    p.add_argument("--end", default=None, help="结束起报时间 YYYYMMDDHH（含，默认=--start）")
    p.add_argument("--freq", type=int, default=None, help="相邻起报间隔小时（默认=步长 interval）")
    p.add_argument("--spec", default="fuxi_ens.json", help="模型 spec JSON 路径")
    p.add_argument("--loader", default="era", choices=["era", "zarr", "era5_store"],
                   help="输入数据源：era=ERA 逐变量文件，zarr=打包好的 zarr store，"
                        "era5_store=新 ERA5 基础库（多组 zarr，根目录由 --zarr 指定）")
    p.add_argument("--zarr", default=None,
                   help="可选：覆盖数据源默认地址（--loader zarr 必传单 store 路径；"
                        "--loader era/era5_store 不传则用各自 loader 内置默认地址）")
    p.add_argument("--steps", type=int, default=10, help="预报步数")
    p.add_argument("--members", type=int, default=None,
                   help="集合成员总数（缺省读 spec 的 model.members；确定性=1）")
    p.add_argument("--history", type=int, default=None, help="输入历史帧数（默认用 spec）")
    p.add_argument("--interval", type=int, default=None, help="时间步长小时（默认用 spec）")
    p.add_argument("--device", type=int, default=None, help="GPU 设备号（默认 0；多卡时由 CUDA_VISIBLE_DEVICES 隔离）")
    p.add_argument("--world-size", type=int, default=None, help="卡数（默认读 WORLD_SIZE）")
    p.add_argument("--gpu-mem", type=float, default=0.7, help="显存占用比例")
    p.add_argument("--out", default=None, help="输出目录；不写则只做输入校验")
    p.add_argument("--vars", default=None,
                   help="要保存的输出变量，逗号分隔（如 z500,u200,v200,msl,tp）；不传则保存全部通道")
    p.add_argument("--verbose", action="store_true", help="打印详细日志（每步耗时、输入通道统计）")
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = args.world_size if args.world_size is not None \
        else int(os.environ.get("WORLD_SIZE", "1"))
    # 多卡时 run_fuxi_ens.sh / run_fuxi_pt2.sh 已用 CUDA_VISIBLE_DEVICES 把每个进程隔离到单卡，device 恒为 0；
    # 单进程想指定别的卡仍可用 --device 覆盖。
    device_id = args.device if args.device is not None else 0

    # 先读原始 JSON 确定 model.class（--backend 可覆盖），据此判断状态表示：
    # ckpt 后端走 field 字典（AIFS，N320 节点），onnx/pt2 走通道张量。
    with open(args.spec, encoding="utf-8") as _fh:
        raw_spec = json.load(_fh)
    sel = args.backend or raw_spec.get("model", {}).get("class")
    if sel is None:
        raise SystemExit("spec 的 model 块没写 \"class\"，也没给 --backend，无法确定用哪个模型类")
    cls = MODEL_REGISTRY.get(sel) or BACKEND_REGISTRY.get(sel)
    if cls is None:
        raise SystemExit(f"未知模型/后端 {sel!r}（模型: {', '.join(MODEL_REGISTRY)}；"
                         f"后端: {', '.join(BACKEND_REGISTRY)}）")
    is_field = getattr(cls, "backend", "") == "ckpt"

    # 按表示加载 spec：FuXi 走 build_input.load_spec（补 _channels/_parse/单位等），
    # AIFS 走 build_input_aifs 的 plain json.load。
    if is_field:
        from build_input_aifs import load_spec as load_spec_aifs, build_aifs_fields
        spec = load_spec_aifs(args.spec)
    else:
        spec = load_spec(args.spec)

    interval = args.interval or spec["model"].get("hour_interval", 6)
    history = args.history or spec["model"].get("history_steps", 2)
    # 集合/确定性显式化：members 缺省从 spec 读；类型与成员数不一致时告警（不报错）
    forecast_type = spec["model"].get("forecast_type", "deterministic")
    members = args.members if args.members is not None else spec["model"].get("members", 1)
    if forecast_type == "deterministic" and members > 1:
        print(f"[警告] spec 声明确定性模型（{spec.get('name')}），但 --members={members}>1；"
              f"确定性模型应跑 1 个成员，请检查 --members / spec")
    elif forecast_type == "ensemble" and members <= 1:
        print(f"[警告] spec 声明集合模型（{spec.get('name')}），但 --members={members}；"
              f"集合成员数应 >1，请检查 --members / spec")
    # 一个进程内复用同一个 loader（只给 FuXi 用；AIFS 的 build_aifs_fields 内部自建 loader）
    loader = None if is_field else _create_loader(args, spec)

    init_times = _init_times(args, interval)

    # 多卡分工：有多个起报时间就把所有起报时间连续切块分给各卡（互不重叠，
    # 如 rank0 报 1-4 月、rank1 报 4-8 月），每卡跑全部成员；只有单个起报则按成员拆。
    if len(init_times) > 1:
        n = len(init_times)
        base, rem = divmod(n, world_size)
        start_i = local_rank * base + min(local_rank, rem)
        span = base + (1 if local_rank < rem else 0)
        init_times = init_times[start_i:start_i + span]
        member_indices = list(range(members))
        member_start, member_stride = 0, 1
    else:
        member_indices = list(range(local_rank, members, world_size))
        member_start, member_stride = local_rank, world_size

    if not init_times:
        print(f"[rank {local_rank}] 没有分配到起报时间，退出。")
        return 0
    if not member_indices:
        print(f"[rank {local_rank}] 没有分配到成员，退出。")
        return 0
    print(f"[rank {local_rank}/{world_size}] 起报 {len(init_times)} 个："
          f"{init_times[0]:%Y%m%d%H} .. {init_times[-1]:%Y%m%d%H}，成员 {len(member_indices)}")

    if args.out is None:
        init = init_times[0]
        if is_field:
            build_aifs_fields(init.strftime("%Y%m%d%H"), spec=spec,
                              history_steps=history, hour_interval=interval, do_interp=True)
        else:
            x = build_input(init.strftime("%Y%m%d%H"), spec=spec,
                            history_steps=history, hour_interval=interval, loader=loader)
            _print_input_summary(x, spec)
        print("未指定 --out，仅做输入构建校验（只校验首个起报时间）。")
        return 0

    os.makedirs(args.out, exist_ok=True)
    # 模型选择：sel 已在上面从原始 spec 解析并校验过（∈ MODEL_REGISTRY ∪ BACKEND_REGISTRY）
    if sel in MODEL_REGISTRY:
        model = create_model(sel, device_id=device_id, gpu_mem_fraction=args.gpu_mem)
    else:
        model = create_backend(sel, device_id=device_id, gpu_mem_fraction=args.gpu_mem)
        print(f"[警告] 使用裸后端 {sel!r}（无模型钩子/归一化）；正规用法是 spec 写 model.class")
    print(f"[rank {local_rank}/{world_size}] 加载模型 (backend={model.backend}, device={device_id}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}) ...")
    model.load(args.model)
    print(model.describe())

    # 输出变量选择：--vars 指定则用，否则保存全部（FuXi 走通道解析，AIFS 走 field 名）
    requested = [v for v in args.vars.split(",") if v.strip()] if args.vars else None
    if is_field:
        save_names = requested                       # AIFS：field 名，None=全部字段
        names_desc = ", ".join(save_names) if save_names else "(全部字段)"
    else:
        save_indices = _resolve_output_indices(spec, requested)
        save_names = [spec["_channels"][ci] for ci in save_indices]
        names_desc = ", ".join(save_names)
    print(f"[rank {local_rank}/{world_size}] 输出变量: {names_desc}")

    # FuXi 输出网格坐标来自 spec；AIFS 的 N320 经纬度由输出 state 自带（to_dataset 里读）
    lat, lon = (None, None) if is_field else grid_coords(spec)
    writer = _AsyncWriter(args.out)

    total_t0 = perf_counter()
    for i, init in enumerate(init_times):
        t0 = perf_counter()
        if is_field:
            state = build_aifs_fields(init.strftime("%Y%m%d%H"), spec=spec,
                                      history_steps=history, hour_interval=interval,
                                      do_interp=True)
        else:
            state = build_input(init.strftime("%Y%m%d%H"), spec=spec,
                                history_steps=history, hour_interval=interval, loader=loader)
            if args.verbose:
                _print_input_summary(state, spec)

        def on_step(s, step_state, init=init):
            step_idx = s + 1                        # 预测步序号，1-based
            if is_field:
                # AIFS：单成员，一步一个 field 字典，直接转 Dataset 落盘（N320 节点）
                ds = model.to_dataset(step_state, spec, save_names=save_names)
                writer.put(f"{init:%Y%m%d}/{step_idx:03d}.nc", ds)
            else:
                multi_member = members > 1
                for m_local, m_id in enumerate(member_indices):
                    ds = model.to_dataset(step_state[m_local], spec,
                                          save_names=save_names, lat=lat, lon=lon)
                    # 确定性（单成员）不套 member_xxx 目录，直接 {起报日}/{step}.nc
                    if multi_member:
                        fname = f"{init:%Y%m%d}/member_{m_id:03d}/{step_idx:03d}.nc"
                    else:
                        fname = f"{init:%Y%m%d}/{step_idx:03d}.nc"
                    writer.put(fname, ds, raw=(step_state, s, init))

        model.run(state, steps=args.steps, members=members,
                      hour_interval=interval, init_time=init,
                      member_start=member_start, member_stride=member_stride,
                      on_step=on_step, log_step=args.verbose,
                      progress=True,
                      progress_label=f"[rank {local_rank}/{world_size}] {init:%m%d%H}")
        # 等本起报的输出全部写完，再读下一个起报的输入——避免写线程与主线程
        # 并发访问 HDF5 触发段错误（netCDF4 非线程安全，见 onnx_infer_dfens.py 注释）
        writer.flush()
        # 释放本次起报的输入（张量或 field 字典）：多卡连续起报若不及时回收会顶爆系统内存
        del state
        gc.collect()
        pct = (i + 1) / len(init_times) * 100
        print(f"[rank {local_rank}/{world_size}] 起报 {i + 1}/{len(init_times)} "
              f"({pct:3.0f}%) 完成，耗时 {perf_counter() - t0:.1f}s")

    errors = writer.close()
    if errors:
        for e in errors:
            print(f"[写入错误] {e}")

    print(f"[rank {local_rank}/{world_size}] 全部完成：{len(init_times)} 个起报 x "
          f"{args.steps} steps x {len(member_indices)} members -> {args.out}，"
          f"总耗时 {perf_counter() - total_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
