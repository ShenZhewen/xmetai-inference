#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理NC -> 每日起报Zarr -> 合并大Zarr
  第一步: 将每个起报日期下的所有NC文件按 member / lead_time 拼接，存为每日Zarr
  第二步: 将每个每日Zarr按 time 维度追加写入一个大Zarr
chunk 全程保持一致: (member=1, time=1, lead_time=1, level=4, lat=721, lon=1440)
"""

import os
import re
import glob
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

# ==================== 配置 ====================
SOURCE_DIR   = "/gpu/zhouchg/FUXI_S2S/seasongsspinfer"
TMP_DIR      = "/gpu/zhouchg/FUXI_S2S/seasongssp/seasontmp"
OUTPUT_PATH  = "/gpu/zhouchg/FUXI_S2S/seasongssp/seasontmp/seasongssp.zarr"
LOG_FILE     = "/gpu/zhouchg/FUXI_S2S/seasongssp/seasontmp/seasongssp.log"

NUM_WORKERS  = 6
VAR_NAME     = "__xarray_dataarray_variable__"
LEAD_LEN     = 180
DTYPE        = np.float16
NC_CHUNKS    = {'lat': 121, 'lon': 240}                  # 读NC时的dask分块
ZARR_CHUNKS  = (1, 1, 180, 4, 121, 240)                  # 写Zarr的存储分块

GPU_MEM_MB   = 256          # 欲占用的显存大小（MiB）


# ==================== 显存占用线程 ====================
def gpu_keepalive(mem_mb=GPU_MEM_MB):
    """
    后台线程：持续执行 GPU 矩阵乘法以保持显存占用，防止集群误判空闲。
    若 GPU 不可用或 PyTorch 未安装，该函数静默返回。
    """
    try:
        import torch
    except ImportError:
        return   # 没有 PyTorch，无法进行 GPU 操作

    if not torch.cuda.is_available():
        return

    device = torch.device('cuda')
    # 计算方阵维度 n，使得两个矩阵（a, b）及结果（c）总占用约 mem_mb MiB
    # 总字节数 ≈ 3 * n^2 * 4（float32）
    n = int((mem_mb * 1024 * 1024 / (3 * 4)) ** 0.5)
    if n < 1:
        n = 1024

    a = torch.randn(n, n, device=device, dtype=torch.float32)
    b = torch.randn(n, n, device=device, dtype=torch.float32)

    print(f"[GPU-keepalive] 启动，占用约 {mem_mb} MiB 显存 (n={n})")
    while True:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        time.sleep(0.1)


# ==================== 第一步：NC -> 每日Zarr ====================
def list_source_dates(source_dir):
    """源目录下所有 YYYYMMDD 子目录"""
    if not os.path.isdir(source_dir):
        return []
    return sorted(d for d in os.listdir(source_dir)
                  if len(d) == 8 and d.isdigit() and os.path.isdir(os.path.join(source_dir, d)))


def list_done_dates(tmp_dir):
    """临时目录下已生成的 YYYYMMDD.zarr"""
    if not os.path.isdir(tmp_dir):
        return set()
    out = set()
    for p in glob.glob(os.path.join(tmp_dir, "*.zarr")):
        name = os.path.basename(p).replace('.zarr', '')
        if len(name) == 8 and name.isdigit():
            out.add(name)
    return out


def nc_to_zarr_one_date(args):
    """单日期: NC按member、lead_time拼接 -> 存为 YYYYMMDD.zarr"""
    date_str, source_dir, output_dir = args
    print(f"[{date_str}] 开始处理")

    files = glob.glob(os.path.join(source_dir, date_str, "**/*.nc"), recursive=True)
    if not files:
        print(f"[{date_str}] 未找到NC文件")
        return None

    # 按lead_time分组（文件名即lead_time编号）
    groups = defaultdict(list)
    for fp in files:
        lead = int(os.path.splitext(os.path.basename(fp))[0])
        groups[lead].append(fp)

    # 每组沿member拼接，再沿lead_time拼接
    lead_ds = []
    for lead, fps in sorted(groups.items()):
        members = [xr.open_dataset(fp, chunks=NC_CHUNKS, cache=False) for fp in sorted(fps)]
        lead_ds.append(xr.concat(members, dim='member'))
        for ds in members:
            ds.close()

    if not lead_ds:
        print(f"[{date_str}] 无有效数据")
        return None

    combined = xr.concat(lead_ds, dim='lead_time')

    os.makedirs(output_dir, exist_ok=True)
    zarr_path = os.path.join(output_dir, f"{date_str}.zarr")
    encoding = {VAR_NAME: {"dtype": DTYPE, "chunks": ZARR_CHUNKS}}
    # ========== 修复：添加 align_chunks=True 以自动对齐 Dask 与 Zarr 分块 ==========
    combined.to_zarr(zarr_path, mode='w', consolidated=True, encoding=encoding, align_chunks=True)

    combined.close()
    for ds in lead_ds:
        ds.close()
    print(f"[{date_str}] 完成 -> {zarr_path}")
    return date_str


def step1_nc_to_zarr():
    """并行: 所有日期 NC -> 每日Zarr（断点续跑）"""
    print("=" * 60)
    print("第一步：NC -> 每日Zarr")
    print("=" * 60)

    source_dates = list_source_dates(SOURCE_DIR)
    done = list_done_dates(TMP_DIR)
    todo = [d for d in source_dates if d not in done]
    print(f"源 {len(source_dates)} 个日期，已处理 {len(done)}，待处理 {len(todo)}")
    if not todo:
        print("无需处理")
        return

    tasks = [(d, SOURCE_DIR, TMP_DIR) for d in todo]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for r in executor.map(nc_to_zarr_one_date, tasks):
            if r:
                print(f"  ✓ {r}")


# ==================== 第二步：每日Zarr -> 大Zarr ====================
def load_merged_log(log_file):
    """读取已合并日期日志"""
    done = set()
    if log_file and os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s:
                    done.add(s)
    return done


def append_date_to_big_zarr(date_str, zarr_path, output_path, first, log_file):
    """把单日期Zarr读取为(M,1,L,C,Y,X)块，追加到大Zarr的time维度；缺失lead用NaN占位"""
    ds = xr.open_zarr(zarr_path, chunks=None, consolidated=False,
                      mask_and_scale=False, use_zarr_fill_value_as_mask=False)
    if VAR_NAME not in ds.data_vars:
        print(f"  ⚠ {date_str} 缺变量 {VAR_NAME}，跳过")
        ds.close()
        return False

    arr = ds[VAR_NAME].values  # (member, time=1, lead_time, level, lat, lon)
    n_member, _, _, n_chan, n_lat, n_lon = arr.shape

    block = np.full((n_member, 1, LEAD_LEN, n_chan, n_lat, n_lon), np.nan, dtype=DTYPE)
    if arr.shape[2] == LEAD_LEN:
        block[:, 0] = arr[:, 0].astype(DTYPE, copy=False)
    else:
        print(f"  ⚠ {date_str} lead_time={arr.shape[2]} != {LEAD_LEN}，NaN占位")

    time_val = ds["time"].values[0]
    da = xr.DataArray(
        block,
        dims=("member", "time", "lead_time", "level", "lat", "lon"),
        coords={
            "member":    ds["member"].values,
            "time":      [time_val],
            "lead_time": np.arange(1, LEAD_LEN + 1),
            "level":     ds["level"].values,
            "lat":       ds["lat"].values,
            "lon":       ds["lon"].values,
        },
        name=VAR_NAME,
    )
    ds.close()

    if first:
        encoding = {VAR_NAME: {"dtype": DTYPE, "chunks": ZARR_CHUNKS}}
        da.to_dataset().to_zarr(output_path, mode="w", encoding=encoding)
    else:
        da.to_dataset().to_zarr(output_path, mode="a", append_dim="time")

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{date_str}\n")
    return True


def step2_merge_to_big_zarr():
    """顺序: 每日Zarr按time追加到大Zarr（断点续跑）"""
    print("=" * 60)
    print("第二步：每日Zarr -> 大Zarr")
    print("=" * 60)

    paths = sorted(glob.glob(os.path.join(TMP_DIR, "*.zarr")))
    if not paths:
        raise RuntimeError(f"目录下无 .zarr 文件: {TMP_DIR}")
    print(f"找到 {len(paths)} 个 zarr 文件")

    pat = re.compile(r"(\d{8})\.zarr$")
    groups = {}
    for p in paths:
        m = pat.match(os.path.basename(p))
        if m:
            groups.setdefault(m.group(1), []).append(p)
    all_dates = sorted(groups.keys())
    print(f"共有 {len(all_dates)} 个起报时间")

    # 断点续跑：日志 + 已存在输出的time
    done = load_merged_log(LOG_FILE)
    first = True
    if os.path.exists(OUTPUT_PATH):
        try:
            ds_exist = xr.open_zarr(OUTPUT_PATH, chunks=None, consolidated=False)
            for t in ds_exist["time"].values:
                done.add(str(np.datetime64(t, "D")).replace("-", ""))
            ds_exist.close()
            first = False
            print(f"已合并 {len(done)} 个日期")
        except Exception as e:
            print(f"  ⚠ 读取已有输出失败，重新覆盖: {e}")
            shutil.rmtree(OUTPUT_PATH, ignore_errors=True)

    for idx, date_str in enumerate(all_dates):
        if date_str in done:
            continue
        print(f"--- [{idx + 1}/{len(all_dates)}] {date_str} ---")
        if append_date_to_big_zarr(date_str, groups[date_str][0], OUTPUT_PATH, first, LOG_FILE):
            first = False

    print(f"✅ 完成：{OUTPUT_PATH}")


# ==================== 主程序 ====================
def main():
    # 启动 GPU 显存占用守护线程，避免集群因无 GPU 活动而杀任务
    keep_thread = threading.Thread(target=gpu_keepalive, daemon=True)
    keep_thread.start()

    #step1_nc_to_zarr()
    step2_merge_to_big_zarr()
    print("=" * 60)
    print("全部完成")
    print("=" * 60)


if __name__ == "__main__":
    main()