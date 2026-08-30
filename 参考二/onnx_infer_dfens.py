"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-28 19:12:10
Description: Only for evaluation with onnx model.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import argparse
import gc
import glob
import logging
import os
import queue
import threading
from time import perf_counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)
__all__ = ["OnnxInferModel"]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

# ---- Custom ops library resolution (referenced from test_infer.py) ----
# Search order: explicit env var -> /gpu/zhaochy/testfdp/V260715 -> /workspace/cma -> project root.
_PLUGIN_DIRS = [
    "/gpu/zhaochy/testfdp/V260715",
    "/workspace/cma",
    _PROJECT_ROOT,
]
_custom_ops_candidates = [
    os.environ.get("CUSTOM_OPS_LIB_PATH", ""),
]
for _d in _PLUGIN_DIRS:
    _custom_ops_candidates.append(os.path.join(_d, "xmetai_onnx_plugins_gcc9_cuda12_ort1.24.4.so"))
    _custom_ops_candidates.append(os.path.join(_d, "xmetai_onnx_plugins.so"))
    _custom_ops_candidates.append(os.path.join(_d, "xmetai_onnx_plugins.cpython-311-x86_64-linux-gnu.so"))
CUSTOM_OPS_LIB_PATH = next((p for p in _custom_ops_candidates if p and os.path.exists(p)), None)
get_library_path = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]
    logger.warning("ONNX Runtime is not available. ONNX inference requires onnxruntime or onnxruntime-gpu.")

if CUSTOM_OPS_LIB_PATH is not None:
    # use precompiled custom ops library by default
    def get_library_path():
        return CUSTOM_OPS_LIB_PATH

    logger.info(f"Using precompiled custom ops library: {CUSTOM_OPS_LIB_PATH}")
else:
    # use mamba_ssm ops only when onnxruntime-extensions is installed.
    try:
        from onnxruntime_extensions import PyOp, get_library_path, onnx_op  # type: ignore[no-redef]

        @onnx_op(
            op_type="SelectiveScanFn",
            inputs=[
                PyOp.dt_float,
                PyOp.dt_float,
                PyOp.dt_float,
                PyOp.dt_float,
                PyOp.dt_float,
                PyOp.dt_float,
                PyOp.dt_float,
                PyOp.dt_float,
            ],
            outputs=[PyOp.dt_float],
            attrs={"delta_softplus": PyOp.dt_int64, "return_last_state": PyOp.dt_int64},
        )
        def selective_scan_onnx(u, delta, A, B, C, D, z, delta_bias, **kwargs):
            # Lazy imports: torch and mamba_ssm are only needed at execution time
            # of this fallback op, not at module import. Keeps the module's
            # top-level deps identical to test_infer.py.
            import torch
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

            delta_softplus = kwargs.get("delta_softplus", 0) == 1
            return_last_state = kwargs.get("return_last_state", 0) == 1
            print(f"u:{u.shape}, delta:{delta.shape}, A:{A.shape}, B:{B.shape}, C:{C.shape}, D:{D.shape}, delta_bias:{delta_bias.shape}")
            print(f"delta_softplus:{delta_softplus}, return_last_state:{return_last_state}")
            # convert all inputs to torch tensors
            u = torch.from_numpy(u).to(device="cuda")
            delta = torch.from_numpy(delta).to(device="cuda")
            A = torch.from_numpy(A).to(device="cuda")
            B = torch.from_numpy(B).to(device="cuda")
            C = torch.from_numpy(C).to(device="cuda")
            D = torch.from_numpy(D).to(device="cuda")
            z = None
            delta_bias = torch.from_numpy(delta_bias).to(device="cuda")
            return selective_scan_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

        logger.info("Using onnxruntime_extensions PyOp for SelectiveScanFn custom op")
    except ImportError as e:
        logger.debug("ONNX custom op plugin is not available: %s", e)


#############################################################################
# Real-time data loading & output helpers (ported from test_infer.py)
#############################################################################
try:
    import cfgrib
except ImportError:  # cfgrib is only required for real-time GRIB ingestion
    cfgrib = None  # type: ignore[assignment]

try:
    from eccodes import (
        codes_grib_new_from_file,
        codes_get_double_array,
        codes_get_long,
        codes_get_values,
        codes_release,
        codes_set_long,
    )
except ImportError:  # eccodes is only required for real-time GRIB ingestion
    codes_grib_new_from_file = None  # type: ignore[assignment]

# Unit scaling and pressure levels (consistent with test_infer.py / CMA-RA-V1.5)
UNIT_SCALE = dict(
    ciwc=1000, clwc=1000, crwc=1000, cswc=1000, q=1000, q2m=1000, tp=1000, gh=9.80665,
    ttr=1 / 3600, ssr=1 / 3600, ssrd=1 / 3600, fdir=1 / 3600,
)
LEVEL_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # hPa


def open_precip_grib(path: str) -> xr.DataArray:
    with open(path, "rb") as f:
        gid = codes_grib_new_from_file(f)
        if gid is None:
            raise ValueError("文件里没有 GRIB message")
        try:
            codes_set_long(gid, "jScansPositively", 0)
            nx = codes_get_long(gid, "Ni")
            ny = codes_get_long(gid, "Nj")
            lats = codes_get_double_array(gid, "latitudes")
            lons = codes_get_double_array(gid, "longitudes")
            values = codes_get_values(gid)
            data = values.reshape(ny, nx)
            da = xr.DataArray(
                data,
                dims=("lat", "lon"),
                coords={
                    "lat": lats.reshape(ny, nx)[:, 0],
                    "lon": lons.reshape(ny, nx)[0, :],
                },
                attrs={"GRIB_shortName": "tp"},
            )
            return da
        finally:
            codes_release(gid)


def bilinear_resize_numpy(data: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError("输入 data 必须是三维 (C, H, W)")

    c, h, w = data.shape
    if h == out_h and w == out_w:
        return data.copy()

    scale_y = h / out_h
    scale_x = w / out_w

    # 目标网格对应的源坐标（align_corners=False）
    y = (np.arange(out_h) + 0.5) * scale_y - 0.5
    x = (np.arange(out_w) + 0.5) * scale_x - 0.5

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    y0 = np.clip(y0, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)

    ly = y - y0
    lx = x - x0
    hy = 1.0 - ly
    hx = 1.0 - lx

    # 计算四个邻点的权重
    w_tl = hy[:, None] * hx[None, :]
    w_tr = hy[:, None] * lx[None, :]
    w_bl = ly[:, None] * hx[None, :]
    w_br = ly[:, None] * lx[None, :]

    # 利用广播获取四个邻点的值并加权求和
    out = (
        data[:, y0[:, None], x0[None, :]] * w_tl[None, :, :]
        + data[:, y0[:, None], x1[None, :]] * w_tr[None, :, :]
        + data[:, y1[:, None], x0[None, :]] * w_bl[None, :, :]
        + data[:, y1[:, None], x1[None, :]] * w_br[None, :, :]
    )

    return out.astype(data.dtype, copy=False)


def update_units(v):
    if isinstance(v, xr.Dataset):
        return v.map(update_units)
    if v.name in UNIT_SCALE:
        scale = UNIT_SCALE[v.name]
        v *= scale
        logger.info(f"scaling {v.name} by {scale:.6f}")
    return v


def load_datasets(path: str) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Load datasets from a GRIB2 file (ANAL/PRECIP/SURFACE from CMA-RA-V1.5).

    Returns:
        data (C, H, W), channels, latitudes, longitudes
    """
    logger.debug(f"Loading datasets from {path}")
    if path.find("ANAL") >= 0:
        short_names = ["z", "t", "u", "v", "q"]
        datasets = xr.open_dataset(
            path,
            engine="cfgrib",
            filter_by_keys={
                'typeOfLevel': 'isobaricInhPa',
                'shortName': ['gh', 't', 'u', 'v', 'q'],
            },
            backend_kwargs={"indexpath": ""},
        )
        datasets = update_units(datasets)
        datasets = datasets.rename_vars({"gh": "z"})
        datasets = datasets.sel(isobaricInhPa=LEVEL_13)  # 13,721,1440
        msl_data = xr.open_dataset(
            path,
            engine="cfgrib",
            filter_by_keys={
                'shortName': 'msl'
            },
            backend_kwargs={"indexpath": ""},
        )
        datasets = xr.merge([datasets, msl_data], compat="override")
        channels = [f"{var}{level}" for var in short_names for level in LEVEL_13] + ["msl"]
        data = np.concatenate([datasets[var].values for var in short_names], axis=0)
        data = np.concatenate([data, msl_data["msl"].values[np.newaxis, :, :]], axis=0)
        latitudes = datasets['latitude'].values
        longitudes = datasets['longitude'].values
        logger.debug(f"{channels}, {data.shape}")
    elif path.find("PRECIP") >= 0:
        # PRECIP is grib file, not grib2
        dataarray = open_precip_grib(path)
        channels = ['tp6']  # 重命名为tp6，对应6小时降水
        data = dataarray.data[np.newaxis, :, :]
        latitudes = dataarray['lat'].values
        longitudes = dataarray['lon'].values
        logger.debug(f"{channels}, {data.shape}")
    elif path.find("SURF") >= 0:
        data = None
        channels = []
        datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
        for _, ds in enumerate(datasets):
            for key, value in ds.variables.items():
                if key not in ["q", "t2m", "u10", "v10"]:
                    continue
                logger.debug(f"{key} with shape {value.shape}")
                if data is None:
                    data = value.values[np.newaxis, :, :]
                else:
                    data = np.concatenate([data, value.values[np.newaxis, :, :]], axis=0)
                if key == "q":
                    key = "q2m"
                    data = data * 1000  # convert from kg/kg to g/kg
                channels.append(key)
        latitudes = datasets[0]['latitude'].values
        longitudes = datasets[0]['longitude'].values
        # convert from [d2m,t2m,u10,v10] to [t2m,d2m,u10,v10]
        data = data[[1, 0, 2, 3], :, :]
        channels = [channels[1], channels[0]] + channels[2:]
        logger.debug(f"{channels}, {data.shape}")
    else:
        logger.error(f"Unknown GRIB2 file type for path: {path}")
        return np.array([]), [], np.array([]), np.array([])

    longitudes = np.linspace(0, 360, 1441, dtype=np.float32)[:-1]  # 0.25 degree interval
    latitudes = latitudes.astype(np.float32)

    # convert data from latitude from descending to ascending
    if data is not None and latitudes[0] > latitudes[-1]:
        data = data[:, ::-1, :]
        latitudes = latitudes[::-1]
        logger.debug("Reversed latitude order from descending to ascending.")
        logger.debug(f"Latitude range after reversal: {latitudes[0]} to {latitudes[-1]}")

    if len(latitudes) != 721:
        latitudes = np.linspace(-90, 90, 721, dtype=np.float32)
        if data is not None:
            data = bilinear_resize_numpy(data, 721, 1440)
        logger.debug(f"new latitudes is :{latitudes}")
        logger.debug(f"longitudes is :{longitudes}")
        logger.debug(f"Resized data to (721, 1440), new shape: {data.shape}")

    return data, channels, latitudes, longitudes


def create_dummy_precip_data(data_anal, channels_precip=['tp6']):
    """
    创建模拟的降水数据（全零数组），保持输入通道和顺序不变
    """
    # 从 ANAL 数据获取空间维度 (H, W)
    if data_anal is not None and len(data_anal.shape) >= 2:
        # 如果是 3D 数据 (C, H, W)，取最后两个维度
        if len(data_anal.shape) == 3:
            h, w = data_anal.shape[-2:]
        else:
            h, w = data_anal.shape
    else:
        # 默认使用标准网格
        h, w = 721, 1440

    # 创建全零数组，模拟降水数据
    dummy_precip = np.zeros((1, h, w), dtype=np.float32)
    logger.debug(f"Created dummy precip data with shape: {dummy_precip.shape}")
    return dummy_precip


def load_input(input_path: str, current_time: str, freq: int = 6):
    # find current and current - freq file from input_path
    current_datetime = pd.to_datetime(current_time, format="%Y%m%d%H")
    hist_datetime = current_datetime - pd.Timedelta(hours=freq)
    hist_time = hist_datetime.strftime("%Y%m%d%H")

    cur_anal_files = glob.glob(os.path.join(input_path, "**", f"*ANAL*{current_time}*.grib2"), recursive=True)
    hist_anal_files = glob.glob(os.path.join(input_path, "**", f"*ANAL*{hist_time}*.grib2"), recursive=True)
    cur_surf_files = glob.glob(os.path.join(input_path, "**", f"*SURFACE*{current_time}*.grib"), recursive=True)
    hist_surf_files = glob.glob(os.path.join(input_path, "**", f"*SURFACE*{hist_time}*.grib"), recursive=True)

    cur_files = [*cur_anal_files, *cur_surf_files]
    hist_files = [*hist_anal_files, *hist_surf_files]

    assert len(cur_files) == 2, f"Cannot find all 2 files (ANAL and SURFACE) for current time {current_time} in {input_path}"
    assert len(hist_files) == 2, f"Cannot find all 2 files (ANAL and SURFACE) for hist time {hist_time} in {input_path}"

    logger.info(f"Current files: {cur_files}")
    logger.info(f"Hist files: {hist_files}")

    # load current data
    data_anal, channels_anal, latitudes, longitudes = load_datasets(cur_files[0])
    data_surf, channels_surf, _, _ = load_datasets(cur_files[1])

    # 创建模拟的降水数据（全零）
    data_precip_dummy = create_dummy_precip_data(data_anal)
    channels_precip = ['tp6']  # 降水通道名保持不变

    # 保持原始数据拼接顺序：data_anal[:-1] + data_surf + data_anal[-1:] + data_precip
    current_data = np.concatenate([data_anal[:-1], data_surf, data_anal[-1:], data_precip_dummy], axis=0).astype(np.float32)
    channels = channels_anal[:-1] + channels_surf + channels_anal[-1:] + channels_precip

    # load hist data
    data_anal_hist, _, _, _ = load_datasets(hist_files[0])
    data_surf_hist, _, _, _ = load_datasets(hist_files[1])

    # 创建历史时刻的模拟降水数据
    data_precip_dummy_hist = create_dummy_precip_data(data_anal_hist)

    hist_data = np.concatenate([data_anal_hist[:-1], data_surf_hist, data_anal_hist[-1:], data_precip_dummy_hist], axis=0).astype(np.float32)

    logger.info(f"Input data shape: current={current_data.shape}, hist={hist_data.shape}")
    logger.info(f"Total channels: {len(channels)}")
    logger.info(f"Channel order: ANAL({len(channels_anal) - 1}) + SURF({len(channels_surf)}) + msl(1) + PRECIP({len(channels_precip)})")

    return np.stack([hist_data, current_data], axis=0), channels, latitudes, longitudes


def save_single_step_pred(output, channel_list: List[str], init_time: pd.Timestamp,
                          forecast_hour: int, latitudes: np.ndarray, longitudes: np.ndarray,
                          save_dir: str, total_member: int = 1):
    """
    保存单个预报时次的结果为独立的NetCDF文件
    文件名格式：W2S_V1_GLB_0P25_ENS_HOUR_YYYYMMDDHH_LT.nc

    Args:
        output: 模型输出数据。total_member=1时shape=(C, H, W)；多成员时shape=(member, C, H, W)
        channel_list: 要素通道列表
        init_time: 起报时间（datetime对象）
        forecast_hour: 预报时效（小时）
        latitudes: 纬度数组
        longitudes: 经度数组
        save_dir: 保存目录
        total_member: 集合成员数，默认1
    """
    # 构造文件名
    init_time_str = init_time.strftime("%Y%m%d%H")
    forecast_hour_str = f"{forecast_hour:03d}"
    ens_prefix = "ENS_" if total_member > 1 else ""
    filename = f"W2S_V1_GLB_0P25_{ens_prefix}HOUR_{init_time_str}_{forecast_hour_str}.nc"
    save_path = os.path.join(save_dir, filename)

    # 创建xarray数据数组
    pred_time = init_time + pd.Timedelta(hours=forecast_hour)
    member_coords = list(range(1, total_member + 1))

    if total_member > 1:
        # output shape: (member, C, H, W) -> (1, member, C, H, W)
        data = output[np.newaxis, ...]
    else:
        # output shape: (C, H, W) -> (1, 1, C, H, W) — 统一4维
        data = output[np.newaxis, np.newaxis, ...]

    pred = xr.DataArray(
        name="data",
        data=data,
        dims=['time', 'member', 'channel', 'lat', 'lon'],
        coords=dict(
            time=[pred_time],
            member=member_coords,
            channel=channel_list,
            lat=latitudes,
            lon=longitudes,
        ),
        attrs={
            "init_time": init_time_str,
            "forecast_hour": forecast_hour,
            "total_member": total_member,
            "resolution": "0.25x0.25 degree",
            "domain": "global"
        }
    ).astype(np.float32)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存NetCDF文件
    pred.to_netcdf(save_path)
    logger.info(f"已保存预报时效 {forecast_hour}h 结果({total_member}成员)至: {save_path}")


def _select_netcdf_engine() -> str:
    """Pick the fastest available NetCDF write engine.

    Avoids scipy/NETCDF3 which is extremely slow for large arrays.
    Preference: netcdf4 > h5netcdf > scipy (last resort).

    Note: xarray uses lowercase engine names ("netcdf4", "h5netcdf"), but the
    actual importable module names differ in case: "netCDF4" (capital F) and
    "h5netcdf". We must probe the correct import name, not the engine name.
    """
    # (xarray engine name, python import name)
    for eng, import_name in (("netcdf4", "netCDF4"), ("h5netcdf", "h5netcdf")):
        try:
            import importlib

            importlib.import_module(import_name)
            return eng
        except ImportError:
            continue
    logger.warning("netcdf4/h5netcdf not installed; falling back to scipy (slow for large arrays).")
    return "scipy"


def save_step_member_nc4(
    save_path: str,
    step_data: np.ndarray,
    member_idx: int,
    channel_list: List[str],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    init_time: pd.Timestamp,
    forecast_hour: int,
    total_member: int,
):
    """Append one member's data to a per-step NetCDF file (netCDF4 incremental mode).

    Creates the file on first member (idx=0), appends on subsequent members.
    Uses unlimited 'member' dimension to avoid holding all members in memory.

    Args:
        save_path: Output NetCDF file path.
        step_data: (C, H, W) array for one member at one step.
        member_idx: 0-based member index.
        channel_list: Channel names.
        latitudes/longitudes: Spatial coords.
        init_time: Init time.
        forecast_hour: Forecast lead hours.
        total_member: Total ensemble members (for metadata).
    """
    import netCDF4 as nc

    pred_time = init_time + pd.Timedelta(hours=forecast_hour)
    init_time_str = init_time.strftime("%Y%m%d%H")

    if member_idx == 0:
        # ---- Create new file with unlimited member dimension ----
        with nc.Dataset(save_path, 'w', format='NETCDF4') as f:
            f.createDimension('time', 1)
            f.createDimension('member', None)  # unlimited
            f.createDimension('channel', len(channel_list))
            f.createDimension('lat', len(latitudes))
            f.createDimension('lon', len(longitudes))

            # Coordinate variables
            time_var = f.createVariable('time', 'f8', ('time',))
            time_var.units = 'hours since 1970-01-01 00:00:00'
            time_var.calendar = 'proleptic_gregorian'
            from netCDF4 import date2num
            dt = pred_time.to_pydatetime()
            time_var[:] = date2num(
                [dt], units=time_var.units, calendar='proleptic_gregorian')

            member_var = f.createVariable('member', 'i4', ('member',))
            member_var[0] = 1

            channel_var = f.createVariable('channel', str, ('channel',))
            for i, ch in enumerate(channel_list):
                channel_var[i] = ch

            lat_var = f.createVariable('lat', 'f4', ('lat',))
            lat_var[:] = latitudes

            lon_var = f.createVariable('lon', 'f4', ('lon',))
            lon_var[:] = longitudes

            # Data variable (zlib compression to save disk)
            data_var = f.createVariable(
                'data', 'f4',
                ('time', 'member', 'channel', 'lat', 'lon'),
                zlib=True, complevel=4)
            data_var.init_time = init_time_str
            data_var.forecast_hour = forecast_hour
            data_var.total_member = total_member
            data_var.resolution = "0.25x0.25 degree"
            data_var.domain = "global"

            data_var[0, 0, :, :, :] = step_data
    else:
        # ---- Append to existing file along member dimension ----
        with nc.Dataset(save_path, 'a') as f:
            data_var = f.variables['data']
            data_var[0, member_idx, :, :, :] = step_data
            if 'member' in f.variables:
                f.variables['member'][member_idx] = member_idx + 1


def save_realtime_outputs(
    all_outputs: np.ndarray,
    channel_list: List[str],
    init_time: pd.Timestamp,
    hour_interval: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    save_dir: str,
    total_member: int = 1,
    num_workers: int = 4,
):
    """Write per-step NetCDF files (one file per forecast hour, all members inside).

    Decoupled from the inference loop so that the GPU hot path never waits on
    disk IO. Safety-first design:
      1. First dump the whole result array to a single .npy so that even if the
         NetCDF step crashes (e.g. netcdf4/HDF5 segfault under parallel writes),
         the 65-min inference result is NOT lost and can be converted later.
      2. Then write per-step NetCDF files SERIALLY. netcdf4's HDF5 backend is
         not thread-safe; concurrent writes from a ThreadPoolExecutor can trigger
         a C-level segfault (core dumped) with no Python traceback. Serial writes
         are slightly slower but reliable.

    Output filename: W2S_V1_GLB_0P25_ENS_HOUR_{init}_{lead:03d}.nc
    Output dims:     (time, member, channel, lat, lon)

    Args:
        all_outputs: (total_member, total_step, C, H, W) ensemble predictions.
        channel_list: Channel names for the C dimension.
        init_time: Init time (datetime).
        hour_interval: Hours per forecast step.
        latitudes/longitudes: Spatial coordinates.
        save_dir: Output directory.
        total_member: Number of ensemble members.
        num_workers: Kept for API compatibility; NetCDF writes are now serial.
    """
    os.makedirs(save_dir, exist_ok=True)
    total_step = all_outputs.shape[1]
    init_time_str = init_time.strftime("%Y%m%d%H")
    member_coords = list(range(1, total_member + 1))
    ens_prefix = "ENS_" if total_member > 1 else ""

    # ---- 1) Safety net: dump raw array + metadata to .npy so inference is never lost ----
    npy_path = os.path.join(save_dir, f"all_outputs_{init_time_str}.npy")
    np.save(npy_path, all_outputs.astype(np.float32))
    # save metadata alongside
    meta_path = os.path.join(save_dir, f"all_outputs_{init_time_str}.meta.npz")
    np.savez(meta_path,
             channel_list=np.array(channel_list, dtype=object),
             latitudes=latitudes.astype(np.float32),
             longitudes=longitudes.astype(np.float32),
             init_time_str=np.array(init_time_str),
             hour_interval=np.array(hour_interval),
             total_member=np.array(total_member))
    logger.info(f"安全兜底: 推理结果已存至 {npy_path} (可用 convert_npy_to_netcdf.py 转换)")

    # ---- 2) Serial NetCDF writes (avoid netcdf4/HDF5 threading segfault) ----
    engine = _select_netcdf_engine()
    netcdf_format = "NETCDF3_64BIT" if engine == "scipy" else "NETCDF4"
    logger.info(f"Saving {total_step} step files ({total_member} members each) "
                f"with engine='{engine}' (format={netcdf_format}, SERIAL) ...")

    def _write_one(t: int) -> int:
        forecast_hour = (t + 1) * hour_interval
        filename = f"W2S_V1_GLB_0P25_{ens_prefix}HOUR_{init_time_str}_{forecast_hour:03d}.nc"
        save_path = os.path.join(save_dir, filename)

        pred_time = init_time + pd.Timedelta(hours=forecast_hour)
        # all_outputs[:, t] shape: (member, C, H, W) -> (1, member, C, H, W)
        step_data = all_outputs[:, t][np.newaxis, ...]
        pred = xr.DataArray(
            name="data",
            data=step_data,
            dims=['time', 'member', 'channel', 'lat', 'lon'],
            coords=dict(
                time=[pred_time],
                member=member_coords,
                channel=channel_list,
                lat=latitudes,
                lon=longitudes,
            ),
            attrs={
                "init_time": init_time_str,
                "forecast_hour": forecast_hour,
                "total_member": total_member,
                "resolution": "0.25x0.25 degree",
                "domain": "global",
            },
        ).astype(np.float32)

        pred.to_netcdf(save_path, engine=engine, format=netcdf_format)
        return forecast_hour

    save_start = perf_counter()
    ok_count = 0
    for t in range(total_step):
        try:
            fh = _write_one(t)
            logger.info(f"已保存预报时效 {fh}h 结果({total_member}成员)至: {save_dir}")
            ok_count += 1
        except Exception as e:  # noqa: BLE001
            logger.error(f"保存第 {t + 1} 步结果失败: {e} (原始数据已在 .npy 兜底)")

    logger.info(f"Save phase completed in {perf_counter() - save_start:.2f} secs. "
                f"成功 {ok_count}/{total_step} 个文件。"
                + ("" if ok_count == total_step else f" 失败步骤可从 {npy_path} 转换。"))


def split_netcdf_by_time(total_file: str, save_dir: str):
    """Split a combined NetCDF (time, [member], channel, lat, lon) into per-step files
    following the W2S_V1_GLB_0P25_ENS_HOUR_YYYYMMDDHH_LT.nc naming rule.

    Use this when predictions were first dumped as one big file and need to be
    fragmented afterwards -- fully decoupling inference from file splitting.

    Args:
        total_file: Path to the combined NetCDF file (must have a 'time' dim
                    and an 'init_time' attr to derive forecast hours).
        save_dir: Directory for the split per-step files.
    """
    engine = _select_netcdf_engine()
    netcdf_format = "NETCDF3_64BIT" if engine == "scipy" else "NETCDF4"
    os.makedirs(save_dir, exist_ok=True)

    ds = xr.open_dataset(total_file, engine=engine)
    da = ds["data"] if "data" in ds else ds[list(ds.data_vars)[0]]

    init_time_str = da.attrs.get("init_time")
    if init_time_str is None:
        init_time_str = pd.Timestamp(da["time"].values[0]).strftime("%Y%m%d%H")
        logger.warning("Total file missing 'init_time' attr; inferred from first time.")
    init_time = pd.Timestamp(init_time_str)

    times = da["time"].values
    ens_prefix = "ENS_" if da.sizes.get('member', 1) > 1 else ""
    for i, t in enumerate(times):
        forecast_hour = int(round((pd.Timestamp(t) - init_time).total_seconds() / 3600))
        filename = f"W2S_V1_GLB_0P25_{ens_prefix}HOUR_{init_time_str}_{forecast_hour:03d}.nc"
        save_path = os.path.join(save_dir, filename)
        sub = da.isel(time=i).expand_dims("time")
        sub.to_netcdf(save_path, engine=engine, format=netcdf_format)
        logger.info(f"拆分时效 {forecast_hour}h 至: {save_path}")
    ds.close()


class OnnxInferModel:
    """Standalone ONNX inference model (no BasePredictor inheritance).

    Top-level dependencies are identical to test_infer.py. The old config-driven
    zarr evaluation path (inference()/test_cascade()/forward()) has been removed;
    use run_realtime() for real-time GRIB2 ingestion.
    """

    def __init__(self, onnx_path, **kwargs):
        self.gpu_mem_fraction = kwargs.pop("gpu_mem_fraction", 0.7)
        self.use_cpu_initializers = kwargs.pop("use_cpu_initializers", True)
        # Attributes consumed by run_realtime(); accepted as kwargs so the
        # __main__ entry can construct the model directly without BasePredictor.
        self.in_frames = kwargs.pop("in_frames", 2)
        self.out_frames = kwargs.pop("out_frames", 1)
        self.test_frames = kwargs.pop("test_frames", [1])
        self.test_chans = kwargs.pop("test_chans", [])
        self.test_names = kwargs.pop("test_names", [])
        self.step_range = kwargs.pop("step_range", [1])
        self.members = kwargs.pop("members", 1)
        self.freq = kwargs.pop("freq", 6)
        self.save_dir = kwargs.pop("save_dir", "./output")
        self.logger = logger
        self.device_id = int(os.environ.get("LOCAL_RANK", 0))
        self.load_model(onnx_path)

    def load_model(self, onnx_path):
        if ort is None:
            raise ImportError("OnnxInferModel requires onnxruntime. Install onnxruntime or onnxruntime-gpu before using ONNX inference.")
        # ort.set_default_logger_severity(0)  # 0 = VERBOSE
        options = self._build_session_options()

        providers = []
        cuda_provider_options = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
        }
        gpu_mem_limit = self._default_gpu_mem_limit()
        if gpu_mem_limit is not None:
            cuda_provider_options["gpu_mem_limit"] = gpu_mem_limit
            self.logger.info(f"Limiting CUDA provider memory to {gpu_mem_limit / (1024 ** 3):.2f} GB")
        cuda_provider = ("CUDAExecutionProvider", cuda_provider_options)

        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(cuda_provider)
        else:
            self.logger.warning(f"CUDA Execution Provider not available for device {self.device_id}. Falling back to CPU.")

        # Always add CPU provider as fallback
        if len(providers) == 0:
            self.logger.error("No suitable execution providers found. Ensure ONNX Runtime is installed with CUDA support.")
        else:
            self.logger.info(f"Using CUDA provider on device {self.device_id}")

        providers.append(("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))

        # register custom op library for Mamba if get_library_path() is available
        if get_library_path is not None:
            # The precompiled custom-ops .so dynamically links libonnxruntime.so.1,
            # which lives inside the onnxruntime package (capi/) and is NOT on the
            # default loader path. Add it to LD_LIBRARY_PATH before dlopen so the
            # .so can resolve its onnxruntime dependency without manual env setup.
            self._ensure_ort_lib_path()
            options.register_custom_ops_library(get_library_path())
            self.logger.info(f"Registered custom op library from {get_library_path()}")

        self.session = ort.InferenceSession(onnx_path, sess_options=options, providers=providers)

        self.logger.info(f"Device {self.device_id}: Using providers {self.session.get_providers()}")
        self.logger.info(f"Loaded ONNX model from {onnx_path}")
        return

    def _build_session_options(self):
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.add_session_config_entry("cudnn_conv_algo_search", "HEURISTIC")
        if self.use_cpu_initializers:
            options.add_session_config_entry("session.use_device_allocator_for_initializers", "0")
            options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
            self.logger.info("Keeping model initializers on host memory to reduce GPU usage")
        return options

    def _default_gpu_mem_limit(self):
        # Lazy import: torch is only used here to query total GPU memory.
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        try:
            props = torch.cuda.get_device_properties(self.device_id)
        except (AssertionError, RuntimeError):
            return None
        limit = int(props.total_memory * self.gpu_mem_fraction)
        return max(limit, 0)

    def _ensure_ort_lib_path(self):
        """Make libonnxruntime.so.1 resolvable so the custom-ops .so can be dlopened.

        The precompiled .so links against libonnxruntime.so.1, but pip-installed
        onnxruntime-gpu keeps that library somewhere under site-packages that is
        NOT on the default loader path. We do two things here:

        1. Locate libonnxruntime*.so anywhere under the onnxruntime package (and
           under the conda env lib dir as a fallback), prepend its dir to
           LD_LIBRARY_PATH.
        2. As a hard guarantee, manually dlopen the found library with ctypes
           using RTLD_GLOBAL so its symbols are available to subsequently
           loaded shared objects, regardless of LD_LIBRARY_PATH quirks.
        """
        import ctypes
        import glob as _glob
        import sys

        try:
            import onnxruntime as _ort

            ort_dir = os.path.dirname(_ort.__file__)
        except ImportError:
            return

        # Build a broad candidate list: onnxruntime package dirs + conda env lib dirs.
        search_roots = [
            os.path.join(ort_dir, "capi"),
            os.path.join(ort_dir, "capi", "lib"),
        ]
        prefix = os.path.dirname(os.path.dirname(os.path.dirname(ort_dir)))  # site-packages parent
        search_roots.append(os.path.join(prefix, "lib"))
        # conda env typical layout: <env>/lib
        env_root = sys.prefix
        search_roots.append(os.path.join(env_root, "lib"))

        candidates = []
        for root in search_roots:
            if os.path.isdir(root):
                candidates += _glob.glob(os.path.join(root, "libonnxruntime.so*"))

        # Prefer versioned SONAME (libonnxruntime.so.1) first, then .so.1.x, then plain .so
        def _sort_key(p):
            name = os.path.basename(p)
            return (0 if name == "libonnxruntime.so.1" else
                    1 if name.startswith("libonnxruntime.so.1.") else
                    2 if name == "libonnxruntime.so" else 3)
        candidates.sort(key=_sort_key)

        lib_path = None
        for cand in candidates:
            if os.path.exists(cand):
                lib_path = cand
                break

        if lib_path is None:
            self.logger.warning(
                "Could not locate libonnxruntime.so under onnxruntime package (%s) or env lib. "
                "Custom op .so may fail to load. Set LD_LIBRARY_PATH manually if needed.", ort_dir
            )
            return

        lib_dir = os.path.dirname(lib_path)
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in cur.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + cur if cur else "")
            self.logger.info(f"Prepended {lib_dir} to LD_LIBRARY_PATH for custom op loading")

        # The .so links against SONAME "libonnxruntime.so.1", but pip may only
        # ship the fully-versioned file "libonnxruntime.so.1.24.4" without the
        # ".so.1" symlink. Create the symlink so the loader can resolve the SONAME.
        soname_link = os.path.join(lib_dir, "libonnxruntime.so.1")
        if lib_path != soname_link and not os.path.exists(soname_link):
            try:
                os.symlink(os.path.basename(lib_path), soname_link)
                self.logger.info(f"Created symlink {soname_link} -> {os.path.basename(lib_path)}")
            except OSError as e:
                self.logger.warning(f"Could not create symlink {soname_link}: {e}")

        # Hard guarantee: pre-load with RTLD_GLOBAL so symbols are globally visible.
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            self.logger.info(f"Pre-loaded {lib_path} with RTLD_GLOBAL")
        except OSError as e:
            self.logger.warning(f"Could not pre-load {lib_path}: {e}")

    def run_realtime(
        self,
        input_path: str,
        current_time: str,
        save_dir: str = None,
        total_step: int = None,
        total_member: int = None,
        hour_interval: int = None,
        channels: str = "",
        data: np.ndarray = None,
        all_channels: List[str] = None,
        latitudes: np.ndarray = None,
        longitudes: np.ndarray = None,
    ):
        """Real-time inference entry point.

        Loads CMA-RA-V1.5 GRIB2 inputs (ANAL/SURFACE) following the same data
        processing logic and paths as ``test_infer.py``, runs an ensemble
        rollout with the loaded ONNX session, and writes per-step NetCDF
        files (``W2S_V1_GLB_0P25_ENS_HOUR_YYYYMMDDHH_LT.nc``) containing all
        ensemble members.

        Args:
            input_path: Directory containing the real-time GRIB2 files.
            current_time: Init time in YYYYMMDDHH format (-freq and 0h).
            save_dir: Directory for prediction outputs. Defaults to self.save_dir.
            total_step: Number of forecast steps. Defaults to max(self.test_frames).
            total_member: Ensemble size. Defaults to self.members.
            hour_interval: Hours per forecast step. Defaults to self.freq.
            channels: Comma-separated channel names to save. Defaults to all.
            data: Pre-loaded input array (in_frames, C, H, W). If None, load from input_path.
            all_channels: Pre-loaded channel name list. Required if data is provided.
            latitudes/longitudes: Pre-loaded coords. Required if data is provided.
        """
        save_dir = save_dir or self.save_dir or "./output"
        total_step = int(total_step) if total_step else max(self.test_frames)
        total_member = int(total_member) if total_member else self.members
        hour_interval = int(hour_interval) if hour_interval else self.freq

        # ---- Load real-time GRIB2 data (same logic as test_infer.load_input) ----
        if data is None:
            data, all_channels, latitudes, longitudes = load_input(input_path, current_time, hour_interval)
        # data shape: (in_frames, C, H, W)

        # ---- Resolve channels to save ----
        if len(channels.strip()) == 0:
            channels_list = list(all_channels)
        else:
            channels_list = [c for c in channels.split(",") if c.strip()]
        valid_channels = []
        channel_indices = []
        for ch in channels_list:
            if ch in all_channels:
                valid_channels.append(ch)
                channel_indices.append(all_channels.index(ch))
            else:
                logger.warning(f"通道 {ch} 不存在于模型输出中，将跳过该通道")
        if len(channel_indices) == 0:
            logger.error("没有有效的通道可供保存，终止推理过程")
            return
        channel_indices = np.array(channel_indices, dtype=np.int64)

        init_time = pd.to_datetime(current_time, format="%Y%m%d%H")
        input_names = [x.name for x in self.session.get_inputs()]
        logger.info(f"Model input names: {input_names}")
        logger.info(f"Inference process started at {init_time} ...")

        # (in_frames, C, H, W) -> (1, in_frames, C, H, W) for ONNX
        base_input = data[np.newaxis, :].astype(np.float32)
        in_frames = base_input.shape[1]

        init_time_str = init_time.strftime("%Y%m%d%H")
        os.makedirs(save_dir, exist_ok=True)

        # ---- Inference with ASYNC NetCDF writing (OOM-safe + fast) ----
        # 优化要点 (解决 OOM 修复后推理变慢 7.5x 的问题):
        # 1. 步长外循环 + 成员内循环: 每步推理所有成员后一次性写入该步的 NetCDF 文件
        #    文件操作从 21*60=1260 次降至 60 次, 大幅减少 HDF5 open/close 开销
        # 2. 异步 I/O: 后台线程写文件, GPU 推理不等磁盘, 写入与推理重叠
        # 3. 不使用压缩 (与原始实现一致): complevel=4 的 zlib 压缩 281MB/步耗时数秒,
        #    去掉后写入速度 ~500MB/s, 每步 ~0.6s, 远快于推理 ~1s, 不构成瓶颈
        # 内存: 需保存所有成员的滚动输入 (21 * ~562MB ≈ 12GB), 远低于 OOM 阈值 (371GB)
        engine = _select_netcdf_engine()
        netcdf_format = "NETCDF3_64BIT" if engine == "scipy" else "NETCDF4"
        member_coords = list(range(1, total_member + 1))
        ens_prefix = "ENS_" if total_member > 1 else ""

        write_queue: queue.Queue = queue.Queue(maxsize=2)
        write_errors = []

        def _netcdf_writer():
            """后台线程: 串行写入 per-step NetCDF 文件.
            netCDF4/HDF5 后端非线程安全, 不可并发写同一文件; 单线程串行写是安全的.
            """
            while True:
                item = write_queue.get()
                if item is None:  # sentinel: 推理结束
                    break
                try:
                    save_path, step_data, fh = item
                    pred_time = init_time + pd.Timedelta(hours=fh)
                    da = xr.DataArray(
                        name="data",
                        data=step_data[np.newaxis, ...],  # (1, member, C, H, W)
                        dims=['time', 'member', 'channel', 'lat', 'lon'],
                        coords=dict(
                            time=[pred_time],
                            member=member_coords,
                            channel=valid_channels,
                            lat=latitudes,
                            lon=longitudes,
                        ),
                        attrs={
                            "init_time": init_time_str,
                            "forecast_hour": fh,
                            "total_member": total_member,
                            "resolution": "0.25x0.25 degree",
                            "domain": "global",
                        },
                    ).astype(np.float32)
                    da.to_netcdf(save_path, engine=engine, format=netcdf_format)
                    logger.info(f"已保存预报时效 {fh}h 结果({total_member}成员)至: {save_path}")
                except Exception as e:
                    write_errors.append(e)
                    logger.error(f"NetCDF 写入失败: {e}")

        writer_thread = threading.Thread(target=_netcdf_writer, daemon=True)
        writer_thread.start()

        # 保存所有成员的滚动输入 (21 * ~562MB ≈ 12GB)
        member_inputs = [base_input.copy() for _ in range(total_member)]

        infer_start = perf_counter()
        for t in range(total_step):
            forecast_hour = (t + 1) * hour_interval
            valid_time = init_time + pd.Timedelta(hours=t * hour_interval)

            # 推理所有成员的当前步
            step_all_members = []
            for member in range(total_member):
                inputs = {'input': member_inputs[member]}

                if "step" in input_names:
                    inputs['step'] = np.array([t], dtype=np.float32)

                if "hour" in input_names:
                    inputs['hour'] = np.array([valid_time.hour / 24], dtype=np.float32)

                if "doy" in input_names:
                    inputs['doy'] = np.array([min(365, valid_time.day_of_year) / 365], dtype=np.float32)

                step_start_time = perf_counter()
                pred = self.session.run(None, inputs)[0]
                step_elapsed_time = perf_counter() - step_start_time

                # 提取要保存的通道 (fancy indexing 返回副本, 安全用于异步写入)
                step_data = np.ascontiguousarray(pred[0, 0, channel_indices])
                step_all_members.append(step_data)

                # 更新该成员的滚动输入 (ascontiguousarray 释放拼接产生的临时内存)
                member_inputs[member] = np.ascontiguousarray(
                    np.concatenate([member_inputs[member], pred], axis=1)[:, -in_frames:]
                )

                logger.info(f"Member: {member + 1}, Step {t + 1}/{total_step}, Forecast hour: {forecast_hour}h, Time: {step_elapsed_time:.3f} secs")

            # 堆叠所有成员数据, 异步写入 (不阻塞下一步推理)
            step_data_stacked = np.stack(step_all_members)  # (member, C, H, W)
            filename = f"W2S_V1_GLB_0P25_{ens_prefix}HOUR_{init_time_str}_{forecast_hour:03d}.nc"
            save_path = os.path.join(save_dir, filename)
            write_queue.put((save_path, step_data_stacked, forecast_hour))

            del step_all_members, step_data_stacked

        # 等待写入线程完成
        write_queue.put(None)
        writer_thread.join()

        if write_errors:
            logger.error(f"保存阶段发生 {len(write_errors)} 个写入错误")

        del member_inputs
        gc.collect()

        logger.info(f"Inference + async save phase completed in {perf_counter() - infer_start:.2f} secs.")
        logger.info(f"\nInference process completed. All {total_step} forecast files saved to {save_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Real-time ONNX inference with CMA-RA-V1.5 GRIB2 data.")
    parser.add_argument('--model', "-m", type=str, required=True,
                        help="Path to the ONNX file for the FDP model.")
    parser.add_argument('--input', "-i", type=str, required=True,
                        help="Path to the input grib/grib2 data file. ANAL/PRECIP/SURFACE from CMA-RA-V1.5")
    parser.add_argument('--time', "-t", type=str, default="2025102700",
                        help="Valid time in YYYYMMDDHH format, -6h and 0h from the init time. Default is 2025102700")
    parser.add_argument('--save_dir', "-s", type=str, default="./output",
                        help="Directory where the prediction output will be saved. Default is ./output")
    parser.add_argument("--channels", "-l", default="",
                        help="Comma-separated list of channel names to save. (e.g. t2m,tp,msl). Default is all channels.")
    parser.add_argument('--total_step', "-n", type=int, default=1,
                        help="Total forecast steps to predict. Default is 1")
    parser.add_argument('--total_member', "-e", type=int, default=1,
                        help="Total ensemble members to predict. Default is 1")
    parser.add_argument('--hour_interval', "-f", type=int, default=6,
                        help="Hour interval between each forecast step. Default is 6")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    assert os.path.exists(args.model), f"Model file {args.model} not found!"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    os.makedirs(args.save_dir, exist_ok=True)

    # Load real-time input first to determine channel layout (same as test_infer.py)
    data, channels, latitudes, longitudes = load_input(args.input, args.time, args.hour_interval)

    logger.info(f"Input data shape: {data.shape}")
    logger.info(f"Available channels count: {len(channels)}")
    logger.info(f"First 10 channels: {channels[:10]}")

    # Build the ONNX inference model. Real-time data is fed in physical units
    # (no normalization), consistent with test_infer.py.
    num_chans = len(channels)
    model = OnnxInferModel(
        onnx_path=args.model,
        in_frames=data.shape[0],
        out_frames=1,
        test_frames=list(range(1, args.total_step + 1)),
        test_chans=list(range(num_chans)),
        test_names=list(channels),
        step_range=[args.total_step],
        members=args.total_member,
        freq=args.hour_interval,
        save_dir=args.save_dir,
    )

    logger.info(f'Loading FDP model from {args.model} ...')
    model_load_start = perf_counter()
    logger.info(f'Model loaded successfully, load time: {perf_counter() - model_load_start:.2f} secs')

    model.run_realtime(
        input_path=args.input,
        current_time=args.time,
        save_dir=args.save_dir,
        total_step=args.total_step,
        total_member=args.total_member,
        hour_interval=args.hour_interval,
        channels=args.channels,
        data=data,
        all_channels=channels,
        latitudes=latitudes,
        longitudes=longitudes,
    )
