# -*- coding: utf-8 -*-
"""Zarr data loader for physical or pre-normalized model input stores.

`normalized=True` 时，使用 store 同目录的 mean.npy/std.npy 恢复物理量：
`physical = normalized * std + mean`；tp 再执行 expm1。这样可把训练用的标准化
Zarr 安全送入自身已内嵌归一化的 ONNX 模型，避免二次归一化。

适配的 zarr 布局（zarr v3，参考 s2s.1950-2024.c76）：
    data   (time, level, lat, lon) float16   —— 注意这里 `level` 维其实是「通道」维
    time   (time)   int64, units="days since 1950-01-01"
    level  (76)     字符串，即 z500/z850/.../t2m/... 通道名
    lat    (121)  /  lon (240)
"""
import os

import numpy as np
import pandas as pd
import xarray as xr


class ZarrDataLoader:
    """从 zarr store 读输入状态。

    data_var    zarr 里的数据变量名（默认 "data"）
    channel_dim zarr 里承载「通道名」的维名（默认 "level"；open 后统一改名成
                build_input 认识的 "channel"）
    tol         按 time 切片后，最近帧与目标时刻的最大允许偏差（默认 12h，超了
                说明数据时间分辨率与 spec 的 hour_interval 对不上，直接报错）
    """

    def __init__(
        self,
        path,
        data_var="data",
        channel_dim="level",
        tol="1min",
        normalized=False,
        mean_path=None,
        std_path=None,
        log_channels=("tp",),
    ):
        self.path = path
        self.data_var = data_var
        self.channel_dim = channel_dim
        self.tol = pd.Timedelta(tol)
        self.normalized = normalized
        self.mean_path = mean_path or os.path.join(path, "mean.npy")
        self.std_path = std_path or os.path.join(path, "std.npy")
        self.log_channels = set(log_channels)
        self._ds = None          # 惰性打开的全时间 Dataset
        self._mean = None
        self._std = None

    def _load_stats(self, channel_count):
        if self._mean is not None:
            return
        if not os.path.isfile(self.mean_path):
            raise FileNotFoundError(f"标准化 Zarr 缺少 mean.npy：{self.mean_path}")
        if not os.path.isfile(self.std_path):
            raise FileNotFoundError(f"标准化 Zarr 缺少 std.npy：{self.std_path}")
        mean = np.array(
            np.load(self.mean_path, mmap_mode="r"),
            dtype=np.float32,
            copy=True,
        ).reshape(-1)
        std = np.array(
            np.load(self.std_path, mmap_mode="r"),
            dtype=np.float32,
            copy=True,
        ).reshape(-1)
        if mean.size != channel_count or std.size != channel_count:
            raise ValueError(
                f"mean/std 通道数不匹配：mean={mean.size}, std={std.size}, "
                f"Zarr={channel_count}"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("mean.npy/std.npy 包含 NaN 或 Inf")
        if np.any(std <= 0):
            raise ValueError("std.npy 包含非正数，无法反归一化")
        self._mean = mean
        self._std = std

    def _to_physical(self, frame):
        if not self.normalized:
            return frame
        channels = [str(value) for value in np.atleast_1d(frame["channel"].values)]
        self._load_stats(len(channels))
        values = np.asarray(frame[self.data_var].values, dtype=np.float32)
        values = values * self._std[:, None, None] + self._mean[:, None, None]
        for index, name in enumerate(channels):
            if name in self.log_channels:
                np.clip(values[index], 0.0, 7.0, out=values[index])
                np.expm1(values[index], out=values[index])
        out = frame.copy()
        out[self.data_var] = (frame[self.data_var].dims, values)
        out[self.data_var].attrs = dict(frame[self.data_var].attrs)
        out[self.data_var].attrs["xmetai_denormalized"] = True
        return out

    def _open(self):
        if self._ds is None:
            # 优先 consolidated 元数据；没有就退到普通 open（zarr v3 用内联 metadata）
            try:
                ds = xr.open_zarr(self.path, consolidated=True)
            except Exception:
                ds = xr.open_zarr(self.path)

            if self.data_var not in ds.data_vars:
                raise KeyError(f"zarr 里没有变量 {self.data_var!r}，现有: {list(ds.data_vars)}")
            if "time" not in ds.coords:
                raise KeyError("zarr 里没有 time 坐标，无法按时间切帧")

            # 时间必须是日期；若还是整数/对象说明 CF 解码没生效，给明确提示而不是算错日期
            if not pd.api.types.is_datetime64_any_dtype(ds.time):
                raise ValueError(
                    f"zarr 的 time 坐标未被解码为日期（dtype={ds.time.dtype}）。"
                    f"请确认 time 数组带 units/calendar 属性，且用 decode_times=True 打开")

            # channel 维统一改名成 build_input 认识的 "channel"
            if self.channel_dim in ds.dims and self.channel_dim != "channel":
                ds = ds.rename({self.channel_dim: "channel"})
            self._ds = ds
        return self._ds

    def load(self, time, channels=None):
        """输入时刻，返回该时刻的 xr.Dataset（dims: channel, lat, lon）。"""
        t = pd.to_datetime(time, format="%Y%m%d%H") if isinstance(time, str) \
            else pd.to_datetime(time)

        frame = self._open().sel(time=t, method="nearest")

        # 距离检查：method="nearest" 会掩盖「时间分辨率不匹配」，这里显式校验
        actual = pd.Timestamp(frame.time.item())
        if abs(actual - t) > self.tol:
            raise ValueError(
                f"zarr 里最接近 {t} 的时刻是 {actual}（差 {actual - t}），"
                f"超过容差 {self.tol}；检查数据时间分辨率是否与 spec 的 hour_interval 匹配")

        if channels is not None and not self.normalized:
            requested = [str(name) for name in channels]
            available = set(str(name) for name in frame["channel"].values)
            missing = [name for name in requested if name not in available]
            if missing:
                raise KeyError(f"zarr 缺少通道：{', '.join(missing)}")
            frame = frame.sel(channel=requested)

        # 去掉 time 标量坐标，转成内存 numpy（避免 dask 惰性值污染后续装配）
        frame = frame.drop_vars("time", errors="ignore")
        frame = self._to_physical(frame)
        if channels is not None and self.normalized:
            requested = [str(name) for name in channels]
            available = set(str(name) for name in frame["channel"].values)
            missing = [name for name in requested if name not in available]
            if missing:
                raise KeyError(f"zarr 缺少通道：{', '.join(missing)}")
            frame = frame.sel(channel=requested)
        return frame.load()

    def load_state(self, time, channels=None):
        """返回统一 State dict：通道名 -> 单帧 (lat, lon) 数组。

        与 load(time) 等价，把 (channel, lat, lon) 的 data 变量摊平成 fields 字典；
        通道名用 zarr 的 channel 维坐标原样，lat/lon 保持数据源原始顺序，留给
        geometry 处理器。zarr 打包时已按模型规范单位存好（无 SCALE 换算）。
        """
        ds = self.load(time, channels=channels)
        fields = {}
        names = [str(c) for c in np.atleast_1d(np.asarray(ds["channel"].values))]
        for i, name in enumerate(names):
            fields[name] = ds[self.data_var].isel(channel=i).values
        return {
            "fields": fields,
            "date": pd.to_datetime(time, format="%Y%m%d%H") if isinstance(time, str)
                    else pd.to_datetime(time),
            "latitudes": np.asarray(ds["lat"].values),
            "longitudes": np.asarray(ds["lon"].values),
        }
