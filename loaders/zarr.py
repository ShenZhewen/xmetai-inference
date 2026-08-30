# -*- coding: utf-8 -*-
"""zarr 数据源 loader：把打包好的 zarr（dims: time, channel, lat, lon）读成
build_input 要的「自描述 xr.Dataset」。

只需实现 `load(time) -> xr.Dataset` 一个接口；build_input 会自动接管单位推断、
层级/网格适配、翻转滚动、装配。zarr 只打开一次，之后按 time 切片命中缓存。

适配的 zarr 布局（zarr v3，参考 s2s.1950-2024.c76）：
    data   (time, level, lat, lon) float16   —— 注意这里 `level` 维其实是「通道」维
    time   (time)   int64, units="days since 1950-01-01"
    level  (76)     字符串，即 z500/z850/.../t2m/... 通道名
    lat    (121)  /  lon (240)
"""
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

    def __init__(self, path, data_var="data", channel_dim="level", tol="12h"):
        self.path = path
        self.data_var = data_var
        self.channel_dim = channel_dim
        self.tol = pd.Timedelta(tol)
        self._ds = None          # 惰性打开的全时间 Dataset

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

    def load(self, time):
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

        # 去掉 time 标量坐标，转成内存 numpy（避免 dask 惰性值污染后续装配）
        return frame.drop_vars("time", errors="ignore").load()
