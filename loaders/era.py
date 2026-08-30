# -*- coding: utf-8 -*-
"""ERA 逐变量文件数据源：实现 load(time) -> xr.Dataset 一个接口。

build_input 只依赖 loader.load(time)，不关心 loader 是什么类——换数据源时
提供一个「有 load(time) -> xr.Dataset 方法」的对象即可，其余（单位推断、
层级/网格适配、装配）全部复用。本类把读文件缓存收进实例，避免模块级全局
状态，也方便不同数据源各自隔离。
"""
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_DATA_ROOT = "/workspace/data/xmetai-data/ERA/nc/0p25"


def _parse_name(name):
    """通道名 -> (var, level 或 None)，如 'z1000' -> ('z', 1000)。

    原 naming.py 的 _parse_name 已并入 build_input；本 loader 只需这一小段，
    为避免 build_input ↔ loaders.era 循环导入，这里保留一份最小实现。
    """
    key = str(name).strip()
    m = re.fullmatch(r"([a-zA-Z]+?)(\d+)", key)
    if m:
        return m.group(1).lower(), int(m.group(2))
    return key.lower(), None


class EraDataLoader:
    """从 ERA 逐变量文件读气象状态，返回自描述 xr.Dataset。"""

    def __init__(self, spec, data_root=None):
        # 从 spec 的通道展开结果得知要读哪些变量、哪些是气压层变量（带 level 维）
        self.pressure_vars = {var for var, lv in spec["_parse"].values() if lv is not None}
        self.variables = sorted(spec["_all_vars"])
        self.data_root = data_root or os.environ.get("DATADIR", DEFAULT_DATA_ROOT)
        self._cache = {}

    def _read_var_file(self, var, year, date):
        # ERA 是逐日文件：相邻起报/相邻帧会反复打开同一个 (变量, 日期).nc。netCDF4/
        # HDF5 在部分版本下反复开关同一文件会原生崩溃（"double free or corruption"）。
        # 这里每个文件只读一次，缓存原始数组 + 元信息，并只保留最近两个日期以控制内存。
        key = (var, date)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.data_root, var, year, date + ".nc")
        with xr.open_dataset(path, cache=False) as f:
            da = f["data"]
            entry = {
                "arr": da.values,
                "lat": f["lat"].values,
                "lon": f["lon"].values,
                "channel": ([str(x) for x in np.atleast_1d(f["channel"].values)]
                            if "channel" in f.coords else None),
                "units": da.attrs.get("units"),
            }
        self._cache[key] = entry
        # 只保留最近两个日期（历史窗口最多跨 1 天），避免缓存无限增长
        keep = {date, (pd.to_datetime(date, format="%Y%m%d") - pd.Timedelta(days=1)).strftime("%Y%m%d")}
        for k in list(self._cache):
            if k[1] not in keep:
                self._cache.pop(k, None)
        return entry

    def load(self, time):
        """输入时刻，返回该时刻的气象状态（自描述 xr.Dataset）。

        数据按 变量/年份/日期.nc 存放，例如
            /workspace/data/xmetai-data/ERA/nc/0p25/q/2024/20240101.nc
        每个文件里一个 `data` 变量：
            * 气压变量（z/t/u/v/q）带 channel 维，channel 坐标存 z1000..z50（自底向上）
            * 地面变量是 2D (lat, lon)，标量 channel 坐标存变量名
        这里把 spec 需要的变量合并成一个 Dataset：气压变量转成 (level, lat, lon)，
        地面变量保持 (lat, lon)，`build_input` 会自动推断单位、对齐层级。
        """
        # 将输入的时间转换为pandas的datetime对象，如果是字符串则按指定格式转换
        t = pd.to_datetime(time, format="%Y%m%d%H") if isinstance(time, str) \
            else pd.to_datetime(time)
        # 获取年份和日期字符串，用于构建文件路径
        year = t.strftime("%Y")
        date = t.strftime("%Y%m%d")

        data_vars = {}
        lat = lon = levels = None

        for var in self.variables:
            entry = self._read_var_file(var, year, date)
            arr = entry["arr"]
            if lat is None:
                lat, lon = entry["lat"], entry["lon"]
            if var in self.pressure_vars:
                levels = [_parse_name(n)[1] for n in entry["channel"]]   # [1000,925,...,50]
                data_vars[var] = (("level", "lat", "lon"), arr, {"units": entry["units"]})
            else:
                data_vars[var] = (("lat", "lon"), arr, {"units": entry["units"]})

        return xr.Dataset(data_vars, coords={"lat": lat, "lon": lon,
                                             "level": levels, "time": t})
