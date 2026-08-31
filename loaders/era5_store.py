# -*- coding: utf-8 -*-
"""era5_foundation_store 数据源 loader。

新的 ERA5「基础库」把数据按物理分组拆成多个 zarr store，放在同一个根目录下：

    era5_pl_2025.01-2025.03.c84.p25.h6.zarr      84 通道（z/t/u/v/w/q × 14 层）
    era5_sfc_2025.01-2025.03.c15.p25.h6.zarr     15 通道（地面）
    era5_cldrad_2025.01-2025.03.c8.p25.h6.zarr    8 通道（云/辐射）
    era5_soil_...                                 4 通道（土壤，已 regrid 到 0.25°）
    era5_wave_...                                 4 通道（波浪，部分通道有缺失）
    era5_static.c4.p25.zarr                       4 通道（静态场，无 time 维）

每个 store 内部统一：data(time, channel, lat, lon) float32；channel 坐标存通道名；
lat 北→南（90→-90）；lon -180→180；time 已按 CF 解码成 datetime64[ns]、6 小时一步。

本 loader **不认识模型**，只做「把数据源读成自描述 Dataset」这一件事：
  1. 找到各组 store 并按需打开（默认 pl/sfc/cldrad，groups 参数可扩展）；
  2. 通道名归一化：store 里气压层叫 z_50 / t_1000，转成 build_input 认识的 z50；
  3. 多组沿 channel 维 concat 成一个 data 变量，**组内通道全部保留，不筛选**；
  4. 丢掉 group 级 GRIB units（只描述该组首个通道，对多通道无意义）。

哪些通道、什么顺序、什么单位，全部由 spec 决定，build_input 负责挑。加新模型
只需新 spec，loader 不用动（w/10hPa 层/土壤/波浪/静态这些 fuxi 用不到的通道，
读出来给 build_input 丢弃即可；若想省 IO，可自行用 groups 缩小范围）。

与 ZarrDataLoader 一样，只实现 load(time) -> xr.Dataset 一个接口。静态组
（无 time 维）暂不纳入；若未来模型要 lsm/orog 等静态输入，需单独机制（它们
不随时间变化），不在本 loader 职责内。
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_GROUPS = ("pl", "sfc", "cldrad")

# 默认根目录：地址写死在这里，日常 `--loader era5_store` 一条命令就能跑；
# 需临时换目录时用 ERA5_STORE_ROOT 环境变量覆盖，不必改代码。
DEFAULT_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

# ERA5 累积场：本 store 把这些场按「1h 累积」采样，但 time step 是 6h；读取时 ×6
# 归一化成每步(6h)累积。哪些场是累积场是 ERA5 数据的固有约定，与具体模型无关；
# 物理单位换算（J→Wh/m²、m→mm）仍由各 spec 的 accepts 负责，不在这里做。
_ACCUMULATED_1H = {"ssr", "ssrd", "fdir", "ttr", "tp"}


def _normalize_channel(name):
    """store 通道名 -> build_input 认识的通道名：z_50 -> z50；地面名原样保留。"""
    m = re.fullmatch(r"([a-zA-Z]+)_(\d+)", name)
    if m:
        return m.group(1) + m.group(2)
    return name


class Era5StoreLoader:
    """从 era5_foundation_store 根目录读输入状态（模型无关，读整组通道）。"""

    def __init__(self, root=None, groups=DEFAULT_GROUPS, data_var="data", tol="12h"):
        self.root = root or os.environ.get("ERA5_STORE_ROOT", DEFAULT_ROOT)
        self.groups = list(groups)
        self.data_var = data_var
        self.tol = pd.Timedelta(tol)
        self._ds = None          # 惰性打开的、沿 channel 合并后的 Dataset

    def _store_path(self, group):
        matches = sorted(glob.glob(os.path.join(self.root, f"era5_{group}*.zarr")))
        if not matches:
            return None
        if len(matches) > 1:
            print(f"[era5_store] group {group!r} 匹配到多个 store，取 "
                  f"{os.path.basename(matches[0])}（其余："
                  f"{', '.join(os.path.basename(m) for m in matches[1:])}）")
        return matches[0]

    def _open_group(self, group):
        path = self._store_path(group)
        if path is None:
            return None
        try:
            ds = xr.open_zarr(path, consolidated=True)
        except Exception:
            ds = xr.open_zarr(path)
        if self.data_var not in ds.data_vars:
            raise KeyError(f"{path} 里没有变量 {self.data_var!r}，现有: {list(ds.data_vars)}")
        if "time" not in ds.coords:
            raise KeyError(f"{path} 里没有 time 坐标（静态组不应传给本 loader）")
        if not pd.api.types.is_datetime64_any_dtype(ds.time):
            raise ValueError(f"{path} 的 time 坐标未被解码为日期（dtype={ds.time.dtype}）")
        return ds

    def _normalize_channels(self, ds):
        """归一化通道名（z_50 -> z50），保留组内全部通道，不按 spec 筛选。"""
        raw = [str(x) for x in np.atleast_1d(np.asarray(ds["channel"].values))]
        return ds.assign_coords(channel=[_normalize_channel(r) for r in raw])

    def _open(self):
        if self._ds is not None:
            return self._ds
        parts = []
        for g in self.groups:
            ds = self._open_group(g)
            if ds is not None:
                parts.append(self._normalize_channels(ds))
        if not parts:
            raise ValueError(f"{self.root} 下没有可用的 era5_* store（group={self.groups}）")
        merged = xr.concat(parts, dim="channel")
        # group 级 GRIB units 只描述该组首个通道，对多通道没有意义；清掉，让
        # build_input 按 spec 量程做数值单位推断（否则 z 的 m2/s2 会污染 t/q/u/v…）
        data = merged[self.data_var].copy()
        data.attrs = {}
        # 辐射/降水是 1h 累积、time step 是 6h：×6 归一化成每步(6h)累积（见 _ACCUMULATED_1H）
        chan = np.atleast_1d(np.asarray(merged["channel"].values))
        mask = np.array([str(c) in _ACCUMULATED_1H for c in chan])
        if mask.any():
            scale = xr.DataArray(np.where(mask, 6.0, 1.0).astype(np.float32),
                                 dims="channel", coords={"channel": merged["channel"]})
            data = data * scale
        self._ds = merged.assign({self.data_var: data})
        return self._ds

    def load(self, time):
        """输入时刻，返回该时刻的 xr.Dataset（dims: channel, lat, lon）。"""
        t = pd.to_datetime(time, format="%Y%m%d%H") if isinstance(time, str) \
            else pd.to_datetime(time)

        frame = self._open().sel(time=t, method="nearest")

        # 距离检查：method="nearest" 会掩盖时间分辨率不匹配，这里显式校验
        actual = pd.Timestamp(frame.time.item())
        if abs(actual - t) > self.tol:
            raise ValueError(
                f"era5_store 里最接近 {t} 的时刻是 {actual}（差 {actual - t}），"
                f"超过容差 {self.tol}；检查数据时间分辨率是否与 spec 的 hour_interval 匹配")

        return frame.drop_vars("time", errors="ignore").load()
