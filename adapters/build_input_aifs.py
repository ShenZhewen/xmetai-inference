# -*- coding: utf-8 -*-
"""把 era5_store(0.25°) 读成 AIFS 1.1 的输入 field 字典（N320 缩减高斯网格）。

AIFS 不走 build_input 的张量装配：它是「命名 field 字典 + N320 非结构化节点」，
输入必须**预先插值到 N320**（earthkit-regrid），再由 SimpleRunner 喂给模型。
所以这里单独一条通路，严格照抄官方 run_AIFS_v1.1.ipynb 的做法：

  1. loader 读 pl/sfc/soil（时间维），static（lsm/z_sfc/slor/sdor）单独读（无时间维）；
  2. 逐场把 store 通道名映射成 checkpoint 里的变量名（z50→z_500、t2m→2t、sot1→stl1、z_sfc→z）；
  3. lon 从 -180:180 滚到 0:360（earthkit 约定），lat 保持北→南；
  4. 每个场叠成 (2, …)（[t-6h, t0] 两帧，时间正序）；
  5. 最后（do_interp=True 时）插值 0.25°→N320。

关键正确性（务必守住）：
  * 字段名与 checkpoint 变量名**一字不差**，共 94 个：78 个气压层(z/t/u/v/w/q×13)
    + 8 地面(sp/msl/skt/2t/2d/10u/10v/tcw) + 4 土壤(stl1/stl2/swvl1/swvl2)
    + 4 静态(lsm/z/slor/sdor，是模型要的**常数强迫**，SimpleRunner 不会自己算)；
  * era5_store 存的是 ERA5 原生单位，AIFS 也训在 ERA5 上，所以**不做任何单位换算**：
    z 位势 m²/s²、q 比湿 kg/kg、t/skt/2t/2d/stl K、u/v/w m/s、msl/sp Pa、tcw kg/m²、
    swvl m³/m³、z_sfc 位势 m²/s²。这里只做**量级自检**，量级不对直接报错，不偷偷换算；
  * 量级自检在**插值前的 0.25° 上**做（与 N320 量级一致），因此 do_interp=False 时
    不依赖 earthkit 也能验证单位 —— 无 GPU / 无 earthkit 的机器也能先验 90% 的正确性；
  * 土壤 swvl/stl 海上是 NaN，官方模型内部 InputImputer 会填（stl→均值、swvl→最小值），
    这里**不填**，NaN 原样透传（与官方一致）；
  * 9 个 computed 强迫（cos_lat/sin_lat/cos_lon/sin_lon/cos_julian_day/sin_julian_day/
    cos_local_time/sin_local_time/insolation）由 anemoi 内部算，这里不管。

用法：
  无卡校验（只验输入装配/单位/网格，不加载模型）：
    python -c "from build_input_aifs import build_aifs_fields; build_aifs_fields('2025010600', do_interp=False)"
  完整输入（含 N320 插值，给 aifs_minimal.py / infer_aifs.py 用）：
    build_aifs_fields('2025010600', do_interp=True)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

# 直接 `python adapters/build_input_aifs.py` 跑时，脚本目录是 adapters/、仓库根目录
# 不在 sys.path，下面的 `from loaders.era5_store import ...` 会 ImportError。作为脚本
# 运行时把根目录补进 sys.path；作为包模块被 runner.py 导入时 __package__ 非空，跳过。
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.era5_store import Era5StoreLoader


# ---------------------------------------------------------------------------
# spec 加载 + 字段映射生成
# ---------------------------------------------------------------------------
def load_spec(path="specs/aifs11.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_mapping(spec):
    """全量 (store 通道名, AIFS 字段名, 是否静态) 有序列表，共 94 项。

    PL:  z50 -> z_50, ...   t50 -> t_50, ...   (var{level} -> var_{level})
    地面/土壤/静态按 spec 的显式映射表。
    """
    mapping = []
    for var in spec["pl_vars"]:
        for lv in spec["levels"]:
            mapping.append((f"{var}{lv}", f"{var}_{lv}", False))
    for store, aifs in spec["surface"].items():
        mapping.append((store, aifs, False))
    for store, aifs in spec["soil"].items():
        mapping.append((store, aifs, False))
    for store, aifs in spec["static"].items():
        mapping.append((store, aifs, True))
    return mapping


def self_check(spec):
    """字段映射自检：94 场、名字无重复、关键字段齐全。不碰数据，本地可跑。"""
    mapping = _build_mapping(spec)
    names = [a for _, a, _ in mapping]
    assert len(mapping) == 94, f"期望 94 个输入场，得到 {len(mapping)}"
    assert len(set(names)) == len(names), "字段名重复"
    for must in ("z_500", "z_1000", "t_850", "u_50", "v_850", "w_500", "q_1000",
                 "sp", "msl", "skt", "2t", "2d", "10u", "10v", "tcw",
                 "stl1", "stl2", "swvl1", "swvl2", "lsm", "z", "slor", "sdor"):
        assert must in names, f"缺少字段 {must}"
    print(f"[aifs] 字段映射自检通过：{len(mapping)} 个输入场")
    return mapping


# ---------------------------------------------------------------------------
# 网格 → N320 插值
# ---------------------------------------------------------------------------
def _align_025(arr2d, lat, lon):
    """统一到「北→南、0:360」，仍在 0.25°(721×1440)。"""
    arr = np.asarray(arr2d, dtype=np.float64)
    if lat.size > 1 and lat[0] < lat[-1]:          # 南→北，翻到北→南
        arr = arr[::-1, :]
    if lon.size > 1 and lon[0] < 0:                # -180:180 → 0:360
        arr = np.roll(arr, -arr.shape[1] // 2, axis=1)
    return arr


def _interp_n320(arr2d):
    """(721,1440) 北→南 / 0:360 → N320 一维节点数组（照抄 notebook 的 earthkit 调用）。"""
    import earthkit.regrid as ekr
    return np.asarray(
        ekr.interpolate(arr2d, {"grid": (0.25, 0.25)}, {"grid": "N320"}), dtype=np.float64
    )


# ---------------------------------------------------------------------------
# 量级自检（单位不对直接报错，不偷偷换算）
# ---------------------------------------------------------------------------
def _check_magnitude(name, arr):
    v = np.asarray(arr, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    med = float(np.median(v))
    mx = float(np.max(v))

    if name == "z_500":
        # 500hPa 位势 ~54000 m²/s²（specs/fuxi21.json 量程 43149~60311）；位势高度才 ~5500 m
        if med < 20000:
            raise ValueError(
                f"[单位错误] z_500 中位数 {med:.0f}，像是位势高度(m)不是位势(m²/s²)。"
                f"AIFS 要 m²/s²；若数据源是 gpm 请 ×9.80665（官方 gh→z 正是 ×9.80665）。")
    elif name == "z":
        # 地形位势 z_sfc ~ 0..7.8e4 m²/s²；若是高度(m)只有 ~0..8e3
        if mx < 20000:
            raise ValueError(
                f"[单位错误] 地形位势 z_sfc 最大值 {mx:.0f}，像是高度(m)不是位势(m²/s²)。"
                f"AIFS 要 m²/s²。")
    elif name == "q_850":
        # 比湿 kg/kg：850hPa 中位数 ~5e-3；若是 g/kg 会大 1000 倍
        if med > 0.05:
            raise ValueError(
                f"[单位错误] q_850 中位数 {med:.4f}，像是 g/kg 不是 kg/kg。"
                f"AIFS 要 kg/kg（ERA5 原生，无需换算）。")


# ---------------------------------------------------------------------------
# 静态 store 读取
# ---------------------------------------------------------------------------
def _read_static(root, data_var="data"):
    matches = sorted(glob.glob(os.path.join(root, "era5_static*.zarr")))
    if not matches:
        raise FileNotFoundError(f"{root} 下找不到 era5_static*.zarr")
    ds = xr.open_zarr(matches[0], consolidated=True)
    if data_var not in ds.data_vars:
        raise KeyError(f"{matches[0]} 里没有变量 {data_var!r}，现有: {list(ds.data_vars)}")
    return ds


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_aifs_fields(init_time, spec=None, root=None, history_steps=None,
                      hour_interval=None, do_interp=True, verbose=True):
    """构建 AIFS 输入 state，返回 {"date": datetime, "fields": {aifs名: (2, …)}}。

    init_time: 'YYYYMMDDHH' 字符串或 datetime。spec 缺省读 specs/aifs11.json。
    do_interp=True  → 字段为 (2, N320)（喂模型）；
    do_interp=False → 字段为 (2, 721, 1440)，只做装配+量级自检，不依赖 earthkit/GPU。
    """
    if spec is None:
        spec = load_spec()
    if history_steps is None:
        history_steps = spec["model"].get("history_steps", 2)
    if hour_interval is None:
        hour_interval = spec["model"].get("hour_interval", 6)
    mapping = self_check(spec)

    init = pd.to_datetime(init_time, format="%Y%m%d%H") if isinstance(init_time, str) \
        else pd.to_datetime(init_time)
    times = [init - pd.Timedelta(hours=(history_steps - 1 - i) * hour_interval)
             for i in range(history_steps)]          # 时间正序 [t-6h, t0]

    loader = Era5StoreLoader(root=root, groups=("pl", "sfc", "soil"))
    frames = [loader.load(t) for t in times]         # 每个: Dataset(data: channel,lat,lon)
    # 用 loader 解析后的根目录（含 DEFAULT_ROOT 兜底），而不是可能为 None 的 root 参数
    static = _read_static(loader.root)                # Dataset(data: channel,lat,lon)，无 time

    lat = np.asarray(frames[0]["lat"].values, dtype=np.float64).ravel()
    lon = np.asarray(frames[0]["lon"].values, dtype=np.float64).ravel()
    slat = np.asarray(static["lat"].values, dtype=np.float64).ravel()
    slon = np.asarray(static["lon"].values, dtype=np.float64).ravel()

    fields = {}
    ref_shape = None
    for store_name, aifs_name, is_static in mapping:
        if is_static:
            a = np.asarray(static["data"].sel(channel=store_name).values, dtype=np.float64)
            a = _align_025(a, slat, slon)
            a = np.repeat(a[np.newaxis, :], history_steps, axis=0)   # 常数场复制两帧
        else:
            seq = [_align_025(np.asarray(f["data"].sel(channel=store_name).values,
                                         dtype=np.float64), lat, lon)
                   for f in frames]
            a = np.stack(seq, axis=0)                                # (history, lat, lon)
        _check_magnitude(aifs_name, a)                                # 0.25° 上先查量级
        if do_interp:
            a = np.stack([_interp_n320(a[i]) for i in range(history_steps)], axis=0)
        if ref_shape is None:
            ref_shape = a.shape
        # 每个场网格必须一致：do_interp=True 为 (history, N320)，False 为 (history, lat, lon)
        assert a.shape == ref_shape, (aifs_name, a.shape, ref_shape)
        fields[aifs_name] = a.astype(np.float32)

    if verbose:
        _report(fields, ref_shape, do_interp)
    return {"date": init.to_pydatetime(), "fields": fields}


def _report(fields, ref_shape, do_interp):
    grid_desc = (f"N320 节点数 {ref_shape[-1]}" if do_interp
                 else f"0.25° 格点 {ref_shape[1]}×{ref_shape[2]}")
    print(f"[aifs] 输入 {len(fields)} 个场，{grid_desc}")
    keys = ["z_500", "z_1000", "t_850", "u_50", "v_850", "w_500", "q_850", "q_1000",
            "sp", "msl", "skt", "2t", "2d", "10u", "10v", "tcw",
            "stl1", "stl2", "swvl1", "swvl2", "lsm", "z", "slor", "sdor"]
    print(f"  {'field':<8} {'min':>12} {'median':>12} {'max':>12} {'NaN':>8}")
    for k in keys:
        v = np.asarray(fields[k][-1], dtype=np.float64)   # 只看 t0 帧
        fin = v[np.isfinite(v)]
        n_nan = int(v.size - fin.size)
        if fin.size:
            print(f"  {k:<8} {fin.min():>12.4g} {np.median(fin):>12.4g} {fin.max():>12.4g} {n_nan:>8}")
        else:
            print(f"  {k:<8} {'—':>12} {'—':>12} {'—':>12} {n_nan:>8}")


if __name__ == "__main__":
    # 只做字段映射自检（本地可跑，不碰数据、不依赖 earthkit/GPU）
    spec = load_spec("specs/aifs11.json")
    self_check(spec)
