# -*- coding: utf-8 -*-
"""第①步：把数据变成模型输入。

你只需要提供一个「有 `load(time) -> xr.Dataset` 方法」的对象（缺省用 ERA 数据源
`EraDataLoader`），剩下的事情这里全做了：单位换算、层级/网格适配、翻转滚动，
最后输出模型要的 (1, 历史步数, 通道数, lat, lon) float32。

**模型的通道排列、单位、物理量程全部在 spec JSON 里**（路径由调用方传入），
不写死在代码里 —— 换一个模型就换一份 JSON，代码不用动。本文件只做"规则引擎"：
读 JSON，拿它的规则做单位换算、对齐层级、装配成张量。

单位换算**不猜**：写死单位的 loader（如 era5_store）直接在 `SCALE` 表里给出每个
通道的换算系数（q×1000、辐射×1/3600、tp×1000，其余 ×1，无 offset）；没有 `SCALE`
的 loader（era 等，单位来自文件 attrs）才回退 spec 的 accepts 查表。都不做数值量程猜测。
只做单位换算、纬度翻转、经度滚动；分辨率对不上直接报错（不做插值）。
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# 直接 `python adapters/build_input.py` 跑时，脚本目录是 adapters/、仓库根目录不在
# sys.path，下面的 `from loaders.era import ...` 会 ImportError。作为脚本运行时把
# 根目录补进 sys.path；作为包模块被 runner.py 导入时 __package__ 非空，这行跳过。
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.era import EraDataLoader


# ---------------------------------------------------------------------------
# 坐标/通道名解析（原 naming.py 并入本文件，去掉只有几行的独立模块）
# ---------------------------------------------------------------------------
LAT_NAMES = ["lat", "latitude", "y"]
LON_NAMES = ["lon", "longitude", "x"]
LEVEL_NAMES = ["level", "isobaricInhPa", "plev", "pressure_level", "lev"]
CHANNEL_NAMES = ["channel", "variable", "var"]


def _level_hpa(raw):
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return int(round(v / 100.0)) if v > 2000 else int(round(v))


def _parse_name(name):
    """通道名/变量名 -> (var, level 或 None)。如 'z1000' -> ('z', 1000)，'t2m' -> ('t2m', None)。"""
    key = str(name).strip()
    m = re.fullmatch(r"([a-zA-Z]+?)(\d+)", key)
    if m:
        return m.group(1).lower(), int(m.group(2))
    return key.lower(), None


# ---------------------------------------------------------------------------
# 单位名称规范化（字符串层面的，和具体模型无关）
# ---------------------------------------------------------------------------
def _normalize_unit(text):
    """把单位字符串规范化：小写、去 **/^/_、压空白，便于静态查表时精确比较。"""
    if text is None:
        return None
    s = str(text).strip().lower().replace("**", "").replace("^", "").replace("_", " ")
    return re.sub(r"\s+", " ", s)


# ---------------------------------------------------------------------------
# spec 加载与通道展开
# ---------------------------------------------------------------------------
def load_spec(path):
    """读 spec JSON，展开成运行时结构。返回 spec dict（已附加 _channels 等）。"""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)

    levels = spec.get("levels", [])
    channels = []            # 顺序即模型通道顺序
    parse = {}               # 通道名 -> (var, level 或 None)
    all_vars = set()
    for group in spec["layout"]:
        glevels = levels if group.get("levels") == "@levels" else None
        for var in group["vars"]:
            if glevels:
                for lv in glevels:
                    name = f"{var}{lv}"
                    channels.append(name)
                    parse[name] = (var, lv)
            else:
                channels.append(var)
                parse[var] = (var, None)
            all_vars.add(var)

    spec["_channels"] = channels
    spec["_parse"] = parse
    spec["_n_channels"] = len(channels)
    spec["_all_vars"] = all_vars
    return spec


# ---------------------------------------------------------------------------
# 单位换算（静态查表，不做数值猜测）
# ---------------------------------------------------------------------------
def _resolve_conversion(spec, var, source_unit):
    """源单位 → 模型单位的换算系数，查 spec 的 unit/accepts 声明，返回 (scale, offset)。

    source_unit 为 None 或等于 spec 的 unit（规范单位）时，直接 (1.0, 0.0)；
    否则必须命中 accepts 里声明的某个源单位，否则报错（说明数据源单位或 spec 有误）。
    单位名只做字符串规范化（_normalize_unit），不做任何数值量程猜测。
    """
    v = spec["variables"][var]
    target = v["unit"]
    su = _normalize_unit(source_unit)
    if su is None or su == _normalize_unit(target):
        return 1.0, 0.0
    for au, d in v.get("accepts", {}).items():
        if _normalize_unit(au) == su:
            return d.get("scale", 1.0), d.get("offset", 0.0)
    raise ValueError(
        f"通道 {var} 源单位 {source_unit!r} 既不是规范单位 {target!r}，"
        f"也不在 accepts {list(v.get('accepts', {}))} 里；"
        f"请检查 loader 的 SCALE 表或数据 attrs 单位、spec 的 accepts 声明")


# EraDataLoader 已移到 loaders/era.py（见本文件顶部 import）。


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------
def _find_coord(obj, names):
    coords = getattr(obj, "coords", {})
    for n in names:
        if n in coords:
            return n
    dims = set(getattr(obj, "dims", []))
    for n in names:
        if n in dims:
            return n
    return None


def _to_2d(da, channel_index=None, level_index=None):
    """把一个 DataArray 压成 2D (lat, lon)。"""
    values = da.values
    idx = []
    for dim in da.dims:
        if dim in CHANNEL_NAMES and channel_index is not None:
            idx.append(channel_index)
        elif dim in LEVEL_NAMES and level_index is not None:
            idx.append(level_index)
        elif dim in LAT_NAMES or dim in LON_NAMES:
            idx.append(slice(None))
        else:
            idx.append(-1)          # 时间等多余维取最后一帧
    return np.squeeze(np.asarray(values[tuple(idx)], dtype=np.float64))


def _inventory(ds):
    """把 Dataset 里所有字段扫成 {(var, level): 2D数组}，并记录单位。"""
    fields = {}
    units = {}

    # 1) 打包的 channel 变量（fuxiens input.nc 这种）
    for name in ds.data_vars:
        da = ds[name]
        cc = _find_coord(da, CHANNEL_NAMES)
        if cc is None:
            continue
        names = [str(x) for x in np.atleast_1d(np.asarray(da.coords[cc].values))]
        for i, ch in enumerate(names):
            var, level = _parse_name(ch)
            fields[(var, level)] = _to_2d(da, channel_index=i)
            units[(var, level)] = da.attrs.get("units")
        return fields, units

    # 2) 逐变量（ERA5 这种，pressure 变量带 level 维）
    for name in ds.data_vars:
        da = ds[name]
        var, name_level = _parse_name(name)
        lc = _find_coord(da, LEVEL_NAMES)
        raw_units = da.attrs.get("units")
        if lc is not None:
            vals = np.atleast_1d(np.asarray(da.coords[lc].values))
            for i, raw in enumerate(vals):
                level = _level_hpa(raw)
                if level is None:
                    continue
                fields[(var, level)] = _to_2d(da, level_index=i)
                units[(var, level)] = raw_units
        else:
            fields[(var, name_level)] = _to_2d(da)
            units[(var, name_level)] = raw_units
    return fields, units


def grid_coords(spec):
    """从 spec["grid"] 生成 lat / lon 坐标数组（模型要求的顺序由 spec 决定）。"""
    glat = spec["grid"]["lat"]
    glon = spec["grid"]["lon"]
    lat = np.arange(glat["size"], dtype=np.float64) * glat["step"] + glat["start"]
    lon = np.arange(glon["size"], dtype=np.float64) * glon["step"] + glon["start"]
    return lat, lon


def _grid_shape(spec):
    """返回模型要求的网格 (nlat, nlon)，从 spec 读，不写死。"""
    return spec["grid"]["lat"]["size"], spec["grid"]["lon"]["size"]


def _geometry(ds, nlat, nlon):
    """返回 (lat翻转?, lon滚动量)，模型要求 lat 北→南、lon 0→360。"""
    lat_name = _find_coord(ds, LAT_NAMES)
    lon_name = _find_coord(ds, LON_NAMES)
    if lat_name is None or lon_name is None:
        raise ValueError("数据里找不到 lat / lon 坐标")
    lat = np.asarray(ds[lat_name].values, dtype=np.float64).ravel()
    lon = np.asarray(ds[lon_name].values, dtype=np.float64).ravel()

    flip = bool(lat.size > 1 and lat[0] < lat[-1])          # 南→北 需要翻转

    if lon.size != nlon or lat.size != nlat:
        raise ValueError(f"网格 {lat.size}x{lon.size} 不是 {nlat}x{nlon}，需要先插值到该分辨率")
    roll = 0
    dlon = 360.0 / nlon
    if not np.allclose(lon % 360.0, (np.arange(nlon) * dlon) % 360.0, atol=1e-4):
        diff = np.abs((lon % 360.0 - 0.0 + 180.0) % 360.0 - 180.0)
        roll = int(np.argmin(diff))
    return flip, roll


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_input(init_time, spec, history_steps=None, hour_interval=None,
                loader=None, verbose=False):
    """构建模型输入 (1, history_steps, N_channel, nlat, nlon) float32。

    init_time 支持 'YYYYMMDDHH' 字符串或 datetime。
    spec 必须由调用方传入（load_spec 展开过的 dict）。
    loader 是「有 load(time) -> xr.Dataset 方法」的对象；不传则用 ERA 数据源。
    多个起报时间复用同一个 loader 时能命中它的读文件缓存。
    history_steps / hour_interval 不传则用 spec 里的 model 字段。
    """
    def log(*a):
        if verbose:
            print(*a)

    if history_steps is None:
        history_steps = spec["model"].get("history_steps", 2)
    if hour_interval is None:
        hour_interval = spec["model"].get("hour_interval", 6)
    if loader is None:
        loader = EraDataLoader(spec)

    nlat, nlon = _grid_shape(spec)

    init = pd.to_datetime(init_time, format="%Y%m%d%H") if isinstance(init_time, str) \
        else pd.to_datetime(init_time)
    times = [init - pd.Timedelta(hours=(history_steps - 1 - i) * hour_interval)
             for i in range(history_steps)]

    log(f"起报时间 {init} | 历史 {history_steps} 帧 | 步长 {hour_interval}h")
    datasets = []
    for t in times:
        ds = loader.load(t)
        datasets.append(ds)
        log(f"  load({t:%Y-%m-%d %H:00}) -> {len(ds.data_vars)} 个变量")

    flip, roll = _geometry(datasets[-1], nlat, nlon)
    log(f"网格: lat 翻转={'是' if flip else '否'}, lon 滚动={roll}")

    # 每个通道的换算系数：loader 写死 SCALE（era5_store）或回退 accepts 查表（era），
    # 不做数值猜测。数据源可能带 spec 用不到的通道（era5_store 的 w/10hPa/土壤/波浪），
    # 只对 spec 声明的通道换算，其余字段直接跳过。
    last_fields, units = _inventory(datasets[-1])
    log(f"字段数: {len(last_fields)} (spec 需要 {spec['_n_channels']})")
    needed = set(spec["_parse"].values())
    # 换算系数：era5_store 等「已写死单位」的 loader 直接用它的 SCALE 表（源→模型单位，
    # 无 offset）；没有 SCALE 的 loader（era 等，单位来自文件 attrs）回退 accepts 查表。
    scale_map = getattr(loader, "SCALE", None)
    conversion = {}
    n_ok = n_convert = 0
    for (var, level) in sorted(last_fields):
        if (var, level) not in needed:
            continue
        if scale_map is not None:
            s, o = scale_map.get(var, 1.0), 0.0
        else:
            s, o = _resolve_conversion(spec, var, units.get((var, level)))
        conversion[(var, level)] = (s, o)
        if s != 1.0 or o != 0.0:
            n_convert += 1
            log(f"  换算 {var}{level or '':<4}: ×{s}" + (f" +{o}" if o else ""))
        else:
            n_ok += 1
    log(f"单位换算: {n_ok} 个原样通过, {n_convert} 个需要换算")

    # 换算系数已拿到，units 这张元信息表不再需要，先释放再装配输出。
    # last_fields 还要在装配时复用（最后一帧不再重复 _inventory）。
    del units

    out = np.empty((history_steps, spec["_n_channels"], nlat, nlon), dtype=np.float32)
    nan_filled = {}
    missing = set()
    for ti, ds in enumerate(datasets):
        # 最后一帧的字段表上面已算过，直接复用，省一次 _inventory
        f = last_fields if ti == len(datasets) - 1 else _inventory(ds)[0]
        for ci, channel in enumerate(spec["_channels"]):
            var, level = spec["_parse"][channel]
            arr = f.get((var, level))
            if arr is None:
                missing.add(channel)
                continue
            scale, offset = conversion[(var, level)]
            arr = arr * scale + offset
            # fuxiens fuxi_ens / fuxi2.1 的 input.nc 里，sst 通道陆地就是 NaN（共 703752 个，
            # 且是唯一含 NaN 的通道），模型自带 land-sea mask 门控，官方不填充、直接喂 NaN。
            # 这里保持一致：不填、不替换，NaN 原样透传。
            bad = ~np.isfinite(arr)
            if bad.any():
                nan_filled[(var, level)] = nan_filled.get((var, level), 0) + int(bad.sum())
            if flip:
                arr = arr[::-1, :]
            if roll:
                arr = np.roll(arr, roll, axis=-1)
            out[ti, ci] = arr

    if missing:
        raise ValueError(f"数据里找不到 {len(missing)} 个通道: "
                         f"{', '.join(sorted(missing, key=spec['_channels'].index))}")

    if nan_filled:
        detail = ", ".join(f"{var}{level or ''}:{n}" for (var, level), n in sorted(nan_filled.items()))
        log(f"[build_input] NaN 保留未填充（与官方一致）-> {detail}")

    result = out[np.newaxis, ...]     # (1, history_steps, N_channel, nlat, nlon)
    log(f"输出 shape: {result.shape} dtype={result.dtype}")
    finite = np.isfinite(result)
    log(f"数值范围: min={result[finite].min():.6g}  max={result[finite].max():.6g}  "
        f"NaN={int((~finite).sum())}")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("用法: python build_input.py <spec.json> [起报时间] [历史帧数] [步长小时]")
    spec = load_spec(sys.argv[1])
    init = sys.argv[2] if len(sys.argv) > 2 else "2024010200"
    history = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    interval = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    x = build_input(init, spec=spec, history_steps=history,
                    hour_interval=interval, verbose=True)
    print("\n构建成功 ✓")
    print("用法: python build_input.py <spec.json> [起报时间] [历史帧数] [步长小时]")
    print(f"例如: python build_input.py {sys.argv[1]} {init} {history} {interval}")
