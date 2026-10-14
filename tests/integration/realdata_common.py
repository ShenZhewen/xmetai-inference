# -*- coding: utf-8 -*-
"""realdata 集成档的共享原语：读取、对齐、分档、对拍。

不带 ``test_`` 前缀、不 import pytest：import 任何 ``test_`` 模块都会触发它的
模块级 fixture 自检。分档阈值从老 ``util/compare_outputs.py`` 原样冻结，**不要改**。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: rtol 容忍 float32 往返与代数化简的舍入；atol 给大片为 0 的场（tp）兜底。
RTOL = 1e-5
ATOL = 1e-6
EXCEED_FRACTION_CLOSE = 1e-3      # 万分之一以下：随机舍入
EXCEED_FRACTION_DRIFT = 0.5       # 半数以上：多数点都变了
DRIFT_STD_FRACTION = 1e-2         # max|diff| 超场自身变率这个比例 ⇒ 物理上可分辨

VERDICT_RANK = {"IDENTICAL": 0, "CLOSE": 1, "DRIFT": 2, "DIFFER": 3}

_INIT_RE = re.compile(r"^\d{8}(?:\d{2})?$")
_STEP_RE = re.compile(r"^(\d{3})\.nc$")


def init_dirs(root: Path) -> dict:
    """子目录名 -> 起报时刻。兼容 YYYYMMDD 与 YYYYMMDDHH 两种命名。"""
    found = {}
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        if _INIT_RE.fullmatch(entry.name):
            fmt = "%Y%m%d" if len(entry.name) == 8 else "%Y%m%d%H"
            found[entry.name] = pd.to_datetime(entry.name, format=fmt)
    return found


def steps_in(directory: Path) -> list:
    """目录里的 ``NNN.nc`` 步序号。"""
    return sorted(
        int(match.group(1))
        for path in directory.iterdir()
        if path.is_file() and (match := _STEP_RE.fullmatch(path.name))
    )


def variables_in(path: Path) -> set:
    """NetCDF 里的二维变量名（小写）。"""
    with xr.open_dataset(path) as ds:
        return {str(name).lower() for name in ds.data_vars if ds[name].ndim == 2}


def align_grid(values, lat, lon):
    """把场摆成「北→南、经度 0..360 升序」。shape 与坐标数不符直接报错。"""
    values = np.asarray(values, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)
    if values.shape[-2:] != (lat.size, lon.size):
        raise ValueError(
            f"数据 shape {values.shape[-2:]} 与坐标 {(lat.size, lon.size)} 不一致")
    if lat.size > 1 and lat[0] < lat[-1]:
        lat = lat[::-1]
        values = np.flip(values, axis=-2)
    lon360 = np.mod(lon, 360.0)
    order = np.argsort(lon360)
    return np.take(values, order, axis=-1), lat, lon360[order]


def load_fields(path: Path, variables):
    """读一个预测文件里的二维场 -> (``{变量: (H,W)}``, lat, lon)。变量名大小写不敏感。"""
    with xr.open_dataset(path) as ds:
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError(f"{path} 缺少 lat/lon 坐标")
        lower = {str(name).lower(): name for name in ds.data_vars}
        fields, lat, lon = {}, None, None
        for variable in variables:
            if variable not in lower:
                raise KeyError(f"{path} 缺少变量 {variable!r}")
            values = np.asarray(ds[lower[variable]].values, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"{path} 的 {variable} 不是二维规则网格")
            values, lat, lon = align_grid(values, ds["lat"].values, ds["lon"].values)
            fields[variable] = values
        return fields, lat, lon


def classify(max_diff, ref_std, exceed_fraction):
    """给一档判定。主判据是超容差点**占比**而非 max|diff|；max|diff| 超场自身空间
    变率 1%（``DRIFT_STD_FRACTION``）时无论占比多少直接 DIFFER。
    """
    if max_diff == 0.0:
        return "IDENTICAL"
    if ref_std > 0 and max_diff > DRIFT_STD_FRACTION * ref_std:
        return "DIFFER"
    if exceed_fraction < EXCEED_FRACTION_CLOSE:
        return "CLOSE"
    if exceed_fraction < EXCEED_FRACTION_DRIFT:
        return "DRIFT"
    return "DIFFER"


def field_metrics(a, b, rtol=RTOL, atol=ATOL):
    """两个 (H,W) 场 -> 指标 dict。容差 ``|a-b| <= atol + rtol*|b|``；NaN 位置不一致
    单独计数，一侧缺测的格子不进超容差统计。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    nan_mismatch = int(np.count_nonzero(finite_a != finite_b))
    both = finite_a & finite_b
    if not both.any():
        return {
            "verdict": "DIFFER",
            "max_abs_diff": float("nan"),
            "exceed_frac": float("nan"),
            "diff_over_std": float("nan"),
            "nan_mismatch": nan_mismatch,
        }

    diff = np.abs(a - b)[both]
    max_diff = float(diff.max())
    ref_std = float(b[both].std())  # 标准答案自身的空间变率
    exceed_frac = float(np.count_nonzero(diff > atol + rtol * np.abs(b[both]))
                        / diff.size)
    return {
        "verdict": classify(max_diff, ref_std, exceed_frac),
        "max_abs_diff": max_diff,
        "exceed_frac": exceed_frac,
        "diff_over_std": max_diff / ref_std if ref_std > 0 else float("nan"),
        "nan_mismatch": nan_mismatch,
    }


def compare_dirs(root_a: Path, root_b: Path, step_limit=None,
                 rtol=RTOL, atol=ATOL):
    """两份输出根目录 -> ``(rows, skipped)``。确定性档（无成员维度）专用。

    ``rows`` 每个 (起报, step, 变量) 一条；``skipped`` 记只在一侧存在的
    ``(级别, 什么, 为什么)`` —— 覆盖面断言靠它抓「产物少写一步反而更容易绿」。
    """
    inits_a, inits_b = init_dirs(root_a), init_dirs(root_b)
    if not inits_a:
        raise ValueError(f"{root_a} 下没有起报目录")
    if not inits_b:
        raise ValueError(f"{root_b} 下没有起报目录")

    skipped = [("起报", name, "只在 A 侧") for name in sorted(set(inits_a) - set(inits_b))]
    skipped += [("起报", name, "只在 B 侧") for name in sorted(set(inits_b) - set(inits_a))]

    common_inits = sorted(set(inits_a) & set(inits_b))
    if not common_inits:
        raise ValueError(
            f"两侧没有共同的起报目录。\n  A: {sorted(inits_a)}\n  B: {sorted(inits_b)}")

    rows = []
    for init_dir in common_inits:
        dir_a, dir_b = root_a / init_dir, root_b / init_dir
        steps_a, steps_b = set(steps_in(dir_a)), set(steps_in(dir_b))
        for step in sorted(steps_a - steps_b):
            skipped.append(("步", f"{init_dir}/{step:03d}", "只在 A 侧"))
        for step in sorted(steps_b - steps_a):
            skipped.append(("步", f"{init_dir}/{step:03d}", "只在 B 侧"))

        steps = sorted(steps_a & steps_b)
        if step_limit is not None:
            steps = [s for s in steps if s <= step_limit]

        names = None
        for step in steps:
            path_a = dir_a / f"{step:03d}.nc"
            path_b = dir_b / f"{step:03d}.nc"
            if names is None:
                vars_a, vars_b = variables_in(path_a), variables_in(path_b)
                for name in sorted(vars_a ^ vars_b):
                    skipped.append(("变量", name, "只在一侧"))
                names = sorted(vars_a & vars_b)
                if not names:
                    continue

            fields_a, lat_a, lon_a = load_fields(path_a, names)
            fields_b, lat_b, lon_b = load_fields(path_b, names)
            # 网格对不上说明网格本身变了，逐格相减只有假「漂移」。
            if lat_a.shape != lat_b.shape or lon_a.shape != lon_b.shape:
                raise ValueError(f"网格 shape 不一致：{path_a} 与 {path_b}")
            if not (np.allclose(lat_a, lat_b, atol=1e-4)
                    and np.allclose(lon_a, lon_b, atol=1e-4)):
                raise ValueError(f"网格坐标不一致：{path_a} 与 {path_b}")

            for name in names:
                rows.append({
                    "init": init_dir,
                    "step": step,
                    "var": name,
                    **field_metrics(fields_a[name], fields_b[name], rtol, atol),
                })
    return rows, skipped


def expected_vars(config_path: Path) -> set:
    """配方声明要落盘的变量集合（小写）。从配方读，不在测试里再抄一份。"""
    from xmetai_inference.configs.base import load_config

    cfg = load_config(str(config_path))
    return {v.strip().lower() for v in str(cfg.vars).split(",") if v.strip()}
