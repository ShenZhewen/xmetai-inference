# -*- coding: utf-8 -*-
"""Shared primitives for the realdata integration suites: loading, alignment,
classification, and comparison.

Deliberately has no ``test_`` prefix and does not import pytest: importing any
``test_`` module triggers its module-level fixture checks. Classification thresholds
are frozen verbatim from the legacy ``util/compare_outputs.py``; do not modify them.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

#: Fixture data root for the realdata suites (placeholder; replace before running integration tests).
DATA_ROOT = "/path/to/data/inference_test_data"

FGVP_FIXTURE_ROOT = Path(DATA_ROOT) / "fgvp_test_data"
FGVP_ENS_FIXTURE_ROOT = Path(DATA_ROOT) / "fgvp_ens_test_data"
FUXI_FIXTURE_ROOT = Path(DATA_ROOT) / "fuxi_test_data"


def data_root_hint(root) -> str:
    """Return a uniform reminder when the fixture root is unavailable; return an empty string otherwise.

    pytest is deliberately not imported here so the module can be imported by plain scripts.
    """
    root = Path(root)
    if str(root).startswith("/path/to/"):
        return (
            f"数据根仍是占位地址：{root}\n"
            "请先打开 tests/integration/realdata_common.py，把顶部的 DATA_ROOT 改成"
            "真实的 fixture 目录（其下应有 fgvp_test_data / fgvp_ens_test_data / "
            "fuxi_test_data），再跑 pytest -m realdata。"
        )
    if not root.is_dir():
        return (
            f"数据目录不存在：{root}\n"
            "请检查 tests/integration/realdata_common.py 顶部的 DATA_ROOT 是否指向"
            "真实 fixture 目录；放好切片数据和 golden 后再跑 pytest -m realdata。"
        )
    return ""


#: rtol absorbs float32 round-trip and algebraic-simplification rounding; atol covers
#: fields that are zero over large areas (tp).
RTOL = 1e-5
ATOL = 1e-6
EXCEED_FRACTION_CLOSE = 1e-3      # below 0.01%: random rounding
EXCEED_FRACTION_DRIFT = 0.5       # above half: most points changed
DRIFT_STD_FRACTION = 1e-2         # max|diff| exceeding this fraction of field variability is physically resolvable

VERDICT_RANK = {"IDENTICAL": 0, "CLOSE": 1, "DRIFT": 2, "DIFFER": 3}

_INIT_RE = re.compile(r"^\d{8}(?:\d{2})?$")
_STEP_RE = re.compile(r"^(\d{3})\.nc$")


def init_dirs(root: Path) -> dict:
    """Subdirectory name -> initialization time. Accepts both YYYYMMDD and YYYYMMDDHH naming."""
    found = {}
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        if _INIT_RE.fullmatch(entry.name):
            fmt = "%Y%m%d" if len(entry.name) == 8 else "%Y%m%d%H"
            found[entry.name] = pd.to_datetime(entry.name, format=fmt)
    return found


def steps_in(directory: Path) -> list:
    """Step indices of the ``NNN.nc`` files in a directory."""
    return sorted(
        int(match.group(1))
        for path in directory.iterdir()
        if path.is_file() and (match := _STEP_RE.fullmatch(path.name))
    )


def variables_in(path: Path) -> set:
    """Two-dimensional variable names in a NetCDF file (lowercase)."""
    with xr.open_dataset(path) as ds:
        return {str(name).lower() for name in ds.data_vars if ds[name].ndim == 2}


def align_grid(values, lat, lon):
    """Orient a field north-to-south with longitude ascending over 0..360. Raises if the shape does not match the coordinates."""
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
    """Read the two-dimensional fields of a prediction file -> (``{variable: (H,W)}``, lat, lon). Variable names are case-insensitive."""
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
    """Assign a verdict. The primary criterion is the **fraction** of tolerance-exceeding
    points, not max|diff|; if max|diff| exceeds 1% of the field's own spatial
    variability (``DRIFT_STD_FRACTION``), the verdict is DIFFER regardless of that fraction.
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
    """Two (H,W) fields -> metrics dict. Tolerance is ``|a-b| <= atol + rtol*|b|``; NaN
    position mismatches are counted separately, and cells missing on one side are excluded
    from the exceed statistics.
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
    ref_std = float(b[both].std())  # spatial variability of the reference field itself
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
    """Compare two output root directories -> ``(rows, skipped)``. Intended for deterministic suites (no member dimension).

    ``rows`` holds one entry per (initialization, step, variable); ``skipped`` records
    ``(level, what, why)`` for entries present on only one side -- the coverage assertions
    rely on it to catch outputs that pass more easily by writing fewer steps.
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
            # A grid mismatch means the grid itself changed; cell-wise subtraction would only yield spurious drift.
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
    """The set of variables the recipe declares for output (lowercase). Read from the recipe rather than duplicated in the test."""
    from xmetai_inference.configs.base import load_config

    cfg = load_config(str(config_path))
    return {v.strip().lower() for v in str(cfg.vars).split(",") if v.strip()}
