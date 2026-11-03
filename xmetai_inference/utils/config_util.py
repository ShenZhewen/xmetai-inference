# -*- coding: utf-8 -*-
"""Config-assembly utilities: model declarations, geometry helpers, and init-time parsing.

One small grab-bag module that the config system needs while assembling a recipe:
the grids/level/channel primitives shared across models, the pure helpers that build
or interpret them (``_expand_channels``, ``grid_coords``, ``parse_times``).

Per-model surface-variable lists and the channel lists they build live in each model's
own ``configs/<name>.py``; only what several models genuinely share stays here.

Deliberately free of onnx/torch imports: importing configs / these constants needs
nothing but numpy. Pure Python, no state.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np


# Shared 0.25 degree global regular grid for FuXi-Ens, FuXi-2.1, and FGVP.
GRID_025 = {
    "lat": {"start": 90.0, "step": -0.25, "size": 721},
    "lon": {"start": 0.0, "step": 0.25, "size": 1440},
}

# South-to-north variant (native CRA store orientation; used by the Tongyun-CRA graph).
# Do NOT pair it with geometry: that processor unconditionally flips south-to-north to
# north-to-south, feeding fields in reverse. Output lat coordinates also come from this grid.
GRID_025_S = {
    "lat": {"start": -90.0, "step": 0.25, "size": 721},
    "lon": {"start": 0.0, "step": 0.25, "size": 1440},
}

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PL_VARS = ["z", "t", "u", "v", "q"]

# FuXi-Ens surface channels (shared by FuXi-Ens and the FGVP family inputs).
FUXI_ENS_SFC_VARS = [
    "t2m", "d2m", "sst", "u10m", "v10m", "u100m", "v100m",
    "msl", "ssr", "ssrd", "fdir", "ttr", "tp",
]


def _expand_channels(pl_vars, levels, surface_vars):
    return [
        f"{variable}{level}"
        for variable in pl_vars
        for level in levels
    ] + list(surface_vars)


FUXI_ENS_CHANNELS = _expand_channels(PL_VARS, LEVELS, FUXI_ENS_SFC_VARS)


def grid_coords(grid):
    """Convert a model grid contract to latitude and longitude coordinate arrays."""
    latitude = grid["lat"]
    longitude = grid["lon"]
    lat = (
        np.arange(latitude["size"], dtype=np.float64) * latitude["step"]
        + latitude["start"]
    )
    lon = (
        np.arange(longitude["size"], dtype=np.float64) * longitude["step"]
        + longitude["start"]
    )
    return lat, lon


def _parse_dt(token):
    """Parse one initialization-time token into a datetime (YYYYMMDD or YYYYMMDDHH)."""
    token = token.strip()
    if re.fullmatch(r"\d{8}", token):
        return datetime.strptime(token, "%Y%m%d")
    if re.fullmatch(r"\d{10}", token):
        return datetime.strptime(token, "%Y%m%d%H")
    raise SystemExit(f"无法解析起报时间 {token!r}（应 YYYYMMDD 或 YYYYMMDDHH）")


def parse_times(spec, hour_interval=24):
    """Expand the unified `times` string into initialization times (datetime, ascending, deduplicated).

    Syntax:
      ""              → []
      "2025010600"    → single initialization
      "20250106,20250112"                → explicit list (comma-separated)
      "2025010600..2025020500"           → closed interval, step hour_interval (default 24h)
      "2025010600..2025020500:24"        → closed interval, `:N` overrides the step (hours)
    Singles and intervals may be mixed in one comma-separated list.
    """
    if spec is None or str(spec).strip() == "":
        return []
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d{8}(?:\d{2})?)\s*\.\.\s*(\d{8}(?:\d{2})?)(?::(\d+))?", tok)
        if m:
            start = _parse_dt(m.group(1))
            end = _parse_dt(m.group(2))
            step = int(m.group(3)) if m.group(3) else hour_interval
            if step <= 0:
                raise SystemExit(f"区间步长必须 >0：{tok!r}")
            t = start
            while t <= end:
                out.append(t)
                t += timedelta(hours=step)
        else:
            out.append(_parse_dt(tok))
    if not out:
        raise SystemExit(f"times 解析后为空：{spec!r}")
    return sorted(set(out))