"""Variable definitions for FuXi weather forecasting models.

Defines the C85 variable table (85 channels) used by FuXi HR:
  - 5 pressure-level variables × 13 levels = 65 channels
  - 20 surface variables
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Normalization epsilon — single source of truth for all (x - mean) / (std + eps) paths.
# 1e-8 is safe for float32; 1e-12 risks denormals.
NORM_EPSILON: float = 1e-8


@dataclass(frozen=True)
class VariableInfo:
    """Metadata for a single forecast variable (one channel)."""

    short_name: str
    long_name: str
    level: int | None = None        # hPa for PL vars, None for surface
    units: str = ""
    is_accumulated: bool = False
    is_log_transformed: bool = False
    channel_index: int = -1         # position in C85 channel ordering

    @property
    def channel_name(self) -> str:
        """Canonical channel name: '{var}{level}' for PL, '{var}' for SFC."""
        if self.level is not None:
            return f"{self.short_name}{self.level}"
        return self.short_name


# ── Pressure levels (13) ─────────────────────────────────────────

PRESSURE_LEVELS: list[int] = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
]

# ── Pressure-level variables ─────────────────────────────────────

PL_VARIABLES: list[tuple[str, str, str]] = [
    ("z", "geopotential", "m²/s²"),
    ("t", "temperature", "K"),
    ("u", "u_component_of_wind", "m/s"),
    ("v", "v_component_of_wind", "m/s"),
    ("q", "specific_humidity", "g/kg"),  # scaled ×1000 from ERA5 kg/kg
]

# ── Surface variables (20) ───────────────────────────────────────

_SFC_DEFS: list[tuple[str, str, str, bool, bool]] = [
    # (short_name, long_name, units, is_accumulated, is_log_transformed)
    # Order matches hres_input.py CHANNELS_85 — this IS the canonical ordering.
    ("msl",    "mean_sea_level_pressure",                     "Pa",    False, False),
    ("t2m",    "2m_temperature",                              "K",     False, False),
    ("d2m",    "2m_dewpoint_temperature",                     "K",     False, False),
    ("sst",    "sea_surface_temperature",                     "K",     False, False),
    ("ws10m",  "10m_wind_speed",                              "m/s",   False, False),
    ("ws100m", "100m_wind_speed",                             "m/s",   False, False),
    ("u10m",   "10m_u_component_of_wind",                     "m/s",   False, False),
    ("v10m",   "10m_v_component_of_wind",                     "m/s",   False, False),
    ("u100m",  "100m_u_component_of_wind",                    "m/s",   False, False),
    ("v100m",  "100m_v_component_of_wind",                    "m/s",   False, False),
    ("lcc",    "low_cloud_cover",                             "0-1",   False, False),
    ("mcc",    "medium_cloud_cover",                          "0-1",   False, False),
    ("hcc",    "high_cloud_cover",                            "0-1",   False, False),
    ("tcc",    "total_cloud_cover",                           "0-1",   False, False),
    ("ssr",    "surface_net_solar_radiation",                  "J/m²",  True,  False),
    ("ssrd",   "surface_solar_radiation_downwards",            "J/m²",  True,  False),
    ("fdir",   "total_sky_direct_solar_radiation_at_surface",  "J/m²",  True,  False),
    ("ttr",    "top_net_thermal_radiation",                    "J/m²",  True,  False),
    ("tcw",    "total_column_water",                          "kg/m²", False, False),
    ("tp",     "total_precipitation",                         "m",     True,  True),
]


# ── Build the full C85 table ─────────────────────────────────────

def _build_c85() -> list[VariableInfo]:
    """Construct the ordered list of 85 VariableInfo entries."""
    table: list[VariableInfo] = []
    idx = 0

    # Pressure-level channels: var × level (65 channels)
    for short, long, units in PL_VARIABLES:
        for lev in PRESSURE_LEVELS:
            table.append(VariableInfo(
                short_name=short,
                long_name=long,
                level=lev,
                units=units,
                channel_index=idx,
            ))
            idx += 1

    # Surface channels (20 channels)
    for short, long, units, accum, logt in _SFC_DEFS:
        table.append(VariableInfo(
            short_name=short,
            long_name=long,
            units=units,
            is_accumulated=accum,
            is_log_transformed=logt,
            channel_index=idx,
        ))
        idx += 1

    return table


C85_VARIABLES: list[VariableInfo] = _build_c85()
"""All 85 variables in channel order."""

C85_CHANNEL_NAMES: list[str] = [v.channel_name for v in C85_VARIABLES]
"""Channel names in order: ['z50', 'z100', ..., 'msl', 't2m', ..., 'tp']."""

# Quick-lookup maps
_NAME_TO_VAR: dict[str, VariableInfo] = {v.channel_name: v for v in C85_VARIABLES}
_NAME_TO_IDX: dict[str, int] = {v.channel_name: v.channel_index for v in C85_VARIABLES}

# Sets for fast membership tests
ACCUMULATED_CHANNELS: frozenset[str] = frozenset(
    v.channel_name for v in C85_VARIABLES if v.is_accumulated
)
LOG_TRANSFORMED_CHANNELS: frozenset[str] = frozenset(
    v.channel_name for v in C85_VARIABLES if v.is_log_transformed
)


def get_variable(channel_name: str) -> VariableInfo:
    """Look up variable metadata by channel name.

    Raises:
        KeyError: if *channel_name* is not in C85.
    """
    return _NAME_TO_VAR[channel_name]


def get_channel_index(channel_name: str) -> int:
    """Return the 0-based index of *channel_name* in C85 ordering.

    Raises:
        KeyError: if *channel_name* is not in C85.
    """
    return _NAME_TO_IDX[channel_name]


# ── Constant fields ──────────────────────────────────────────────

CONST_CHANNEL_NAMES: list[str] = [
    "geo",        # normalized geopotential (orography)
    "lsm",        # land-sea mask
    "cos_lat",    # cos(latitude)
    "sin_lat",    # sin(latitude)
    "cos_lon",    # cos(longitude)
    "sin_lon",    # sin(longitude)
]
N_CONST_CHANNELS: int = len(CONST_CHANNEL_NAMES)
