"""Plot FuXi forecast output as weather maps.

Self-contained visualization using matplotlib + cartopy with discrete
color scales for key weather variables (precipitation, wind speed).

Usage:
    python plot.py --output_dir ./output --channels t2m z500 tp
    python plot.py --output_dir ./output --steps 1 3 5 --discrete
"""

import argparse
import logging
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

mpl.use("Agg")

from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Discrete color scales for weather variables ──────────────────────────────

VAR_COLORS = {
    "tp": {
        "levels": [0, 0.1, 1, 2, 5, 10, 15, 20, 30, 40, 50, 100, 300, 1000],
        "colors": [
            "#FFFFFF", "#F0E6C3", "#B6F391", "#52ED52", "#95CFFF",
            "#368EFF", "#1061FF", "#0033FF", "#FFFF00", "#FFA500",
            "#FF0000", "#8B2500", "#FF00FF",
        ],
    },
    "gs": {
        "levels": [0.0, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7],
        "colors": [
            "#FFFFFF", "#8FCEF0", "#489B9F", "#49B154", "#9FCE51",
            "#FAE159", "#F8B547", "#F26429", "#DC3328", "#B01A20",
        ],
    },
}
VAR_COLORS["ws"] = VAR_COLORS["ws10m"] = VAR_COLORS["ws100m"] = VAR_COLORS["gs"]

DEFAULT_CMAP = "viridis"


# ── Plotting core ────────────────────────────────────────────────────────────

def _strip_altitude(name):
    """Strip trailing digits: 'z500' → 'z', 'tp' → 'tp'."""
    return re.sub(r"\d+$", "", name)


def get_color_kwargs(var_name, data_values, discrete=False):
    """Get cmap/norm kwargs for pcolormesh."""
    base = _strip_altitude(str(var_name).lower().split("_")[0])
    use_discrete = discrete or (base in VAR_COLORS)

    if use_discrete and base in VAR_COLORS:
        cfg = VAR_COLORS[base]
        n = min(len(cfg["levels"]), len(cfg["colors"]))
        cmap = ListedColormap(cfg["colors"][:n])
        norm = BoundaryNorm(cfg["levels"][:n], ncolors=cmap.N, clip=True)
        return {"cmap": cmap, "norm": norm}

    vals = data_values.astype(np.float32)
    return {"cmap": DEFAULT_CMAP, "vmin": np.nanmin(vals), "vmax": np.nanmax(vals)}


def plot_field(
    data_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    var_name: str,
    title: str = "",
    save_path: str | None = None,
    discrete: bool = False,
    coastline: bool = True,
    gridline: bool = True,
    dpi: int = 150,
):
    """Plot a single 2D weather field on a map."""
    lon_range = lons[-1] - lons[0]
    lat_range = abs(lats[0] - lats[-1])
    aspect = lon_range / max(lat_range, 1)
    figsize = (8 * aspect, 6)

    projection = ccrs.PlateCarree(180) if HAS_CARTOPY else None
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection}, dpi=dpi)

    color_kwargs = get_color_kwargs(var_name, data_2d, discrete=discrete)

    if HAS_CARTOPY:
        img = ax.pcolormesh(
            lons, lats, data_2d,
            transform=ccrs.PlateCarree(),
            shading="auto",
            **color_kwargs,
        )
        extent = [float(lons[0]), float(lons[-1]),
                  max(float(min(lats[0], lats[-1])), -89.5),
                  min(float(max(lats[0], lats[-1])), 89.5)]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        if coastline:
            ax.add_feature(cfeature.COASTLINE, edgecolor="k", linewidth=0.5)
        if gridline:
            gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.5,
                              linewidth=1, linestyle="--", crs=ccrs.PlateCarree())
            gl.top_labels = gl.right_labels = False
    else:
        img = ax.pcolormesh(lons, lats, data_2d, shading="auto", **color_kwargs)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(var_name.upper(), fontsize=14, fontweight="bold")

    # Colorbar
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.015, pos.height])
    cbar = plt.colorbar(img, cax=cax, orientation="vertical", extend="both", extendfrac=0.03)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(var_name, size=10)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    return save_path


# ── Batch plotting from output directory ─────────────────────────────────────

def plot_forecast(
    output_dir: str | Path,
    channels: list[str] | None = None,
    steps: list[int] | None = None,
    plot_dir: str | Path | None = None,
    discrete: bool = False,
    coastline: bool = True,
    gridline: bool = True,
    dpi: int = 150,
):
    """Plot forecast fields from saved NetCDF output files."""
    output_dir = Path(output_dir)
    if plot_dir is None:
        plot_dir = output_dir / "plots"
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted([f for f in output_dir.glob("*.nc") if f.stem.isdigit()])
    if not nc_files:
        logger.error(f"No step .nc files (001.nc, 002.nc, ...) found in {output_dir}")
        return

    if steps:
        nc_files = [f for f in nc_files if int(f.stem) in steps]

    for nc_path in nc_files:
        da = xr.open_dataarray(nc_path)
        step_idx = int(nc_path.stem)
        valid_time = da.attrs.get("valid_time", "")

        available = list(da.coords["channel"].values)
        plot_channels = channels if channels else available
        valid_ch = [c for c in plot_channels if c in available]
        if not valid_ch:
            logger.warning(f"None of {plot_channels} found in {nc_path.name}, skipping")
            continue

        lats = da.coords["lat"].values
        lons = da.coords["lon"].values

        for ch in valid_ch:
            field = da.sel(channel=ch).values
            title = f"{ch.upper()} | Step {step_idx}"
            if valid_time:
                title += f" | {valid_time}"
            save_path = str(plot_dir / f"step{step_idx:03d}_{ch}.png")
            plot_field(
                field, lats, lons, ch,
                title=title, save_path=save_path,
                discrete=discrete, coastline=coastline,
                gridline=gridline, dpi=dpi,
            )
            print(f"Saved: {save_path}")

    logger.info(f"All plots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot FuXi forecast fields")
    parser.add_argument("--output_dir", required=True, help="Directory with .nc output files")
    parser.add_argument("--channels", nargs="+", default=None,
                        help="Channels to plot (default: all)")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="Step indices to plot (default: all)")
    parser.add_argument("--plot_dir", default=None, help="Output directory for plots")
    parser.add_argument("--discrete", action="store_true", help="Use discrete color scales")
    parser.add_argument("--no-coastline", action="store_true", help="Hide coastlines")
    parser.add_argument("--no-gridline", action="store_true", help="Hide gridlines")
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution")
    args = parser.parse_args()
    plot_forecast(
        args.output_dir,
        channels=args.channels,
        steps=args.steps,
        plot_dir=args.plot_dir,
        discrete=args.discrete,
        coastline=not args.no_coastline,
        gridline=not args.no_gridline,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
