"""FuXi 2.1 — Minimal PT2 inference with autoregressive rollout.

Runs a torch.export model for N steps, saving denormalized predictions as NetCDF.
GPU-resident rollout: recurrence state stays on GPU between steps.

Usage:
    python inference.py --model_dir ./model --input input.nc \
        --output_dir ./output --steps 40 --forecast_time 2024092900
"""

import argparse
import inspect
import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import xarray as xr

from data_util import load_input, load_norm_stats, postprocess
from variables import C85_CHANNEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_ORDER = ["input", "step", "hour", "doy"]


def load_model(model_path: str, device: torch.device) -> tuple:
    """Load torch.export model onto the specified device.

    Returns (module, input_names, dtype) where dtype is inferred from model params.
    """
    logger.info(f"Loading model from {model_path}")
    with torch.cuda.device(device):
        ep = torch.export.load(model_path)
        module = ep.module()
        del ep

    sig = inspect.signature(module.forward)
    sig_names = list(sig.parameters.keys())
    if sig_names and sig_names[0].startswith("args_"):
        input_names = CANONICAL_ORDER[: len(sig_names)]
    else:
        input_names = sig_names
    logger.info(f"Model input signature: {input_names}")

    # Infer dtype from model parameters
    dtype = torch.float32
    for p in module.parameters():
        dtype = p.dtype
        break
    logger.info(f"Model dtype: {dtype}")

    torch.cuda.empty_cache()
    return module, input_names, dtype


def prepare_features(
    valid_time: pd.Timestamp, step: int, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Compute temporal conditioning features for one step."""
    tod = (valid_time.hour * 60 + valid_time.minute) / 1440.0
    doy = min(365, valid_time.day_of_year) / 365.0
    return {
        "step": torch.tensor([step], dtype=dtype, device=device),
        "hour": torch.tensor([tod], dtype=dtype, device=device),
        "doy": torch.tensor([doy], dtype=dtype, device=device),
    }


def print_dataarray(da: xr.DataArray):
    """Print per-channel value ranges for verification."""
    channels = da.coords["channel"].values
    msg = f"shape: {da.shape}"
    if "lat" in da.dims and "lon" in da.dims:
        lat = da.lat.values
        lon = da.lon.values
        msg += f", latlon: ({lat[0]:.2f}~{lat[-1]:.2f}) x ({lon[0]:.2f}~{lon[-1]:.2f})"
    print(msg)
    for ch in channels:
        x = da.sel(channel=ch).values
        print(f"  {ch:>6s}: {x.min():.4f} ~ {x.max():.4f}")


def save_step(
    prediction: np.ndarray,
    step_idx: int,
    valid_time: pd.Timestamp,
    lats: np.ndarray,
    lons: np.ndarray,
    channels: list[str],
    output_dir: Path,
):
    """Save one denormalized prediction step as NetCDF."""
    da = xr.DataArray(
        prediction,
        dims=["channel", "lat", "lon"],
        coords={
            "channel": channels,
            "lat": lats,
            "lon": lons,
        },
        attrs={"valid_time": str(valid_time)},
    )
    print_dataarray(da)
    path = output_dir / f"{step_idx:03d}.nc"
    da.to_netcdf(path)
    return path


def run(args):
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input (pre-normalized)
    logger.info(f"Loading input from {args.input}")
    input_da = load_input(args.input)
    lats = input_da.coords["lat"].values
    lons = input_da.coords["lon"].values
    channels = list(input_da.coords["channel"].values)

    # Load normalization stats for output denormalization
    mean, std = load_norm_stats(args.model_dir)
    logger.info(f"Norm stats loaded: mean={mean.shape}, std={std.shape}")

    # Load model — find .pth or .pt2 in model_dir
    model_path = None
    for ext in (".pt2", ".pth"):
        candidates = list(Path(args.model_dir).glob(f"*{ext}"))
        if candidates:
            model_path = str(candidates[0])
            break
    if model_path is None:
        raise FileNotFoundError(f"No .pt2 or .pth model found in {args.model_dir}")
    module, input_names, model_dtype = load_model(model_path, device)
    np_dtype = np.float16 if model_dtype == torch.float16 else np.float32

    # Initialize recurrence state on GPU
    state = torch.from_numpy(input_da.values[None].astype(np_dtype)).to(device)
    logger.info(f"Initial state shape: {state.shape} dtype={state.dtype} on {device}")

    # Rollout
    forecast_time = pd.to_datetime(args.forecast_time, format="%Y%m%d%H")
    dt = pd.Timedelta(args.frame_interval)
    total_time = 0.0

    logger.info(
        f"Starting rollout: {args.steps} steps, interval={args.frame_interval}"
    )

    for t in range(args.steps):
        valid_time = forecast_time + (t + 1) * dt
        features = prepare_features(forecast_time + t * dt, t, device, model_dtype)

        # Build positional args
        ordered_args = [state]
        for name in CANONICAL_ORDER[1:]:
            if name in input_names and name in features:
                ordered_args.append(features[name])

        # Run model
        t0 = perf_counter()
        with torch.no_grad():
            result = module(*ordered_args)
        infer_time = perf_counter() - t0
        total_time += infer_time

        # Extract prediction: last frame of the 2-frame output
        prediction = result[:, -1].clone().cpu().float().numpy()  # (1, 85, 721, 1440) float32

        # Denormalize
        postprocess(prediction, mean, std)

        # Save (prints per-channel value ranges before writing)
        logger.info(f"  Step {t+1}/{args.steps}: {infer_time:.2f}s")
        path = save_step(prediction[0], t + 1, valid_time, lats, lons, channels, output_dir)
        logger.info(f"  Saved {path.name}")

        # Feed back for next step
        state = result

    logger.info(
        f"Done. {args.steps} steps in {total_time:.1f}s "
        f"({total_time/args.steps:.2f}s/step). Output: {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="FuXi 2.1 PT2 Inference")
    parser.add_argument(
        "--model_dir", required=True,
        help="Directory containing fuxi-2.1.pth, mean.nc, std.nc",
    )
    parser.add_argument("--input", required=True, help="Path to input .nc file")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--steps", type=int, default=40, help="Number of rollout steps")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--forecast_time",
        required=True,
        help="Forecast init time (YYYYMMDDHH)",
    )
    parser.add_argument(
        "--frame_interval", default="6h", help="Time interval between steps"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
