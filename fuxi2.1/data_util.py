"""Data utilities for FuXi 2.1 inference.

Handles: input loading, normalization stats, output denormalization.
"""

import os

import numpy as np
import xarray as xr


def load_input(path: str) -> xr.DataArray:
    """Load pre-normalized input from NetCDF.

    Expected shape: (time=2, channel=85, lat=721, lon=1440).
    Input must already be z-score normalized.
    """
    ds = xr.open_dataset(path)
    return ds["input"]


def load_norm_stats(model_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load mean.nc and std.nc for output denormalization.

    Returns (mean, std) each shape (85,).
    """
    mean = xr.open_dataarray(os.path.join(model_dir, "mean.nc")).values
    std = xr.open_dataarray(os.path.join(model_dir, "std.nc")).values
    return mean, std


def postprocess(
    output: np.ndarray, mean: np.ndarray, std: np.ndarray, exp_channel: int = -1
) -> np.ndarray:
    """Denormalize output in-place: output = output * std + mean.

    The tp channel (last, index 84) was log1p-normalized during training,
    so expm1 is applied to reverse it. Pass exp_channel=None to skip.

    Args:
        output: array with shape (..., C, H, W)
        mean: shape (C,)
        std: shape (C,)
        exp_channel: channel index for expm1 reversal (-1 = last)
    """
    ndim = output.ndim
    shape = [1] * (ndim - 3) + [mean.shape[0], 1, 1]
    m = mean.reshape(shape).astype(output.dtype)
    s = std.reshape(shape).astype(output.dtype)

    np.multiply(output, s, out=output)
    np.add(output, m, out=output)

    if exp_channel is not None:
        ch = output[..., exp_channel, :, :]
        np.clip(ch, None, 20, out=ch)
        np.expm1(ch, out=ch)
        np.clip(ch, 0, None, out=ch)

    return output
