"""Dataset：一个类型一个模块，这里导出各自的 ``create_*_dataset()``。"""
from __future__ import annotations

from .processed_multi_zarr import create_processed_multi_zarr_dataset

__all__ = [
    "create_processed_multi_zarr_dataset",
]
