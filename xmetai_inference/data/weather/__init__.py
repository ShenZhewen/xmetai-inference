"""Dataset: one module per type; each module's ``create_*_dataset()`` is exported here."""
from __future__ import annotations

from .processed_multi_zarr import create_processed_multi_zarr_dataset
from .example_nc import create_example_nc_dataset, example_nc_init_times

__all__ = [
    "create_processed_multi_zarr_dataset",
    "create_example_nc_dataset",
    "example_nc_init_times",
]
