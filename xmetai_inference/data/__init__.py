"""Dataset and DataLoader construction for inference."""

from .contract import DatasetSource
from .factory import create_dataset_source

__all__ = [
    "DatasetSource",
    "create_dataset_source",
]
