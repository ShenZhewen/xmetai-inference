"""Shared contracts between datasets, DataLoader, and model processing."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DatasetSource:
    """Runtime view of a Dataset, its DataLoader, and source metadata.

    需要单位换算系数的地方（如评测要与推理对齐）直接读 config 的
    ``dataset.processors[unit_convert].scales``，不要从这里取。
    """

    dataset: Any
    dataloader: Any
    channel_names: list[str]
    latitudes: np.ndarray
    longitudes: np.ndarray

    def __iter__(self) -> Iterator[Any]:
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
