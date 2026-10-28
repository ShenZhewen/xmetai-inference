"""Dataset 与 DataLoader 的构造。

- ``loader``：配置 → DataLoader（按 ``dataset.type`` 分发）
- ``weather/``：气象类 Dataset 的实现（当前只有 ``processed_multi_zarr``）

加类型 = 在 ``data/<category>/`` 下新建 ``<type>.py``，再在
``loader.create_dataloader`` 里加一条分支。
"""
from __future__ import annotations

from .loader import create_dataloader
from .weather import create_processed_multi_zarr_dataset

__all__ = [
    "create_dataloader",
    "create_processed_multi_zarr_dataset",
]
