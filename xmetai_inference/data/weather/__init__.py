"""气象类 Dataset 实现：一个类型一个模块，这里导出各自的 ``create_*_dataset()``。

加类型 = 在本目录新建 ``<type>.py``，在下面补一行导出，再去
``data/loader.py`` 的 ``create_dataloader`` 加一条分支。
"""
from __future__ import annotations

from .processed_multi_zarr import create_processed_multi_zarr_dataset

__all__ = [
    "create_processed_multi_zarr_dataset",
]
