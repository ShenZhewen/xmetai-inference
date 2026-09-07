"""Shared contracts between datasets, DataLoader, and model processing."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DatasetSource:
    """Runtime view of a Dataset, its DataLoader, and source metadata.

    以下字段已移除（曾经存在但全项目无人读取）：

      ``input_space``   恒为 "physical"，无消费者
      ``preprocessed``  恒为 True，无消费者
      ``unit_scales``   + ``SCALE`` 别名：单位换算已由 dataset.processors 的
                        ``unit_convert`` 在 DataLoader worker 里完成，这里再暴露
                        一份只会让人以为下游还要自己乘一次
      ``static_source`` 工厂从不设置它，``load_static_state()`` 必然抛异常

    需要单位换算系数的地方（如评测要与推理对齐）应直接读 config 的
    ``dataset.processors[unit_convert].scales``（见 util/eval_common.py 的
    ``defaults_from_config``），而不是从这里取 —— 那样才是同一个出处。
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
