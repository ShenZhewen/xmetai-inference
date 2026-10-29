"""配置 → DataLoader：按 ``dataset.type`` 分发到 ``data/<category>/`` 下的实现。

加一种类型 = 新建 ``data/<category>/<type>.py`` 并在下面加一条分支。
"""
from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import DataLoader

from ..data.weather import create_processed_multi_zarr_dataset


def create_dataloader(
    spec,
    *,
    model_cls,
    init_times,
    history_steps,
    hour_interval,
    model_path=None,
    dataloader_options=None,
):
    """按 ``config["dataset"]`` 声明的类型构建推理 DataLoader。"""
    if not isinstance(spec, Mapping):
        raise TypeError("dataset 必须是包含 type 和参数的配置字典")
    name = spec.get("type")

    if name == "processed_multi_zarr":
        dataset = create_processed_multi_zarr_dataset(
            spec,
            model_cls=model_cls,
            init_times=init_times,
            history_steps=history_steps,
            hour_interval=hour_interval,
            model_path=model_path,
        )
    else:
        raise ValueError(
            f"未知 Dataset {name!r}（当前仅支持 processed_multi_zarr）"
        )

    options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }
    options.update(dict(dataloader_options or {}))
    if options["shuffle"]:
        raise ValueError("推理 DataLoader 不允许 shuffle=True")

    return DataLoader(dataset, **options)
