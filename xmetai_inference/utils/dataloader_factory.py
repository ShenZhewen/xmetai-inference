"""Config → DataLoader: dispatch on ``dataset.type`` to the implementation under ``data/<category>/``.

Adding a type means creating ``data/<category>/<type>.py`` and adding a branch below.
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
    """Build an inference DataLoader for the type declared in ``config["dataset"]``."""
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
