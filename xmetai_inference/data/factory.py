"""Configuration-driven Dataset and DataLoader factory."""
from __future__ import annotations

from collections.abc import Mapping

from .contract import DatasetSource
from .datasets.processed_multi_zarr import create_processed_multi_zarr_dataset


def create_dataset_source(
    spec,
    *,
    model_cls,
    init_times,
    history_steps,
    hour_interval,
    data_root=None,
    model_path=None,
    dataloader=None,
):
    """Build a Dataset source without applying model transformations."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise ImportError("Dataset 推理需要安装 torch") from error

    if not isinstance(spec, Mapping):
        raise TypeError("dataset 必须是包含 type 和参数的配置字典")
    config = dict(spec)
    name = config.get("type")
    if name != "processed_multi_zarr":
        raise ValueError(
            f"未知 Dataset {name!r}（当前仅支持 processed_multi_zarr）"
        )

    dataset, metadata = create_processed_multi_zarr_dataset(
        config,
        model_cls=model_cls,
        init_times=init_times,
        history_steps=history_steps,
        hour_interval=hour_interval,
        data_root=data_root,
        model_path=model_path,
    )
    options = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }
    options.update(dict(dataloader or {}))
    if options["shuffle"]:
        raise ValueError("推理 DataLoader 不允许 shuffle=True")

    loader = DataLoader(dataset, **options)
    return DatasetSource(
        dataset=dataset,
        dataloader=loader,
        channel_names=metadata["channel_names"],
        latitudes=metadata["latitudes"],
        longitudes=metadata["longitudes"],
    )
