# -*- coding: utf-8 -*-
"""数据源包：每种数据源只实现 load(time) -> xr.Dataset 一个接口。

加新数据源两步：
  1) 新建 loaders/<name>.py，实现 load(time) -> xr.Dataset。
  2) 在 LOADER_REGISTRY 里加一行 {数据源名: 类}，并在 create_loader 里加一条
     构造分支（不同数据源构造参数不同，故此处显式分发）。
"""

# ERA5 原生单位 → 模型物理单位。累积量的 1h→6h 转换仍由具体 loader 负责。
ERA5_TO_MODEL_SCALE = {
    "q": 1000.0,
    "ssr": 1.0 / 3600.0,
    "ssrd": 1.0 / 3600.0,
    "fdir": 1.0 / 3600.0,
    "ttr": 1.0 / 3600.0,
    "tp": 1000.0,
}


from .era import EraDataLoader
from .era5_store import Era5StoreLoader
from .zarr import ZarrDataLoader

LOADER_REGISTRY = {
    "era": EraDataLoader,
    "zarr": ZarrDataLoader,
    "zarr_normalized": ZarrDataLoader,
    "era5_store": Era5StoreLoader,
}


def create_loader(name, model_cls=None, path=None, groups=None):
    """按数据源名构造 loader。

    era 需要 model_cls 从 input_channels 推导变量/层级；zarr / era5_store
    不需要模型契约。
    """
    if name == "era":
        return EraDataLoader(model_cls=model_cls, data_root=path)
    if name == "zarr":
        if not path:
            raise ValueError("zarr loader 需要 Zarr store 路径")
        return ZarrDataLoader(path)
    if name == "zarr_normalized":
        if not path:
            raise ValueError("zarr_normalized loader 需要 Zarr store 路径")
        return ZarrDataLoader(path, normalized=True)
    if name == "era5_store":
        # 根目录写死在 era5_store.py 的 DEFAULT_ROOT；--zarr 传了才覆盖
        kwargs = {"root": path}
        if groups is not None:
            kwargs["groups"] = groups
        return Era5StoreLoader(**kwargs)
    raise ValueError(f"未知数据源 {name!r}（可选 {', '.join(LOADER_REGISTRY)}）")
