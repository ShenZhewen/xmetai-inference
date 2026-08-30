# -*- coding: utf-8 -*-
"""数据源包：每种数据源只实现 load(time) -> xr.Dataset 一个接口。

加新数据源两步：
  1) 新建 loaders/<name>.py，实现 load(time) -> xr.Dataset。
  2) 在 LOADER_REGISTRY 里加一行 {数据源名: 类}，并在 create_loader 里加一条
     构造分支（不同数据源构造参数不同，故此处显式分发）。
"""
from .era import EraDataLoader
from .era5_store import Era5StoreLoader
from .zarr import ZarrDataLoader

LOADER_REGISTRY = {
    "era": EraDataLoader,
    "zarr": ZarrDataLoader,
    "era5_store": Era5StoreLoader,
}


def create_loader(name, spec=None, path=None, data_root=None):
    """按数据源名构造 loader。spec 是 load_spec 展开过的 dict（era/era5_store 需要）。"""
    if name == "era":
        return EraDataLoader(spec, data_root=data_root)
    if name == "zarr":
        if not path:
            raise ValueError("--loader zarr 需要 zarr store 路径（--zarr）")
        return ZarrDataLoader(path)
    if name == "era5_store":
        # 根目录写死在 era5_store.py 的 DEFAULT_ROOT；--zarr 传了才覆盖
        return Era5StoreLoader(root=path)
    raise ValueError(f"未知数据源 {name!r}（可选 {', '.join(LOADER_REGISTRY)}）")
