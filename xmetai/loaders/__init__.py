# -*- coding: utf-8 -*-
"""数据源包：内置 Loader 注册和外部 Loader 工厂构造。

外部 config 可以直接传 Loader 类或工厂函数。构造函数可按需接收关键字参数
``path``、``model_cls`` 和 ``groups``，返回对象必须实现 ``load_state()``。
"""
import inspect

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


def _validate_loader(loader, source):
    if not callable(getattr(loader, "load_state", None)):
        raise TypeError(
            f"Loader {source!r} 构造结果 {type(loader).__name__} "
            "没有实现 load_state(time, channels=None)")
    return loader


def _create_external_loader(factory, model_cls=None, path=None, groups=None):
    if not callable(factory):
        if path:
            raise TypeError("config 传入 Loader 实例时不能再使用 --data-root 覆盖路径")
        return _validate_loader(factory, factory)

    signature = inspect.signature(factory)
    available = {
        "path": path,
        "model_cls": model_cls,
        "groups": groups,
    }
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = {
        name: value
        for name, value in available.items()
        if accepts_kwargs or name in signature.parameters
    }
    return _validate_loader(factory(**kwargs), factory)


def create_loader(name, model_cls=None, path=None, groups=None):
    """按内置数据源名、外部 Loader 类、工厂函数或实例构造 Loader。

    era 需要 model_cls 从 input_channels 推导变量/层级；zarr / era5_store
    不需要模型契约。
    """
    if not isinstance(name, str):
        return _create_external_loader(
            name,
            model_cls=model_cls,
            path=path,
            groups=groups,
        )
    if name == "era":
        return _validate_loader(
            EraDataLoader(model_cls=model_cls, data_root=path), name)
    if name == "zarr":
        if not path:
            raise ValueError("zarr loader 需要 Zarr store 路径")
        return _validate_loader(ZarrDataLoader(path), name)
    if name == "zarr_normalized":
        if not path:
            raise ValueError("zarr_normalized loader 需要 Zarr store 路径")
        return _validate_loader(ZarrDataLoader(path, normalized=True), name)
    if name == "era5_store":
        # 根目录写死在 era5_store.py 的 DEFAULT_ROOT；--zarr 传了才覆盖
        kwargs = {"root": path}
        if groups is not None:
            kwargs["groups"] = groups
        return _validate_loader(Era5StoreLoader(**kwargs), name)
    raise ValueError(f"未知数据源 {name!r}（可选 {', '.join(LOADER_REGISTRY)}）")
