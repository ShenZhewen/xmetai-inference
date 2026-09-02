# -*- coding: utf-8 -*-
"""模型包：具体模型的固定运行契约。

引擎在 xmetai.backends，模型在 xmetai.models，语义不混：
  * backends 只提供共享运行契约以及 onnx/pt2/ckpt 三种执行机制；
  * MODEL_REGISTRY 是唯一注册表：模型名 → 具体模型类。模型继承对应引擎，只声明
    通道、网格、时间窗口和输入表示；完整 Processor 流程由 configs 声明。

加新模型三步：
  1) 新建 models/<name>.py，继承对应引擎（backends.onnx.OnnxInferModel /
     backends.pt2.Pt2InferModel / backends.ckpt.CkptInferModel / …）。
  2) 在模型类声明 input_channels/input_fields 和 input_assembler。
  3) 在 MODEL_REGISTRY 注册，并新增一份包含完整 Processor 流程的 config。

模型选择主路径：config/CLI 的 model_class → create_model()。
"""
from importlib import import_module

import numpy as np


# FuXi-Ens、FuXi-2.1 和 FGVP 共用的 0.25° 全球规则网格。
GRID_025 = {
    "lat": {"start": 90.0, "step": -0.25, "size": 721},
    "lon": {"start": 0.0, "step": 0.25, "size": 1440},
}

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PL_VARS = ["z", "t", "u", "v", "q"]
FUXI_ENS_SFC_VARS = [
    "t2m", "d2m", "sst", "u10m", "v10m", "u100m", "v100m",
    "msl", "ssr", "ssrd", "fdir", "ttr", "tp",
]
FUXI21_SFC_VARS = [
    "msl", "t2m", "d2m", "sst", "ws10m", "ws100m", "u10m", "v10m",
    "u100m", "v100m", "lcc", "mcc", "hcc", "tcc",
    "ssr", "ssrd", "fdir", "ttr", "tcw", "tp",
]


def _expand_channels(pl_vars, levels, surface_vars):
    return [
        f"{variable}{level}"
        for variable in pl_vars
        for level in levels
    ] + list(surface_vars)


FUXI_ENS_CHANNELS = _expand_channels(
    PL_VARS, LEVELS, FUXI_ENS_SFC_VARS)
FUXI21_CHANNELS = _expand_channels(
    PL_VARS, LEVELS, FUXI21_SFC_VARS)


def grid_coords(grid):
    """模型网格契约 → 纬度、经度坐标数组。"""
    latitude = grid["lat"]
    longitude = grid["lon"]
    lat = (
        np.arange(latitude["size"], dtype=np.float64) * latitude["step"]
        + latitude["start"]
    )
    lon = (
        np.arange(longitude["size"], dtype=np.float64) * longitude["step"]
        + longitude["start"]
    )
    return lat, lon


# 模型层：名字 → (模块, 类)。按需导入，避免使用 PT2/AIFS 时强制安装 ONNX Runtime。
MODEL_REGISTRY = {
    "fuxi_ens_onnx": (
        "xmetai.models.fuxi_ens_onnx",
        "FuxiEnsOnnxModel",
    ),
    "fuxi21_pt2": (
        "xmetai.models.fuxi21_pt2",
        "Fuxi21Pt2Model",
    ),
    "aifs11_ckpt": (
        "xmetai.models.aifs11_ckpt",
        "Aifs11CkptModel",
    ),
    "iwc_fgvp_gdn2_onnx": (
        "xmetai.models.iwc_fgvp_gdn2",
        "IwcFgvpGdn2Model",
    ),
}


def get_model_class(model_name):
    """按模型名导入并返回具体模型类。"""
    target = MODEL_REGISTRY.get(model_name)
    if target is None:
        raise ValueError(f"未知模型 {model_name!r}（可选 {', '.join(MODEL_REGISTRY)}）")
    module_name, class_name = target
    return getattr(import_module(module_name), class_name)


def create_model(model_name, device_id=0, gpu_mem_fraction=0.7):
    """按模型名构造推理模型。"""
    cls = get_model_class(model_name)
    return cls(device_id=device_id, gpu_mem_fraction=gpu_mem_fraction)


__all__ = [
    "FUXI_ENS_CHANNELS",
    "FUXI21_CHANNELS",
    "GRID_025",
    "MODEL_REGISTRY",
    "create_model",
    "get_model_class",
    "grid_coords",
]
