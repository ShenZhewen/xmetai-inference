# -*- coding: utf-8 -*-
"""FuXi-2.1 确定性推理配置（era5_foundation_store2 h6 多店）。

新 foundation store 是原始物理量；输入前处理由 dataset.processors 显式声明：
  unit_convert -> geometry -> normalize -> fill_missing
tp 在归一化前 log1p，输出 denormalize 继续用 expm1 恢复。
"""
import os

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = os.environ.get(
    "ERA5_STORE_ROOT",
    "/workspace/data/liujunjie/era5_foundation_store2",
)

FUXI21_DYNAMIC_STORES = [
    os.path.join(ERA5_ROOT, "era5_pl_2025.01-2026.07.c84.p25.h6.zarr"),
    os.path.join(ERA5_ROOT, "era5_sfc_2025.01-2026.07.c15.p25.h6.zarr"),
    os.path.join(ERA5_ROOT, "era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr"),
    os.path.join(ERA5_ROOT, "era5_soil_2025.01-2026.07.c4.p25.h6.zarr"),
    os.path.join(ERA5_ROOT, "era5_wave_2025.01-2026.07.c5.p25.h6.zarr"),
]

cfg = InferConfig(
    name="fuxi21",
    # 权重与归一化统计量（mean.nc/std.nc）**必须在同一个目录**：normalize/denormalize
    # 按 dirname(model_path) 找统计量（processing/tensor_processors.py），分开放会直接
    # FileNotFoundError。本机三样都在 xmetai-inference2 下。
    # 2026-09-30 更正：原默认指的「旧仓库 xmetai-inference（无 2）」在这台机器上不存在
    # （当时以为权重留在旧目录），实测权重/mean/std/input.nc 全在 2 号目录里，改过来。
    # 换机器用 FUXI21_MODEL_PATH 覆盖。
    model_path=os.environ.get(
        "FUXI21_MODEL_PATH",
        "/workspace/szwCode/xmetai-inference2/model_artifacts/fuxi2.1/fuxi-2.1.pt2",
    ),
    model_class="fuxi21_pt2",

    # 数据层：5 个动态 store 原样交给 processed_multi_zarr；
    # 物理量 -> 单位换算 -> 模型 mean/std 归一化 -> 归一化后填缺测 0。
    dataset={
        "type": "processed_multi_zarr",
        "paths": FUXI21_DYNAMIC_STORES,
        "processors": [
            {
                "name": "unit_convert",
                "scales": {
                    "q": 1000.0,
                    "ssr": 6.0 / 3600.0,
                    "ssrd": 6.0 / 3600.0,
                    "fdir": 6.0 / 3600.0,
                    "ttr": 6.0 / 3600.0,
                    "tp": 6000.0,
                },
            },
            {"name": "geometry"},
            {
                "name": "normalize",
                "mean_file": "mean.nc",
                "std_file": "std.nc",
                "log1p_channels": ["tp"],
            },
            {
                "name": "fill_missing",
                "rules": {"*": {"method": "constant", "value": 0.0}},
                "unconfigured": "keep",
            },
        ],
    },

    # Loader 层：只负责把不同起报时间组成 [B,T,C,H,W]。
    dataloader={
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    # 模型层：Dataset 已完成输入前处理；这里只保留循环清零和物理量输出恢复。
    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[
            {
                "name": "zero_channels",
                "channels": ["ssr", "ssrd", "fdir", "ttr", "tp"],
            },
        ],
        output=[
            {
                "name": "denormalize",
                "mean_file": "mean.nc",
                "std_file": "std.nc",
                "expm1_channels": ["tp"],
                "nonnegative": ["tp"],
            },
        ],
    ),
    times="2025010200..2025010500:24",
    steps=60,
    hour_interval=6,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,d2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/fuxi_single_output_new",
)
