# -*- coding: utf-8 -*-
"""Pangu-Weather 确定性推理（方案B：6h 步进 + 24h 跳步）。

模型契约见 models/pangu_onnx.py 模块 docstring（双输入上层 5×13 + 地面 4 通道、
物理量进/出、图内归一化、q=kg/kg 不 ×1000、z=m²/s² geopotential、网格 = GRID_025）。

数据源 = era5_foundation_store2 的 pl + sfc（与 FengQing 同源），但**没有** tp 补充
store —— Pangu 无降水通道。dataset.processors 只留 geometry（把 lon 从 [-180,180)
重排到 [0,360)），不做单位换算 / 归一化。

方案B 由模型类在 forward 内实现：framework step 粒度 = 6h，第 4 步（lead 24h）起用
24h 模型从锚点直接跳，详见 pangu_onnx.py。所以这里的 steps 是 6h 步数，不是 24h 步数。

License：BY-NC-SA 4.0，商用禁止（见 model_artifacts/Pangu-Weather-main/README.md）。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

PANGU_ROOT = "/workspace/szwCode/xmetai-inference2/model_artifacts/Pangu-Weather-main"
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

PANGU_STORES = [
    ERA5_ROOT + "/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ERA5_ROOT + "/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]

cfg = InferConfig(
    name="pangu",
    # 6h 步进模型；24h 模型由模型类从同目录推导（pangu_weather_24.onnx），
    # 可用环境变量 PANGU_ONNX_24 覆盖。
    model_path=PANGU_ROOT + "/pangu_weather_6.onnx",
    model_class="pangu_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": PANGU_STORES,
        "processors": [
            {"name": "geometry"},
        ],
    },

    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    # 物理量进/出：输入归一化（input）、回填（recurrent）、反归一化（output）全空。
    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[],
    ),

    # ---- 首轮实测：2 个起报 × 4 步（lead 6/12/18/24h，正好走完一轮 24h 跳步）----
    # 验证点：
    #   1) z500 ≈ 5e4、msl ≈ 1e5（物理量量级），证明「图内归一化」成立、q 没误 ×1000；
    #   2) step 4（lead 24h）走了 24h 分支没报错、量级不崩。
    # 实测通过后，把 times 扩到整段、steps 改成 40（= 10 天）跑正式预报。
    times="2025010200..2025010300:24",
    steps=60,
    hour_interval=6,
    # 确定性模型必须为 1（同 FengQing：>1 会落到基类成员路径，喂错键名）。
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl",
    gpus=1,
    cuda_devices="0",
    output_dir="/workspace/data/shenzw/pangu_output",
)