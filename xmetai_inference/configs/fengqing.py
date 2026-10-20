# -*- coding: utf-8 -*-
"""FengQing V1.5Beta（归一化前移）确定性推理配置。

tp 走补充 store（6h 累积 m，×1000 转 mm）。输入归一化在数据层，图出归一化残差，
反演在模型类内（需上一帧 + residual std），故 processing 只留输出端 denormalize。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

# 服务器绝对路径：直接 `python cli.py --config fengqing` 启动。
FENGQING_ROOT = "/workspace/szwCode/xmetai-inference2/model_artifacts/FengQing"
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

# 顺序敏感：MultiZarrDataset 按 channel_values.index(v) 取「第一个同名」通道，
# 新 tp store（6h 累积 m）排最前，旧 sfc store 里的 tp 会被顶掉。
TP_ROOT = "/workspace/data/liujunjie/era5_tp_zarr"
FENGQING_STORES = [
    TP_ROOT + "/era5.202501-202607.c1.p25.h6.v1.lon180.zarr",
    ERA5_ROOT + "/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ERA5_ROOT + "/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]

cfg = InferConfig(
    name="fengqing",
    model_path=FENGQING_ROOT + "/onnx/fengqing_pre.onnx",
    model_class="fengqing_pre_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": FENGQING_STORES,
        "processors": [
            {
                "name": "unit_convert",
                "scales": {
                    "tp": 1000.0,  # 新 tp 已是 6h 累积 m，只 ×1000 转 mm
                },
            },
            {"name": "geometry"},
            {"name": "normalize_fengqing"},
        ],
    },

    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    # 输入归一化在 dataset.processors（normalize_fengqing），反归一化在这里。
    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[{"name": "denormalize_fengqing"}],
    ),

    # 起报数必须 = gpus × batch_size = 4（cli 按起报时间切；起报数 < gpus 时部分
    # rank 分不到任务直接退出）。`:` 后是起报间隔，与 steps/hour_interval 无关。
    times="2025010200..2025122800:24",
    # hour_interval=6 是模型契约（FengQing 类属性同为 6），不可改：改成 24 会让
    # 数据层按 24h 取历史窗、rollout 的 valid_time 也跟着错。
    steps=60,          # 预报时长 15 天 = 60 × 6h
    hour_interval=6,
    # 确定性模型必须为 1：>1 会落到基类成员路径，喂错键名（"input"）。
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    # eval 时必须显式传 --forecast 指向这里，否则会评到默认目录的旧数据。
    output_dir="/workspace/data/shenzw/fengqing_output",
)
