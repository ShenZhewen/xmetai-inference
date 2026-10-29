# -*- coding: utf-8 -*-
"""fengshun2.0 集合推理（87 通道、日步长、图内自带归一化）。

输入盘 ``cra.1979_2024.c87.p25.h24.v1``，通道顺序与图内烘焙的 mean/std 逐位一致。
盘是 zscore 归一化态：数据侧 ``denormalize``（stats_path=盘），tp 走 ``expm1`` 还原
成 mm，图头/图尾自己归一化。不配 geometry（盘原生 90→−90 与 GRID_025 一致），
不配 zero_channels（tp 是纯输出通道）。两帧 [t-1,t] 进、单帧出。

备选盘 ``era5.1959_2025.c87`` 的 tp 在 2024-01-01 前后换口径（1h → 24h，约 24×），
sst 陆地还有 33.89% NaN，不要用。
"""
from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    FENGSHUN_CHANNELS, GRID_025, InferConfig, ModelProcessingConfig)


Fengshun20OnnxModel = OnnxInferModel.declare(
    "Fengshun20OnnxModel",
    model_name="fengshun20_onnx",
    channels=FENGSHUN_CHANNELS,
    grid=GRID_025,
    hour_interval=24,         # 日步长；history_steps=2 / offset=0 用默认值
    forecast_type="ensemble",
    members=11,
    windowed_output=False,    # 单帧输出，滑窗归 Rollout
    # hour/doy 喂原值（分母 1，doy 不截 365）；step / perturb_std_scale 用默认口径。
    time_features={
        "hour": {"divisor": 1.0},
        "doy": {"divisor": 1.0},
    },
)


FENGSHUN_ROOT = "/workspace/tmp/douzsh/models"
# 1979-01-01 … 2024-12-31，每日 00Z，zscore fp16 单 store。
FENGSHUN_STORE = "/workspace/data/zarr_refresh_v3/cra.1979_2024.c87.p25.h24.v1"

cfg = InferConfig(
    name="s2s_cra_0p25_fengshun20_ens",
    model_path=f"{FENGSHUN_ROOT}/fengshun2.0_260901.onnx",
    model_class="fengshun20_onnx",
    # 图里有 6 个 Mamba3SisoFn 自定义算子，只有 CUDA 实现，且与 ORT 1.24 ABI 绑定。
    ops_library=f"{FENGSHUN_ROOT}/xmetai_onnx_plugins.ort1_24.cuda12.so",

    dataset={
        "type": "processed_multi_zarr",
        "paths": [FENGSHUN_STORE],
        "processors": [
            # 归一化态 → 物理量；tp 由 log1p(mm) 还原成 mm。
            {
                "name": "denormalize",
                "stats_path": FENGSHUN_STORE,
                "expm1_channels": ["tp"],
            },
            # 兜底：sst 填气候均值（287.222 K = 归一化 0），其它通道出 NaN 报错。
            {
                "name": "fill_missing",
                "rules": {"sst": {"method": "constant", "value": 287.222}},
                "unconfigured": "error",
            },
        ],
    },
    dataloader={
        # 单样本 ≈ 0.7 GB，且每成员各持一份回填状态；batch_size 保持 1。
        "batch_size": 1,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 起报日可改这一行；最后一个 20241101 + 60 天正好到盘尾 2024-12-31。
    times="2024090100,2024100100,2024110100",
    steps=60,             # 一天一步，往后 60 天
    hour_interval=24,
    members=51,           # 一次前向一个成员
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    # 成员按 range(rank, 51, 4) 切：13/13/13/12。
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fengshun_output",
)
