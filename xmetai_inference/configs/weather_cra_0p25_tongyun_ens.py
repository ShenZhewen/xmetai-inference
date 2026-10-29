# -*- coding: utf-8 -*-
r"""Tongyun 集合图（CRA 版），71 通道，数据源是 CRA c71 store。

store 存归一化值、图内烘焙归一化：数据侧先 ``denormalize``（stats_path=store），
输出已是物理量；tp 停在 log1p 空间，**不要 expm1**。纬度原生南→北（GRID_025_S），
**不要配 geometry**（会把场喂反：+6h z500 RMSE 90→486）。

单帧输出 ⇒ gpu_state 必须 False；成员由图内 RandomNormalLike 自带。
cudnn_algo_search 必须 EXHAUSTIVE（HEURISTIC 在 stem 卷积上失败）。
"""
from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    CRA_CHANNELS, GRID_025_S, InferConfig, ModelProcessingConfig)


TongyunCraEnsModel = OnnxInferModel.declare(
    "TongyunCraEnsModel",
    model_name="tongyun_cra_ens_onnx",
    channels=CRA_CHANNELS,
    grid=GRID_025_S,                 # 南→北，与 CRA store 原生朝向一致
    forecast_type="ensemble",
    members=51,
    windowed_output=False,           # 单帧输出，滑窗归 Rollout；与 gpu_state=False 绑定
)


# CRA-40 c71 store（zarr v3，lat −90→90 南→北）
CRA_STORE = "/workspace/data/zarr_refresh_v3/cra.19790101_20260714.c71.p25.h6.v1"
SRC_GRAPH = "/workspace/tmp/douzsh/models/20261019/tongyun_ens_cra_fp16.onnx"
OPS_LIBRARY = "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"

cfg = InferConfig(
    name="weather_cra_0p25_tongyun_ens",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,  # 单帧输出图，不能开 GPU 常驻
    model_class="tongyun_cra_ens_onnx",
    perturb_graphs=0,
    model_batch=6,
    gpu_mem=0.9,

    dataset={
        "type": "processed_multi_zarr",
        "paths": [CRA_STORE],
        # store(归一化) -> 物理量（tp 仍是 log1p 空间，图要的就是这个）；不配 geometry。
        "processors": [
            {"name": "denormalize", "stats_path": CRA_STORE},
        ],
    },
    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 起报时段到 2026-07-14 06Z；smoke 用 --times 覆盖。
    times="2026070200",
    steps=60,
    hour_interval=6,
    members=51,
    vars="z500,t2m,msl,tp",
    gpus=1,
    cuda_devices="0",
    intra_op_threads=0,
    cudnn_algo_search="EXHAUSTIVE",  # Tongyun stem 卷积在 HEURISTIC 下建不起来
    resident_members=2,
    output_dir="/workspace/data/shenzw/tongyun_cra",
)
