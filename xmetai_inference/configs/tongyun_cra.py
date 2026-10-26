# -*- coding: utf-8 -*-
r"""Tongyun 集合图（CRA 版），71 通道，数据源是 CRA c71 zarr store。

store 存归一化值、图内烘焙归一化，故 processors 里先 denormalize（stats_path=该 store）；
输出已是物理量，model_processing 三阶段都空。纬度朝向南→北（GRID_025_S），**不要配
geometry**（会把场喂反，+6h z500 RMSE 90→486）。cudnn_algo_search 必须 EXHAUSTIVE。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

# CRA-40 再分析 c71 store（zarr v3；lat −90→90 南→北，与模型的 GRID_025_S 一致）
CRA_STORE = "/workspace/data/zarr_refresh_v3/cra.19790101_20260714.c71.p25.h6.v1"
SRC_GRAPH = "/workspace/tmp/douzsh/models/20261019/tongyun_ens_cra_fp16.onnx"  # 20261019 导出
OPS_LIBRARY = "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"

cfg = InferConfig(
    name="tongyun_cra",
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
        # store(归一化) -> 物理量（tp 仍是 log1p 空间，图要的就是这个）；
        # 不配 geometry（图就在南→北朝向上训练/导出）。
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