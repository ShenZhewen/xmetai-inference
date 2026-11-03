# -*- coding: utf-8 -*-
"""Tongyun-det single-nc example: infer straight from one business-supplied ``.nc``.

tongyun_det == FGVP 改名同图（registry ``iwc_fgvp_gdn2_onnx``）：
78 通道（FUXI_ENS_CHANNELS / GRID_025 北→南 / 确定性单成员 / 物理进·物理出，
图内烘焙归一化）。这里输入走 ``dataset.type="example_nc"``：每个 nc = 一个起报窗，
全部历史帧都在 nc 内；init 时间从 nc 的 ``time`` 坐标末帧读，故 ``cfg.times`` 留空。

nc 必须是图的输入：物理量按 ``FUXI_ENS_CHANNELS`` 顺序、``GRID_025`` 朝向（北→南）、
float32、shape (time=history, channel=78, 721, 1440)，sst 已补缺测。用
``tools/make_tongyun_sample_nc.py --config weather_era_0p25_fgvp --store <c116归一化盘>``
从归一化 store 反归一化抽取得到（与 store 推理逐位一致）。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

# Placeholder paths; replace before deployment (the inference Job yaml overrides --out).
SRC_GRAPH = "/path/to/models/20261013/tongyun_det.onnx"
OPS_LIBRARY = "/path/to/models/xmetai_onnx_plugins.ort1_24.cuda12.so"
TONGYUN_NC_INPUT = "/path/to/input/input.nc"
OUTPUT_DIR = "/path/to/output/tongyun_example"


cfg = InferConfig(
    name="weather_example_tongyun",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,
    model_class="iwc_fgvp_gdn2_onnx",
    model_batch=1,
    gpu_mem=0.9,

    dataset={
        "type": "example_nc",
        "paths": [TONGYUN_NC_INPUT],
        "mode": "model_ready",
    },
    dataloader={
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # example_nc derives init time from the nc itself; times is intentionally empty.
    times="",
    steps=60,
    hour_interval=6,
    members=1,  # deterministic model — no ensemble members
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=1,
    cuda_devices="0",
    intra_op_threads=0,
    cudnn_algo_search="EXHAUSTIVE",
    output_dir=OUTPUT_DIR,
)