# -*- coding: utf-8 -*-
"""Tongyun 集合图 + SPA（潜空间 AR(1) 强迫）。

78 通道 / 2 帧输入 / 1 帧输出 / 0.25°；图内自带 RandomNormalLike，
SPA 在其上增加时间相关的潜空间强迫。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

FGVP_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]
# Tongyun 20261013 导出（不是 20261008 的 iwc_gssp_im 那张图）
SRC_GRAPH = "/workspace/tmp/douzsh/models/20261013/tongyun_ens_era5_fp16.onnx"
OPS_LIBRARY = "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

cfg = InferConfig(
    name="fgvp_ens_spa",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,  # 单帧输出图，不能开 GPU 常驻
    model_class="iwc_fgvp_mamba_ens_onnx",
    weight_perturb=None,
    latent_forcing={
        "targets": ("add_35722", "add_35727", "add_35732"),  # 三组 decoder 入口
        "sigma": 0.04,
        "tau_days": 3.0,
        "rank": 16,
        "length_scale": 32.0,
        "seed": 0,
        "hour_interval": 6.0,
        "verify_on_load": True,
        "cache_dir": MODEL_ONNX_DIR,
        "amp_profile": (
            (0, 1.76),
            (4, 2.62),
            (8, 2.34),
            (12, 2.36),
            (20, 2.68),
            (28, 3.31),
            (40, 5.10),
            (60, 7.05),
        ),
    },
    perturb_graphs=0,
    model_batch=6,
    gpu_mem=0.9,
    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_DYNAMIC_STORES,
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
                "name": "fill_missing",
                "rules": {
                    "sst": {
                        "method": "constant",
                        "value": 287.066,
                    },
                },
                "unconfigured": "error",
            },
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
    times="2025070200..2025073000:24,2025110100..2025113000:24",
    steps=60,
    hour_interval=6,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    intra_op_threads=0,
    cudnn_algo_search="EXHAUSTIVE",  # Tongyun stem 卷积在 HEURISTIC 下建不起来
    resident_members=2,
    output_dir="/workspace/data/shenzw/fgvp_spa_tongyun_spa",
)
