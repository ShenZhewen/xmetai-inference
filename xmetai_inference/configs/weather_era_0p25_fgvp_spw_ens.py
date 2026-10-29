# -*- coding: utf-8 -*-
"""确定性 FGVP + SPW（权重扰动，m1a）的标准档，51 成员。

成员集按指纹复用 S2B；只在该配方上改时段 / 卡数 / 输出目录。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
MODEL_DIR = "/workspace/tmp/douzsh/models"
SRC_GRAPH = f"{MODEL_DIR}/iwc_fgvp_gdn2.onnx"
OPS_LIBRARY = f"{MODEL_DIR}/xmetai_onnx_plugins.so"  # 进成员指纹；与 S2B 一致才能复用成员
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"  # 权重根目录（member_dir 不进指纹）

FGVP_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# S2B 的配方（drop 为空）。
MEMBERS = 51
BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")


cfg = InferConfig(
    name="weather_era_0p25_fgvp_spw_ens",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,  # 单步型图，GPU 常驻会绕过滑窗回填
    model_class="iwc_fgvp_gdn2_onnx",
    weight_perturb={
        "groups": ("latent_mixer",),  # 携带状态的 recurrent 核
        "include": (),
        "sigma_full": 0.027,  # S2B 反推的靶心档
        "balance": "energy",
        "exclude": BASE_EXCLUDE,
        "noise_struct": "channel",  # 逐输出通道共享一个 xi（进指纹）
        "min_elements": 64,
        "member_dir": MODEL_ONNX_DIR,
        "seed": 0,
    },
    perturb_graphs=0,  # 确定性模型必须为 0（>0 走集合图编舞，会被挡下）

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
                "rules": {"sst": {"method": "constant", "value": 287.066}},
                "unconfigured": "error",
            },
        ],
    },
    dataloader={
        # batch_size 只是 I/O 装载分组（起报数）；前向批见 model_batch。
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 夏季 29 起报 + 冬季 30 起报（与 fgvp_spa / fgvp_ens_spw 同窗口）。
    times="2025070200..2025073000:24,2025110100..2025113000:24",
    steps=60,
    hour_interval=6,
    members=MEMBERS,
    vars="z500,u200,v200,msl,tp",
    gpus=4,  # 成员按 range(rank, members, 4) 切
    cuda_devices="0,1,2,3",
    intra_op_threads=0,
    cudnn_algo_search="",
    model_batch=1,  # m1a 下批的是起报；GDN2 图的 batch 轴未核，保持逐条前向
    resident_members=2,
    gpu_mem=0.9,
    output_dir="/workspace/data/shenzw/fgvp_spw_std",
)
