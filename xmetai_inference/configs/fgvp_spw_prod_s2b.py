# -*- coding: utf-8 -*-
r"""fgvp_spw_prod_s2b：确定性 FGVP + SPW（m1a）的 S2B 生产档。

51 成员 / 29 起报（20250702-0730）/ 60 步 / 5 变量。成员集按扰动指纹落盘，
源图或扰动规格一变就整体重造；本配方与 fgvp_spw.py 共用同一套成员。
"""
from __future__ import annotations

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
MODEL_DIR = "/workspace/tmp/douzsh/models"
# 成员权重落盘根目录；框架在其下按指纹建子目录。member_dir 不进指纹，
# 换父目录不会触发重造。
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

FGVP_SPW_PROD_S2B_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# S2B 的 exclude（drop 为空 -> 就是这 7 个）；与 fgvp_spw_ab 一致才能命中同一成员集。
BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")


cfg = InferConfig(
    name="fgvp_spw_prod_s2b",
    # 原始确定性图；框架按指纹找 <模型名>.spw-members-<指纹>/<模型名>.mNNN.onnx。
    model_path=f"{MODEL_DIR}/iwc_fgvp_gdn2.onnx",
    ops_library=f"{MODEL_DIR}/xmetai_onnx_plugins.so",
    # 必须 False：图只出预报那一帧（单步型），GPU 常驻分支会绕过滑窗回填。
    gpu_state=False,
    model_class="iwc_fgvp_gdn2_onnx",
    weight_perturb={
        # 注入点：latent_mixer（携带状态的 recurrent 核）。
        "groups": ("latent_mixer",),
        # 显式写空，与 A/B 那一轮逐字段一致（() 与不写等价）。
        "include": (),
        # 改它会换指纹 -> 另造一整套 51 成员，必须留痕。
        "sigma_full": 0.027,
        "balance": "energy",
        "exclude": BASE_EXCLUDE,
        # 逐输出通道共享一个 xi；非默认值进指纹，不会命中 iid 的成员集目录。
        "noise_struct": "channel",
        "min_elements": 64,
        # 不在 budgets() 里：换父目录不改指纹。
        "member_dir": MODEL_ONNX_DIR,
        "seed": 0,
    },
    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_SPW_PROD_S2B_DYNAMIC_STORES,
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
        # m1a 是成员外层、数据集内层；预取队列 ~= workers × 2 × batch × 645 MB/样本。
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },
    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[],
    ),
    # 29 个起报（20250702-0730，日间隔 24h）。
    times="2025070200..2025073000:24",
    # 60 步 x 6 h = +360 h：要覆盖论文报告失效模式的 +240/+360 h。
    steps=60,
    hour_interval=6,
    # 51 成员 = 落盘 51 套权重；同一张卡同时驻留几份由 resident_members 定。
    members=51,
    # 五个变量；每帧 20.8 MB，产物约 1.85 TB（只存 z500 是 368 GB）。
    # vars 进 plan 指纹，改它等于换一份计划、整批重算。
    vars="z500,u200,v200,msl,tp",
    # 容器里真实可见的卡号（跑前 nvidia-smi -L 确认）；更换卡数用 --gpus 覆盖，不进
    # plan 指纹，不影响续跑。
    gpus=2,
    cuda_devices="0,1",
    # 每个成员一份权重 + session；claim_order=member 下一时刻只活一个成员，
    # 留 2 个槽位作余量即可。
    resident_members=2,
    # 正式输出目录（与 F 的生产目录分开，也与试跑目录分开）。
    output_dir="/workspace/data/shenzw/fgvp_spw_prod_s2b",
)
