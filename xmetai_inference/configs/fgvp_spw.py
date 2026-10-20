# -*- coding: utf-8 -*-
r"""fgvp_spw：确定性模型 + SPW（权重扰动，m1a）的标准档。

扰动 N 份权重 = N 个成员，每个成员是独立 onnx；成员集按指纹复用 S2B 那套。
配方与 fgvp_spw_prod_s2b.py 一致，本档只改时段 / 卡数 / 输出目录。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
# 源确定性图 + 插件（只读），这两个路径进成员指纹：与 S2B 一致才能认回已有成员。
MODEL_DIR = "/workspace/tmp/douzsh/models"
SRC_GRAPH = f"{MODEL_DIR}/iwc_fgvp_gdn2.onnx"
OPS_LIBRARY = f"{MODEL_DIR}/xmetai_onnx_plugins.so"
# 成员权重根目录；框架按指纹在下面建 <模型名>.spw-members-<指纹>/ 子目录。
# member_dir 不进指纹，换父目录不会重造。
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

FGVP_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# S2B 的配方（= fgvp_spw_ab 的 S2B 臂，drop 为空）。
MEMBERS = 51
BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")


cfg = InferConfig(
    name="fgvp_spw",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    # 必须 False：单步型图，GPU 常驻分支会绕过滑窗回填。
    gpu_state=False,
    model_class="iwc_fgvp_gdn2_onnx",
    weight_perturb={
        # 注入点：latent_mixer（携带状态的 recurrent 核）。
        "groups": ("latent_mixer",),
        # 显式写空，与 S2B 逐字段一致（() 与不写等价，但肉眼可对）。
        "include": (),
        # S2B 的压强：短时效 CRPS 代价对 sigma 的斜率反推出的靶心档。
        "sigma_full": 0.027,
        "balance": "energy",
        "exclude": BASE_EXCLUDE,
        # 定义性字段：逐输出通道共享一个 xi，不是逐元素 iid（非默认值进指纹）。
        "noise_struct": "channel",
        "min_elements": 64,
        "member_dir": MODEL_ONNX_DIR,
        "seed": 0,
    },
    # 确定性模型必须为 0：>0 走「集合图 x 图内成员」的编舞，会被模型类型挡下。
    perturb_graphs=0,
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
        # batch_size 只是 I/O 装载分组（起报数）；前向批是 model_batch。
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },
    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[],
    ),
    # 夏季 29 起报 + 冬季 30 起报（与 fgvp_spa / fgvp_ens_spw 同窗口）。
    times="2025070200..2025073000:24,2025110100..2025113000:24",
    steps=60,
    hour_interval=6,
    members=MEMBERS,
    vars="z500,u200,v200,msl,tp",
    # 4 卡：成员按 range(rank, members, 4) 切 -> 13/13/13/12。
    gpus=4,
    cuda_devices="0,1,2,3",
    intra_op_threads=0,
    cudnn_algo_search="",
    # m1a 下批的是起报；GDN2 图的 batch 轴未核，保持生产档的逐条前向。
    model_batch=1,
    # 一时刻只活一个成员，池子只需要一个余量槽。
    resident_members=2,
    gpu_mem=0.9,
    # 新时段 / 新成员编排，另开目录，避免与 s2b 的 resume 计划串档。
    output_dir="/workspace/data/shenzw/fgvp_spw_std",
)
