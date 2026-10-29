# -*- coding: utf-8 -*-
"""集合图 + SPW（权重扰动）标准档。

成员 = n 份扰动图 × 每份图同 session 跑 M 次前向；同一份图的 M 个成员必须同进程
同 session 连跑，不能按成员拆进程。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
MODEL_DIR = "/workspace/tmp/douzsh/models"
SRC_GRAPH = f"{MODEL_DIR}/20261008/iwc_gssp_im_fp16.onnx"  # fp16、动态 batch 轴
OPS_LIBRARY = f"{MODEL_DIR}/xmetai_onnx_plugins.ort1_24.cuda12.so"
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"  # 扰动权重根目录（不进指纹）

FGVP_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# 成员结构：n 份扰动图 × 每份 M 次图内前向
PERTURB_GRAPHS = 8      # 扰动图份数（每卡 2 份）
RUNS_PER_GRAPH = 6      # 每份图每起报的轨迹数
MEMBERS = PERTURB_GRAPHS * RUNS_PER_GRAPH   # 48，必须 = M

# 与确定性档 s2b 逐字相同的 exclude。
BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")


cfg = InferConfig(
    name="weather_era_0p25_fgvp_ens_spw",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,  # 单步型图，GPU 常驻会绕过滑窗回填
    model_class="iwc_fgvp_mamba_ens_onnx",
    model_batch=6,  # 一份图的 M 条轨迹合成一个前向批
    gpu_mem=0.9,
    weight_perturb={
        "groups": ("latent_mixer",),
        "sigma_full": 0.027,
        "balance": "energy",
        "exclude": BASE_EXCLUDE,
        "noise_struct": "channel",  # 逐输出通道共享一个 xi
        "min_elements": 64,
        "member_dir": MODEL_ONNX_DIR,
        "seed": 0,
    },
    perturb_graphs=PERTURB_GRAPHS,  # >0 才走图外层编舞；确定性模型配了会报错

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
        # batch_size 只是 I/O 装载分组；前向批是上面的 model_batch。
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 夏季 29 起报 + 冬季 30 起报（与 fgvp_spa / fgvp_spw_ens 同窗口）。
    times="2025070200..2025073000:24,2025110100..2025113000:24",
    steps=60,
    hour_interval=6,
    members=MEMBERS,
    vars="z500,u200,v200,msl,tp",
    gpus=4,  # 每卡领 2 份图（range(rank, G, 卡数)），卡间轨迹数完全相等
    cuda_devices="0,1,2,3",
    intra_op_threads=0,
    cudnn_algo_search="",
    resident_members=2,  # 编舞逐图 load/release，同一时刻一卡只有一份图活着
    output_dir="/workspace/data/shenzw/fgvp_ens_spw_std",
)
