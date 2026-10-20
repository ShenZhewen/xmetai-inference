# -*- coding: utf-8 -*-
r"""fgvp_ens_spw：集合模型 + SPW（权重扰动）的标准档。

成员 = n 份扰动图 × 每份图在同一 session 里跑 M 次前向（n=8, M=6 -> 48）。
同一份图的 M 个成员必须在同一进程同一 session 里连跑，不能按成员拆进程。
权重扰动与图内随机是两条独立轴；换卡数后不要与旧档逐位横比。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
# 只读源集合图（fp16、动态 batch 轴）与插件。
MODEL_DIR = "/workspace/tmp/douzsh/models"
SRC_GRAPH = f"{MODEL_DIR}/20261008/iwc_gssp_im_fp16.onnx"
OPS_LIBRARY = f"{MODEL_DIR}/xmetai_onnx_plugins.ort1_24.cuda12.so"
# 扰动权重的落盘根目录；框架按指纹在下面建子目录（member_dir 不进指纹）。
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

FGVP_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# 成员结构：n 份扰动图 x 每份 M 次图内前向
PERTURB_GRAPHS = 8      # n：扰动图份数（每卡 2 份）
RUNS_PER_GRAPH = 6      # M：每份图在每个起报上跑几条轨迹（= 图内随机样本）
MEMBERS = PERTURB_GRAPHS * RUNS_PER_GRAPH   # 48：总成员数，必须等 M

# 与确定性档 s2b 逐字相同（exclude 那 7 个）。
BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")


cfg = InferConfig(
    name="fgvp_ens_spw",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    # 必须 False：单步型图，GPU 常驻分支会绕过滑窗回填。
    gpu_state=False,
    # 一份图的 M 条轨迹按这个宽度做前向批；M=6 时一批正好跑完。
    model_batch=6,
    gpu_mem=0.9,
    model_class="iwc_fgvp_mamba_ens_onnx",
    weight_perturb={
        # 注入点与幅度沿用 s2b。
        "groups": ("latent_mixer",),
        "sigma_full": 0.027,
        "balance": "energy",
        "exclude": BASE_EXCLUDE,
        # 定义性字段：逐输出通道共享一个 xi，不是逐元素 iid。
        "noise_struct": "channel",
        "min_elements": 64,
        "member_dir": MODEL_ONNX_DIR,
        "seed": 0,
    },
    # >0 才走图外层编舞（集合模型专有；确定性模型配了会直接报错）。
    perturb_graphs=PERTURB_GRAPHS,
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
        # batch_size 只是 I/O 装载分组；前向批是上面的 model_batch。
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
    # 夏季 29 起报 + 冬季 30 起报（与 fgvp_spa / aBt15 同窗口）。
    times="2025070200..2025073000:24,2025110100..2025113000:24",
    steps=60,
    hour_interval=6,
    members=MEMBERS,
    vars="z500,u200,v200,msl,tp",
    # 4 卡：每卡领 2 份图（range(rank, G, 卡数)），卡间轨迹数完全相等。
    gpus=4,
    cuda_devices="0,1,2,3",
    intra_op_threads=0,
    cudnn_algo_search="",
    # 编舞逐图 load/release，同一时刻一张卡只有一份图是活的；2 留一个余量槽。
    resident_members=2,
    # 新成员数（48）与旧 29 起报 x 51 成员档不同，另开目录避免 resume 串档。
    output_dir="/workspace/data/shenzw/fgvp_ens_spw_std",
)
