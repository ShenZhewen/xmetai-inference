# -*- coding: utf-8 -*-
r"""fgvp_spw：**确定性模型 + SPW（权重扰动）** 的标准档 —— 完整自包含。

SPW 有两种用法，本文件是第一种：

  1) **确定性模型 + SPW**（m1a，成员在外层，本文件）
     扰动 N 份权重 = N 个成员；每份冻结权重要跑完全部起报，每个起报只出 1 条
     轨迹，一个成员就是一份 onnx。本档 N = ``MEMBERS`` = **51**。

  2) **集合模型 + SPW**（图在外层）
     成员 = n 份扰动图 x 每份图跑 M 次前向 = n x M，见 ``configs/fgvp_ens_spw.py``。
     那条路要求模型类 ``forecast_type="ensemble"``，确定性模型配 ``perturb_graphs>0``
     会直接报错 —— 所以本文件显式写 ``perturb_graphs=0``。

---- 配方：S2B，逐字段与 ``fgvp_spw_prod_s2b.py`` 相同 ----

  groups="latent_mixer" + noise_struct="channel" + sigma_full=0.027 + seed=0。
  这是五轮实验里唯一「两头都不输」的配置（细节与判据见那份 config 的文件头 /
  perturbation/README 2F 节）。**不要为了不重建权重去改这些字段以外的写法**：
  groups / sigma_full / balance / exclude / include / min_elements / seed /
  noise_struct / 源 .onnx 的 size/mtime 都在 ``weights.spw`` 的指纹里。

  指纹一致 ⇒ **直接复用 S2B 那套已落盘的 51 个成员**（member_dir 相同、
  源模型相同），不重造。反过来，换 ``model_path`` / ``ops_library`` 指向的源文件
  会改指纹 ⇒ 重造 51 x ~1.49 GB ≈ 76 GB，还要重跑成员权重复现性检查。

---- 与生产档 ``fgvp_spw_prod_s2b`` 的差别（只有跑法，没有扰动规格）----

  * 时段：标准两季窗口 —— 夏季 20250702-0730（29 起报）+ 冬季 20251101-1130
    （30 起报）= **59 起报**。评测侧必须同步；
  * 4 卡（s2b 是 2 卡），成员按 rank 切 13/13/13/12；
  * 输出目录另开（``fgvp_spw_std``），与 s2b 的产物、计划标记互不干扰；
  * ``model_batch=1``：m1a 是「成员外层、起报在内层」，这里批的是**起报**。
    s2b 的生产跑没有开它；GDN2 这张图的 batch 轴是否是符号维没有核过
    （后端 load 时实测，静态就 WARNING 退回 1）。要提速先确认动态 batch 再改。
  * dataloader 收到 2 x 2（s2b 是 4 x 4）：4 个 worker 进程各带一套预取队列，
    每条样本约 645 MB，4 x 2 x 4 的队列在 4 卡并发下会把主机内存吃满。

---- 预算 ----

  输出：59 x 51 x 60 x 20.8 MB ≈ **3.8 TB**（5 变量 z500/u200/v200/msl/tp）。
  成员权重：复用 S2B 已有集合则不新增；缺失时 51 x 1.49 GB ≈ 76 GB。
  前向次数：59 x 51 x 60 = 180 540。

---- 跑法（服务器）----

    cd /workspace/szwCode/xmetai-inference2
    nvidia-smi -L                      # 确认真实可见卡号与 cuda_devices 一致

    # smoke：1 起报 / 3 成员 / 6 步，务必给 --out（否则会写进正式目录）
    xmetai-infer --model fgvp_spw --times 2025070200 --members 3 --steps 6 \
        --gpus 1 --cuda-devices 0 --out /tmp/smoke_spw

    # 正式
    xmetai-infer --model fgvp_spw
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

# ---------------------------------------------------------------------------
# 路径（服务器上的固定位置，直接改这里）
# ---------------------------------------------------------------------------
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
# 源确定性图 + 插件（只读）。**这两个路径进成员指纹**：与 S2B 生产档逐字相同，
# 才能认回已有的 51 个成员；挪文件 = 改 size/mtime = 整份重造。
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
    # 成员外层 + claim_order=member：同一时刻每个 rank 只有一个成员是活的，
    # 池子只需要一个余量槽（s2b 的实测结论，见那份 config）。
    resident_members=2,
    gpu_mem=0.9,
    # 新时段 / 新成员编排，另开目录，避免与 s2b 的 resume 计划串档。
    output_dir="/workspace/data/shenzw/fgvp_spw_std",
)
