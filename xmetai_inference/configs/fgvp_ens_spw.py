# -*- coding: utf-8 -*-
r"""fgvp_ens_spw：**集合模型 + 权重扰动** —— 把 SPW 叠到 FGVP 集合图（图内随机算子）上。

思路（2026-10-12 定档）：fgvp_ens 的 51 个成员全靠图内随机算子，而「不同 session 的
随机序列逐位相同」（README 坑 1）意味着多卡拆成员后，成员多样性只剩每进程的
ceil(51/卡数) 条独立轴——这是「FGVP 集合 spread 只有 AIFS 的 1/4」的头号嫌疑
（fgvp_ens.py 文件头的待验怀疑）。本配置把 m1a 的冻结权重作为**第二条独立轴**叠上去：

    成员 m = 第 g 份扰动图（冻结权重，g = 摊分序号）× 该图 session 内的第 j 次前向
    （图内随机算子），m = g * J + j 由 cli._graph_member_split 摊好（51/5 = 11+10x4）。

即使所有 session 的图内随机流逐位相同，不同权重图出的成员照样不同——成员多样性
不再赌跨进程 RNG 分歧。编舞是「图外层」（cli 的 _run_graph_outer）：每份图一个
session 驱动 J 条轨迹（复用 _run_plain_block 的 members>1 分支，含 rollout_batch 前向批），
多卡按**图**切（range(rank, G, 卡数)），一图的成员永不跨进程。

消融轴就在 `PERTURB_GRAPHS` 一个字面量上：0/不写 = 纯 fgvp_ens（对照）；
1 = 扰动一份图服务全部成员；5 = 生产档；10 = 「10 图 × ~5 成员」的另一种拆法。

⚠⚠ 开跑前必须先在服务器上做的三件事（groups / σ 是从确定性 gdn2 图标定的，
**没有**在集合图上验证过）：

  1) 看集合图的注入组（latent_mixer 是 gdn2 的张量名前缀，集合图是 Mamba 骨干，
     组名结构未必相同）：
       P="python -m xmetai_inference.perturbation"
       $P groups /workspace/tmp/douzsh/models/20261008/iwc_gssp_im_fp16.onnx
     下面的 weight_perturb["groups"] 按这个输出改。
  2) σ 不许直接抄 0.027：集合图自带图内离散度，σ̂ 目标值与确定性档不同。先用探针
     量基线再定（$P probe 的 --m1a-members 8）。
  3) 零成本旧账：fgvp_ens.py 文件头那条「挑一个已完整的起报日，比对
     member_000/001/002/003 的 060.nc 是否逐字节相同」还没人做。做了它才知道图内
     随机轴的真实自由度——若跨 rank 序列相同，(g, j) 两轴是**耦合**的（各图共享同一
     套图内噪声序列），方案照样成立但解读要变。

预算与算力（2 x H100 口径可参考 fgvp_spw_prod_s2b 的实测锚点）：

  * 成员集：PERTURB_GRAPPS 份（不是 51 份！）× 每份被扰动张量字节数 —— 具体多大
    取决于集合图的组结构（第 1 步的输出），gdn2 上 latent_mixer 是 1.49 GB/份，
    集合图按 groups 实测后再算。这是本方案相对 51 份成员集（~76 GB）的主要节省。
  * 前向总量与 fgvp_ens 完全相同：起报 × 51 × 60，一个不多。
  * 输出 29 起报 × 51 成员 × 60 帧 × 20.8 MB（5 变量）≈ **1.85 TB**，与 S2B 生产档
    同口径（开跑前确认卷装得下）。
  * 每 rank 显存：1 份图 + J 条轨迹 state（J=11 × 645 MB ≈ 7 GB）+ rollout_batch 的
    前向批（k x 648 MB），与 fgvp_ens 每 rank 7 条成员一个量级。

跑法（服务器，先 smoke 再正式 —— **smoke 务必加 --out**，别写进生产目录）：

    cd /workspace/szwCode/xmetai-inference2
    nvidia-smi -L                     # 卡号与 cuda_devices 对齐

    # 先验链路（1 起报 / 全部图 × 少量成员 / 6 步，几分钟）：
    xmetai-infer --model fgvp_ens_spw --times 2025070200 --steps 6 \
        --members 5 --gpus 1 --cuda-devices 0 \
        --out /workspace/data/shenzw/fgvp_ens_spw_smoke
    # members=5 / graphs=5 => 每图 1 个成员会被「每图至少 2 成员」的校验拦下，
    # smoke 想造满 G 份图就用 --members 10（每图 2 个）。

    # 正式：
    xmetai-infer --model fgvp_ens_spw

跑完之后（顺序：先对账、再谈指标）：

  1. 体检：python -m xmetai_inference.perturbation check <out_dir>（成员互异、帧数齐全）。
  2. 跨图独立性抽查：同一图内两成员差 vs 跨图两成员差（后者应显著大于前者；
     若跨图成员差≈0，说明成员集没接上、所有图加载了同一份权重——加载期自检
     本该拦住，这里双保险）。
  3. spread 曲线 + 与 fgvp_ens 原档并排（$P spread 多目录对比），看 1/4 欠离散
     是否被第二条轴抬起来。
  4. 评测走 xmetai-evaluation-v2 现成裁判（口径同 spw_prod_s2b 那套）。

本配置与 fgvp_ens.py 的 diff 只有：weight_perturb 块、PERTURB_GRAPHS、
member_dir / output_dir、resident_members、times 窗口。数据集 / dataloader /
rollout_batch / gpu 旋钮逐字照抄 fgvp_ens（第 5 轮已验证的配方）。
"""
from __future__ import annotations

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
MODEL_DIR = "/workspace/tmp/douzsh/models"
# 扰动权重（<stem>.mNNN.onnx / perturbed.mNNN.data）的落盘根目录。框架按指纹在下面
# 建子目录（与 fgvp_spw_prod_s2b 同一套协议；换 member_dir 不改指纹）。
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

FGVP_ENS_SPW_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

MEMBERS = 51
# 扰动图数 G：members 摊到 G 份图上（51/5 = 11+10x4）。改这个值会**改成员集份数**
# 但不改指纹（指纹只含 spec：groups/sigma/seed/...，不含 G）——同一 spec 目录下
# 造满 G 份即可，加 G 不需要重造已有的那些份。
# 卡数对齐：G >= gpus 才不空卡（5 图配 5 卡；8 卡想用满就 G=8 或 10）。
PERTURB_GRAPHS = 5

# 与 fgvp_spw_prod_s2b 逐字相同（exclude 那 7 个）。是否适用于集合图见文件头 ⚠ 第 1 步。
BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")


cfg = InferConfig(
    name="fgvp_ens_spw",
    # ⚠ 源是**集合图**（iwc_gssp_im 的 **fp16** 版，图内随机算子出成员），不是 s2b
    # 用的确定性 iwc_fgvp_gdn2.onnx。源进指纹（size/mtime），挪动 = G 份图全部重造。
    # fp16 + 动态 batch 轴：一次 forward 堆 k 条轨迹（下面的 rollout_batch）。
    # 批的仍是**同图的成员**（图外层编舞下每 rank 一份图、J 条轨迹），k 的有效上限
    # 就是 J（51/5 -> 11），再大最后一块只装余数。显存按 k 倍涨（fp16 减半），
    # k 超出 batch_axis_dynamic 的实测能力时 runner 会打 WARNING 退回 1。
    model_path=f"{MODEL_DIR}/20261008/iwc_gssp_im_fp16.onnx",
    ops_library=f"{MODEL_DIR}/xmetai_onnx_plugins.ort1_24.cuda12.so",
    # 必须 False：单步型图，GPU 常驻分支绕过滑窗回填（同 fgvp_ens 的理由）。
    gpu_state=False,
    rollout_batch=6,
    gpu_mem=0.9,
    model_class="iwc_fgvp_mamba_ens_onnx",
    weight_perturb={
        # ⚠⚠ 占位值：latent_mixer 是 gdn2 图的张量组名。集合图（Mamba 骨干）的组
        # 结构没探过 —— 先跑文件头 ⚠ 第 1 步的 groups 命令，按输出改这里。
        "groups": ("latent_mixer",),
        # ⚠⚠ 占位值：0.027 是确定性档在「短时效 CRPS 代价 vs sigma 斜率」上的靶心。
        # 集合图自带图内离散度，sigma 应显著**小于**它 —— 先 probe 再定，改值留痕。
        "sigma_full": 0.027,
        "balance": "energy",
        "exclude": BASE_EXCLUDE,
        "noise_struct": "channel",
        "min_elements": 64,
        "member_dir": MODEL_ONNX_DIR,
        "seed": 0,
    },
    # 集合模型 + 权重扰动的编舞开关（见 configs/base.py 的字段说明与本文件头）。
    perturb_graphs=PERTURB_GRAPHS,
    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_ENS_SPW_DYNAMIC_STORES,
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
        "batch_size": 4,
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
    # 与 fgvp_spw_prod_s2b 同窗口（20250702-0730 的 29 个起报）：spw_prod_s2b / AIFS /
    # fgvp_ens 归档都在这个窗口上有数，三方并排读。全年口径照 fgvp_ens.py 写
    # "2025010200..2025121500:24"。
    times="2025070200..2025073000:24",
    steps=60,
    hour_interval=6,
    members=MEMBERS,
    vars="z500,u200,v200,msl,tp",
    # 图外层按图切卡：G=5 配 5 卡正好一卡一图；卡数 > G 会有卡空转（cli 会打印
    # 「没有分到图」退出，不报错，但别这么配）。8 卡想用满改 PERTURB_GRAPHS=8/10。
    gpus=5,
    cuda_devices="0,1,2,3,4",
    intra_op_threads=0,
    cudnn_algo_search="",
    # 图外层的池子语义与 m1a 相同：换图 = 换权重 = 重装 session。本配方下同一张卡
    # 只有一份图常驻（一 rank 一图），2 = 留一个余量槽（同 s2b 的给法）。
    resident_members=2,
    output_dir="/workspace/data/shenzw/fgvp_ens_spw",
)
