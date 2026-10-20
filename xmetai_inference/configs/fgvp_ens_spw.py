# -*- coding: utf-8 -*-
r"""fgvp_ens_spw：**集合模型 + SPW（权重扰动）** 的标准档 —— 完整自包含。

SPW 有两种用法，本文件是第二种：

  1) **确定性模型 + SPW**（m1a，成员在外层）
     扰动 N 份权重 = N 个成员；每份冻结权重要跑完全部起报，每个起报只出 1 条
     轨迹，一个成员就是一份 onnx。见 ``configs/fgvp_spw_prod_s2b.py``（N=51）。

  2) **集合模型 + SPW**（图在外层，本文件）
     集合图自带图内随机算子，同一份权重每次前向给一个不同样本，所以：

         成员 = n 份扰动图 x 每份图在同一 session 里跑 M 次前向 = n x M

     本档 n = ``PERTURB_GRAPHS`` = 8、M = ``RUNS_PER_GRAPH`` = 6 -> **48 成员**。
     成员 m 的权重身份是 g = m // M，图内序号 j = m % M。同一份图的 M 个成员
     必须在同一进程的同一 session 里连跑（``cli._run_graph_outer``），不能按
     成员拆到不同进程——否则图内随机算子按「worker 内第几次前向」对齐的序号
     就断了。

---- 为什么取 8 x 6 ----

  * 8 份图 / 4 卡 = 每卡 2 份图；每卡每起报跑 2 x 6 = 12 条轨迹，卡间完全均衡。
  * M = 6 与 ``model_batch`` = 6 对齐：一份图的 6 条轨迹在每步只需要一次前向批。
  * 48 与 FGVP 原生 51 成员只差 3，CRPS / spread 的成员数偏差可以忽略。
    要严格 51 只能拆成 17 x 3（17 图 / 4 卡 = 5+4+4+4，卡间约 25% 尾巴），或者
    放弃等 M（51/8 = 7x3 + 6x5）；本档不这么做。

改 n / M 时保持 ``MEMBERS = PERTURB_GRAPHS * RUNS_PER_GRAPH``，否则
``_graph_member_split`` 会按 51/8 那种「前几份多一个」摊分，M 不再相等。

---- 两条随机轴 ----

权重扰动（本文件的 SPW）与图内随机算子是两条独立轴：换卡数只改「图分给哪个
rank」，不改 (g, j) -> 成员号的映射；但图内随机序列按 session 内的前向顺序对齐，
**换卡数 / 换 session 次序后不要拿新产物与在盘旧档逐位横比**（同批同卡数才可比）。
``perturb_graphs`` 不进入权重指纹，加/减图不会重造已有的那几份成员权重。

---- groups / sigma 的来历 ----

沿用确定性档 s2b 的 ``latent_mixer`` + ``noise_struct="channel"`` +
``sigma_full=0.027``。README 2O 记录：集合图（Mamba 骨干）的注入组尚未单独做
审计记录。这是本臂现行配方；若服务器上还没核过 groups，SPW 这条腿的解读要打
折扣，不是「效果不好」而是「跑的可能不是想跑的臂」。

---- 时段、预算、跑法 ----

时段：夏季 20250702-0730（29 起报）+ 冬季 20251101-1130（30 起报）= **59 起报**，
评测侧必须同步。60 步 x 6 h = +360 h，5 变量 z500/u200/v200/msl/tp。

输出预算：59 x 48 x 60 x 20.8 MB ≈ **3.5 TB**；成员权重集 8 份（不随成员数变，
按 latent_mixer 与 gdn2 同量级估算约 12 GB）。

    cd /workspace/szwCode/xmetai-inference2
    nvidia-smi -L

    # smoke：1 起报 / 8 图 x 2 成员 / 6 步，务必给 --out
    xmetai-infer --model fgvp_ens_spw --times 2025070200 --steps 6 \
        --members 16 --gpus 1 --cuda-devices 0 --out /tmp/smoke_ens_spw

    # 正式
    xmetai-infer --model fgvp_ens_spw

smoke 的 ``--members`` 必须 >= 2 x PERTURB_GRAPHS = 16（每图至少 2 成员的校验）。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

# ---------------------------------------------------------------------------
# 路径（服务器上的固定位置，直接改这里）
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 成员结构：n 份扰动图 x 每份 M 次图内前向
# ---------------------------------------------------------------------------
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
        # 注入点与幅度沿用 s2b；集合图上的组审计见文件头。
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
