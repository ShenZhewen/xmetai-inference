# -*- coding: utf-8 -*-
r"""集合图 `iwc_gssp_im_fp16.onnx` 上的扰动臂梯队 —— **共用体**，各臂是 2 行的兄弟模块。

这条线做的是「在自带随机轴的集合图上再加扰动」。图上实测的随机轴（逐跳追过）：

    cat_9 [b,16200,1536] 潜状态 → pad → ConvTranspose(mean/logvar) → μ, σ
        μ = Slice(convolution)          σ = Exp(Mul(Slice(convolution_1), 0.5))
        ε = RandomNormalLike(μ)（全图唯一随机算子，scale=1，每步重抽 = 时间上 iid）
        normal = μ + σ⊙ε                ← 51 个成员的唯一来源

  * **A 轴（图内自带噪声的幅度 σ × k）已于 2026-10-20 移除**——实测无收益：
    `ens_k20` 判负（spread ×2.4 的同时 RMSE ×7.5，`activity_ratio` 掉到 0.61），
    实验记录见 `docs/PAPER.md` §3.3.10。本文件只用图上原有的幅度。
  * **C 轴 = 时间相关**（AR(1) 潜空间强迫）：锚点两处可选，见下面的 `AR1_ANCHORS`。
    **"prior"（= cat_9）在编码器侧**，后面还要过采样 + 第二趟编码 + 整条骨干才到预报，
    杠杆远低于 gdn2；**"head" 才是 gdn2 锚点的对应物**（预测头前 4 层）。
    框架加载时自动在图旁生成/复用强迫图，不用手工 build。
  * B 轴（SPW 权重扰动）在 `fgvp_ens_spw.py` 里，本文件不动它。

为什么打 C 轴（2026-10-12 诊断，z500、共同起报日）：SPW 已把 spread 在 24-72h 抬了
39-46%、CRPS 8/9 档更低，但 SER（spread/rmse@ensemble）仍只有 0.46/0.55，**缺口集中在
24-72h**（360h 反而 0.695/0.776，长时效不缺）。轴 C 的作用是「把 iid 噪声换成有时间
相关的噪声」，在 gdn2 上已验证（A32 vs A32T0：同 spread 下 +48/72h SER 0.57/0.51 →
1.09/1.09，P1 成立）。

⚠ 开跑前一件零成本的事：fgvp_ens 原档 `member_000..003` 的 060.nc 逐字节比对
（fgvp_ens.py 文件头的老账）：它定「图内随机轴」的真实自由度，也定 AR(1) 的成员独立性
解读。跨 rank 序列相同时，(图, 图内序号) 两轴是耦合的。

跑法（服务器；一条 2 卡）：

    cd /workspace/szwCode/xmetai-inference2
    xmetai-infer --model fgvp_ens_ar1 --cuda-devices 0,1 --out /tmp/smoke_ar1     # 先 smoke
    xmetai-infer --model fgvp_ens_ar1                                             # 正式（写 output_dir）

smoke 务必给 `--out`，别写进生产目录。

预算：筛查窗口 3 起报 x 51 成员 x 60 步 x z500 ≈ **38 GB/臂**。
上正式档改本文件的 `TIMES` / `VARS` 两个字面量（29 起报 5 变量 ≈ 1.85 TB/臂，与
s2b 生产档同窗口，才能三方并排读）。

判据（写在最前，免得看错）：

  * **只看 SER 与 CRPS**，别看 spread 绝对值（A32 的 spread 比 s2b 小却标定更好）。
  * 目标量是 **N_eff**：8 图 ANOVA 实测 Z500 中段只有 12.9-17.5（MSL 12.4）——
    51 个成员实际自由度只有 1/3~1/4。抬 N_eff 才算成功，抬 spread 不算。
  * 自己的缺口头号是 **24h SER**（0.391→0.461，最欠的一档）。
  * 每臂先按筛查窗口判方向，赢了再上 29 起报对齐 S2B 窗口复判。
"""
from __future__ import annotations

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
MODEL_DIR = "/workspace/tmp/douzsh/models"
# **所有生成物统一落这里**（2026-10-12 用户指定；与 fgvp_ens_spw / s2b 同一套协议）：
#   AR(1) 强迫图 MODEL_ONNX_DIR/iwc_gssp_im_fp16.ar1-<指纹>.onnx  ← AR1_SPEC.cache_dir
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"
GRAPH_DIR = f"{MODEL_DIR}/20261008"
# 只读源集合图 —— MODEL_DIR 是别人的目录，一个字节都不往那儿写。
SRC_GRAPH = f"{GRAPH_DIR}/iwc_gssp_im_fp16.onnx"
OUT_ROOT = "/workspace/data/shenzw"

# 筛查窗口：3 起报、z500（≈38 GB/臂）。上正式档改这两个字面量：
#   TIMES = "2025070200..2025073000:24"        # 29 起报，与 s2b / fgvp 归档同窗口
#   VARS  = "z500,u200,v200,msl,tp"            # 5 变量（≈1.85 TB/臂）
TIMES = "2025070200..2025070400:24"
VARS = "z500"
MEMBERS = 51
# 一条臂 4 卡。cli.py 对集合（members=51 ≥ 卡数）固定**按成员切**：
#   每卡领 range(rank, 51, 4) = 13/13/13/12 个成员，各卡独立跑完整起报。
# 所以 2 卡→4 卡就是这条轴上的 ~2× 加速；顺带把每卡的 rollout state 从
# 26×645MB≈17GB 压到 13×645MB≈8.4GB（cli.py 多卡分工注释里那个 OOM 的由来）。
# 成员不会因为换卡数而重新编号：成员身份 = 绝对的 member_index
# （latent.py 的 _rank_entropy = (seed, 7919, member_index, slot)，slot 只是步内分块的
# 计数，不是成员号），cli 传的也是 range(local_rank, members, world_size) 这个绝对序列。
GPUS = 4
CUDA_DEVICES = "0,1,2,3"

FGVP_ENS_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# AR(1) 潜空间强迫的**两个锚点位**（2026-10-12 把图结构追完之后加的）。
#
#   "prior" = cat_9 [b,16200,1536] —— 三个 VAE **编码器**出口的 Concat = 潜先验的入口。
#       它下游是一整段：prior 头(mean/logvar) → RandomNormalLike 采样 → 写回输入场
#       → patch_embed(conv2d_6) → **加到主输入的 token 嵌入上**
#       (add_16206 = conv2d_1 + conv2d_6) → LayerNorm → 整条 Mamba 骨干
#       → latent_mixer → 预测头。**之前所有臂用的都是它。**
#
#   "main"  = add_35782 / add_35787 / add_35792 [b,16200,512] —— **gdn2 锚点的精确
#       对应物**。图取证（2026-10-12，本机只读 trace）：
#         · 三者分别来自 latent_mixer.group_projs.{0,1,2}.bias，三个组一一对应
#           gdn2 的 /latent_mixer/Add{,1,2}_output_0；
#         · 第一个消费者都是 LayerNorm（= blocks.0/norm1 的入口签名），与 gdn2 锚点
#           「→ decoder_group.N/layers.0/blocks.0/norm1」逐字同构；
#         · 往下 46 跳用到 decoder_group.0 的权重，vae_encoder_group 一次都不出现
#           ⇒ 它确实是「进 decoder 之前」。
#         本地实测 resolve_targets + detect_layout 通过（tokens=16200 channels=512）。
#
#   "head"  = add_40939 / add_46096 / add_51253 —— 我先前按「离预测头最近的 3 维张量」
#       选的，**选深了**：它在 decoder 之后、joint_prediction_head 之前（离 pred 15 跳，
#       gdn2 是 20 跳但结构是 decoder **入口**）。留着当第三个位置刻度，不是主臂。
#
#       两个候选同在一条链上，相隔约 31 层（decoder 本体）。当初落到 cat_9 是因为
#       TARGET_ALIASES["latent_state"] 写死了 gdn2 的 TorchScript 名，别名解析不到 ——
#       那是个假的二选一：锚点在图上，只是换了名字。
#
# 当初为什么落在 cat_9：TARGET_ALIASES["latent_state"] 写死的是 gdn2 的导出名
# （/latent_mixer/Add_output_0 这类 TorchScript 名），集合图是 add_40939 这种被拍平的
# 名字，别名解析不到。当时的结论是「退而求其次用 cat_9」——**那是个假的二选一**：
# 锚点在图上，只是换了名字。
#
# sigma/tau_days/rank/length_scale 取自 A32 主臂（那一组是已验证的配方，原样搬过来）。
AR1_ANCHORS = {
    "prior": ("cat_9",),
    "main": ("add_35782", "add_35787", "add_35792"),
    "head": ("add_40939", "add_46096", "add_51253"),
}
AR1_ANCHOR = "prior"
AR1_SPEC = {
    "targets": AR1_ANCHORS[AR1_ANCHOR],
    "sigma": 0.04,
    "tau_days": 3.0,
    "rank": 16,
    "length_scale": 32.0,
    "seed": 0,
    "hour_interval": 6.0,
    "verify_on_load": True,
    "cache_dir": MODEL_ONNX_DIR,
}


def build(name, *, ar1=False, sigma=None, rollout_batch=None, anchor=None):
    """造一条臂的配置。ar1 = 叠不叠时间相关强迫；
    sigma = 覆写 AR1_SPEC 的强迫强度（None = 用共用体里的 0.04）；
    rollout_batch = 一次前向堆几条轨迹（None = 6，与 fgvp_ens 同）；
    anchor = AR(1) 打在哪个潜状态（None = 共用体缺省 "prior"）：
        "prior" = cat_9（编码器出口 / 潜先验入口，下游还有采样 + 第二趟编码 + 整条骨干）
        "main"  = add_35782/35787/35792（**gdn2 锚点的精确对应物**，进 decoder 之前）
        "head"  = add_40939/46096/51253（decoder 之后、预测头之前；我先前选深的那个）
        取值见上面的 AR1_ANCHORS。**两条只有 anchor 不同时才是干净的位置对照。**

    AR(1) 由框架在加载时叠加（latent_forcing），不落盘新图。

    sigma 存在的理由（2026-10-12）：第 8 轮实测 σ=0.04 的 AR(1) 与裸跑差 <1%，
    落在判读分辨力以下；要先用一条**宽跨度**的 σ 阶梯区分「幅度太小」与「锚点没生效」。

    rollout_batch（2026-10-12 放开）。旧版这里写死 `1 if ar1 else 6`：开了强迫就必须
    降到 1 —— 当时 onnx 后端喂 ξ 用的是 `ar1.row(row)`（宽度恒 1），一次前向堆 k 条
    轨迹会让块内 k 条**共用相邻行**，块内成员几乎重复、集合静默欠散（不报错）。
    现在宽度已经接通（`backends.spa.LatentStepStream.__call__(width=k)`
    + 后端喂 `[k,T,R]` 图输入，
    该图的 batch 轴本来就是符号维），而且**逐位等价**：行的抽样顺序与 rb=1 完全相同
    ⇒ rb 不再是集合统计的变量，`pert`(6) 与 AR(1) 臂之间那条混淆项消失了。

    所以缺省 6（快 ~6 倍：前向次数 ∝ 轨迹数/rb）。要复现 σ 阶梯那一批（rb=1）就显式
    传 1 —— 两者**逐位同流**，跑出来的数应当一模一样，不一样就是哪里不对，
    正好当回归。pt2 后端仍不支持 k>1（ξ 走无 batch 轴的 spw_xi 缓冲，会当场报错）。
    """
    spec = dict(AR1_SPEC)
    if anchor is not None:
        spec["targets"] = AR1_ANCHORS[anchor]
    if sigma is not None:
        spec["sigma"] = float(sigma)
    batch = 6 if rollout_batch is None else int(rollout_batch)
    return InferConfig(
        name=name,
        model_path=SRC_GRAPH,
        ops_library=f"{MODEL_DIR}/xmetai_onnx_plugins.ort1_24.cuda12.so",
        # 必须 False：单步型图，GPU 常驻分支绕过滑窗回填（同 fgvp_ens 的理由）。
        gpu_state=False,
        # 前向批：图已重导为动态 batch 轴，ξ 输入侧的宽度也已接通（见 build docstring），
        # 所以两条轴都吃 6。默认可被 rollout_batch= 覆写。
        rollout_batch=batch,
        gpu_mem=0.9,
        model_class="iwc_fgvp_mamba_ens_onnx",
        # 本梯队不使用 B 轴（权重扰动）；要用见 fgvp_ens_spw.py。
        weight_perturb=None,
        latent_forcing=spec if ar1 else None,
        # 0 = 不走图外层编舞（那是 SPW 的编舞）。集合语义由模型类自己的
        # forecast_type="ensemble" + members 决定：51 条轨迹共享一份图。
        perturb_graphs=0,
        dataset={
            "type": "processed_multi_zarr",
            "paths": FGVP_ENS_DYNAMIC_STORES,
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
        model_processing=ModelProcessingConfig(
            input=[],
            recurrent=[],
            output=[],
        ),
        times=TIMES,
        steps=60,
        hour_interval=6,
        members=MEMBERS,
        vars=VARS,
        gpus=GPUS,
        cuda_devices=CUDA_DEVICES,
        intra_op_threads=0,
        cudnn_algo_search="",
        resident_members=2,
        output_dir=f"{OUT_ROOT}/{name}",
    )


# 裸跑对照：原图 + 集合语义，不加任何扰动机制。同一个筛查窗口上与各臂逐项对照，
# 也是「(图, 图内序号) 两轴耦合吗」那条老账的取证目录。
cfg = build("fgvp_ens_pert")
