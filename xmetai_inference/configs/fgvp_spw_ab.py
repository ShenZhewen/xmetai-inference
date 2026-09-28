# -*- coding: utf-8 -*-
r"""fgvp_spw_ab：把论文的三阶段消融缩成「一次跑一个变体」。

用法（同一份 config，用环境变量选变体；每个变体的输出目录与成员集目录都独立）：

    SPW_AB=S xmetai-infer --model fgvp_spw_ab      # channel 结构（第三轮主臂，已跑完）
    SPW_AB=T xmetai-infer --model fgvp_spw_ab      # 只扰 dt_bias + A_log（已跑完）
    SPW_AB=V xmetai-infer --model fgvp_spw_ab      # scalar 结构（整张量一个 xi）
    SPW_AB=E3 xmetai-infer --model fgvp_spw_ab     # 分段 rollout（3 段 x 20 步）
    SPW_AB=D0 xmetai-infer --model fgvp_spw_ab     # 同源确定性基准（sigma_full=0.001）
    SPW_AB=T02 xmetai-infer --model fgvp_spw_ab    # T 的 sigma 减半（0.02）
    SPW_AB=T08 xmetai-infer --model fgvp_spw_ab    # T 的 sigma 翻倍（0.08）
    SPW_AB=S2 xmetai-infer --model fgvp_spw_ab     # channel + 降幅 0.0341（生产候选）
    python -m xmetai_inference.perturbation spread `
        /workspace/data/shenzw/spw_ab/F /workspace/data/shenzw/spw_ab/S

每个变体 = 若干成员 x 3 起报 x 60 步（+360 h）/ 只存 z500。这是 2026-09-23 定下的
A/B 协议：够算 sigma_hat 曲线与尺度分解，成本是同规模生产跑（51 成员）的约 1/10，
且和基线 A 逐时效一一对应。

--------------------------------------------------------------------------
历史（第一、二轮，A..H 全部跑完；config 保留以便复现，结论见 perturbation/README
的 2C / 2D / 2E / 2I 节）：

  A  decoder_group, sigma_full=0.01  —— 基线（= D:\xmetai-evaluation-result\
      fgvp_spw_m1a_test）。+240 h 的 sigma_hat/rmse 只有 0.138（参考带 fgvp_hou 是
      0.881），而且这个比值随效**下降**（0.198 -> 0.138）：缺的是**增长率**，不是幅度。
  B  latent_mixer（含 dt_bias）      —— 把扰动从「输出渲染那一级」挪到携带状态的 recurrent 核。
  C  latent_mixer（排除 dt_bias）    —— 拆开「换组」与「放开 dt_bias」两个因素。
  D  decoder_group, sigma_full=0.02  —— 只看 sigma 是否线性（结论：<=120 h 精确 x2.00）。
  E/F latent_mixer, sigma_full=0.02/0.04 —— 补幅度。F 是唯一把标定比推到 0.65-1.47 的变体。
  G  latent_mixer + decoder_group    —— 多组注入（总注入方差被摊薄，无收益）。
  H  all（全权重）                    —— 论文 GraphCast 的生产配置；FGVP 上无收益。

第一、二轮的三条硬结论（后面所有设计都建立在它们上）：

  1. **增长率由注入点决定，sigma 只决定幅度**（D/A 同组 sigma x2 在 <=120 h 精确 x2.00）。
  2. **幅度能补、形状补不了**：F 把 +240 h 的绝对离散度推到参考集合的 0.72，但 1-3 天
     过冲到 2.8-3.2 倍；单一标量 sigma 的响应指数随时效从 1.01 掉到 0.47（README 2L.4）
     => 想用 sigma 同时标定两头在数学上不可能。
  3. **4-6 成员的 sigma_hat 由单个成员主导**（拿掉第 3 号掉 22-34%）=> 比 10% 级的差异
     要 8 成员；比 2 倍以上的差异 6 成员够用（README 2E.5）。

--------------------------------------------------------------------------
2026-09-23 第三轮改档：**同预算、只换「注入结构」**（变体 T / S）

原定的第三轮（I/J/K/L/M：sigma 工作点 + 块级扫描）取消，原因：

  * I/J（sigma 工作点）已被更硬的口径回答过：sigma=0.04 的代价是 1-3 天过冲，而 sigma
    是**唯一**的幅度旋钮且长时效饱和 —— 再扫 0.025/0.03 只是把过冲从 2.8 倍挪到 1.8 倍，
    不改变形状病；
  * K/L（按块类型分开扰）的假设（「粗尺度离散度只能来自 recurrent 回路」）已被 2L.6/2L.8
    的尺度证据取代：被系统抑制的是**中小尺度通道**（21-40 / 41-64 带），不是某一半块；
  * M（B 的 8 成员重跑）随着 B 退出主线的对照位而没有价值。

新臂的设计原则：**与 F 只差一个自由度，且注入方差完全相同**（`balance="energy"` 下两份
预算都 = sigma_full^2 x SigmaW^2_候选池 = 0.04^2 x 459257.48）。所以任何形状变化都只能
归因于「注入结构」，不能归因于「幅度」。

  T  dt_bias + A_log（偶数块的 GatedDeltaNet，12 张量 / 9360 元素）
     注入点：Delta 网络的**逐通道时间常数**（dt = softplus(dt_bias + ...)，A_log 是记忆
     衰减的对数刻度）。这是全模型里少数「扰一次、随步累积」的参数；README 2L.6 说论文四个
     骨干的优胜点都紧邻「控制潜模长时间演化」的那一层，FGVP 上对应的就是它。
     A_log 只有 24 个元素，**必须写进 include**，否则会被 min_elements=64 挡掉。
     注意：C（排除 dt_bias）与 B 的 sigma_hat 差 <=10%，所以这个臂的真正目的是把 C 的结论
     从「幅度无差别」升级成「形状也无差别」——T 的 p(lead) 明显低于 F 说明这条通道是死的；
     反而更高则说明 C 当初被「2.5 倍补偿方差」掩盖了。

  S  latent_mixer 全组（210 张量）+ noise_struct="channel"
     注入结构：每个**输出通道**共享一个 xi，等价于给每个潜通道一个空间上均匀的增益扰动
     （delta_W = diag(xi) . W），而不是逐元素 iid。这是论文 Phase 3（SFNO 取最低十个球谐
     模、sigma=0.25）在**没有谱域可切**时的对应做法：让噪声只在少数自由度上变化。
     三种噪声结构的每元素方差都是 1，**注入总方差不变**——它是「同预算换结构」，不是又一次
     sigma 扫描。1 维张量（bias / norm / dt_bias）自动退化为 iid。
     S vs F 是**唯一一个只差结构**的对照（组、sigma、成员数、种子全同）。

判据（三条都从**成员场**算，不需要实况；1+2 同时成立才算改了形状）：

  1. **长时效更散**：spread_新 / spread_F 在 +240..360 h > 1；
  2. **短时效不更散**：同一比值在 +6..48 h <= 1；
  3. **分歧落在哪个尺度**：把响应指数按带分解 p(lead, band)（README 2L.8 的表），看增益
     落在 1_4 / 5_20（唯一没被系统抑制的通道），还是又落在 21_40 / 41_64（注多少都被吃掉）。

  三条都重合 => 结构无收益，创新①（噪声相关化）否决，剩下的唯一方向是 2J 的
  「轨迹内刷新 + 时间相关」（E3）。

--------------------------------------------------------------------------
2026-09-23 第四轮改档：把两个自由度各推到**极端**（变体 V / E3）

两者都不改注入机制（仍是 m1a），只改「噪声落在哪些自由度」与「什么时候注入」。

  V  latent_mixer 全组（210 张量）+ noise_struct="scalar"
     每个**张量**共享一个 xi —— 一个成员只有 210 个随机数。它依然守恒「注入方差 =
     sigma^2 x ||W||^2」（每元素 E[xi^2]=1），但把随机自由度压到最低：一个成员能张成的
     扰动子空间秩 <= #被扰张量。所以 V 问的不是 S 那个问题（噪声落在少数**空间**自由度上
     行不行），而是「离散度的**秩**降到最低还够不够」。
     配平后 sigma 与 F / S 逐位相同（0.0457845）—— V vs S vs F 是同一份预算下三种秩的
     对照：iid 3.73e8 / channel 210 x 成员 / scalar 210 x 成员。
     统计学后果要记住：单张量的注入方差比 = sigma^2 x chi^2_1（只有一个随机数），离散度
     统计量天然比 iid 抖，所以 V 的 sigma_hat 必须与 F/S 同池同口径比，不能看单次抽样。

  E3  latent_mixer 全组 + 分段（config 的 spw_segments：3 段 x 20 步）
     注入的**时间结构**：整条轨迹不再冻结一组权重，而是每 20 步换一次（换段 = 换一份成员
     权重；帧号连续、成员号不变、输出布局不变）。同一张分段表能表达三种写法，只改 config：
       (a) 同 sigma 同种子 -> 三段目录相同 = 什么也没变（验证「分段机制本身不动结果」）；
       (b) 同 sigma 不同种子 -> 纯**刷新**（预算不变、只是重抽），论文的「沿轨迹刷新噪声」；
       (c) 每段 sigma = 冻结臂 sqrt(1/K) -> 同预算的**分段 sigma**，K 段各摊 1/K 份。
     本文档取 (c)+(b)：sf = 0.04/sqrt(3) = 0.0230940、种子 0/1/2，于是三段注入方差之和与
     F 的那一份**完全相等**（= 0.04^2 x SigmaW^2），差别只有「什么时候注入 + 重抽几次」。
     可证伪的预测（写下来是为了接受反驳）：spread^2(lead) = Sum_k p_k x 增长(lead-k)^2，
     p_k 是第 k 段的注入方差，而增长随时效单调递增 —— 于是「预算全放在 t=0」是**极值**，
     刷新只会**降低**长时效离散度。所以预期 E3 在 +240..360 h 的 spread 低于 F，在
     +6..48 h 也低于 F（首段只有 1/sqrt(3) 的幅度）。若实测相反，说明「时间结构」不是靠
     iid 重抽能补的，得换成真正时间相关的强迫（AR(1) 潜空间残差，README 2J 第三条）。
     成本：3 套成员集（跑完留一套，其余可删）；每成员多 K-1 次 session 建立（段边界），
     按 e2e 实测这是分段唯一的开销。

**第四轮实测判决（2026-09-24，详细见 perturbation/README 的 2L.12）**

两个极端都**没改形状**，两条轴一起关闭：

  V  spread 全时效 x1.69-2.66（3/3 天），但换算成等效 sigma 倍数后 kV 平在 **2.33-2.74**
     => 「把秩降到 1」等价于把 sigma 乘 2.4（iid 1.0 / channel 1.47 / scalar 2.4 是同一条
     **单调幅度旋钮**）。代价是模型被推坏：短时效标定比 2.04、CRPS 全时段 x1.21-1.72、
     RMSE x1.256、ACC x0.814，41-64 / 65-128 带功率暴涨 x5.6 / x7.5。
     **结论：结构轴关闭；这个方向不安全，不要拿 V 上生产。**

  E3  预测**命中**：spread 在**所有**时效都低于 F（+240..360 只有 0.60 倍，0/3 天上升），
     kE3 平在 0.37-0.49。段边界指纹（+120 h 局部极大 0.503、+240 h 之后斜率由平转升）
     证明**换段确实注入了新鲜方差**，只是首段只剩 1/sqrt(3) 的幅度 => 「iid 重抽」换不来
     时间相关性。**结论：时间轴关闭，唯一还开着的门是 AR(1) 潜空间强迫（2J 第三条）。**

  意外收获（比上面两条更有用）：**不去扩大离散度的臂，集合均值反而更准** —— E3 的
     RMSE 412.7 / ACC 0.7905 / activity 0.8574 是五档最好（对 F 是 x0.950 / x1.040 / x1.116，
     逐箱 3/3 天；CRPS 短中期 3/3 天更好 3.5-10.6%），T 紧随其后，而放大离散度的 S / V 都更差
     => 扰动本身在替模型做「多成员平均」，把系统性偏差抹平了一点。**这就是 D0 存在的理由**：
     它能一次分离「平均的收益」与「扰动的代价」。

--------------------------------------------------------------------------
2026-09-24 第五轮定档：**补同源基准 + 两条判决性扫描**（变体 D0 / T02 / T08 / S2）

背景是第三轮 S / T 的实测判决（见 perturbation/README 的 2L.11）：S 与 T 在
k(lead) = (spread_X/spread_F)^(1/p(lead)) 口径下**都是幅度等价物**（S ≡ sigma x1.47、
T ≡ sigma x0.37），**形状没改**；但 T 的集合均值反而三档里最准（RMSE 415.4 vs F 434.6、
ACC 全程 3/3 天更优、activity +11-28%、5-20/21-40 带功率 x1.47/x1.62）。

  D0  latent_mixer, sigma_full=0.001（配平后 0.0011446）
      同源确定性基准：离散度只有 F 的 ~2.5%、均值漂移 ~1e-6（可忽略），所以它就是确定性
      预报，但走**同一个 reader、同一份 eval config、同一 3 个起报**。**为什么需要它**：T 的
      均值结论现在建立在跨源对照上（single_fgvp_20260916 是别的产出目录，2H 的 #11 已点过），
      有 D0 才能变成同源结论。判读：T 的 rmse / acc / act 若仍全面优于 D0，则
      「DeltaNet 的衰减常数有系统性偏差」成立 —— 这是本轮最值钱的线索。

  T02 / T08  T 的 sigma_full=0.02 / 0.08（配平后 0.0295329 / 0.1181316）
      验「锐化是 sigma^2 的二阶效应」（对称噪声 + 非线性响应 => 均值漂移 ~ sigma^2）。
      判读：4 倍 sigma 跨度上，act/rmse 的增益按 sigma^2 缩放 => 二阶机制成立；按 sigma
      线性缩放 => 一阶（更值得追）；T08 反而变差 => 0.04 附近就是均值最优点。

  S2  latent_mixer + channel, sigma_full=0.0341（配平后 0.0390313）
      S 的降幅档 = **目前唯一可能上生产**的配置：0.0341 是按 2L.11 的 k(+6h)=1.342 反推的，
      把 S 的幅度压到「短时效与 F 等 spread」，于是短时效过冲（ratio 1.63）回到 F 的水平，
      而长时效仍净赚 3.5-12.7% spread。判读：+6..48 的 spread/ratio 回到 ~1.0，且
      +240..360 的 spread >= 1.04，否则那点红利是噪声。

  故意**没排**的：论文式「不摊薄」的分段（三段都写 0.04）—— 它的总注入方差是 F 的 3 倍，
  长时效 spread 一定会涨，但那涨的是**幅度**不是形状，会和 E3 的判决搅在一起没法归因。

--------------------------------------------------------------------------
跑之前先核对配平 sigma（本地只跑「选择 + 配平」，不推理；服务器日志里的 `sigma[...]`
应与下表逐位一致，对不上就是选组口径变了）：

| 变体 | 组 / 结构 | 张量 | 元素/成员 | 能量占比 | 配平后 sigma | 预算 |
| --- | --- | --- | --- | --- | --- | --- |
| F（参考，已有） | latent_mixer / iid | 210 | 373019520 | 76.33% | 0.0457845 | 1 份 |
| T | dt_bias+A_log（偶数块）/ iid | 12 | 9360 | 45.92% | 0.0590658 | 1 份 |
| S | latent_mixer / channel | 210 | 373019520 | 76.33% | 0.0457845 | 1 份 |
| V | latent_mixer / scalar | 210 | 373019520 | 76.33% | 0.0457845 | 1 份 |
| E3 | latent_mixer / iid，3 段 x 20 步 | 210 | 373019520 | 76.33% | 0.0264337（每段） | 3 x 1/3 份 |
| D0 | latent_mixer / iid | 210 | 373019520 | 76.33% | 0.0011446 | ~1e-6 份 |
| T02 | dt_bias+A_log / iid | 12 | 9360 | 45.92% | 0.0295329 | 1/4 份 |
| T08 | dt_bias+A_log / iid | 12 | 9360 | 45.92% | 0.1181316 | 4 份 |
| S2 | latent_mixer / channel | 210 | 373019520 | 76.33% | 0.0390313 | 0.73 份 |

「1 份」= sigma_full^2 x 候选池 SigmaW^2（0.04^2 x 459257.48）；候选池 = 888 张量 /
731452800 元素。T 的 12 个张量里 6 个 dt_bias（1536 元素，外置）走 perturbed.mNNN.data、
6 个 A_log（24 元素，留在图内）直接替换 —— 所以 T 的成员集只有几十 KB 的 perturbed 数据。

E3 有**三套**成员集（每段一套，sigma 相同但种子不同 -> 指纹不同 -> 三个目录），每套
6 个成员；跑完只需留 E3 需要的那套，其余删掉。三段各摊 1/3 份预算，和起来正好是 F 那一份。

挑别的子组用 `groups --under/--depth`（`--depth` 默认 2，只到 `latent_mixer/layers` 这一级）：

    python -m xmetai_inference.perturbation groups <onnx> --energy `
        --under latent_mixer.layers.0.blocks --depth 5 --top 20

并集 token：一个 token 里用逗号分隔多个前缀，它们仍然算**一个组**（一份预算）。FGVP 的
12 个 block 是交错的（偶数是 GatedDeltaNet、奇数是 attn+mlp），没有单个前缀能覆盖某一半；
写成 6 个 token 会逐组配平 => 总注入方差变 6 倍（G 已经踩过 2 倍的那个坑）。

调度（多进程 / 多卡）超参数见下面的 `SCHEDULING`：任务单元是 `(起报, 成员块)`，worker 自己
去 `{output_dir}/.xmetai_plan/` 认领，跑完打完成标记（重启跳过已完成的、崩掉的 worker 手里的
任务被活着的人捡走）。命令行只留 `--schedule` / `--procs-per-gpu` / `--resume` 三个日常覆盖
项，其余全部跟配方走进版本控制。字段含义见 README 的「多卡运行」。
"""
from __future__ import annotations

import os

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
MODEL_DIR = "/workspace/tmp/douzsh/models"
OUT_ROOT = "/workspace/data/shenzw/spw_ab"

FGVP_SPW_AB_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# 默认排除（与 fgvp_spw_m1a 一致）：归一化统计量、掩码、烘焙常量不是权重。
# 注意 dt_bias **不在**默认排除里，它在 latent_mixer 下默认就是候选 —— 所以「排除它」必须显式写。
BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")

# FGVP 的 latent_mixer 有 12 个交错 block：偶数号是 GatedDeltaNet（dt_bias / A_log 只在这里）。
# 一个组的注入点要跨这 6 个块，只能写并集 token（逗号分隔 -> 仍是**一个组**、一份预算）。
_MIXER_BLOCKS = (0, 2, 4, 6, 8, 10)


def _mixer_token(*leaves):
    """把「偶数块 mixer 里的若干叶子张量」写成一个并集 token。"""
    return ",".join(
        f"latent_mixer.layers.0.blocks.{block}.mixer.mixer.{leaf}"
        for block in _MIXER_BLOCKS for leaf in leaves
    )


# T 那一族的注入集（dt_bias + A_log，偶数块的 GatedDeltaNet）。T / T02 / T08 **共用同一个
# 常量**，避免「改了一处、三臂悄悄分叉」——这是 config 里最容易犯的错。
_T_GROUPS = (_mixer_token("dt_bias", "A_log"),)


# E3 的分段表：3 段 x 20 步 = 60 步（与 A/B 协议的 steps 一致）。每段 sigma_full 取
# 0.04/sqrt(3)，于是三段注入方差之**和**与 F 的那一份完全相等（同预算）；三段种子 0/1/2
# 保证段与段之间是**不同的抽样** —— 同 sigma 同种子会落进同一个成员集目录（= 同一份权重），
# 那就等于没刷新（这是分段表最容易写错的地方，selftest 里有一条专门钉它）。
_E3_SF = 0.04 / 3 ** 0.5
_E3_SEGMENTS = [
    {"sigma_full": _E3_SF, "steps": 20},
    {"sigma_full": _E3_SF, "steps": 20, "seed": 1},
    {"sigma_full": _E3_SF, "steps": 20, "seed": 2},
]

VARIANTS = {
    # ---- 第一轮（2026-09-23 已跑完，结果见 perturbation/README 的 2D 节）----
    "A": {"groups": ("decoder_group",), "drop": (), "sigma_full": 0.01, "members": 6,
          "why": "基线：只扰 decoder（= 已跑完的那次，6 成员）"},
    "B": {"groups": ("latent_mixer",), "drop": (), "sigma_full": 0.01, "members": 4,
          "why": "换到 recurrent 核，含 dt_bias（**第一轮胜者**）"},
    "C": {"groups": ("latent_mixer",), "drop": ("dt_bias",), "sigma_full": 0.01, "members": 4,
          "why": "recurrent 核但排除 dt_bias（实测与 B 无差别，且带整场偏移）"},
    "D": {"groups": ("decoder_group",), "drop": (), "sigma_full": 0.02, "members": 4,
          "why": "基线 + sigma 翻倍（实测 <=120h 精确 x2.00 -> sigma 是线性的幅度旋钮）"},
    # ---- 第二轮：在 B 的基础上补幅度（E/F）与试形状（G/H）----
    "E": {"groups": ("latent_mixer",), "drop": (), "sigma_full": 0.02, "members": 6,
          "why": "B 的 sigma 翻倍（B 的 sigma_group 只有 A 的 45%，还有幅度余量）"},
    "F": {"groups": ("latent_mixer",), "drop": (), "sigma_full": 0.04, "members": 6,
          "why": "**当前参考臂**：唯一把标定比推到 0.65-1.47 的变体（代价：1-3 天过冲 2.8-3.2 倍）"},
    "G": {"groups": ("latent_mixer", "decoder_group"), "drop": (), "sigma_full": 0.01,
          "members": 6, "why": "多尺度种子：latent 给相干的粗尺度、decoder 给天气尺度（无收益）"},
    "H": {"groups": ("all",), "drop": (), "sigma_full": 0.01, "members": 6,
          "why": "论文 GraphCast 的生产配置（全权重）——FGVP 上无收益"},
    # ---- 第三轮（2026-09-23 改档）：同预算、只换注入结构。设计说明见文件头。----
    "T": {"groups": _T_GROUPS,
          "include": ("A_log",),
          "drop": (), "sigma_full": 0.04, "members": 6,
          "why": "只扰 GatedDeltaNet 的逐通道时间常数（dt_bias + A_log，12 张量 / 9360 元素）："
                 "论文四个骨干的优胜点都紧邻「控制潜模长时间演化」的那层。"
                 "A_log 只有 24 元素，必须 include 才不会被 min_elements 挡掉。"
                 "配平后 sigma=0.0590658（被扰元素比 F 少 4 万倍），"
                 "目的：把 C 的「幅度无差别」升级成「形状也无差别」。"
                 "**实测（2026-09-24，README 2L.11）**：spread 只有 F 的 0.36-0.60（等效 sigma x0.37，"
                 "kT 平在 0.31-0.40）=> 作为离散度旋钮**低效、形状也没改**；但集合均值 "
                 "RMSE 415.4 vs F 434.6（3/3 天更优）、ACC 全时效更好、activity +11-28%、"
                 "5-20/21-40 带功率 x1.47/x1.62 => 修的是**系统性锐度（偏差）**，不是不确定度"},
    "S": {"groups": ("latent_mixer",),
          "noise_struct": "channel",
          "drop": (), "sigma_full": 0.04, "members": 6,
          "why": "同 F 同组同 sigma，**只把逐元素 iid 换成逐输出通道共享**（noise_struct=channel）："
                 "等价于给每个潜通道一个空间均匀的增益扰动 delta_W = diag(xi).W，"
                 "是论文 Phase 3（粗尺度）在没有谱域可切时的对应做法。"
                 "配平后 sigma=0.0457845（与 F 逐位相同）——S vs F 是唯一只差结构的对照。"
                 "**实测（2026-09-24，README 2L.11）**：spread 全时效 x1.21-1.51（3/3 天同向），"
                 "等效 sigma 倍数 k(lead) 平在 1.41-1.52 => S ≡ 把 sigma 乘 ~1.47，**形状没改**；"
                 "短时效标定比 1.10->1.63（过离散，CRPS 3/3 天劣 8-23%），长时效 0.59->0.77；"
                 "21-40 带反而 -17% => 判据 1 过、2/3 不过。降幅到 sigma_full=0.0341 可与 F 短时效"
                 "对齐、长时效净赚 3.5-12.7%"},
    # ---- 第四轮：把「结构秩」与「时间结构」各推到极端（设计说明见文件头）----
    "V": {"groups": ("latent_mixer",),
          "noise_struct": "scalar",
          "drop": (), "sigma_full": 0.04, "members": 6,
          "why": "整张量共享一个 xi（一个成员 = 210 个随机数）：同预算、**最低秩**。"
                 "配平后 sigma 与 F/S 逐位相同（0.0457845），所以它是「注入方差不变、随机"
                 "自由度最少」的对照 —— 用来判断 F 的收益到底需要多少秩。"
                 "**实测（2026-09-24，README 2L.12）**：spread 全时效 x1.69-2.66（3/3 天同向），"
                 "但等效 sigma 倍数 kV **平在 2.33-2.74**（不是预估的 1.4-1.5）=> 秩降到 1 等价于"
                 "把 sigma 乘 2.4，iid / channel / scalar 是**同一条单调幅度旋钮**（1.0 / 1.47 / 2.4），"
                 "三个点都没改形状 => **结构轴整体关闭**。代价：短时效标定比 2.04（过离散）、"
                 "CRPS 全时段 x1.21-1.72、RMSE x1.256、ACC x0.814、41-64 与 65-128 带功率"
                 "暴涨 x5.6 / x7.5 => **把模型推坏了**，不要上生产"},
    "E3": {"groups": ("latent_mixer",),
           "drop": (), "sigma_full": 0.04, "members": 6,
           "segments": _E3_SEGMENTS,
           "why": "同预算的分段 sigma + 轨迹内刷新（3 段 x 20 步，sf=0.04/sqrt(3)=0.0230940，"
                  "种子 0/1/2）：三段注入方差之和 = F 的那一份，差别只有「什么时候注入 + "
                  "重抽几次」。预测长时效 spread **低于** F（预算放在 t=0 才是极值）。"
                  "**实测（2026-09-24，README 2L.12）**：预测**命中** —— 所有时效都低于 F"
                  "（+240..360 只有 0.60 倍，0/3 天上升），kE3 平在 0.37-0.49 => **时间轴也关闭**。"
                  "段边界指纹（+120 h 局部极大 0.503、+240 h 之后斜率由平转升）证明换段确实"
                  "注入了新鲜方差，只是幅度太小。**意外收获**：集合均值是五档最好"
                  "（RMSE 412.7 / ACC 0.7905 / act 0.8574，对 F 是 x0.950 / x1.040 / x1.116，"
                  "逐箱 3/3 天同向；CRPS 短中期 3/3 天更好 3.5-10.6%）=> 扰动在替模型做"
                  "「多成员平均」、抹平系统性偏差，**这正是 D0 存在的理由**"},
    # ---- 第五轮（2026-09-24 定档）：补**同源基准**与两条**判决性**扫描。设计说明见文件头。----
    "D0": {"groups": ("latent_mixer",), "drop": (), "sigma_full": 0.001, "members": 6,
           "why": "**同源确定性基准**。sigma_full=0.001 => 配平后 sigma=0.0011446，离散度只有 F 的"
                  " ~2.5%、均值漂移 ~1e-6（可忽略），所以它就是「确定性预报」，但走**同一个 reader、"
                  "同一份 eval config、同一 3 个起报** —— 用来把 T 的「集合均值比确定性档还准」从"
                  "跨源指示升级为同源结论（原对照 single_fgvp_20260916 是别的产出目录，见 2H 的 #11）。"
                  "判读：T 的 rmse/acc/act vs D0 若仍全面更优，则「DeltaNet 衰减常数有系统性偏差」成立"},
    "T02": {"groups": _T_GROUPS, "include": ("A_log",), "drop": (),
            "sigma_full": 0.02, "members": 6,
            "why": "T 的 sigma 减半（配平后 0.0295329）。与 T08 一起验「锐化是 sigma^2 的二阶效应」"
                   "（对称噪声 + 非线性响应 => 均值漂移 ~ sigma^2）：4 倍 sigma 跨度上，若 act/rmse 的"
                   "增益按 sigma^2 缩放（1/4 与 4 倍）则二阶机制成立；按 sigma 线性缩放则是一阶"},
    "T08": {"groups": _T_GROUPS, "include": ("A_log",), "drop": (),
            "sigma_full": 0.08, "members": 6,
            "why": "T 的 sigma 翻倍（配平后 0.1181316，即衰减常数被扰 ~12%）。若均值质量继续变好，"
                   "说明 0.04 还没到拐点、值得找均值最优点；若反而变差，0.04 附近就是最优"},
    "S2": {"groups": ("latent_mixer",), "noise_struct": "channel",
           "drop": (), "sigma_full": 0.0341, "members": 6,
           "why": "**S 的降幅档 = 目前唯一可能上生产的配置**：0.0341 是按 2L.11 的 k(+6h)=1.342 反推的"
                  "——把 S 的幅度压到「短时效与 F 等 spread」，于是短时效的过冲（ratio 1.63）回到 F 的"
                  "水平，而长时效仍净赚 3.5-12.7% spread。判读：+6..48 的 spread/ratio 应回到 ~1.0，"
                  "+240..360 的 spread 应 >=1.04（否则那点红利是噪声）"},
    "S2B": {"groups": ("latent_mixer",), "noise_struct": "channel",
            "drop": (), "sigma_full": 0.027, "members": 6,
            "why": "**S2 的定量靶心档（2M.8 第 1 顺位）**：2M.7 ⑤ 从 S / S2 两点量出「短时效 CRPS 代价对"
                   "sigma_full 的斜率 ≈ 14.9 / 单位 sigma」，S2 的短时效是 1.108（+10.8%），所以 sigma_full"
                   "从 0.0341 降到 ≈0.027 就能把短时效 CRPS 压回 F 的水平，而长时效按同一斜率只从 0.938"
                   "走到 ≈0.910（仍净赚 -9%）。考虑到 latent_mixer 的响应是次线性（2L.4），真实拐点可能"
                   "偏保守，所以这一档取激进端 0.027（保守端是 0.030）。"
                   "判读：短时效 CRPS 比 → ~1.00（3/3 天）、长时效仍 < 0.95，且判据 1 / 3 仍 ✅、判据 2 转 ✅"},
}


VARIANT = os.environ.get("SPW_AB", "S").strip().upper()
if VARIANT not in VARIANTS:
    raise SystemExit(
        f"SPW_AB={VARIANT!r} 不是已知变体；可选：{', '.join(sorted(VARIANTS))}"
        "（用法：SPW_AB=S xmetai-infer --model fgvp_spw_ab）")
variant = VARIANTS[VARIANT]
exclude = tuple(BASE_EXCLUDE) + tuple(variant["drop"])

cfg = InferConfig(
    name=f"fgvp_spw_ab_{VARIANT}",
    # 原始确定性图（不是成员模型）。框架会按规格找 <模型名>.spw-members-<指纹>/ 下的
    # <模型名>.mNNN.onnx（扁平布局）。
    model_path=f"{MODEL_DIR}/iwc_fgvp_gdn2.onnx",
    ops_library=f"{MODEL_DIR}/xmetai_onnx_plugins.so",
    # 必须 False：图只出预报那一帧（单步型），GPU 常驻分支会绕过滑窗回填。
    gpu_state=False,
    model_class="iwc_fgvp_gdn2_onnx",
    weight_perturb={
        # 注入点：groups（可写成并集 token）+ include（名字不像 weight/bias 的可学习参数）
        "groups": variant["groups"],
        "include": variant.get("include", ()),
        "sigma_full": variant["sigma_full"],
        "balance": "energy",
        "exclude": exclude,
        # 注入结构：iid（论文原式）/ channel（逐输出通道共享）/ scalar（整张量共享一个 ξ）。
        # 三种结构的每元素方差都是 1，所以「换结构不换预算」；非 iid 会进指纹 -> 另落一个目录。
        "noise_struct": variant.get("noise_struct", "iid"),
        "min_elements": 64,
        # 成员集落盘目录。变体的 groups/sigma/noise_struct 不同 -> 指纹不同 -> 各自一个
        # <模型名>.spw-members-<指纹>/ 子目录，互不串味；每个目录各有一份 base.data
        # （硬链接不可用时是一整份复制），跑完一个变体确认没用了就删掉它的目录。
        "member_dir": f"{MODEL_DIR}/spw_ab",
        "seed": 0,
    },
    # 分段 rollout（E3）：只有 E3 这一臂给了分段表，其余臂是 None（不分段 -> 行为与以前逐位
    # 相同）。解析 / 校验 / 段边界见 perturbation.segments.parse_segment_plan；跑之前会先
    # 把各段的成员集都造好（段边界是在推理中途到的，那里不能等生成）。
    spw_segments=variant.get("segments"),
    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_SPW_AB_DYNAMIC_STORES,
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
        # m1a 是成员外层、数据集内层：M 成员 = 数据集走 M 遍。真正要控的是主机内存
        # （预取队列 ≈ workers x 2 x batch x 645MB/样本）。内存小的机器先降 num_workers。
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
    # A/B 协议：3 个起报（与基线一致）、60 步 = +360 h、只存 z500。
    times="2025070100..2025070300:24",
    steps=60,
    hour_interval=6,
    # 成员数由变体表决定（历史臂 4-6，第三轮 6）。建成员集仍按 rank 分片（6 成员 2 卡 ->
    # 每个 worker 造 3 个），跑什么由任务表动态认领，与分片无关。
    members=variant.get("members", 6),
    vars="z500",
    # 容器里真实可见的卡号（跑之前 nvidia-smi -L 确认）；个数必须 == gpus。
    gpus=2,
    cuda_devices="0,1",
    # 常驻 session 池：本配方 6 个成员各一份权重（latent_mixer 1.49 GB）x 6 ≈ 9 GB 显存，
    # 显存够 -> 全常驻，换成员时不重建 session、不重预热 cuDNN。claim_order=member 已经
    # 让连续任务落在同一成员上，池子补的是「被另一个 worker 插队后切回来」与「失败重派」。
    # 显存紧就降到 2-3；1 = 一换一放（改造前的口径）。
    resident_members=6,
    output_dir=f"{OUT_ROOT}/{VARIANT}",
)
