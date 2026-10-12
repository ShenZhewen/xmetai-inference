# -*- coding: utf-8 -*-
r"""fgvp_ar1：AR(1) 潜空间强迫 —— 拧「噪声的时间相关」这条轴（前四轮之后唯一开着的门）。

用法（同一份 config，用环境变量选臂；每臂的输出目录独立）：

    SPW_AR1=A32   xmetai-infer --model fgvp_ar1     # 主臂：行星尺度 + tau=3 天
    SPW_AR1=A1    xmetai-infer --model fgvp_ar1     # 尺度轴：所有尺度一起扰
    SPW_AR1=A32t1 xmetai-infer --model fgvp_ar1     # 时间轴：tau=1 天
    SPW_AR1=A32t0 xmetai-infer --model fgvp_ar1     # iid 端点：tau=0.05 天（≈ 每步重抽）
    SPW_AR1=W32   xmetai-infer --model fgvp_ar1     # 叠加臂：F 的 m1a 权重 + AR(1)

    python -m xmetai_inference.perturbation verdict F A32 A1 A32t1 A32t0 W32

跑前冒烟（可选；2026-09-29 起 .data 跨目录落位已由 _localize_external_data 自动处理
——硬链接进图目录，跨卷/权限不可用则整份复制**一次**。正常应直接通过）：

    python -m xmetai_inference.perturbation latent build \
        /workspace/tmp/douzsh/models/iwc_fgvp_gdn2.onnx \
        --sigma 0.04 --tau-days 3 --length-scale 32 --rank 16 \
        --cache-dir /workspace/data/shenzw/model_onnx
    python - <<'EOF'
    import onnxruntime as ort, glob
    p = sorted(glob.glob("/workspace/data/shenzw/model_onnx/*.ar1-*.onnx"))[-1]
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])  # 只建会话，不跑前向
    print("session OK:", [i.name for i in s.get_inputs()])
    EOF

为什么是这条轴（前四轮的诊断链，细节见 perturbation/README 的 2L 节）：

  1. sigma 只是**幅度**旋钮：D/A 同组 sigma x2 在 <=120 h 精确 x2.00；S / T / V 三个
     「换注入结构」的臂在 k(lead) = (spread_X/spread_F)^(1/p(lead)) 口径下全部是平的
     （S = x1.47、V = x2.4、T = x0.37）=> **结构轴关闭**。
  2. 用 sigma 同时标定两头在数学上不可能：单一标量的响应指数 p(lead) 从 +6 h 的 1.007
     掉到 +342 h 的 0.466，而 F 自己的缺口要求长时效等效 k >= 3.41（五个臂最大只给到 2.8）。
  3. E3（3 段 x 20 步刷新）实测**降低**长时效离散度（+240..360 h 只有 F 的 0.60）：
     换段确实注入了新鲜方差，但 iid 重抽换不来**时间相关性**。

  于是剩下的唯一一扇门是：让噪声在时间上相关（AR(1)），并且注在耐用的尺度上。

本 config 实现的就是这一条：不动权重，只在图上加一个 `spw/xi` 输入，把
`z (x) (1 + sigma * U diag(xi_t) V^T)` 接到低分辨率的潜状态上（/latent_mixer/Add*_output_0
—— 三个 VAE 组各自的潜状态出口）；xi_t 由**驱动侧**的 AR(1) 过程给出：

    xi_{t+1} = rho * xi_t + sqrt(1 - rho^2) * eps_t,     rho = exp(-dt / tau)

两个自由度（与论文 Phase 2/3 的「张量组 / 尺度」两阶段消融一一对应）：

| 自由度 | 本文怎么实现 | 论文的对应物 |
| --- | --- | --- |
| 时间相关 | tau（rho = exp(-dt/tau)） | 「沿轨迹刷新噪声」（论文只用「刷 / 不刷」这个二值） |
| 空间尺度 | length_scale（高斯低通相关长度，token 数） | Phase 3：SFNO 只扰最低十个球谐模 |

5 个臂 = 2（尺度）x 2（时间）+ 1（叠加），其余全部照抄 F（sigma、成员数、起报、步长、变量）：

  A32    length_scale=32, tau=3.0   —— 主臂。32 token ≈ 64 度经度（波数 ~3），只扰行星尺度；
         3 天与 IFS-ENS 的 SPPT 相关时间同量级。
  A1     length_scale=1,  tau=3.0   —— 尺度轴：不平滑，所有尺度一起扰。
  A32t1  length_scale=32, tau=1.0   —— 时间轴：相关时间短一档。
  A32t0  length_scale=32, tau=0.05  —— **端点对照**：rho = exp(-0.25/0.05) = 0.0067 ≈ 每步
         重抽 = m3 的时间结构。它与 A32 只差 tau，而 tau **不进图指纹**（只在驱动侧用），
         所以两臂**共用同一份强迫图**：A32 vs A32t0 是「时间相关 vs 无记忆」的唯一变量对照。
  W32    同 A32，另加 F 的 m1a 权重扰动 —— 回答「AR(1) 能不能叠在权重扰动上」。权重规格
         直接取 fgvp_spw_ab 的 VARIANTS["F"]（不复制，见下面的导入注释）：**指纹相同 =>
         复用 F 已经落盘的那份成员集**，不重新生成 6 x 6.6 GB，只多 6 份 ~85 MB 的强迫图。

可证伪的预测（写下来是为了接受反驳；跑完用 verdict 一把判）：

  P1（时间相关就是缺的那一块）：tau=3 的两臂在 +240..360 h 的 k 明显高于同 tau=0.05 臂
     （A32 > A32t0）。若 A32 ≈ A32t0，说明连时间相关也换不来增长率 —— 潜空间强迫这条腿
     也关闭，F 的欠离散只能靠初值扰动或多成员平均去补。
  P2（注在哪一档尺度）：A32 的增益应落在 1-4 / 5-20 带，21-40 / 41-64 不涨。A1 若把增益
     摊到 41-64，说明「注在哪一档尺度」这条判据（README 2L.8 的 p(lead, band)）在没有
     谱域的骨干上也成立。
  P3（叠加）：W32 的长时效 k 高于 F（>1.0），但不会比 A32 高出太多；W32 的短时效 k 一定
     > 1 —— F 自己在 1-3 天就过冲 2.8-3.2 倍，再叠噪声只会更散。

怎么读（判决页）：

    python -m xmetai_inference.perturbation verdict F A32 A1 A32t1 A32t0 W32

  * 读法一句话：**k 平 ⇒ 这族只是幅度**（换个 sigma 重跑同一个臂即可，别的自由度不必再扫）；
    k 上升 ⇒ 真的改了形状。判据 1/2/3 三条同时成立才算改了形状。
  * [2] 节给出「基准臂缺口 <-> 需要多少 k」的门槛（F 的长时效是 3.41）。

⚠️ 读法修正（2026-09-24，D0 判决后、开跑之前落笔）：

  * **P1 / P3 照原样读**：主判据 k 建在 spread 上（product_kind=ensemble），那是成员分布的
    量，不受集合平均阻尼影响。
  * **P2 与「均值质量」必须按残差读，不能看原始值。** spherical_band_power_* 与
    activity_ratio 都是 specialized 类指标，而 metrics/base.py 对 DETERMINISTIC_FIELD 的校验
    会直接拒绝带 member 维的输入（要求先做 ensemble_mean 降维）⇒ 它们量的是**集合均值场**。
    集合平均本身压低带功率，散得越少越「好看」。跨 9 臂实测：

        corr(r_long, 5_20 带增益) = -0.951   gain = +72.7% - 62.0% * r_long
        corr(r_long, act)         = -0.928   act  = 0.9107 - 0.1178 * r_long

    决定性对照是 D0（sigma=0.001，就是没扰动的原模型）：它的 21-40 带功率是 F 的 +227%，
    且落在拟合线上 —— 原模型自己「最尖」，所以那些数字量的是 F 的均值被压暗了多少。
    本 config 的 4 个纯 latent 臂没有权重扰动，spread 完全来自潜空间强迫，与 F 的差距可能
    很大 ⇒ 原始带功率之间的比较是被污染的。**做法：画 (r_long, 带增益) 散点看谁偏离趋势线**
    （先例：S2 离散度比 F 高 13.6%，5-20 残差却是全臂最大的 +8.9%）。
  * **均值质量一律拿 D0 当同源对照，不是拿 F。** D0（`fgvp_spw_ab` 的 sigma_full=0.001 档）
    是「不扰动」的锚点，已评测归档、直接进 verdict 即可，不必重跑；A32 的 rmse 若连 D0
    都赢不了，就不能说「均值改善了」。注意 D0 仍有 sigma=0.001 的残余离散度，而本 config
    的 sigma=0 是**逐位恒等**的空转（见上文），两者都可用，前者免费、后者与 AR(1) 图完全同源。
  * **不要把「带功率变亮」读成「尺度选择性」**：T 曾被评为「整体锐化 21-40 +61%」，
    而 D0 在同一带是 +227%、T 相对趋势线还是负残差（−28%）。默认是伪影，只有残差为正才算数。
    详见 README 2N.7 的读法修正与 2M.7 ①/⑥。

成本与口径：

  * 4 个纯 latent 臂**不需要成员集**：同一份强迫图服务全部成员，成员之间的差别只来自 xi
    的种子（由 member_index / batch / 进程 rank 决定）。这比 F 便宜一个「生成 6 x 6.6 GB
    扰动权重」的量级；
  * 每臂 6 成员 x 3 起报 x 60 步、只存 z500 => 与 F/S/T/V/E3 的 A/B 协议逐时效一一对应，
    同一个 eval config 就能直接比；
  * sigma 所有臂都取 0.04（与 F 的 sigma_full 同数字，但它作用在潜变量上、不参与配平）。
    按结论 1，sigma 只是幅度旋钮：k 平而幅度不够时直接调 sigma 重跑同一个臂，不要加新自由度。
    sigma=0 是合法的空转对照（gain 恒为 1、输出与源图逐位相同，selftest 里有一条专门钉它）。
"""
from __future__ import annotations

import os

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
MODEL_DIR = "/workspace/tmp/douzsh/models"
OUT_ROOT = "/workspace/data/shenzw/spw_ab"
# 强迫图（.ar1-<指纹>.onnx）集中落这里，不落在源模型目录（2026-09-29 改）。
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

FGVP_AR1_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# F 的权重规格只有一个来源：fgvp_spw_ab 的 VARIANTS["F"]。**不复制**那份 dict ——
# m1a 的成员集是按规格指纹落在磁盘上的，复制出来的副本一旦与 F 分叉（哪怕只差 seed），
# W32 就会悄悄用上另一套权重，而「W32 vs F」的全部结论都会失真。导入前给 SPW_AB 一个
# 缺省值：fgvp_spw_ab 在导入期就按它装配一份 cfg（纯 dataclass，无副作用）。
os.environ.setdefault("SPW_AB", "F")

from xmetai_inference.configs.fgvp_spw_ab import (  # noqa: E402  (必须在 setdefault 之后)
    BASE_EXCLUDE as _AB_EXCLUDE,
)
from xmetai_inference.configs.fgvp_spw_ab import VARIANTS as _AB_VARIANTS  # noqa: E402

_F = _AB_VARIANTS["F"]

# 与 fgvp_spw_ab 里构造 F 的那段逐字段一致（含 balance / min_elements / member_dir / seed），
# 所以指纹一致、成员集直接复用。
F_WEIGHT_PERTURB = {
    "groups": _F["groups"],
    "include": _F.get("include", ()),
    "sigma_full": _F["sigma_full"],
    "balance": "energy",
    "exclude": tuple(_AB_EXCLUDE) + tuple(_F["drop"]),
    "noise_struct": _F.get("noise_struct", "iid"),
    "min_elements": 64,
    "member_dir": f"{MODEL_DIR}/spw_ab",
    "seed": 0,
}

# latent_forcing 的目标：FGVP 三个 VAE 组各自送进 decoder 的潜状态（512 通道 / 16200 token）
# —— 论文四个骨干的优胜注入点都紧邻各架构的低分辨率潜空间（README 2M.4 的 Q2）。
_TARGET = "latent_state"
_RANK = 16
_SIGMA = 0.04

ARMS = {
    "A32": {"length_scale": 32.0, "tau_days": 3.0, "stack": False,
            "why": "主臂：只扰行星尺度（32 token ≈ 64 度经度）+ 3 天相关时间"},
    "A1": {"length_scale": 1.0, "tau_days": 3.0, "stack": False,
           "why": "尺度轴：不平滑，所有尺度一起扰 —— 与 A32 只差 length_scale"},
    "A32T1": {"length_scale": 32.0, "tau_days": 1.0, "stack": False,
              "why": "时间轴：tau 短一档（rho = 0.778）"},
    "A32T0": {"length_scale": 32.0, "tau_days": 0.05, "stack": False,
              "why": "iid 端点：rho = 0.0067 ≈ 每步重抽 = m3 的时间结构；"
                     "与 A32 共用同一份图、只差驱动侧的 tau"},
    "W32": {"length_scale": 32.0, "tau_days": 3.0, "stack": True,
            "why": "叠加臂：F 的 m1a 权重（复用已有成员集）+ A32 的潜空间强迫"},
}

ARM = os.environ.get("SPW_AR1", "A32").strip().upper()
if ARM not in ARMS:
    raise SystemExit(
        f"SPW_AR1={ARM!r} 不是已知的臂；可选：{', '.join(sorted(ARMS))}"
        "（用法：SPW_AR1=A32 xmetai-infer --model fgvp_ar1）")
arm = ARMS[ARM]

cfg = InferConfig(
    name=f"fgvp_ar1_{ARM}",
    # 纯 latent 臂直接用原始确定性图（框架会在它旁边生成/复用一份 .ar1-<指纹>.onnx）；
    # W32 走 m1a：框架先解析出成员图，再在成员图上叠加 spw/xi。
    model_path=f"{MODEL_DIR}/iwc_fgvp_gdn2.onnx",
    ops_library=f"{MODEL_DIR}/xmetai_onnx_plugins.so",
    # 必须 False：图只出预报那一帧（单步型），GPU 常驻分支会绕过滑窗回填。
    gpu_state=False,
    model_class="iwc_fgvp_gdn2_onnx",
    weight_perturb=F_WEIGHT_PERTURB if arm["stack"] else None,
    latent_forcing={
        "targets": (_TARGET,),
        "sigma": _SIGMA,
        "tau_days": arm["tau_days"],
        "rank": _RANK,
        "length_scale": arm["length_scale"],
        # 图指纹里的种子（决定空间/通道基的图案）。驱动侧的 xi 是 (seed, member_index,
        # batch, 进程 rank) 的函数，与这里无关 —— 所以各臂的基可以保持一致。
        "seed": 0,
        # 只给驱动侧算 rho 用；不进图指纹，所以换 tau 复用同一份图。
        "hour_interval": 6.0,
        # 加载时确认图里真的有 spw/xi 输入且 metadata 对得上（抓「图没被改写」）。
        "verify_on_load": True,
        # 强迫图集中落 MODEL_ONNX_DIR（2026-09-29 改）。跨目录时 _localize_external_data
        # 会把 .data 硬链接（跨卷/权限不可用时整份复制**一次**）进该目录、location 改裸
        # 文件名 —— onnxruntime 校验的是外部数据的**真实路径**必须在图目录内，相对路径
        # 指回源目录和符号链接都过不了（坑 6，A32 首跑实测）。首跑日志看「外部数据落位」
        # 一行确认是 link 还是 copy。
        # 注意 W32 的强迫图也落这里（届时成员集的 perturbed.mNNN.data 也会各复制/链接
        # 一份进来）；m1a 成员集仍在 MODEL_DIR/spw_ab（见 F_WEIGHT_PERTURB 的 member_dir，
        # 挪它 = 指纹失配 = 重建 6 x 6.6 GB，别动）。
        "cache_dir": MODEL_ONNX_DIR,
        },
    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_AR1_DYNAMIC_STORES,
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
        # 纯 latent 臂是「一个 session + 一批轨迹」（成员就是批里的位置）：真正要控的是
        # 每条轨迹一份 ~645 MB 的 state。W32 走 m1a（成员外层），代价是数据集走 6 遍。
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
    # A/B 协议：3 个起报（与 F 一致）、60 步 = +360 h、只存 z500。
    times="2025070100..2025070300:24",
    steps=60,
    hour_interval=6,
    # 成员数固定 6：与 F 同一口径（4-6 成员的 sigma_hat 由单个成员主导，比 10% 级差异
    # 要 8 成员；本轮的判据是 k(lead) 的**形状**，6 成员够用，见 README 2E.5）。
    members=6,
    vars="z500",
    # 容器里真实可见的卡号（跑之前 nvidia-smi -L 确认）；个数必须 == gpus。
    gpus=2,
    cuda_devices="0,1",
    output_dir=f"{OUT_ROOT}/{ARM}",
)