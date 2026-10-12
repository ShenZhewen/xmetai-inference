# -*- coding: utf-8 -*-
"""FuXi-2.1 潜空间 AR(1) 强迫：把确定性 pt2 模型变成集合（4 臂对照的最小集）。

目的：把 fgvp_ar1（FGVP/GDN 骨干）验证过的 AR(1) 潜空间强迫迁到第二个架构
（FuXi transformer）上 —— PAPER.md §2M「通用扰动方案」的泛化性证据。机制与
fgvp_ar1 完全同构（同一 LatentForcingSpec / 基构造 / Ar1Driver / LatentStepStream），
只是注入实现换成了 pt2 后端的 torch.fx 图改写（perturbation/latent_pt2.py，
改图不落盘：加载时内存改写，τ 留驱动侧 ⇒ 换 τ 复用同一份改写逻辑）。

## 臂（环境变量 SPW_FUXI_AR1，默认 F32）

| 臂 | sigma | tau_days | 回答什么 |
| --- | --- | --- | --- |
| S0    | 0     | 3.0  | 底噪对照：不改图，成员间差异只来自 GPU 非确定性（fuxi2.1 实测高空通道每次 ~0.2% 场变率）。**先跑这个**，它给后面所有臂的 spread 划出「零强迫下限」 |
| F32   | 0.04  | 3.0  | 主臂：行星尺度（length_scale=32 token ≈ 64° 经度）+ 3 天相关（ρ=0.92） |
| F32T0 | 0.04  | 0.05 | iid 端点（ρ=0.0067 ≈ 每步重抽）：与 F32 唯一变量 = 驱动侧 τ。P1 的 FuXi 版 —— 两者持平则「时间相关是增长率来源」这条腿在第二个架构上也关闭 |

det 基线 = 现有 configs/fuxi21.py（不改）；判断标准不是「集合均值打赢 det」
（任何扰动都能靠平均抵消误差），而是 CRPS / spread / 标定比 / k(lead)
（见 perturbation/README.md 2N.7）。

## 注入点

targets 用 "layer_stream:0,10,20"：第 0/10/20 层末的残差流（潜状态
[1, 28800, 1536]）。3 处对齐 FGVP-A32 的 3 个 latent_state（σ=0.04 口径可比）；
要全 30 层把 targets 改成 ("layer_stream",) 即可。FuXi-2.1 实测布局：
tokens=28800（120×240）、channels=1536、30 个 decoder 层、fp16。

## 与 fgvp_ar1 的量级差异

- 只 latent 不落盘：**没有成员权重集**（对比 F 的 6 x 6.6 GB）——一张 pt2、
  一批轨迹，成员身份 = ξ 的行号（熵含进程 rank，多卡按成员切不重复）；
- 首跑规模照抄 fgvp_ar1：6 成员 x 3 起报 x 60 步、只 z500 —— 先验通接线与
  σ 量级，再放大到 51 成员对齐评测（届时改 members / times 两个数字）。

用法（与 fgvp_ar1 同式）：

    SPW_FUXI_AR1=F32   xmetai-infer --model fuxi_ar1
    SPW_FUXI_AR1=S0    xmetai-infer --model fuxi_ar1
    SPW_FUXI_AR1=F32T0 xmetai-infer --model fuxi_ar1
"""
from __future__ import annotations

import os

from xmetai_inference.configs.base import InferConfig

# 数据管线 / 前后处理 / 模型路径全部**从 fuxi21 导入**（不复制，防分叉）：
# unit_convert -> geometry -> normalize -> fill_missing，recurrent 清零诊断通道
# （ssr/ssrd/fdir/ttr/tp），denormalize + tp 非负 —— 扰动只碰潜空间，与这些正交。
from xmetai_inference.configs.fuxi21 import cfg as _BASE

OUT_ROOT = "/workspace/data/shenzw/fuxi_ar1"

# 注入层：0/10/20（3 处，对齐 FGVP-A32 的 3 个 latent_state）。全 30 层改
# targets 为 ("layer_stream",)。层号必须存在于图里（FuXi-2.1 是 0..29）。
_TARGETS = ("layer_stream:0,10,20",)
_GRID = (120, 240)          # FuXi-2.1 潜网格（28800 token = 120x240），fx 图推不出来，必须显式
_RANK = 16
_LENGTH_SCALE = 32.0        # token；32 ≈ 64° 经度 ≈ 波数 3，与 fgvp_ar1 的 A32 同口径

ARMS = {
    "F32": {"sigma": 0.04, "tau_days": 3.0,
            "why": "主臂：行星尺度 + 3 天相关（rho=0.92）"},
    "S0": {"sigma": 0.0, "tau_days": 3.0,
           "why": "底噪对照：sigma=0 不改图，成员差异只来自 GPU 非确定性"},
    "F32T0": {"sigma": 0.04, "tau_days": 0.05,
              "why": "iid 端点（rho=0.0067）：与 F32 唯一变量 = 驱动侧 tau"},
}

ARM = os.environ.get("SPW_FUXI_AR1", "F32").strip().upper()
if ARM not in ARMS:
    raise SystemExit(
        f"SPW_FUXI_AR1={ARM!r} 不是已知的臂；可选：{', '.join(sorted(ARMS))}"
        "（用法：SPW_FUXI_AR1=F32 xmetai-infer --model fuxi_ar1）")
arm = ARMS[ARM]

cfg = InferConfig(
    name=f"fuxi_ar1_{ARM.lower()}",
    model_path=_BASE.model_path,
    model_class="fuxi21_pt2",
    # 潜空间 AR(1)（pt2：加载后对 fx 图内存改写；grid 必须显式给，tokens/channels
    # 留空 = 从图推再校验）。tau 不进改写口径 ⇒ F32 与 F32T0 共同一份改写逻辑。
    latent_forcing={
        "targets": _TARGETS,
        "sigma": arm["sigma"],
        "tau_days": arm["tau_days"],
        "rank": _RANK,
        "length_scale": _LENGTH_SCALE,
        "grid": _GRID,
        "seed": 0,
        "hour_interval": 6.0,
        "verify_on_load": True,
    },
    dataset=_BASE.dataset,
    dataloader=_BASE.dataloader,
    model_processing=_BASE.model_processing,
    # 与 fgvp_ar1（A32）同一批起报：跨骨干可比。
    times="2025070100..2025070300:24",
    steps=60,
    hour_interval=6,
    members=51,
    vars="z500",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir=f"{OUT_ROOT}/{ARM.lower()}",
)
