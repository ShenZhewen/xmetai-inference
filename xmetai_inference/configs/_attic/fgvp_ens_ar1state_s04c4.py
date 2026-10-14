"""四卡枢轴：与 `fgvp_ens_ar1state_s04` **除卡数外逐字相同**。

## 为什么要有这一档（2026-10-13）

`state_s00 / s02 / s04` 那三条是在 **2 卡**上跑出来的；从那以后本条线统一改到 4 卡
（`fgvp_ens_ar1state_s04.py` 里一次改动，六条兄弟臂同时生效）。

换卡数不改变成员身份（成员 = 绝对的 `member_index`），但有两个通道会变：

  1. **批长**：每卡内一次前向堆几条轨迹（26 个成员 → `6+6+6+6+2`，13 个 → `6+6+1`）。
     批长变了，cuBLAS/cuDNN 可能选到不同 kernel ⇒ 逐位不同、60 步自回归后放大。
  2. **⚠ 成员与图内随机流的配对**：见 `perturbation/README.md` **坑 1** —— onnxruntime 的
     随机算子没给 seed 也是确定性的，**两个 session 的第 k 次前向拿到逐位相同的 ε**。
     本族的成员只靠驱动侧的 ξ 区分，图里那份原生 ε 是**不受控的第二来源**，按「worker 内
     第几次前向」对齐 ⇒ 换卡数把 51 个成员重新分组到不同数量的 ε 类里。坑 1 记的那次实测
     后果是离散度系统性偏小约 **√(51/13) ≈ 2 倍**，不是百分位。

⇒ **不预设卡数效应小，也不预设 2 卡的归档可用。** 本档是量它的**桥**，一个档干两件事：

  * `s04c4` ↔ `state_s04`（除卡数外逐字相同）⇒ **直接量出卡数效应**。
    这比 "s00 ↔ 裸跑" 更干净：两边是同一条改写链、同一份输入图，
    差出来的只有卡数 + 两次运行之间的抽样底噪。
  * 判读规则（事先定死，别事后找理由）：
      - 差 **≲0.5%** 且 spread/SER 同向 ⇒ 卡数可忽略，4 卡的臂（`prior_s04 / head_s04 /
        t005 / s08`）可以对 2 卡的 σ 阶梯一起读，`s00/s02` 不用重跑；
      - 差 **1% 量级或更大**、或 spread 系统性偏移 ⇒ **2 卡的 σ 阶梯作废**，
        `s00/s02` 也按 4 卡重跑（各加一条 `_c4` 档，**不要覆盖同名归档**）。

⚠ **不要改名复用 `state_s04`**：那会覆写已出数的 2 卡归档，桥就没了。
"""
from copy import deepcopy

from xmetai_inference.configs.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.configs.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_s04c4"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
# s04 已经把 gpus 设成 4；这里显式重写一遍，为的是让"本档是 4 卡档"在本文件里自证，
# 不依赖读者去翻父文件（本档的全部意义就在这一行）。
cfg.gpus = 4
cfg.cuda_devices = "0,1,2,3"
