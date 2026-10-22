"""AR(1) 去均值 + a(t) 臂：σ=0.04 / τ=3d，a(t) 照 σ*(t) 表（1.76x → 6.13x）（第 14 轮）。

= aB 的包络 + zm 的去均值，两个自由度同时开。与 aB 的差别只有 `demean_members = 51`。

---- 为什么这条是"免费加力"的检验 ----

aB 的包络是照 σ*(t) 实测表反推的，360h 处有效 σ = 0.04 × 6.13 = 0.245。它一直没敢
当真，因为在**长时效**上加大 σ 要赔平均场精度（见 zm 臂 docstring：312/360h 两批都是
σ=0 最好）。去均值如果真把那笔代价拆掉，那么 aB 的激进包络在长时效就应当**只赚不赔**：

  ZA1  zm_aB 的 312/360h 集合平均 RMSE ≤ aB（不该再为大幅 σ 付长时效代价）；
  ZA2  zm_aB 的 spread 在长时效显著高于 aB（0.245 的有效 σ 真的注进去了）；
  ZA3  ⇒ SER 在长时效最接近 1 的一条。

若 ZA1 不成立（zm_aB 与 aB 的长时效 RMSE 同档甚至更差），说明那个代价不是均值踢
而是别的东西（非线性把大扰动放大成偏向），那 a(t) 这条轴的上限就是被那个机制封住的，
不是被均值踢封住的 —— 这本身也是个结论。

⚠ 这条臂**不**指望在 24~240h 上比 aB 更好：那段 σ 越大越好（整流项为正），去掉均值踢
  会同时削掉那个整流项里来自"共性分量"的部分。所以判据只看长时效，短时效只要求不劣化
  超过跨批偏移（~1%）。

⚠ 循环论证边界与 aB 相同：σ*(t) 是从 07-02 自己的 rmse 反推的，拿它评 07-02 是循环的。
正式判据落在 07-10 / 07-18 两天上。
"""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_zmaB"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
# 与 aB 逐点相同（288h 那点已剔除，理由见 aB 的 docstring）。
cfg.latent_forcing["amp_profile"] = (
    (0, 1.76), (4, 2.62), (8, 2.34), (12, 2.36),
    (20, 2.68), (28, 3.21), (40, 4.74), (60, 6.13),
)
cfg.latent_forcing["demean_members"] = 51
