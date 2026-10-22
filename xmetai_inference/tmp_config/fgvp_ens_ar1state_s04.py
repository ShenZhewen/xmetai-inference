"""AR(1) main arm: latent-mixer output / decoder entry, sigma=0.04, tau=3d.

Shared screening recipe for the eleven anchor/time/amplitude/scale ablations.
See perturbation/README.md, section 0, for smoke runs and paired evaluation.
Targets are specific to the 20261008 iwc_gssp_im_fp16.onnx export.
"""
from xmetai_inference.tmp_config.fgvp_ens_pert import build

cfg = build("fgvp_ens_ar1state_s04", ar1=True, sigma=0.04, rollout_batch=6)
cfg.latent_forcing["targets"] = ("add_35782", "add_35787", "add_35792")
cfg.latent_forcing["tau_days"] = 3.0
# 起报 3（2026-10-13 深夜定稿）。三天 = 20250702 / 07-10 / 07-18。
#
# 为什么退回三起报：第 13 轮为了省时间砍成单起报，当时代价写得很清楚 ——
# **丢掉"逐起报符号一不一致"这道闸**。那一轮所有读数都是 n=1。把在盘档案摊开
# 量过之后，代价比原先记的还大：同一个臂只换起报，z500 RMSE 在 24h 差 37~40%、
# 168h 差 27~29%、360h 差 9~14%（ar1state_s00/s02/s04 的三层报档），
# **比要读的臂间效应（1~3%）大一个量级**。更要命的是最优 σ 本身就随起报跳：
# 360h 上 07-02 是 s04 最好、07-10 是 s02 最好、07-18 是**不扰动**最好。
# ⇒ n=1 给不出误差条，"定一版"就定不下来。
#
# 为什么是这三天（深夜改过一次）：**对齐在盘已有的三层报档案**。
#   fgvp_ens_ar1state_s00 / _s02 / _s04 用的就是 07-02 / 07-10 / 07-18，
#   评测协议已核实一致（同 VARS=["z500"] / ddof=0 / centered=False /
#   max_wavenumber=720 / n_workers=9 / writers=csv_long+spectrum），
#   实测跨批偏移中位 0.6~2.9% ⇒ 那三条**可以直接复用，不必重跑**，本批只跑
#   "没有三层报的"五条（s08/aA/aB/l8/tinf），省 9 趟。
#   （原定的 07-03 是为了接 fgvp_ens_spw 的机制级对照，但那份只有 07-02/07-03
#    两天、本来就只有 n=2；真要读那个对照得在 SPW 侧补跑 07-10/07-18，
#    那是另一趟 Job，不在本批。）
# 代价：推理量 ×3（本轮 5 臂 × 3 起报 = 15 趟）。
#
# ⚠ 起报数是**整族共享**的：全部兄弟臂 deepcopy 本文件，这里改一次整族生效。
# ⚠ 评测侧 periods 必须同步（weather_rmse_ens_fgvp_ar1v14.py）。不同步的后果不是
#   报错，是 reader 去找一份不存在的场：**出空样本、静默少一行**，最难查。
cfg.times = "2025070200,2025071000,2025071800"
cfg.members = 51
cfg.steps = 60
cfg.vars = "z500"
# 一条臂 4 卡（2026-10-14：**8 卡 → 4 卡**，配额不够，整批十条臂一起降）。
#   cli 对集合固定**按成员切**：每卡领 range(rank, 51, 4) = 13/13/13/12 个成员，
#   各卡独立跑完整起报。成员身份 = 绝对的 member_index，不因换卡数重新编号，
#   cli 传的也是 range(local_rank, members, world_size) 这个绝对序列。
#
# ⚠ 但卡数会改集合统计量，**跨卡数不能混读**，原因是两类：
#   1. 每卡内一次前向堆几条轨迹会变（4 卡下 13 成员 → 6+6+1）。
#   2. 图内那份原生 ε 不受控、且按「worker 内第几次前向」对齐
#      ——见 perturbation/README.md「坑 1」（两个 session 的第 k 次前向拿到逐位相同的 ε）。
#      本族成员只靠驱动侧的 ξ 区分，换卡数会把 51 个成员**重新分组到不同数量的 ε 类**里：
#      8 卡是 7 个类（每类 ~7.3 个成员），4 卡是 13 个类（每类 ~3.9 个）。
#   3. **驱动侧 ξ 的流本身也换**（比 1、2 更容易被忘）：_rank_entropy =
#      (seed, 7919, member_index, slot)，而 slot = m % world_size
#      （latent.py 的 _world_size / _build_cohort 按它重放）。8 卡 slot = m%8，
#      4 卡 slot = m%4 ⇒ **51 条流全部重抽**。latent.py 自己写着「卡数不对，
#      重放出来的就不是那 51 条流」——去均值臂（zm15/zmaB）尤其依赖这条。
# ⇒ 一切结论只在本批**同卡数**内部成立。在盘的 ab_v3 / ab_v3_rerun / 三条起报档
#   （fgvp_ens_ar1state_s00/s02/s04）**全是 8 卡口径** ⇒ 本批的**绝对量**（SER 数值、
#   σ*≈0.095）不能与它们对拍；**批内臂间差值不受影响** —— 卡数效应的一阶项是共模的，
#   两条臂共享同一套 ξ 流与 ε 分组，减一下就掉了。
#   先例（PAPER.md §4.1 的 P1）：2 卡 → 8 卡，24h spread 只动 **+0.19%**，
#   而那次的重组幅度（26 类 → 7 类）比这次的 8→4（7 → 13）还大。
# 十条兄弟臂全部 deepcopy 本文件 ⇒ 这里改一次，整族同时生效。
cfg.gpus = 4
cfg.cuda_devices = "0,1,2,3"
