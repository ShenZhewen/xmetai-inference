# -*- coding: utf-8 -*-
r"""``fgvp_ens_spw`` 的两整月 / 4 卡档（第 15 轮，与 ``fgvp_ens_ar1state_aBt15`` 头对头）。

与 ``fgvp_ens_spw.py`` 的 diff **只有四处字面量**：``name`` / ``output_dir`` /
``times`` / ``gpus``(+``cuda_devices``)。扰动配方（``weight_perturb`` 的
groups / sigma_full / exclude / noise_struct / seed）、数据集、dataloader、
``rollout_batch``、``members``、``vars``、``steps`` 全部逐字继承。

---- 为什么另开一个文件，而不是原地改 ----

``fgvp_ens_spw.py`` 是「51 成员 / 8 图 / 8 卡 / 29 起报 20250702-0730」那个**首个配方
的留痕**（``perturbation/README.md`` 2O）。原地改 times 会让那份记录和它的产物对不上；
更实际的风险是**输出根串档** —— 本批是 4 卡口径，与 8 卡混在同一个 root 下，
``resume=True`` 会把同名已算块静默跳过，日志一切正常而口径其实是另一个
（v14 的 ``weather_rmse_ens_fgvp_ar1v14.py`` docstring 里那条老账是同一个坑）。

---- 8 卡 → 4 卡：动的是什么、不动的是什么 ----

``PERTURB_GRAPHS = 8`` **不变**。图外层编舞按**图**切卡（``range(rank, G, 卡数)``），
所以 4 卡下每个 rank 领 2 份图、而不是 1 份：

    8 卡：rank r → 图 {r}          一卡一图（原配方）
    4 卡：rank r → 图 {r, r+4}     一卡两图

**成员集按图摊分（``cli._graph_member_split(51, 8)`` = 7×3 + 6×5），不随卡数变**
⇒ 51 个成员的 (图号 g, 图内序号 j) 二元组**逐个不变**，只是分给哪个进程变了；
「一图的成员永不跨进程」这条约束在 4 卡下依然成立（同一图的 J 条轨迹都在同一个
rank 内跑完），``members >= 2G``（51 >= 16）也照样满足。

⚠ 但「成员集不变」≠「逐位同流」：换卡数会改 session 的 load/release 次序与
  每进程的图内前向计数，而图内随机算子是按「worker 内第几次前向」对齐的
  （README 坑 1）⇒ 本批的数与任何在盘的 8 卡 spw 档**不可静默横比**。
  要横比就同批同卡数（本批两臂都是 4 卡，批内自洽）。
⚠ ``resident_members`` 继承 2。原配方下同一张卡只有一份图常驻（一 rank 一图），
  2 = 留一个余量槽；4 卡下一 rank 两图、但编舞仍是逐图 load/release，池子语义
  不变，所以 2 照样够。**首次 smoke 时盯一眼显存**，不够就抬到 3。

---- 前置（这两条是跑通的前提，不在本文件里）----

``weight_perturb["groups"] = ("latent_mixer",)`` 和 ``"sigma_full": 0.027`` 在
``fgvp_ens_spw.py`` 里是**从确定性 gdn2 图抄的占位值**（README 2O 的「未验/待做」
第 1~3 条）。**本批按"已在服务器上核过"处理** —— 若实际没核，第一批跑出来的
spw 那条腿作废，不是"效果不好"而是"跑的不是想跑的臂"。

---- 窗口与预算 ----

    2025070200..2025073000:24   29 起报   ← 与 fgvp_ens_spw 原配方同窗口
                                             （s2b / fgvp_ens 归档同口径）
    2025110100..2025113000:24   30 起报   ← 冬季整月；L360 → 2025-12-15，
                                             在观测档案边界（2025-12-31）内

59 起报 × 51 成员 × 60 帧 × 5 变量 ≈ **3.8 TB**（原配方 29 起报 ≈ 1.85 TB 的两倍）。
与 ``fgvp_ens_ar1state_aBt15`` 合计 **≈ 7.5 TB** —— 开跑前确认卷装得下。

---- 跑法 ----

    cd /workspace/szwCode/xmetai-inference2
    # 先 smoke：1 起报 / 全部图 × 少量成员 / 6 步，写 /tmp，别碰生产目录
    xmetai-infer --model fgvp_ens_spw_2m --times 2025070200 --steps 6 \
        --members 16 --gpus 1 --cuda-devices 0 --out /tmp/smoke_spw_2m
    # members 必须 >= 2 x PERTURB_GRAPHS = 16（每图至少 2 成员的校验）。
    # 正式：
    xmetai-infer --model fgvp_ens_spw_2m

⚠ smoke 用 ``--gpus 1`` 时 8 份图会串行装 8 遍，慢；只是验链路，别拿它估速度。
"""
from copy import deepcopy

from xmetai_inference.configs.fgvp_ens_spw import cfg as _spw

cfg = deepcopy(_spw)
cfg.name = "fgvp_ens_spw_2m"
cfg.output_dir = "/workspace/data/shenzw/fgvp_ens_spw_2m"

# 两个整月，与 fgvp_ens_ar1state_aBt15 逐字相同的窗口。
cfg.times = "2025070200..2025073000:24,2025110100..2025113000:24"

# 8 卡 → 4 卡（G 仍是 8，每卡领 2 图，见 docstring）。
cfg.gpus = 4
cfg.cuda_devices = "0,1,2,3"
