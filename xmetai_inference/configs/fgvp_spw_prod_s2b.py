# -*- coding: utf-8 -*-
r"""fgvp_spw_prod_s2b：SPW 的**大规模生产跑** —— S2B 配方，51 成员 / 29 起报 / 60 步 /
5 变量（z500, u200, v200, msl, tp）。

跑什么：`xmetai_inference/tmp_config/fgvp_spw_ab.py` 的 **S2B** 臂（`latent_mixer` + `noise_struct="channel"`
+ `sigma_full=0.027`），从 A/B 协议的 6 成员放大到 51 成员、29 起报。

为什么是 S2B（依据 = 判决页 `$P verdict F S2B`，细节见 perturbation/README 与 PAPER.md）：

  * 五轮实验里**唯一**「两头都不输」的配置：长时效 CRPS 比 F 好 6.8%，短时效只差 1.9%
    （F 自己在 1-3 天过冲参考集合 2.8-3.2 倍，S2B 把这个过冲压回 ~1.0 倍）；
  * **判据 3（尺度选择性）全项目唯一通过**：增益落在 1-4 / 5-20 带（5-20 增益 +21.0%），
    21-40 带 -9.0%，而不是「整体锐化」；
  * 判据 1 通过（长时效离散度比 1.022），**判据 2 差 2.7% 未过**（1.027）。

**S2B 不是合格配置，是「现有自由度里最好的那个」。** 长时效缺口需要等效 sigma x3.41，
S2B 只给到 x1.07；而且在 `latent_mixer` 这个注入点上**不存在**能同时过判据 1 / 2 的 sigma
—— 两头的比值几乎相同（1.027 / 1.022），调 sigma 会让它们**一起动**。这条否定定理的适用
范围也是实测出来的：换到**跨组**注入点（`decoder_group` / 全权重 `all`）它就不成立，
那里 k(lead) 确实不平（详见 PAPER.md §2.1.5 的适用范围段）。

所以这次生产跑的定位是：**给「曲线上的最优点」一个 51 成员的总体读数**，不是「过验收」。
A/B 协议只有 6 成员，σ̂ 由单个成员主导（拿掉第 3 号掉 22-34%），51 成员是这一批里第一个
能当基准用的总体值。

与原来那份 F 配方（`configs/fgvp_spw_prod.py`）共用的口径 —— **51 成员 / 29 起报 20250702-0730 /
60 步** —— 的理由**原本写在那份 config 的文件头里，而该文件已于 2026-09-24 删除**。
（变量是本次唯一放开的一项：那份是只存 z500，本次 5 个变量，见下面 `vars` 那段的磁盘代价。）
同样的口径说明、预算表（2F.4）与验收判据（2F.6）存档在 `perturbation/README.md` 的 **2F 节**，
本节不再重复。本文件只记 S2B 独有的部分。

成员集：**本次整份重造 51 个**（不复用第五轮那份）。

  指纹 = (groups, sigma_full, balance, exclude, include, min_elements, seed,
  **noise_struct**, noise_block（冻结键，见 weights.spw.NOISE_BLOCK_FROZEN）, member_format,
  源 .onnx 的 size/mtime)。
  S2B 与 F 只差 `sigma_full`（0.027 vs 0.04）与 `noise_struct`（channel vs iid）；
  `noise_struct` 的**非默认值进指纹**（`weights/spw.py` 的 `budgets()`，且有 selftest
  专门钉住「channel 不许耦合命中 iid 的成员集目录、静默复用错权重」）—— 所以 S2B 走自己的
  成员集目录，与 F **互不串味**。

  第五轮那份 S2B 成员集（`member_000..005`，在旧根 `/workspace/tmp/douzsh/models/spw_ab/`）
  **本次不用**：整份造 51 个 => 成员集预算 51 x 1.49 GB = **76 GB**（而不是复用的 67 GB）。
  `member_dir` 指向 `/workspace/data/shenzw/model_onnx`（下面按指纹分子目录），框架会自己
  makedirs，不需要预先建；`member_dir` **不在 budgets() 里**，换它不改指纹。

  重造本身是一次**免费的权重复现性检查**：规格、seed、源 .onnx 的 size/mtime 都没变，
  所以新造的 member_000..005 应当与第五轮那 6 个**逐字节相同**。旧根若还在，跑完 `cmp` 一下：

      NEW=$(ls -d /workspace/data/shenzw/model_onnx/*spw-members-*/ | head -1)
      OLD=/workspace/tmp/douzsh/models/spw_ab
      for i in 000 001 002 003 004 005; do
        cmp "$NEW/iwc_fgvp_gdn2.m$i.onnx" $OLD/*/iwc_fgvp_gdn2.m$i.onnx && echo "m$i 同"
      done

  全同 => 扰动生成本身在重构前后没动；有不同 => 扰动路径变了，A/B 那几轮的结论要重新掂量。
  （旧根已清掉就跳过，不阻塞。）

  成员集一致时还多一次**零成本端到端对账**：member_000..005 x 起报 0702/0703 x 60 帧，
  应当与 A/B 归档 `D:\xmetai-evaluation-result\spw_ab\S2B` 逐帧吻合。容差按
  |Δ| / (成员间 σ̂) <= 1e-3 —— **不是逐位相等**：29 起报的分批与 3 起报不同，CUDA 归约
  顺序会变。远小于才是对的；若达到成员离散度的量级，说明成员集或前处理串了，停下来查。

预算与算力：

  * 成员集 51 x 1.49 GB = **76 GB**（= 70.9 GiB；2026-09-24 实测 26 个成员合计 36.13 GiB，
    单个 1.39 GiB = 1.49 GB，对得上）；
  * 输出 **29** 起报 x 51 成员 x 60 帧 x **20.8 MB** = **~1.85 TB**
    （每帧 5 个变量 x 4.15 MB；只存 z500 的话是 368 GB）。
    **这是本轮最大的一笔开销 —— 开跑前先确认目标卷装得下**，写盘时间也按 5 倍算。
  * **多一份 base.data**（+2.98 GB，一次性）：成员集里共享的那份 base 是源 `.data` 的
    硬链接，硬链接要求同卷，而源在 `/workspace/tmp/douzsh/`、成员根在 `/workspace/data/`
    —— 不同卷就退化成整份复制。`build_member_set` 会打一条 warning 说明，不是错误。
    想省这 2.98 GB 只有一招：把源 `.onnx`/`.data` 也放到同卷。**但那会改指纹里的
    size/mtime（源身份进指纹）-> 51 个成员全部重造**，得不偿失，别为了 2.98 GB 去动源。
  * **88 740 次前向**（29 x 51 x 60）。按 2026-09-24 的实测锚点现算：
    实测 **3.49 分钟/任务**（1 起报 x 1 成员 x 60 步、5 变量）
    = **0.287 任务/分 / 每个 worker**（当时的原始读数：2 个 worker 合起来 0.57 任务/分）。
    吞吐 ≈ 0.287 x worker 数，**线性外推是上界**，实际会低一些。各档都算好放这儿，
    改 worker 数不用重推：1479 / (0.287 x W) 小时。
      - **W=6**（2 卡 x 3 进程，**本配方现值**）= **14-43 h**，线性外推 14.3 h 只是天花板。
        `procs_per_gpu>1` 的收益**从来没实测过**（见文件头「多进程 / 多卡」）：单卡功耗
        只有 200 W / 700 W，说明内核之间有缝隙可填，但「填空转」填不满是常态；何况
        PCIe / 主机带宽每卡只有一份，3 个进程已经在互抢。**实际期望 1.0-1.4 任务/分
        -> 18-25 h**。跑起来先量「任务/分」，按文件头那三条判据决定留在 3 还是退回 2/1。
      - **W=4**（2 卡 x 2 进程）= **21-43 h**。
      - **W=2**（2 卡 x 1 进程）= **43 h**。**只有这一档有实测支撑**，其余全是外推。
    写盘不是瓶颈：5 变量实测 3.49 s/步，当初只写 z500 时是 3.66 s/步 —— 多 4 个变量没变慢。

跑法（服务器）：

    cd /workspace/szwCode/xmetai-inference2
    nvidia-smi -L                     # 先确认真实可见的卡号与 cfg 的 cuda_devices 一致
    xmetai-infer --model fgvp_spw_prod_s2b            # 2 卡，见 cfg 的 gpus / cuda_devices

    # 只想先验链路（1 起报 / 3 成员 / 6 步，几分钟）。注意 **务必加 --out**：
    # 不加就往生产输出目录写，会和正式跑的产物混在一起。
    # 顺带把 member_000..002 造出来（成员集是共用的，正式跑接着补 003..050）。
    xmetai-infer --model fgvp_spw_prod_s2b --times 2025070200 --members 3 --steps 6 \
        --gpus 1 --cuda-devices 0 --out /workspace/data/shenzw/spw_prod_s2b_smoke

**多进程 / 多卡（本框架新增的任务级动态认领）—— 三条要盯住：**

  * **`procs_per_gpu` = 3（每卡三个 worker 进程），并行度 = 卡数 x procs_per_gpu。**
    **这是本文件里唯一被实测推翻过的判断，动它之前先读这段。** 原来这里写的是「必须保持
    1：m1a 每个成员是一份 1.49 GB 权重 + 一个 session，显存分不出第二份」——2026-09-24
    实测证伪：单进程单卡时 `nvidia-smi` 是 **14153 MiB / 81559 MiB（17%）**，功耗只有
    **142-200 W / 700 W**。显存不是约束，SM 也没吃满（`utilization.gpu` 报的 100% 只说明
    「有 kernel 在跑」，不等于 SM 忙 —— 满载的判据是功耗，H100 SXM 应在 400-600 W）。
    框架本来就支持多进程：`cli.py` 的 `_gpu_mem_per_proc` 把 `gpu_mem` 按份数分（本配方
    没设 gpu_mem，取缺省 0.7 -> 每份 0.233，80G 卡上约 **18.7 GB**），每份 > 0.9 才告警。
    **为什么停在 3 不往 4 走**：单 worker 实测吃 ~14 GB，3 份 = 42 GB / 80 GB，配额
    18.7 GB 也高于实测用量，两头都有余量；4 份的话配额恰好落到 14 GB = 实测用量本身，
    等于把 ORT 的天花板压在实际占用上，容易 OOM。**显存是这条的硬上限，不是 CPU/内存**
    —— 这台机器核和内存都富余，但它们喂不出更多吞吐（见文件头预算段）。
    语义上也安全：m1a 的成员身份绑在**权重文件**上，不像 fgvp_ens 绑在 session 内的前向
    序号上，所以多进程**不会**产生重复成员（「一个任务必须圈住整个起报」那条约束是
    fgvp_ens 的，这里没有）。
    **待补实测**：本配方现在是 **2 卡 x 3 进程 = 6 worker**，而 procs_per_gpu>1 的收益
    从来没量过 —— **3 是没量过就上的，风险最大的一档，跑起来务必前 15 分钟就核一遍**。
    锚点是 **0.287 任务/分 / worker**（2026-09-24 实测，每 worker 1 进程、3.49 分钟/任务），
    6 个 worker 线性外推应该看到 **~1.7 任务/分**，实际大概率 1.0-1.4（PCIe / 主机带宽每卡
    只有一份，3 个进程已经在互抢）。
      - 爬到 **≥1.4** -> 3 赚到了，保持。
      - 只爬到 **~1.15** -> 3 和 2 打平（都是 0.19/worker 的边际），**退回 2**，别白占显存。
      - 只爬到 **~0.86** -> 多进程完全没生效（= 1 进程的水平），退回 1。
    判据是**总吞吐**不是单任务耗时：每卡 3 进程时单个任务必然更慢，别读错。
  * `claim_order="member"`：连续认领同一成员就不必重新装载权重（省一次 1.49 GB 读盘）。
    m1a 下 `auto` 也会这么分流，写死是为了把这个便宜钉住。
  * 计划 / 完成标记落在 `{output_dir}/.xmetai_plan/`，**与 F 的生产目录各一份、互不干扰**；
    `resume="outputs"` => 产物齐全即跳过，中断后重跑同一条命令即可。

跑完之后（**先做权重与对账，再谈指标**）：

  1. 权重复现：新造的 member_000..005 与第五轮那 6 个 `cmp`（命令见「成员集」段）——
     相同才谈得上后面几步。旧根已清掉就跳过。
  2. 对账（上一条通过才做）：member_000..005 在起报 0702 / 0703 上与 `spw_ab/S2B`
     归档比（容差见上）。
  3. 体检：`python -m xmetai_inference.perturbation check <out_dir>`
     —— 每个成员是否都跑满 60 帧（G 就是 member_002/003 只有 31 帧才作废的）。
  4. 成员数偏差：`$P spread --members 0,1,2,3,4,5 <out_dir>`，量 σ̂@6 相对 σ̂@51 差多少 ——
     这条决定前面几轮基于 6 成员的结论能不能直接沿用。
  5. 盘点：`du -sh /workspace/data/shenzw/fgvp_spw_prod_s2b` 对一下 ~1.85 TB 的预算
     （显著偏小说明 NetCDF 开了压缩，那条预算要重算）。
  6. 评测（xmetai-evaluation-v2，裁判是现成的）：

        FUXI_ENS_OUTPUT=/workspace/data/shenzw/fgvp_spw_prod_s2b \
        START_DATE=20250702 END_DATE=20250730 \
        EVAL_OUTPUT=/workspace/.../results/spw_prod_s2b \
        xmetai-eval --config weather_rmse_ens_fgvp

口径提醒（与原来那份 F 配方相同，都是踩过的坑）：

  * `cuda_devices` 必须是容器里**真实可见**的卡号；写了看不见的号，rank 会以
    `CUDA failure 100: no CUDA-capable device is detected` 直接死，不是退到 CPU。
  * 成员集里的 `base.data` 是源 `.data` 的**硬链接**，必须与源文件同卷；不成立就整份复制
    （本次新根与源多半不同卷 -> 会复制，见「预算与算力」）。
  * **源模型路径没动**：`model_path` 仍是 `/workspace/tmp/douzsh/models/iwc_fgvp_gdn2.onnx`。
    本次只挪了成员权重的落盘根目录。如果那边也要挪，那是另一件事 —— 源路径进指纹
    （size/mtime），一挪 51 个成员全部重造，要先说。
  * 别指望符号链接 —— onnxruntime 的外部数据真实路径校验会拒绝（坑 6）。
"""
from __future__ import annotations

from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
# 源模型（确定性 .onnx + 插件 .so）所在处 —— 只读，本次不动。
MODEL_DIR = "/workspace/tmp/douzsh/models"
# **成员权重（m1a 落盘的 .mNNN.onnx / perturbed.mNNN.data）的根目录。**
# 框架在它下面建一个按指纹命名的子目录：
#   /workspace/data/shenzw/model_onnx/iwc_fgvp_gdn2.spw-members-<指纹>/
# 换这个值**不改指纹**（member_dir 不在 spec.budgets() 里），所以子目录名不变、
# 变的只是父目录 —— 本次整份重造，见文件头「成员集」那段。
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

FGVP_SPW_PROD_S2B_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# 与 fgvp_spw_ab 逐字相同（S2B 的 drop 为空 -> exclude 就是这 7 个）。
# 指纹要一样 —— 指纹一样 = 同一份成员权重，所以整份重造出来的仍是 A/B 那一轮的成员集，
# 那几轮的结论照样适用（重建后 `cmp` 一下就能确认，见文件头「成员集」段）。
BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")


cfg = InferConfig(
    name="fgvp_spw_prod_s2b",
    # 原始确定性图（不是成员模型）。框架按规格找 <模型名>.spw-members-<指纹>/ 下的
    # <模型名>.mNNN.onnx（扁平布局）。指纹里的源 .onnx size/mtime 必须与第五轮一致，
    # 否则 S2B 的 6 个旧成员认不回来、要整份重造 51 个。
    model_path=f"{MODEL_DIR}/iwc_fgvp_gdn2.onnx",
    ops_library=f"{MODEL_DIR}/xmetai_onnx_plugins.so",
    # 必须 False：图只出预报那一帧（单步型），GPU 常驻分支会绕过滑窗回填。
    gpu_state=False,
    model_class="iwc_fgvp_gdn2_onnx",
    weight_perturb={
        # 注入点：latent_mixer（携带状态的 recurrent 核）。与 F 相同 —— S2B 只改了
        # 噪声结构与幅度，注入点没动。
        "groups": ("latent_mixer",),
        # 显式写空：S2B 没有需要 include 的小张量（那是 T 族才要的，A_log 只有 24 元素）。
        # 写出来是为了让「S2B 与 A/B 那一轮逐字段一致」肉眼可查 —— 它进 budgets()，
        # 但 () 与不写等价。
        "include": (),
        # **S2B 的压强**：0.027 是按「短时效 CRPS 代价对 sigma 的斜率 ≈ 14.9 / 单位 sigma」
        # 从 S2 的 0.0341 反推的靶心档。这里是**字面量**，不是环境变量 —— 换值必须改文件，
        # 因为改它会换指纹 -> 另造一整套 51 成员，这个动作要留痕。
        "sigma_full": 0.027,
        "balance": "energy",
        "exclude": BASE_EXCLUDE,
        # **S2B 的定义性字段**：逐输出通道共享一个 xi（delta_W = diag(xi).W），而不是
        # 逐元素 iid。非默认值**进指纹**（weights/spw.py 的 budgets()），
        # 所以它会落到自己的成员集目录，不会命中 F 的 iid 成员集。
        "noise_struct": "channel",
        "min_elements": 64,
        # 成员集根目录 = /workspace/data/shenzw/model_onnx（下面按**指纹**分子目录）。
        # ⚠ 这个值不在 budgets() 里 -> 换它**不改指纹**，同名的子目录只换了父目录。
        # 框架会自己 makedirs(子目录)，不需要预先建 model_onnx。
        "member_dir": MODEL_ONNX_DIR,
        "seed": 0,
    },
    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_SPW_PROD_S2B_DYNAMIC_STORES,
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
        # m1a 是成员外层、数据集内层：51 成员 = 数据集走 51 遍。真正要控的是主机内存
        # （预取队列 ~= workers x 2 x batch x 645 MB/样本）。内存小的机器先降 num_workers。
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
    # 29 个起报（日间隔 24h，20250702-0730）。与 fgvp_ens.py 的值一致 —— 那份的 times 也是
    # `..2025073000:24`，虽然它自己的注释还写着「9 个起报」（2026-09-24 发现，未改那份）。
    # ⚠ 原来这里是 9 个起报（到 0710），配的注释是「与 fgvp_hou（51 成员归档）和 M3
    # （51 成员）**同窗口**，三方 CRPS 才能并排读」。窗口从 0710 放到 0730 之后，
    # **「同窗口」这条没有核对过** —— 要把三方并排读之前，先去确认那两份归档的起报范围。
    # S2B 的 A/B 窗口是 0701-03，重叠 0702/0703，对账不受影响 -> 对账用。
    times="2025070200..2025073000:24",
    # 60 步 x 6 h = +360 h：要覆盖论文报告失效模式的 +240/+360 h。
    steps=60,
    hour_interval=6,
    # 51 成员 = 落盘 51 套权重（论文量级 50 + 控制位）。框架按 rank 分片建成员，
    # 建完之后跑什么由任务表动态认领，与建成员的分片无关；同一张卡同时驻留几个成员的
    # session 由下面的 resident_members 定（不再是「只能 1 个」）。
    members=51,
    # 五个变量：z500 给 CRPS / spread 裁判，u200 / v200 / msl / tp 是论文失败模式那一组。
    # **代价是磁盘**：每帧 4.15 MB x 5 = 20.8 MB，产物从 368 GB（只存 z500）涨到 ~1.85 TB。
    # `vars` 进 plan 指纹（run_payload），所以改了它旧产物不会被 resume 认领 —— 本次是首跑，
    # 无所谓；以后想加变量，等于换一份计划、整批重算。
    vars="z500,u200,v200,msl,tp",
    # 容器里真实可见的卡号（跑之前 nvidia-smi -L 确认）；个数必须 == gpus。
    # 本次按 **2 卡**跑（2026-09-24 曾改成 4 卡，当天因没有四卡改回）。换卡数用
    # `--gpus N --cuda-devices ...` 覆盖即可 —— 卡数不进 plan 指纹
    # （见 scheduling/plan.py 的 run_payload），不影响续跑。
    # 并行度 = 卡数 x procs_per_gpu = 2 x 3 = **6 个 worker 进程**（见上面的 SCHEDULING）。
    gpus=2,
    cuda_devices="0,1",
    # 常驻 session 池：m1a 每个成员一份权重（latent_mixer 1.49 GB）+ 一个 session，装载
    # 一次还要重预热 cuDNN。51 份全常驻 = 76 GB，不现实。
    # **本配方给 2，不是原来的 8**：池子的收益是「来回切成员时不重装」，而 claim_order=
    # "member" 下每个 rank 同一时刻只有一个成员是活的（连跑完它的全部起报才换下一个），
    # 换出去的成员这一轮再也不会回来 —— 池子从头到尾没有命中机会。留着 8 只是白占
    # 8 x 1.49 ≈ 12 GB 显存（沿用上面那句的估法），而这份显存正是 procs_per_gpu=3 要用的：
    # 3 份 x (12 GB 权重 + arena) 已经吃掉 42 GB / 80 GB，没有空间再养一个 8 槽的池子。
    # 任务失败时框架会 `_drop_sessions` 整池清空重来，重派同样指望不上池子。
    # 给 2 是留一个余量（正常用 1 个，另一个槽位给「同成员换批」之类的意外）。
    resident_members=2,
    # 正式输出目录（与 F 的生产目录分开，也与试跑目录分开）。
    output_dir="/workspace/data/shenzw/fgvp_spw_prod_s2b",
)
