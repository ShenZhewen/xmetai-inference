# -*- coding: utf-8 -*-
"""fgvp_ens：FGVP 的 Mamba 骨干集合预报版本（51 成员，FP32 导出）。

骨架抄 fuxi_ens（集合路径），dataset processors 抄 fgvp single（同为 IWC 族、
同一套 era5_foundation_store2）。与 fgvp single 的差别只有三处：
  * model_path 指向集合版的 onnx，model_class 换成集合模型类；
  * members=51（fgvp single 是 1），输出落 member_<NNN>/ 子目录；
  * 模型路径与算子库用字面量而非 os.environ.get —— 项目约定：新配置不引入环境
    变量，改路径直接改这个文件（可 diff、可留痕）。其余 config 里的
    os.environ.get 是历史写法，不动。

调度（多进程 / 多卡）超参数见下面的 `SCHEDULING`：任务单元是 `(起报, 成员块)`，worker 自己
去 `{output_dir}/.xmetai_plan/` 认领，跑完打完成标记（重启跳过已完成的、崩掉的 worker 手里的
任务被活着的人捡走）。命令行只留 `--schedule` / `--procs-per-gpu` / `--resume` 三个日常覆盖
项，其余全部跟配方走进版本控制。字段含义见 README 的「多卡运行」。

⚠ 本模型的「成员」是**图内随机算子**给的（框架不加扰动，51 成员 = 同一个 session 里的 51 次
前向），所以一个任务必须圈住「一个起报的全部成员」—— 把一个起报的成员拆到不同进程会静默
产生重复成员，理由见 SCHEDULING 里 `mode` / `members_per_task` 的注释。

本次运行口径（2 x H100 80G / 24 核 / 256 G 内存）：9 个起报 x 51 成员 x 60 步，与
`fgvp_hou`、`M3`（以及 `fgvp_spw_prod_s2b`）**同一窗口**，产物可以并排读。2 卡 x 每卡 2
进程 = 4 个 worker，9 个任务按认领分（3+2+2+2）：一张卡上两个会话同时推理，为的是把单
进程里那 51 次串行前向之间的空转填上（理由见 SCHEDULING 的 procs_per_gpu）。想先来一发
快的：`--times 2025070200 --steps 12`。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

FGVP_ENS_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# 集合成员数：每个起报落 members 个 member_<NNN>/ 目录。下面 SCHEDULING 要用同一个数
# （一个任务圈住一个起报的全部成员），所以抽成常量，避免两处各写 51 走神。
MEMBERS = 51


cfg = InferConfig(
    name="fgvp_ens",
    model_path="/workspace/tmp/douzsh/models/iwc_mamba_260909.onnx",
    ops_library="/workspace/tmp/douzsh/models/xmetai_onnx_plugins.so",
    # 必须 False：图只出预报那一帧（单步型），GPU 常驻分支直接 state=result 回填、
    # 不经过 _next_state 的滑窗，且 _spare buffer 复用假设输入输出同形。见模型类
    # docstring。OnnxInferModel.load 的启动期断言会拦这个组合。
    gpu_state=False,
    model_class="iwc_fgvp_mamba_ens_onnx",
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
        # 集合路径上前向逐条跑（图 batch 维固定 1），batch_size 只影响数据装载
        # 分组，不影响速度。主机内存的大头不是这里的预取队列，而是 Rollout 自己：
        # 51 个成员 = 51 条轨迹，每条持有一份 2 帧 state 拷贝（约 648 MB），一个进程约
        # 33 GB 常驻；每步的零头只有几百 MB（预报帧与输出切片），所以每进程峰值约 35 GB，
        # 4 个进程约 140 GB。上面 SCHEDULING 的 procs_per_gpu 那条记着 66 GB 那个坑。
        # 注意：claim 调度下 DataLoader 被绕过（cli 会打日志明说），num_workers /
        # batch_size 都不生效，输入由认领侧的预取线程按任务直接取（一帧约 650 MB）；这两项
        # 只在 --schedule static 下起作用。
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
    # 起报范围（语法见 base.parse_times）：9 个起报，与 fgvp_hou / M3 / fgvp_spw_prod_s2b
    # 同窗口，产物可以并排读。全年正式跑是 "2025010200..2025121500:24"（348 个起报）——2 卡
    # 不划算，等回到 4 卡再换回来。只想快速自检就把窗口收成 2 个起报。
    times="2025070200..2025073000:24",
    steps=60,
    members=MEMBERS,
    vars="z500,u200,v200,msl,tp",
    # 2 x H100（跑之前 `nvidia-smi -L` 确认卡号；个数必须 == gpus，写错会在 CUDA 初始化时报
    # failure 100）。
    gpus=2,
    cuda_devices="0,1",
    # —— 单进程内部的线程预算 ——
    # ONNX Runtime 的**算子内并行线程**（intra_op）。0 = 自动 = 物理核 / 进程数，
    # 进程数 = gpus x scheduling.procs_per_gpu（本配方 = 4）。auto 就是「4 个 worker 平
    # 分物理核」，所以 24 核的机器每个 worker 拿 6 条线程；不设则每个 session 各自吃满
    # 物理核 -> 4 倍超订，zarr 解码 / CPU 算子 / 拷贝互相抢核。
    # 要显式封顶就写死一个数（例如 6）；CPU 侧前处理重时说明瓶颈在主机侧，可以再往上调
    # （4 个进程 x 6 条线程 = 24，正好把核用满，先按 auto 跑）。
    # inter_op 恒为 1（ORT_SEQUENTIAL 下它本来就不起作用），不需要配。
    intra_op_threads=0,
    # 卷积算法搜索：空 = ORT 默认 EXHAUSTIVE（首次推理慢、之后走缓存，零精度损失）。
    # 只有「一张卡跑多个进程」时每个 session 都要各搜一遍、启动预热翻倍，那时改成
    # "HEURISTIC" 换启动速度（代价是每层未必拿到最优 kernel）。
    cudnn_algo_search="",
    # 常驻 session 池（resident_members）对本配方**无意义**，所以不写：本模型的成员是
    # 图内随机算子给的，一个任务 = 一个起报的全部 51 个成员、全程同一个 session，没有
    # 「换成员 = 换 session」这回事。那个旋钮只服务 m1a（见 configs/fgvp_spw_ab.py）。
    # 自检专用目录：与正式归档分开，免得「对不对」被已有产物搅混（resume=outputs 会把盘上
    # 齐全的任务直接跳过）。测通之后改成 "/workspace/data/shenzw/fgvp_ens_output"，或用
    # `--out` 覆盖。
    output_dir="/workspace/data/shenzw/fgvp_ens_test",
)
