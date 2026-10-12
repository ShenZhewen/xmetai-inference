# -*- coding: utf-8 -*-
"""fgvp_ens：FGVP 的 Mamba 骨干集合预报版本（51 成员，FP32 导出）。

骨架抄 fuxi_ens（集合路径），dataset processors 抄 fgvp single（同为 IWC 族、
同一套 era5_foundation_store2）。与 fgvp single 的差别只有三处：
  * model_path 指向集合版的 onnx，model_class 换成集合模型类；
  * members=51（fgvp single 是 1），输出落 member_<NNN>/ 子目录；
  * 模型路径与算子库用字面量而非 os.environ.get —— 项目约定：新配置不引入环境
    变量，改路径直接改这个文件（可 diff、可留痕）。其余 config 里的
    os.environ.get 是历史写法，不动。

多卡分工见 `cli.py` 的 worker 分片：集合（members>1 且 members >= 卡数）**按成员切**
（`range(rank, members, world_size)` —— 每卡跑**全部**起报、只挂自己那一份成员），
确定性（members==1）才按起报切。一卡一进程（`for rank in range(gpus)`）。命令行只留
`--gpus` / `--cuda-devices` / `--times` / `--steps` / `--out` 等常规覆盖项。

⚠ 本模型的「成员」是**图内随机算子**给的（框架不加扰动，51 成员 = 同一个 session 里的
51 次前向）。注意这与上面那条「按成员切」**是有张力的**：拆到不同进程后 member_<NNN>
互不相同，前提是各进程的随机序列互不相同。见 `members` 字段旁的说明。

本次运行口径（8 x H100 80G）：132 个起报 x 51 成员 x 60 步，8 个 worker（一卡一进程），
每进程只挂 ceil(51/8) = 7 个成员，产物落 fgvp_ens_output_5。
想先来一发快的：`--times 2025072100 --steps 12 --out /workspace/data/shenzw/_ens_smoke`。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

FGVP_ENS_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

# 集合成员数：每个起报落 members 个 member_<NNN>/ 目录。抽成常量只是为了单一出处，
# 避免 times/output 那几处各写一遍 51 走神。
MEMBERS = 51


cfg = InferConfig(
    name="fgvp_ens",
    model_path="/workspace/tmp/douzsh/models/20261008/iwc_gssp_im.onnx",
    # 2026-10-09 换成 GPT 优化过的算子实现（对方称 4x 提速）。旧的
    # xmetai_onnx_plugins.so 留着做回滚与数值对照，别删。
    # ⚠ 文件名里的 ort1_24 = 按 ONNX Runtime 1.24 构建。先对一下容器里的版本
    #   python -c "import onnxruntime; print(onnxruntime.__version__)"
    # 对不上就别用，加载或运行期会崩。
    ops_library="/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so",
    # 必须 False：图只出预报那一帧（单步型），GPU 常驻分支直接 state=result 回填、
    # 不经过 _next_state 的滑窗，且 _spare buffer 复用假设输入输出同形。见模型类
    # docstring。OnnxInferModel.load 的启动期断言会拦这个组合。
    gpu_state=False,
    # 前向批：一次 forward 堆几条轨迹（1 = 逐条，与历史行为一致）。
    # ⚠ 本配方下批的是**成员**，但一个进程并不拿满 51 个成员 —— cli.py 按成员切，
    #   每个 rank 只领 range(rank, 51, 卡数) 那 ceil(51/卡数) 条（8 卡 = 7 条）。
    #   所以 k 的有效上限是那 7 条（再大最后一块也只会是余数），而不是 51。
    # 显存按 k 倍涨：单次前向的输入拷贝 k x 648 MB 加中间激活（OOM 时在 node_Add_*
    #   上炸的就是这部分，随 k 长）。轨迹状态常驻是另一笔、与 k 无关。
    # 图已重导为动态 batch 轴（输入/输出第一维都是符号维）；万一图不是，runner 会打
    #   WARNING 退回 1，不会静默算错。与 gpu_state=True 互斥（直接报错，不退回）。
    # ⚠ 与下面 dataloader 的 batch_size 不是一个东西：那个是 Dataset 一次装几个起报
    #   （I/O 批），跟前向批无关，也不影响速度。
    rollout_batch=6,
    # ORT 的显存天花板，占**卡总显存**的比例（backends/base.py::_default_gpu_mem_limit ->
    # onnx.py 的 cuda_opts["gpu_mem_limit"]）。缺省 0.7 在 80G 卡上只放出约 55.7 GiB，
    # 剩下 24 GiB ORT 根本申请不到 —— k 往上调时报的
    #   bfc_arena ... Available memory of X is smaller than requested bytes of Y
    # 是撞**这个顶**，不是撞卡。0.9 -> 约 73.4 GiB，留 ~6 GB 给 driver / CUDA context /
    # 碎片；框架在 >0.9 时会告警。
    # 一卡一进程（cli.py 的 for rank in range(gpus)），所以每个进程按字面值吃满自己的
    # 卡，不存在「两个进程挤一张卡、比例要减半」的情形。
    gpu_mem=0.9,
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
        # batch_size 只影响数据装载分组、不碰 GPU，所以不影响速度；前向批是上面的
        # rollout_batch（本模型图已重导为动态 batch 轴，不再是「固定 1」）。
        # num_workers 只影响预取深度（每 worker 预取 2 个 batch，每 batch 4 个起报
        # x 645 MB ≈ 2.6 GB），不是主机内存的大头，调小它是为了别把内存往上堆。
        # ⚠ 主机内存的大头是 Rollout，但**不是**「每进程挂 51 条轨迹」——members=51
        # 且 51 >= 卡数，cli.py:941 走的是**按成员切**，每个 rank 只领
        # range(rank, 51, 卡数) 那几条，所以每进程的 state 常驻是
        #     ceil(51/卡数) x 645 MB：4 卡 -> 13 条 -> 约 8 GB；8 卡 -> 7 条 -> 约 4.5 GB。
        # （按起报切才会每进程挂满 51 条 ≈ 33 GB —— 那正是 cli.py:932 记着「把 pod 的
        # cgroup 限额顶穿、两个 rank 被 OOM kill」的那条路，本配方不会走。）
        "batch_size": 4,
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
    # 起报范围（语法见 base.parse_times；逗号分隔的多段区间取并集、自动去重升序）。
    # 本轮：补 2025-07-02 **之后**的空缺 —— 上一批补的是 07-02 之前的 144 个，已单独在跑，
    # 不要把那个窗口再写回来。
    # 盘上 07-02 之后只有两块完整（20250702..20250720、20251104..20251119），
    # 所以只跑夹着它们的这两段：
    #   20250721 .. 20251103    106 个起报
    #   20251120 .. 20251215     26 个起报        合计 132
    # 第一段含残缺的 20250721..0727、第二段含残缺的 20251120..1126，重跑即覆盖；
    # 已完整的 20251104..20251119 跳过不碰。
    # ⚠ 0728..1103 和 1127..1215 是**完全没跑过**的（盘上连目录都没有），不是残缺 ——
    # 这两段才是本次起报量的主体。
    # 全年正式跑是 "2025010200..2025121500:24"（348 个起报）。
    times="2025072100..2025110300:24,2025112000..2025121500:24",
    steps=60,
    # 见文件头 ⚠：本模型的成员是图内随机算子给的，而 cli.py 又把成员切到不同进程。
    # 两件事叠加后，一个起报的 51 个 member_<NNN> 其实是 8 份独立的随机序列、
    # 每份 7 条，**没有跨进程的对齐保证**。待验：挑一个已完整的起报日，比对
    # member_000/001/002/003 的 060.nc 是否逐字节相同 —— 相同就说明这 51 个成员
    # 实际只有 8 个自由度，那 FGVP 集合 spread 只有 AIFS 的 1/4 就有解释了。
    members=MEMBERS,
    vars="z500,u200,v200,msl,tp",
    # 8 x H100（跑之前 `nvidia-smi -L` 确认卡号；cuda_devices 的元素个数必须 >= gpus，
    # 写错会在 CUDA 初始化时报 failure 100）。一卡一进程，8 卡 = 8 个 worker。
    gpus=8,
    cuda_devices="0,1,2,3,4,5,6,7",
    # —— 单进程内部的线程预算 ——
    # ONNX Runtime 的**算子内并行线程**（intra_op）。0 = 自动 = 物理核 / 进程数，
    # 进程数 = gpus（一卡一进程，本配方 8）。auto 就是「8 个 worker 平分物理核」，
    # 24 核的机器每个 worker 拿 3 条线程；不设则每个 session 各自吃满物理核 ->
    # 8 倍超订，zarr 解码 / CPU 算子 / 拷贝互相抢核。
    # 要显式封顶就写死一个数；CPU 侧前处理重时说明瓶颈在主机侧，可以再往上调。
    # inter_op 恒为 1（ORT_SEQUENTIAL 下它本来就不起作用），不需要配。
    intra_op_threads=0,
    # 卷积算法搜索：空 = ORT 默认 EXHAUSTIVE（首次推理慢、之后走缓存，零精度损失）。
    # 只有「一张卡跑多个进程」时每个 session 都要各搜一遍、启动预热翻倍，那时改成
    # "HEURISTIC" 换启动速度（代价是每层未必拿到最优 kernel）。
    cudnn_algo_search="",
    # 常驻 session 池（resident_members）对本配方**无意义**，所以不写：本模型的成员是
    # 图内随机算子给的，一个任务 = 一个起报的全部 51 个成员、全程同一个 session，没有
    # 「换成员 = 换 session」这回事。那个旋钮只服务 m1a（见 configs/fgvp_spw_ab.py）。
    # 本轮（07-02 之后 132 个起报）另起一份，不写回 _2：_2 里已有的 0702..0727 是**旧
    # .so** 出的，这批是新 .so，混在一份里事后按目录分不开。独立成 _5 也好单独验证。
    # 跑完再跟已有的完整日期一起聚合（check_merge_ens.py 把 _5 加进 SRC_DIRS）。
    # ⚠ resume=outputs 会跳过盘上已齐全的任务，所以换目录 = 强制重算，这正是这里要的。
    output_dir="/workspace/data/shenzw/fgvp_ens_output_5",
)
