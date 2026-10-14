# -*- coding: utf-8 -*-
"""fgvp_ens：FGVP 的集合预报版本（51 成员，FP32 导出，骨干自带随机轴出成员）。

权重是 20261008 导出的 iwc_gssp_im（模型类 iwc_fgvp_mamba_ens_onnx 沿用旧名，
输入契约不变）。⚠ 该 onnx 带 .data 外部权重，两个文件必须在同一目录，挪权重要
一起挪。

骨架抄 fuxi_ens（集合路径），dataset processors 抄 fgvp single（同为 IWC 族、
同一套 era5_foundation_store2）。与 fgvp single 的差别：
  * model_path 指向集合版的 onnx，model_class 换成集合模型类；
  * members=51（fgvp single 是 1），输出落 member_<NNN>/ 子目录；
  * 集合图已重导为动态 batch 轴，故可设 model_batch=6 做前向批（见下）；
  * 模型路径与算子库用字面量而非 os.environ.get —— 项目约定：新配置不引入环境
    变量，改路径直接改这个文件（可 diff、可留痕）。其余 config 里的
    os.environ.get 是历史写法，不动。
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

FGVP_ENS_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

cfg = InferConfig(
    name="fgvp_ens",
    model_path="/workspace/tmp/douzsh/models/20261008/iwc_gssp_im.onnx",
    # 2026-10-09 换成 GPT 优化过的算子实现（对方称 4x 提速）。旧的
    # xmetai_onnx_plugins.so 留着做回滚与数值对照，别删。
    # ⚠ 文件名里的 ort1_24 = 按 ONNX Runtime 1.24 构建，跨环境先对版本：
    #   python -c "import onnxruntime; print(onnxruntime.__version__)"
    #   对不上就别用，加载或运行期会崩（ABI 绑定）。
    ops_library="/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so",
    # 必须 False：图只出预报那一帧（单步型），GPU 常驻分支直接 state=result 回填、
    # 不经过 _next_state 的滑窗，且 _spare buffer 复用假设输入输出同形。见模型类
    # docstring。OnnxInferModel.load 的启动期断言会拦这个组合。
    gpu_state=False,
    # 前向批：一次 forward 堆几条成员轨迹（1 = 逐条，与历史行为一致）。
    # ⚠ 一个进程并不拿满 51 个成员 —— cli.py 按成员切，每个 rank 只领
    #   range(rank, 51, 卡数) 那 ceil(51/卡数) 条（4 卡 = 13 条），k 的有效上限
    #   是这 13 条，不是 51。
    # 显存按 k 倍涨：单次前向的输入拷贝 k x 648 MB 加中间激活（OOM 时在
    #   node_Add_* 上炸的就是这部分）。轨迹状态常驻是另一笔、与 k 无关。
    # 图已重导为动态 batch 轴（输入/输出第一维都是符号维）；万一图不是，runner
    #   会打 WARNING 退回 1，不会静默算错。与 gpu_state=True 互斥（本配置 False）。
    # 跑起来到 rank 日志里搜「前向批生效」：有这行才算真批上（每个进程只打一行）；
    #   「模型未声明 batch_axis_dynamic」则是退回了 1，这一趟照跑完但与批量化无关。
    # 批内噪声是否仍逐成员独立（随机算子若按 (1,...) 出噪声再 broadcast，k 行会拿到
    #   同一份扰动、集合塌缩且不报错）由集成档盯：tests/integration/
    #   test_fgvp_ens_realdata.py 的成员互异 + spread>0 两条用例。
    model_batch=6,
    # ORT 的显存天花板，占**卡总显存**的比例（onnx 后端的 cuda_opts
    # gpu_mem_limit = torch 报的 total_memory × 这个比例，见 backends/base.py）。
    # 缺省 0.7 在 80 GiB 卡上只放出 56.0 GiB，k 往上调时报的
    #   bfc_arena ... Available memory of X is smaller than requested bytes of Y
    # 是撞**这个顶**，不是撞卡。0.9 -> 72.0 GiB，留 8 GiB 给 driver /
    # CUDA context / 碎片。一卡一进程（cli.py 按 rank 切），每个进程按字面值
    # 吃满自己的卡，不存在「两个进程挤一张卡、比例要减半」的情形。
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
        # batch_size 只影响数据装载分组、不碰 GPU，不影响速度；前向批是上面的
        # model_batch。num_workers 只影响预取深度（每 worker 预取 2 个 batch），
        # 不是主机内存的大头。真正的大头是 Rollout 的轨迹状态常驻，但按成员切后
        # 每 rank 只挂 ceil(51/卡数) 条：4 卡 = 13 条 × 645MB ≈ 8 GB。
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
    # 起报范围：先按 4 个起报跑通链路，改这里调范围（语法见 base.parse_times）。
    times="2025010200..2025121500:24",
    steps=60,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fgvp_ens_output",
)
