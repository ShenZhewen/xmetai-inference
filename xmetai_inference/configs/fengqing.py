# -*- coding: utf-8 -*-
"""FengQing V1.5Beta (pre-normalized ONNX) deterministic inference config.

Input data is era5_foundation_store2 6-hourly Zarr:
  * tp in this store is 1-hour accumulation in m; the model expects 6-hour
    accumulation in mm, so unit_convert multiplies tp by 6000.
  * Input normalization is done in the data layer (normalize_fengqing). The
    ONNX graph emits a normalized *residual*; reconstructing the physical field
    and re-normalizing for the next step stays in the model class (it needs the
    previous frame + residual std), so model_processing remains empty.
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

# 服务器路径写死：直接 `python cli.py --config fengqing` 启动。
FENGQING_ROOT = "/workspace/szwCode/xmetai-inference2/model_artifacts/FengQing"
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

FENGQING_STORES = [
    ERA5_ROOT + "/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    ERA5_ROOT + "/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]

cfg = InferConfig(
    name="fengqing",
    model_path=FENGQING_ROOT + "/onnx/fengqing_pre.onnx",
    model_class="fengqing_pre_onnx",

    dataset={
        "type": "processed_multi_zarr",
        "paths": FENGQING_STORES,
        "processors": [
            {
                "name": "unit_convert",
                "scales": {
                    "tp": 6000.0,
                },
            },
            {"name": "geometry"},
            {"name": "normalize_fengqing"},
        ],
    },

    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    # 分层：输入归一化在 dataset.processors（normalize_fengqing），反归一化在这里。
    # 模型层只做「一步前向 + 残差反演」，全程留在归一化空间，不碰 mean。
    model_processing=ModelProcessingConfig(
        input=[],
        recurrent=[],
        output=[{"name": "denormalize_fengqing"}],
    ),

    # 框架整体连通性测试：2 卡 × batch 2 × 预报 15 天，起报按天（每天 00Z）。
    #
    # 起报数必须 = gpus × batch_size = 4，理由是 cli.py 的多卡切分逻辑：
    # members=1 时 cost_by_init 恒 ≤ cost_by_member，所以永远「按起报时间切」。
    # 起报数 < gpus 会让部分 rank 分不到任务直接退出（2 卡只有 1 卡干活）；
    # 起报数 = 4 时每卡拿 2 个，恰好凑成 1 个 batch，batch 维才真的被走到。
    #
    # `:24` 是起报**间隔**（每天一个起报），与 steps/hour_interval 的预报步长
    # 无关。显式写出来：cli.py:207 调 parse_times 时没传 hour_interval，缺省
    # 恰好也是 24，但别依赖这个巧合。
    times="2025010200..2025010500:24",
    # steps 与 hour_interval 是两个独立概念，别和起报间隔混在一起：
    #   hour_interval=6  模型每步推进 6h —— 这是**模型契约**（FengQing 类属性
    #                    也是 6），改成 24 会让数据层按 24h 取历史窗（模型要的是
    #                    T-6h 和 T 两帧，会变成 T-24h 和 T），rollout 的
    #                    valid_time 也会跟着错。不可改。
    #   steps=60         预报 60 步 = 360h = 15 天
    #   times 的 `:24`   起报间隔，每天一个起报日
    steps=60,          # 预报时长 15 天 = 60 × 6h
    hour_interval=6,
    # 确定性模型必须为 1：cli.py 按 members==1 才走 run_batch（残差反演路径）；
    # >1 会落到基类 run() → 继承的 OnnxInferModel.forward()，喂错键名（"input"）。
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    # 独立目录：不复用 fengqing_single_output2（那里是单起报的旧结果）。
    # eval 时必须显式传 --forecast 指向这里，否则会评到默认目录的旧数据。
    output_dir="/workspace/data/shenzw/fengqing_2gpu_4day",
)
