# -*- coding: utf-8 -*-
"""fengshun2.0 集合推理（87 通道、日步长 h24 盘、图内自带归一化）。

输入盘用 CRA 的 87 通道日盘 ``cra.1979_2024.c87.p25.h24.v1``（1979-01-01 …
2024-12-31，每日 00Z）：87 通道的顺序、mean/std 与图内烘焙值逐位一致，抽检全盘
无 NaN，tp 口径整段统一。

与其它 config 的差别：

  * **输入盘是 zscore 归一化态**，而图自己会做 ``Sub mean → Div std``，所以数据侧
    要配 ``denormalize``（stats_path = 同一个盘），把归一化值还原成物理量再喂进去。
    tp 在这个盘里是 log1p(累积 mm)，要 ``expm1_channels=["tp"]`` 还原成物理 mm。
  * **不配 geometry**：盘原生就是 lat 90→−90、lon 0→359.75，与图的网格一致。
  * **不配 normalize / 反归一化输出**：归一化全在图内，物理进 / 物理出。
  * **不配 zero_channels**：tp 输入在模型内部自然置 0（纯输出通道），不用框架补刀。
  * 盘自带 ``mask/const.nc/weight.nc``（陆地/海洋掩膜等），推理不用，忽略。

（备选盘 ``era5.1959_2025.c87.p25.h24.v1`` 通道与统计量相同，但它的 tp 在
2024-01-01 前后换了口径——之前是 1h 量级（全球均值 ~0.12 mm），之后是 24h 总量
（~2.9 mm），线性差约 24×；且 sst 陆地有 33.89% NaN。所以不用它。）
"""
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

FENGSHUN_ROOT = "/workspace/tmp/douzsh/models"
# 输入盘：1979-01-01 … 2024-12-31，每日 00Z（h24），zscore fp16 单 store，无 NaN。
FENGSHUN_STORE = "/workspace/data/zarr_refresh_v3/cra.1979_2024.c87.p25.h24.v1"

cfg = InferConfig(
    name="fengshun20",
    model_path=f"{FENGSHUN_ROOT}/fengshun2.0_260901.onnx",
    model_class="fengshun20_onnx",
    # 图里有 6 个 Mamba3SisoFn 自定义算子，只有 CUDA 实现，且与 ORT 1.24 ABI 绑定。
    ops_library=f"{FENGSHUN_ROOT}/xmetai_onnx_plugins.ort1_24.cuda12.so",

    dataset={
        "type": "processed_multi_zarr",
        "paths": [FENGSHUN_STORE],
        "processors": [
            # 归一化态 → 物理量；tp 由 log1p(mm) 还原成 mm。
            {
                "name": "denormalize",
                "stats_path": FENGSHUN_STORE,
                "expm1_channels": ["tp"],
            },
            # 兜底：CRA 盘抽检无 NaN，这条正常不触发；真出 NaN 就按 sst 气候均值
            # （287.222 K = 归一化 0）填，别的通道出 NaN 直接报错。
            {
                "name": "fill_missing",
                "rules": {"sst": {"method": "constant", "value": 287.222}},
                "unconfigured": "error",
            },
        ],
    },
    dataloader={
        # 一个样本 = 2×87×721×1440 fp32 ≈ 0.7 GB，集合还要按成员各持一份回填状态；
        # batch_size 保持 1，别按起报堆批。
        "batch_size": 1,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 3 个起报日（盘覆盖 1979-01-01 … 2024-12-31，换日子就直接改这一行）。
    # 每个 +60 步都落在盘内（最后一个 20241101 + 60 天 = 2024-12-31，正好是盘尾）
    # ⇒ 60 个时效全在盘内，评测可以直接拿同一个盘当实况、不必换源。
    times="2024090100,2024100100,2024110100",
    steps=60,             # 一天一步 ⇒ 往后 60 天
    hour_interval=24,     # 日步长（模型类的 hour_interval 同 24）
    members=51,           # 图内 RandomNormalLike：一次前向一个成员，51 成员 = 51 条轨迹
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    # 4 卡：members 按 rank 切 range(rank, 51, 4) ⇒ 13/13/13/12，每卡 13 条轨迹驻留。
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fengshun_output",
)
