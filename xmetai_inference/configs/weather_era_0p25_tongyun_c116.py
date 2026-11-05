# -*- coding: utf-8 -*-
"""WeatherBench2 ERA5 0.25° Tongyun（确定性）—— 2025–2026 c116 v3 归一化盘。

Tongyun == FGVP（iwc_fgvp_gdn2 改名）：同一张 tongyun_det.onnx、同一套 78 通道
（FUXI_ENS_CHANNELS）北→南 GRID_025、图内烘焙归一化、物理进/物理出。本配方只换
输入盘（c78 → c116 v3 单盘，按名取模型要的 78 个通道）与起报范围，模型契约不变。

喂图前的换算按图内 mean/std 与盘内统计量、以及图里 tp 的专门子链对过：
  - q 盘里已是 g/kg，图内统计量 1:1 → 原样喂。
  - tp 盘里是 log1p(mm)（clip 后 log1p），图内入/出口各带一条 log1p / expm1
    （clip 0–7，见 GatherND 取 77 号通道）→ 图的接口是 mm，喂前先 expm1。
  - ssr/ssrd/fdir/ttr 图内统计量是盘内 W/m² 的 6 倍（图内无 ×6 子链）→ 喂前 ×6
    （6h-Wh/m²）。
  - sst 盘里陆地为 NaN（c78 盘是用全球均值填过的；图里会把 NaN 置 0）→
    fill_missing 287.066 K。
"""
from xmetai_inference.backends.onnx import OnnxInferModel
from xmetai_inference.configs.base import (
    FUXI_ENS_CHANNELS, GRID_025, InferConfig, ModelProcessingConfig)


# ---- 真实部署地址（Tongyun = FGVP 改名） --------------------------------------
SRC_GRAPH = "/workspace/tmp/douzsh/models/20261013/tongyun_det.onnx"
OPS_LIBRARY = "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"
# 2025-01–2026-07 c116 v3 归一化盘：channel / data / mean / std，6h，北→南。
ERA_STORE = "/workspace/data/zarr_refresh_v3/era5.202501_202607.c116.p25.h6.v3"
# 推理原始产物根：{init}/member_XXX/{step:03d}.nc（后续 wb2_export.py 转 Zarr）
OUTPUT_DIR = "/workspace/data/shenzw/wb2/inference_raw_c116"


# 确定性单成员；图契约同 fgvp（不改网格/通道/窗口化）。这里直接声明类对象，
# 不依赖 model_factory 的注册名（各部署树的注册名不一定一致）。
TongyunDetModel = OnnxInferModel.declare(
    "TongyunDetModel",
    model_name="tongyun_det_onnx",
    channels=FUXI_ENS_CHANNELS,
    grid=GRID_025,
    windowed_output=False,
)


cfg = InferConfig(
    name="weather_era_0p25_tongyun_c116",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,
    model_class=TongyunDetModel,
    model_batch=1,
    gpu_mem=0.9,

    dataset={
        "type": "processed_multi_zarr",
        "paths": [ERA_STORE],
        "processors": [
            # 归一化 v3 单盘 → 图内输入空间；tp 由 log1p(mm) 还原成 mm。
            {"name": "denormalize", "stats_path": ERA_STORE,
             "expm1_channels": ["tp"]},
            # 辐射：盘内 W/m² → 图内 6h-Wh/m²。
            {"name": "unit_convert",
             "scales": {"ssr": 6.0, "ssrd": 6.0, "fdir": 6.0, "ttr": 6.0}},
            # sst 陆地 NaN → c78 盘同款全球均值；其它通道若出现 NaN 直接报错。
            {"name": "fill_missing",
             "rules": {"sst": {"method": "constant", "value": 287.066}},
             "unconfigured": "error"},
        ],
    },
    dataloader={
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    # 盘覆盖 2025010100–2026073118；history=2（init 及前一帧）→ 可用起报
    # 2025010106–2026073112。默认 00Z+12Z 全量 ≈1153 起报 × 60 时效；
    # 想先冒烟把 times 改成 "2025010112..2025010512:12" 即可。
    times="2025010112..2026073112:12",
    steps=60,
    hour_interval=6,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    intra_op_threads=0,
    cudnn_algo_search="EXHAUSTIVE",
    output_dir=OUTPUT_DIR,
)
