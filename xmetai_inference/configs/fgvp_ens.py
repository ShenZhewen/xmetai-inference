# -*- coding: utf-8 -*-
"""fgvp_ens：FGVP 的 Mamba 骨干集合预报版本（51 成员，FP32 导出）。

骨架抄 fuxi_ens（集合路径），dataset processors 抄 fgvp single（同为 IWC 族、
同一套 era5_foundation_store2）。与 fgvp single 的差别只有三处：
  * model_path 指向集合版的 onnx，model_class 换成集合模型类；
  * members=51（fgvp single 是 1），输出落 member_<NNN>/ 子目录；
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
    times="2025010200..2025010300:24",
    steps=60,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="/workspace/data/shenzw/fgvp_ens_output",
)
