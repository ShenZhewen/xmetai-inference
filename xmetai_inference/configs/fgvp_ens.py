# -*- coding: utf-8 -*-
"""fgvp_ens：FGVP 的集合预报版本（51 成员，FP32 导出，骨干自带随机轴出成员）。

权重是 /workspace/tmp/douzsh/models/20261008 导出的 iwc_gssp_im（模型类沿用旧名
iwc_fgvp_mamba_ens_onnx，输入契约不变）。⚠ 该 onnx 带 .data 外部权重，两个文件必须
在同一目录，挪权重要一起挪。

与 fgvp single 的差别：members=51（输出落 member_<NNN>/ 子目录）、集合图已重导为动态
batch 轴（故可设 model_batch>1）。模型路径与算子库用字面量而非 os.environ.get。
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
    # 文件名里 ort1_24 = 按 ONNX Runtime 1.24 构建，跨环境先对版本，对不上加载/
    # 运行期会崩（ABI 绑定）。旧的 xmetai_onnx_plugins.so 留着做回滚，别删。
    ops_library="/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so",
    # 必须 False：图只出预报那一帧（单步型），GPU 常驻分支会跳过 _next_state 的
    # 滑窗。OnnxInferModel.load 的启动期断言会拦这个组合。
    gpu_state=False,
    # 前向批：一次 forward 堆几条成员轨迹（1 = 逐条，与历史行为一致）。cli 按成员
    # 切，每 rank 只领 ceil(51/卡数) 条（4 卡 = 13），k 的有效上限是 13，不是 51。
    # 图输入/输出 batch 轴都需为符号维，否则 runner 打 WARNING 退回 1（不会静默算错）；
    # 与 gpu_state=True 互斥。rank 日志里搜「前向批生效」确认真批上，
    # 「模型未声明 batch_axis_dynamic」则是退回了 1。
    model_batch=6,
    # 占卡总显存的比例（ORT gpu_mem_limit = total_memory × 此值；0.9 在 80 GiB 卡上
    # 放出 72 GiB）。调大 model_batch 时若报 bfc_arena 就是撞这个顶。
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
        # batch_size 只影响数据装载分组、不碰 GPU；前向批见上面的 model_batch。
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
    # 起报范围，改这里调范围（语法见 base.parse_times）。
    times="2025010200..2025121500:24",
    steps=60,
    members=51,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fgvp_ens_output",
)
