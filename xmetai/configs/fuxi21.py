# -*- coding: utf-8 -*-
"""FuXi-2.1 确定性推理配置。

PT2 模型，mean.nc/std.nc 须与 fuxi-2.1.pt2 同目录；统一 Processor 管线负责
归一化、反归一化和诊断通道回填清零。输入 NaN 保留，由模型内部处理。
"""
from xmetai.configs.base import InferConfig, ROOT

cfg = InferConfig(
    name="fuxi21",
    model_path=f"{ROOT}/model_artifacts/fuxi2.1/fuxi-2.1.pt2",
    model_class="fuxi21_pt2",
    loader="era5_store",
    pre_processors=[
        {"name": "geometry"},
        {"name": "channel_order"},
        {"name": "unit_convert"},
        {
            "name": "normalize",
            "mean_file": "mean.nc",
            "std_file": "std.nc",
            "log1p_channels": ["tp"],
            "allow_nonfinite": True,
        },
    ],
    recurrent_processors=[
        {
            "name": "zero_channels",
            "channels": ["ssr", "ssrd", "fdir", "ttr", "tp"],
        },
    ],
    output_processors=[
        {
            "name": "denormalize",
            "mean_file": "mean.nc",
            "std_file": "std.nc",
            "expm1_channels": ["tp"],
            "nonnegative": ["tp"],
        },
    ],
    times="2025010200..2025123100:24",
    steps=60,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,d2m,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fuxi_single_output_new",
)
