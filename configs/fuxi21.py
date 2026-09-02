# -*- coding: utf-8 -*-
"""FuXi-2.1 确定性推理（对应旧 scripts/run_fuxi_pt2.sh）。

PT2 模型，mean.nc/std.nc 须与 fuxi-2.1.pt2 同目录；z-score 空间，pre_process /
post_process 做归一化/反归一化，zero_recurrent 回填诊断通道。
"""
from configs.base import InferConfig, ROOT

cfg = InferConfig(
    name="fuxi21",
    model_path=f"{ROOT}/weights/fuxi2.1/fuxi-2.1.pt2",
    spec="specs/fuxi21.json",
    loader="era5_store",
    times="2025010300..2025022600:24",
    steps=60,
    members=1,
    vars="z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,d2m,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    output_dir="/workspace/data/shenzw/fuxi_single_output",
)
