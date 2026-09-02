# -*- coding: utf-8 -*-
"""AIFS 1.1（anemoi GNN，.ckpt）模型与输入契约。

模型类只声明固定字段和输入表示；静态场、字段映射、单位检查与 N320 插值流程由
configs/aifs11.py 声明。
"""
from xmetai.backends.ckpt import CkptInferModel


LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
PL_VARS = ("z", "t", "u", "v", "w", "q")
SURFACE_MAPPING = {
    "msl": "msl",
    "sp": "sp",
    "skt": "skt",
    "t2m": "2t",
    "d2m": "2d",
    "u10m": "10u",
    "v10m": "10v",
    "tcw": "tcw",
}
SOIL_MAPPING = {
    "sot1": "stl1",
    "sot2": "stl2",
    "vsw1": "swvl1",
    "vsw2": "swvl2",
}
STATIC_MAPPING = {
    "lsm": "lsm",
    "z_sfc": "z",
    "slor": "slor",
    "sdor": "sdor",
}
FIELD_MAPPING = {
    **{f"{var}{level}": f"{var}_{level}" for var in PL_VARS for level in LEVELS},
    **SURFACE_MAPPING,
    **SOIL_MAPPING,
    **STATIC_MAPPING,
}
INPUT_FIELDS = tuple(FIELD_MAPPING.values())


class Aifs11CkptModel(CkptInferModel):
    """AIFS 1.1 单模型（deterministic，1 member）。"""
    model_name = "aifs11_ckpt"
    history_steps = 2
    hour_interval = 6
    forecast_type = "deterministic"
    members = 1
    state_representation = "field_dict"
    input_assembler = "field_dict"
    input_fields = INPUT_FIELDS
    loader_groups = ("pl", "sfc", "soil")


assert len(INPUT_FIELDS) == 94
assert len(set(INPUT_FIELDS)) == len(INPUT_FIELDS)
