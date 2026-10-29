# -*- coding: utf-8 -*-
"""模型包：具体模型的固定运行契约（通道 / 网格 / 时间窗口 / 输入表示）。

引擎在 backends，MODEL_REGISTRY 是唯一注册表（模型名 → 模型类），完整 Processor
流程由 configs 声明。新模型 = 新建 models/<name>.py、继承对应引擎、声明
input_channels、在 REGISTRY 注册，并加一份 config。
"""
from importlib import import_module

import numpy as np


# FuXi-Ens、FuXi-2.1 和 FGVP 共用的 0.25° 全球规则网格。
GRID_025 = {
    "lat": {"start": 90.0, "step": -0.25, "size": 721},
    "lon": {"start": 0.0, "step": 0.25, "size": 1440},
}

# 同一个 0.25° 网格的南→北版本：直接照 CRA store 的原生朝向建模的模型用
# （Tongyun-CRA 的图内 mean/std 就是从该 store 烘焙的）。用它就不配 geometry
# 处理器——那个处理器看到 lat 在南→北就无条件翻转，会把这种模型喂反。
# 输出文件的 lat 坐标取自模型的 grid，必须跟着一起切，否则数据与坐标错配。
GRID_025_S = {
    "lat": {"start": -90.0, "step": 0.25, "size": 721},
    "lon": {"start": 0.0, "step": 0.25, "size": 1440},
}

# s2s 系列的 1.5° 全球网格：lat 90→−90（121）、lon 0→358.5（240），
# 与盘 s2s.19500101_20260417.c76.1p5.h24.v1 / cra.1979_2025.c87.1p5.h24.v1 原生朝向一致。
GRID_1P5 = {
    "lat": {"start": 90.0, "step": -1.5, "size": 121},
    "lon": {"start": 0.0, "step": 1.5, "size": 240},
}

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PL_VARS = ["z", "t", "u", "v", "q"]
FUXI_ENS_SFC_VARS = [
    "t2m", "d2m", "sst", "u10m", "v10m", "u100m", "v100m",
    "msl", "ssr", "ssrd", "fdir", "ttr", "tp",
]
FUXI21_SFC_VARS = [
    "msl", "t2m", "d2m", "sst", "ws10m", "ws100m", "u10m", "v10m",
    "u100m", "v100m", "lcc", "mcc", "hcc", "tcc",
    "ssr", "ssrd", "fdir", "ttr", "tcw", "tp",
]


def _expand_channels(pl_vars, levels, surface_vars):
    return [
        f"{variable}{level}"
        for variable in pl_vars
        for level in levels
    ] + list(surface_vars)


FUXI_ENS_CHANNELS = _expand_channels(
    PL_VARS, LEVELS, FUXI_ENS_SFC_VARS)
FUXI21_CHANNELS = _expand_channels(
    PL_VARS, LEVELS, FUXI21_SFC_VARS)

# fengshun2.0：87 通道 = z/t/u/v/q × 13 层（50→1000 升序）+ 22 个地面/诊断通道
# （比 FuXi-2.1 多 t2m_min/t2m_max，共 87）。顺序与输入盘
# cra.1979_2024.c87.p25.h24.v1 的 channel 数组逐位一致（era5.1959_2025.c87 同序
# 同统计量）；图内的 mean/std 也是按这个顺序烘焙的，改顺序不会报错、只会整场错位。
FENGSHUN_SFC_VARS = [
    "msl", "t2m", "t2m_min", "t2m_max", "d2m", "sst", "ws10m", "ws100m",
    "u10m", "v10m", "u100m", "v100m", "lcc", "mcc", "hcc", "tcc",
    "ssr", "ssrd", "fdir", "ttr", "tcw", "tp",
]
FENGSHUN_CHANNELS = _expand_channels(PL_VARS, LEVELS, FENGSHUN_SFC_VARS)


# 风顺 ERA 1.5°（源图 s2s_gssp_gdn2.onnx）：76 通道、1.5°，顺序与训练盘
# s2s.19500101_20260417.c76.1p5.h24.v1 的 channel 数组逐位一致 —— 上层 [z,t,u,v,q]
# × 13 层（**1000→50 降序**，与 fengshun2.0 的 50→1000 升序相反），地面
# [t2m,d2m,sst,ttr,u10m,v10m,u100m,v100m,msl,tcwv,tp]。
# 图内 mean/std 也是按这个顺序烘焙的，改顺序不会报错、只会整场错位。
S2S_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
S2S_PL_VARS = ["z", "t", "u", "v", "q"]
S2S_SFC_VARS = [
    "t2m", "d2m", "sst", "ttr", "u10m", "v10m", "u100m", "v100m",
    "msl", "tcwv", "tp",
]
S2S_CHANNELS = _expand_channels(S2S_PL_VARS, S2S_LEVELS, S2S_SFC_VARS)


FENGQING_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
FENGQING_PL_VARS = ["z", "q", "t", "u", "v"]
FENGQING_SFC_VARS = ["tp", "msl", "u10m", "v10m", "t2m"]
FENGQING_CHANNELS = _expand_channels(
    FENGQING_PL_VARS, FENGQING_LEVELS, FENGQING_SFC_VARS)

# Pangu-Weather：与 FengQing 同为 0.25° 规则网格、同 13 层（1000→50 降序）、
# 同上层变量序 [z,q,t,u,v]，差别只在地面通道：Pangu 无 tp，= [msl,u10m,v10m,t2m]。
PANGU_LEVELS = FENGQING_LEVELS
PANGU_PL_VARS = FENGQING_PL_VARS
PANGU_SFC_VARS = ["msl", "u10m", "v10m", "t2m"]
PANGU_CHANNELS = _expand_channels(
    PANGU_PL_VARS, PANGU_LEVELS, PANGU_SFC_VARS)

# CRA（CRA-40 再分析 c71 store）：上层 z/t/u/v/q × 13 层（50→1000 升序），地面
# [t2m,q2m,u10m,v10m,msl,tp]，共 71 通道。与 FengQing 的层级顺序与上层变量序都不同。
CRA_SFC_VARS = ["t2m", "q2m", "u10m", "v10m", "msl", "tp"]
CRA_CHANNELS = _expand_channels(PL_VARS, LEVELS, CRA_SFC_VARS)


def grid_coords(grid):
    """模型网格契约 → 纬度、经度坐标数组。"""
    latitude = grid["lat"]
    longitude = grid["lon"]
    lat = (
        np.arange(latitude["size"], dtype=np.float64) * latitude["step"]
        + latitude["start"]
    )
    lon = (
        np.arange(longitude["size"], dtype=np.float64) * longitude["step"]
        + longitude["start"]
    )
    return lat, lon


# 模型层：名字 → (模块, 类)。按需导入，避免使用 PT2 时强制安装 ONNX Runtime。
MODEL_REGISTRY = {
    "fuxi_ens_onnx": (
        "xmetai_inference.models.fuxi_ens_onnx",
        "FuxiEnsOnnxModel",
    ),
    "fuxi21_pt2": (
        "xmetai_inference.models.fuxi21_pt2",
        "Fuxi21Pt2Model",
    ),
    "iwc_fgvp_gdn2_onnx": (
        "xmetai_inference.models.iwc_fgvp_gdn2",
        "IwcFgvpGdn2Model",
    ),
    "iwc_fgvp_mamba_ens_onnx": (
        "xmetai_inference.models.iwc_fgvp_mamba_ens",
        "IwcFgvpMambaEnsModel",
    ),
    "fengqing_pre_onnx": (
        "xmetai_inference.models.fengqing_pre_onnx",
        "FengqingPreOnnxModel",
    ),
    "pangu_onnx": (
        "xmetai_inference.models.pangu_onnx",
        "PanguOnnxModel",
    ),
    "tongyun_cra_ens_onnx": (
        "xmetai_inference.models.tongyun_cra_ens",
        "TongyunCraEnsModel",
    ),
    "fengshun20_onnx": (
        "xmetai_inference.models.fengshun20_onnx",
        "Fengshun20OnnxModel",
    ),
    "fengshun_era_1p5_onnx": (
        "xmetai_inference.models.fengshun_era_1p5",
        "FengshunEra1p5Model",
    ),
}


def get_model_class(model_name):
    """按注册名解析内置模型，或验证外部 config 直接提供的模型类。"""
    if not isinstance(model_name, str):
        from xmetai_inference.backends import BaseInferModel

        if not isinstance(model_name, type) or not issubclass(model_name, BaseInferModel):
            raise TypeError(
                "model_class 必须是内置模型注册名或 BaseInferModel 子类")
        return model_name
    target = MODEL_REGISTRY.get(model_name)
    if target is None:
        raise ValueError(f"未知模型 {model_name!r}（可选 {', '.join(MODEL_REGISTRY)}）")
    module_name, class_name = target
    return getattr(import_module(module_name), class_name)


def create_model(model_name, device_id=0, gpu_mem_fraction=0.7,
                 ops_library=None, gpu_state=None, weight_perturb=None,
                 member_index=0, latent_forcing=None, intra_op_threads=0,
                 cudnn_algo_search="", resident_members=1):
    """按模型名构造推理模型。

    weight_perturb（改权重）与 latent_forcing（改图）是两条正交路径，仅 ONNX 可同时开。
    member_index / resident_members 只在配了 weight_perturb 时有意义：决定 load() 加载
    哪份成员、session 在显存里驻留几份（-1 = 全常驻）。
    """
    cls = get_model_class(model_name)
    model = cls(device_id=device_id, gpu_mem_fraction=gpu_mem_fraction)
    if ops_library and hasattr(model, "ops_library"):
        model.ops_library = ops_library
    if gpu_state is not None and hasattr(model, "gpu_state"):
        model.gpu_state = gpu_state
    if intra_op_threads and hasattr(model, "intra_op_threads"):
        model.intra_op_threads = int(intra_op_threads)
    if cudnn_algo_search and hasattr(model, "cudnn_algo_search"):
        model.cudnn_algo_search = str(cudnn_algo_search)
    if resident_members is not None and hasattr(model, "resident_members"):
        model.resident_members = int(resident_members)
    model.member_index = int(member_index or 0)
    if weight_perturb is not None:
        from xmetai_inference.weights.spw import WeightPerturbSpec

        spec = WeightPerturbSpec.from_config(weight_perturb)
        if spec is not None:
            if not getattr(model, "supports_weight_perturb", False):
                raise ValueError(
                    f"{cls.__name__}（backend={model.backend}）不支持权重扰动："
                    "当前只实现了 ONNX 后端的注入（m3 图内随机算子 / m1a 逐成员冻结"
                    "权重），见 xmetai_inference/perturbation/README.md")
            model.weight_perturb = spec
    if latent_forcing is not None:
        from xmetai_inference.weights.spa import LatentForcingSpec

        forcing = LatentForcingSpec.from_config(latent_forcing)
        if forcing is not None:
            # 能力位与 weight_perturb 分开：onnx（weights/spa.py 图改写）与 pt2
            # （weights/spa_pt2.py 的 fx 图改写）各自声明 supports_latent_forcing。
            if not getattr(model, "supports_latent_forcing", False):
                raise ValueError(
                    f"{cls.__name__}（backend={model.backend}）不支持潜空间 AR(1) 强迫："
                    "已实现的是 ONNX 后端（weights/spa.py）与 pt2 后端（weights/spa_pt2.py）的图改写"
                    "注入，见 xmetai_inference/perturbation/README.md 2N 节")
            model.latent_forcing = forcing
    return model


__all__ = [
    "CRA_CHANNELS",
    "CRA_SFC_VARS",
    "FENGSHUN_CHANNELS",
    "FENGSHUN_SFC_VARS",
    "FENGQING_CHANNELS",
    "FUXI_ENS_CHANNELS",
    "FUXI21_CHANNELS",
    "GRID_025",
    "GRID_025_S",
    "GRID_1P5",
    "PANGU_CHANNELS",
    "S2S_CHANNELS",
    "S2S_SFC_VARS",
    "MODEL_REGISTRY",
    "create_model",
    "get_model_class",
    "grid_coords",
]
