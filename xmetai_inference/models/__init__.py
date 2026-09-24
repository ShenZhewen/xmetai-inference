# -*- coding: utf-8 -*-
"""模型包：具体模型的固定运行契约。

引擎在 xmetai.backends，模型在 xmetai.models，语义不混：
  * backends 只提供共享运行契约以及 onnx/pt2 两种执行机制；
  * MODEL_REGISTRY 是唯一注册表：模型名 → 具体模型类。模型继承对应引擎，只声明
    通道、网格、时间窗口和输入表示；完整 Processor 流程由 configs 声明。

加新模型三步：
  1) 新建 models/<name>.py，继承对应引擎（backends.onnx.OnnxInferModel /
     backends.pt2.Pt2InferModel / …）。
  2) 在模型类声明 input_channels。
  3) 在 MODEL_REGISTRY 注册，并新增一份包含完整 Processor 流程的 config。

模型选择主路径：config/CLI 的 model_class → create_model()。
"""
from importlib import import_module

import numpy as np


# FuXi-Ens、FuXi-2.1 和 FGVP 共用的 0.25° 全球规则网格。
GRID_025 = {
    "lat": {"start": 90.0, "step": -0.25, "size": 721},
    "lon": {"start": 0.0, "step": 0.25, "size": 1440},
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

    ops_library / gpu_state / weight_perturb / latent_forcing / member_index 属于运行配置，
    由 config 提供；None 时使用模型类默认值（weight_perturb=None / latent_forcing=None
    即不加扰动）。

    weight_perturb 与 latent_forcing 是两条正交的「确定性 -> 集合」路径：前者按
    `W~ = W (x) (1 + sigma*xi)` 改**权重**，后者按 `z' = z (x) (1 + sigma*U diag(xi_t) V^T)`
    给图加一个 `spw/xi` 输入、改**图**（xi_t 是驱动侧的 AR(1) 过程）。两者可以同时开，
    config 里同时写就是「冻结的扰动权重 + 时间相关的潜空间强迫」。

    member_index 只在配了 weight_perturb 时有意义：它决定 load() 加载
    哪一份成员模型。跑集合时同一个实例会被反复 load（每次换一个成员号）；resident_members
    决定这些 session 在显存里驻留几份——1（缺省）是一换一放，51 个成员 x 3 GB 全挂上去
    会把显存吃干净；显存够大就调大它（-1 = 全常驻），换回来时不重建 session、不重新预热
    cuDNN。见 backends/session_pool.py 与 cli 的 _run_member_block。
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
        from xmetai_inference.perturbation import WeightPerturbSpec

        spec = WeightPerturbSpec.from_config(weight_perturb)
        if spec is not None:
            if not getattr(model, "supports_weight_perturb", False):
                raise ValueError(
                    f"{cls.__name__}（backend={model.backend}）不支持权重扰动："
                    "当前只实现了 ONNX 后端的注入（m3 图内随机算子 / m1a 逐成员冻结"
                    "权重），见 xmetai_inference/perturbation/README.md")
            model.weight_perturb = spec
    if latent_forcing is not None:
        from xmetai_inference.perturbation import LatentForcingSpec

        forcing = LatentForcingSpec.from_config(latent_forcing)
        if forcing is not None:
            if not getattr(model, "supports_weight_perturb", False):
                raise ValueError(
                    f"{cls.__name__}（backend={model.backend}）不支持潜空间 AR(1) 强迫："
                    "当前只实现了 ONNX 后端的图改写注入（latent_forcing），"
                    "见 xmetai_inference/perturbation/README.md")
            model.latent_forcing = forcing
    return model


__all__ = [
    "FENGQING_CHANNELS",
    "FUXI_ENS_CHANNELS",
    "FUXI21_CHANNELS",
    "GRID_025",
    "PANGU_CHANNELS",
    "MODEL_REGISTRY",
    "create_model",
    "get_model_class",
    "grid_coords",
]
