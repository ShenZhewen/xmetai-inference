# -*- coding: utf-8 -*-
"""模型注册表与构造入口：注册名 → 模型类 → 实例（按需导入）。"""
from importlib import import_module


# 模型层：名字 → (模块, 类)。
MODEL_REGISTRY = {
    "fuxi_ens_onnx": (
        "xmetai_inference.configs.weather_era_0p25_fuxi_ens",
        "FuxiEnsOnnxModel",
    ),
    "fuxi21_pt2": (
        "xmetai_inference.configs.weather_era_0p25_fuxi21",
        "Fuxi21Pt2Model",
    ),
    "iwc_fgvp_gdn2_onnx": (
        "xmetai_inference.configs.weather_era_0p25_fgvp",
        "IwcFgvpGdn2Model",
    ),
    "iwc_fgvp_mamba_ens_onnx": (
        "xmetai_inference.configs.weather_era_0p25_fgvp_ens",
        "IwcFgvpMambaEnsModel",
    ),
    "fengqing_pre_onnx": (
        "xmetai_inference.configs.weather_era_0p25_fengqing",
        "FengqingPreOnnxModel",
    ),
    "pangu_onnx": (
        "xmetai_inference.configs.weather_era_0p25_pangu",
        "PanguOnnxModel",
    ),
    "tongyun_cra_ens_onnx": (
        "xmetai_inference.configs.weather_cra_0p25_tongyun_ens",
        "TongyunCraEnsModel",
    ),
    "fengshun20_onnx": (
        "xmetai_inference.configs.s2s_cra_0p25_fengshun20_ens",
        "Fengshun20OnnxModel",
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

    weight_perturb（改权重）与 latent_forcing（改图）正交，仅 ONNX 可同时开；
    member_index / resident_members 只对 weight_perturb 有意义（-1 = 全常驻）。
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
            if not getattr(model, "supports_latent_forcing", False):
                raise ValueError(
                    f"{cls.__name__}（backend={model.backend}）不支持潜空间 AR(1) 强迫："
                    "已实现的是 ONNX 后端（weights/spa.py）与 pt2 后端（weights/spa_pt2.py）的图改写"
                    "注入，见 xmetai_inference/perturbation/README.md 2N 节")
            model.latent_forcing = forcing
    return model


__all__ = ["MODEL_REGISTRY", "create_model", "get_model_class"]
