# -*- coding: utf-8 -*-
"""Model registry and construction entry point: registered name → model class → instance (lazy imports)."""
from importlib import import_module


# Model layer: name → (module, class).
MODEL_REGISTRY = {
    "fuxi_ens_onnx": (
        "xmetai_inference.configs.weather_era_0p25_fuxi_ens",
        "FuxiEnsOnnxModel",
    ),
    "fuxi21_pt2": (
        "xmetai_inference.configs.weather_era_0p25_fuxi21",
        "Fuxi21Pt2Model",
    ),
    "iwc_tongyun_gdn2_onnx": (
        "xmetai_inference.configs.weather_era_0p25_tongyun",
        "IwcTongyunGdn2Model",
    ),
    "fengqing_pre_onnx": (
        "xmetai_inference.configs.weather_era_0p25_fengqing",
        "FengqingPreOnnxModel",
    ),
    "pangu_onnx": (
        "xmetai_inference.configs.weather_era_0p25_pangu",
        "PanguOnnxModel",
    ),
    "tongyun_era_ens_onnx": (
        "xmetai_inference.configs.weather_era_0p25_tongyun_ens",
        "TongyunEraEnsModel",
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
    """Resolve a built-in model by registered name, or validate a model class supplied directly by an external config."""
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
                 ops_library=None, gpu_state=None, intra_op_threads=0,
                 cudnn_algo_search=""):
    """Construct an inference model by name."""
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
    return model


__all__ = ["MODEL_REGISTRY", "create_model", "get_model_class"]
