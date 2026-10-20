# -*- coding: utf-8 -*-
"""推理配置基础设施：InferConfig + load_config()。

config 是「运行配方」：跑哪个模型类、用什么权重、什么起报时间、几张卡、输出到哪。
模型契约（通道/单位/网格）由模型类的类属性声明；每个 configs/<name>.py 定义
全局变量 ``cfg``，起报时间统一用 ``times`` 字段（见 parse_times）。
"""
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

# 项目根目录（绝对）；wheel 安装在外部时可显式设置 XMETAI_INFERENCE_ROOT。
ROOT = os.environ.get("XMETAI_INFERENCE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_dt(token):
    """单个起报时间 token -> datetime（YYYYMMDD 或 YYYYMMDDHH）。"""
    token = token.strip()
    if re.fullmatch(r"\d{8}", token):
        return datetime.strptime(token, "%Y%m%d")
    if re.fullmatch(r"\d{10}", token):
        return datetime.strptime(token, "%Y%m%d%H")
    raise SystemExit(f"无法解析起报时间 {token!r}（应 YYYYMMDD 或 YYYYMMDDHH）")


def parse_times(spec, hour_interval=24):
    """把统一的 `times` 字符串展开成起报时间列表（datetime，升序去重）。

    语法：
      ""                                 → []
      "2025010600" / "20250106"          → 单个起报
      "20250106,20250112"                → 显式列表（逗号分隔）
      "2025010600..2025020500"           → 闭区间，步长 hour_interval（默认 24h=每天一次）
      "2025010600..2025020500:24"        → 闭区间，`:N` 覆盖步长（小时）
    单个与区间可混在同一个逗号列表里，如 "20250106,20250110..20250112:12"。
    """
    if spec is None or str(spec).strip() == "":
        return []
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d{8}(?:\d{2})?)\s*\.\.\s*(\d{8}(?:\d{2})?)(?::(\d+))?", tok)
        if m:
            start = _parse_dt(m.group(1))
            end = _parse_dt(m.group(2))
            step = int(m.group(3)) if m.group(3) else hour_interval
            if step <= 0:
                raise SystemExit(f"区间步长必须 >0：{tok!r}")
            t = start
            while t <= end:
                out.append(t)
                t += timedelta(hours=step)
        else:
            out.append(_parse_dt(tok))
    if not out:
        raise SystemExit(f"times 解析后为空：{spec!r}")
    return sorted(set(out))


@dataclass
class ModelProcessingConfig:
    """Model-owned input, recurrent, and output processing recipe."""

    input: Any = field(default_factory=list)
    recurrent: Any = field(default_factory=list)
    output: Any = field(default_factory=list)


@dataclass
class InferConfig:
    """一次推理的完整运行配方（由 inference.py 加载并执行）。"""
    name: str
    model_path: str                          # 模型文件路径（.onnx/.pt2）
    # 内置模型可写注册名；外部 config 可直接传 BaseInferModel 子类。
    model_class: Any = "fuxi_ens_onnx"
    # Processor 属于模型运行配方，不属于 Dataset/DataLoader。
    model_processing: Any = None
    # 旧配置兼容字段；新配置统一使用 model_processing。
    pre_processors: Any = None
    recurrent_processors: Any = None
    output_processors: Any = None
    ops_library: str = ""                    # 自定义算子库 .so（空 = 不注册，标准模型）
    gpu_state: Any = None                   # None = 使用模型类默认；True/False 可覆盖
    # 权重扰动（SPW）规格；None = 不加。dict 见 weights.spw.WeightPerturbSpec。
    weight_perturb: Any = None
    # 潜空间 AR(1) 强迫规格；None = 不加。dict 见 weights.spa.LatentForcingSpec。
    # 它只给图加一个 `spw/xi` 输入、不改权重，与 weight_perturb 正交、可叠加。
    latent_forcing: Any = None
    # 集合模型 + 权重扰动：把 members 个成员摊到 G 份扰动图上，按图切多卡。
    #   0 = 不启用（weight_perturb 走 m1a）；G>=1 要求模型类 forecast_type="ensemble"。
    perturb_graphs: int = 0
    # intra_op_threads：0 = ORT 默认；cudnn_algo_search："" = EXHAUSTIVE。
    intra_op_threads: int = 0
    cudnn_algo_search: str = ""
    # 一次 forward 堆几条轨迹（前向批，不是 dataset 的 batch_size）。要求图的 batch
    # 轴是符号维且不与 gpu_state=True 同用；显存按此倍数涨。
    model_batch: int = 1
    # 常驻 session 池额度：1 = 一换一放，N = 最多 N 份，-1 = 全常驻。
    # 只在配了 weight_perturb 时有意义（latent_forcing 共享同一份 session）。
    resident_members: int = 1
    # 数据源统一走 xmetai-core Dataset + torch DataLoader 批量读取。
    dataset: Any = None
    dataloader: Any = field(default_factory=dict)
    data_root: str = ""                      # 可选覆盖数据源根目录（保留字段；当前 Dataset 使用绝对路径）
    times: str = ""                          # 起报时间（统一格式，见 parse_times）
    steps: int = 60                          # 预报步数
    hour_interval: int = 0                  # 时间步长小时（0 = 读模型类的 hour_interval）
    members: int = 0                         # 集合成员数（0 = 读模型类的 members）
    vars: str = ""                           # 输出变量，逗号分隔（空 = 保存全部通道）
    gpu_mem: float = 0.7                     # 显存占用比例
    log_level: str = "INFO"                  # 控制台日志级别
    gpus: int = 1                            # 卡数（>1 时按起报时间分摊到各卡）
    cuda_devices: str = "0"                  # 物理卡号，逗号分隔，按序对应各 rank
    output_dir: str = ""                     # 输出目录（空 = 只做输入校验，不跑模型）
    # 仅文档 / K8s Job 用：标明该配置应在哪个镜像里跑（自定义算子 .so 是 ABI 绑定）
    image: str = "registry.bingosoft.net/pytorch/pytorch:2.11.0-cuda12.8-mamba3-20260403"
    _source_path: str = field(init=False, repr=False, default="")

    def resolve_path(self, value):
        """相对外部 config 所在目录解析本地路径；URL 和绝对路径保持不变。"""
        if not value or os.path.isabs(value) or "://" in value:
            return value
        base = os.path.dirname(self._source_path) if self._source_path else ROOT
        return os.path.abspath(os.path.join(base, value))

    def model_path_abs(self, value=None):
        return self.resolve_path(self.model_path if value is None else value)

    def data_root_abs(self, value=None):
        return self.resolve_path(self.data_root if value is None else value)

    def ops_library_abs(self):
        return self.resolve_path(self.ops_library)

    def processor_specs(self):
        """Return model-owned processing stages with legacy config support."""
        if self.model_processing is not None:
            if any(
                value is not None
                for value in (
                    self.pre_processors,
                    self.recurrent_processors,
                    self.output_processors,
                )
            ):
                raise ValueError(
                    "model_processing 不能与旧的 pre/recurrent/output_processors 同时配置"
                )
            if isinstance(self.model_processing, ModelProcessingConfig):
                processing = self.model_processing
            else:
                processing = ModelProcessingConfig(**dict(self.model_processing))
            return processing.input, processing.recurrent, processing.output
        return (
            self.pre_processors,
            self.recurrent_processors,
            self.output_processors,
        )

def _resolve_config_path(value):
    """解析外部配置路径或包内配置名。"""
    value = os.fspath(value)
    direct = os.path.abspath(value)
    if os.path.isfile(direct):
        return direct

    name = value
    if name.startswith("configs/") or name.startswith("configs\\"):
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
    if os.path.sep not in name and (os.path.altsep is None or os.path.altsep not in name):
        if not name.endswith(".py"):
            name += ".py"
        packaged = os.path.join(os.path.dirname(__file__), name)
        if os.path.isfile(packaged):
            return packaged
    raise SystemExit(f"配置文件或内置配置不存在：{value}")


def load_config(path):
    """加载外部配置文件或包内配置名，返回其中的 InferConfig。"""
    path = _resolve_config_path(path)
    if not os.path.isfile(path):
        raise SystemExit(f"配置文件不存在：{path}")
    config_dir = os.path.dirname(path)
    if config_dir not in sys.path:
        # 保留在 sys.path 中：config 导入的本地模型可能在后续按需加载其他同目录模块。
        sys.path.insert(0, config_dir)
    module_name = f"_xmetai_run_config_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        raise SystemExit(
            f"配置文件 {path} 没有定义 `cfg`（应写 cfg = InferConfig(...)）")
    if not isinstance(cfg, InferConfig):
        raise SystemExit(
            f"配置文件 {path} 的 cfg 类型是 {type(cfg).__name__}，应为 InferConfig")
    input_specs, recurrent_specs, output_specs = cfg.processor_specs()
    if input_specs is None or recurrent_specs is None or output_specs is None:
        raise SystemExit(
            f"配置文件 {path} 没有完整声明模型 Processor")
    cfg._source_path = path
    return cfg
