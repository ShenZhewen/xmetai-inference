# -*- coding: utf-8 -*-
"""推理配置基座：公共网格/通道声明 + InferConfig + load_config()。

每个 ``configs/<name>.py`` = 模型类（声明契约）+ 全局 ``cfg``（运行配方）；配方里
公共件统一 ``from xmetai_inference.configs.base import ...``。
"""
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


# FuXi-Ens、FuXi-2.1 和 FGVP 共用的 0.25° 全球规则网格。
GRID_025 = {
    "lat": {"start": 90.0, "step": -0.25, "size": 721},
    "lon": {"start": 0.0, "step": 0.25, "size": 1440},
}

# 南→北版本（CRA store 原生朝向，Tongyun-CRA 图就用它）。用它**不要**配 geometry：
# 该处理器会把南→北无条件翻成北→南，把场喂反。输出 lat 坐标也取自这个 grid。
GRID_025_S = {
    "lat": {"start": -90.0, "step": 0.25, "size": 721},
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

# fengshun2.0：87 通道 = 上层 65 + 22 地面（比 FuXi-2.1 多 t2m_min/t2m_max）。
# 顺序必须与输入盘 channel 数组及图内 mean/std 逐位一致；改顺序不报错、只会整场错位。
FENGSHUN_SFC_VARS = [
    "msl", "t2m", "t2m_min", "t2m_max", "d2m", "sst", "ws10m", "ws100m",
    "u10m", "v10m", "u100m", "v100m", "lcc", "mcc", "hcc", "tcc",
    "ssr", "ssrd", "fdir", "ttr", "tcw", "tp",
]
FENGSHUN_CHANNELS = _expand_channels(PL_VARS, LEVELS, FENGSHUN_SFC_VARS)


FENGQING_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
FENGQING_PL_VARS = ["z", "q", "t", "u", "v"]
FENGQING_SFC_VARS = ["tp", "msl", "u10m", "v10m", "t2m"]
FENGQING_CHANNELS = _expand_channels(
    FENGQING_PL_VARS, FENGQING_LEVELS, FENGQING_SFC_VARS)

# Pangu-Weather：层级与上层变量序同 FengQing；地面通道无 tp，= [msl,u10m,v10m,t2m]。
PANGU_LEVELS = FENGQING_LEVELS
PANGU_PL_VARS = FENGQING_PL_VARS
PANGU_SFC_VARS = ["msl", "u10m", "v10m", "t2m"]
PANGU_CHANNELS = _expand_channels(
    PANGU_PL_VARS, PANGU_LEVELS, PANGU_SFC_VARS)

# CRA c71：上层 z/t/u/v/q × 13 层（50→1000 升序）+ 地面 [t2m,q2m,u10m,v10m,msl,tp]。
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


# 项目根目录（绝对）；wheel 安装在外部时可显式设置 XMETAI_INFERENCE_ROOT。
ROOT = os.environ.get("XMETAI_INFERENCE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))

# 包内配方目录（--config <name> / --model <name> 都解析到这里）。
PACKAGE_RECIPES = os.path.dirname(os.path.abspath(__file__))


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
    """模型侧三段处理配方：input / recurrent / output。"""

    input: Any = field(default_factory=list)
    recurrent: Any = field(default_factory=list)
    output: Any = field(default_factory=list)


@dataclass
class InferConfig:
    """一次推理的完整运行配方（由 cli 加载并执行）。"""
    name: str
    model_path: str                          # 模型文件路径（.onnx/.pt2）
    model_class: Any = "fuxi_ens_onnx"       # 注册名，或直接传 BaseInferModel 子类
    # Processor 属于模型运行配方，不属于 Dataset/DataLoader。
    model_processing: Any = None
    pre_processors: Any = None               # 旧配置兼容；新配置用 model_processing
    recurrent_processors: Any = None
    output_processors: Any = None
    ops_library: str = ""                    # 自定义算子库 .so（空 = 标准模型）
    gpu_state: Any = None                    # None = 用模型类默认；True/False 可覆盖
    weight_perturb: Any = None               # 权重扰动（SPW），dict 见 weights.spw
    # 潜空间 AR(1) 强迫（SPA）：只给图加一个 spw/xi 输入、不改权重，与上者正交。
    latent_forcing: Any = None
    # 集合模型 + 权重扰动：把 members 个成员摊到 G 份扰动图上，按图切多卡。
    #   0 = 不启用（走 m1a）；G>=1 要求模型类 forecast_type="ensemble"。
    perturb_graphs: int = 0
    intra_op_threads: int = 0                # 0 = ORT 默认
    cudnn_algo_search: str = ""              # "" = EXHAUSTIVE
    # 一次 forward 堆几条轨迹（前向批，不是 dataset 的 batch_size）。要求图的 batch
    # 轴是符号维且不与 gpu_state=True 同用；显存按此倍数涨。
    model_batch: int = 1
    # 常驻 session 池额度：1 = 一换一，N = 最多 N 份，-1 = 全常驻。
    # 只在配了 weight_perturb 时有意义（latent_forcing 共享同一份 session）。
    resident_members: int = 1
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

    def ops_library_abs(self):
        return self.resolve_path(self.ops_library)

    def processor_specs(self):
        """取 (input, recurrent, output) 三段；兼容旧的平铺字段。"""
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
    """解析外部配置路径或包内配方名（包内配方 = ``configs/<name>.py``）。"""
    value = os.fspath(value)
    direct = os.path.abspath(value)
    if os.path.isfile(direct):
        return direct

    name = value.replace("\\", "/")
    # 带目录前缀的写法（含 job yaml 里拼的绝对路径片段）继续接受。
    for prefix in ("configs/", "models/"):
        if name.startswith(prefix):
            name = name.rsplit("/", 1)[-1]
            break
    if "/" not in name:
        if not name.endswith(".py"):
            name += ".py"
        packaged = os.path.join(PACKAGE_RECIPES, name)
        if os.path.isfile(packaged):
            return packaged
    raise SystemExit(f"配置文件或内置配方不存在：{value}")


def load_config(path):
    """加载外部配置文件或包内配方名，返回其中的 InferConfig。

    按文件路径逐次 exec（而非 import_module）：重新 load 会得到新一批配置对象，
    文件改动与模块级 ``os.environ`` 读取都生效。因此配方里必须用绝对导入。
    """
    path = _resolve_config_path(path)
    if not os.path.isfile(path):
        raise SystemExit(f"配置文件不存在：{path}")
    config_dir = os.path.dirname(path)
    if config_dir not in sys.path:
        # 不删：config 导入的本地模型可能后续还要加载同目录的其他模块。
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


__all__ = [
    "CRA_CHANNELS",
    "CRA_SFC_VARS",
    "FENGSHUN_CHANNELS",
    "FENGSHUN_SFC_VARS",
    "FENGQING_CHANNELS",
    "FENGQING_LEVELS",
    "FENGQING_PL_VARS",
    "FENGQING_SFC_VARS",
    "FUXI_ENS_CHANNELS",
    "FUXI_ENS_SFC_VARS",
    "FUXI21_CHANNELS",
    "FUXI21_SFC_VARS",
    "GRID_025",
    "GRID_025_S",
    "InferConfig",
    "LEVELS",
    "ModelProcessingConfig",
    "PANGU_CHANNELS",
    "PANGU_LEVELS",
    "PANGU_PL_VARS",
    "PANGU_SFC_VARS",
    "PACKAGE_RECIPES",
    "PL_VARS",
    "ROOT",
    "grid_coords",
    "load_config",
    "parse_times",
]
