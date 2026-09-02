# -*- coding: utf-8 -*-
"""推理配置基础设施：InferConfig + load_config()。

配置是普通 dataclass，不是 detectron2 的 LazyConfig/omegaconf —— 推理项目只要
「运行配方」，用不上训练框架那一堆重依赖（yacs/omegaconf/hydra/iopath）。

职责边界：
  * 模型契约（通道/单位/网格/时间窗口）折叠进 models 包及各模型类的
    input_channels/output_channels/grid/history_steps 等类属性，不再靠 spec JSON；
    config 用 `model_class=` 引用模型类，并完整声明输入、回填和输出 Processor。
  * config 是「运行配方」：跑哪个模型类、用什么权重、什么起报时间、几张卡、输出到哪。

约定：
  * 每个 xmetai/configs/<name>.py 定义全局变量 `cfg`（InferConfig 实例）；
  * config 里的相对路径（model_path）相对仓库根 ROOT 解析；
  * 支持环境变量插值：config 里可用 os.environ.get(...) 覆盖默认（如模型路径）。
  * 起报时间统一用单个 `times` 字段（见 parse_times）。
"""
import importlib.util
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

# 项目根目录（绝对）。优先环境变量，其次从包内 configs/ 向上两级推导。
# 安装为 wheel 后若权重位于外部目录，应显式设置 XMETAI_INFERENCE_ROOT，
# 或在配置中通过模型对应的环境变量指定绝对权重路径。
ROOT = os.environ.get("XMETAI_INFERENCE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


def _abs(p):
    """相对路径 -> 相对 ROOT 的绝对路径；空串 / 已是绝对路径则原样返回。"""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


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
class InferConfig:
    """一次推理的完整运行配方（由 inference.py 加载并执行）。"""
    name: str
    model_path: str                          # 模型文件路径（.onnx/.pt2/.ckpt）
    model_class: str = "fuxi_ens_onnx"       # 模型类名（models/MODEL_REGISTRY 的键）
    # 三个阶段共同构成完整处理配方；可写名称列表或 [{"name": "...", ...}]。
    pre_processors: Any = None
    recurrent_processors: Any = None
    output_processors: Any = None
    ops_library: str = ""                    # 自定义算子库 .so（空 = 不注册，标准模型）
    loader: str = "era5_store"               # 输入数据源 era/zarr/era5_store
    data_root: str = ""                      # 可选覆盖数据源根目录（对应 --zarr）
    times: str = ""                          # 起报时间（统一格式，见 parse_times）
    steps: int = 60                          # 预报步数
    members: int = 0                         # 集合成员数（0 = 读模型类的 members）
    vars: str = ""                           # 输出变量，逗号分隔（空 = 保存全部通道）
    gpu_mem: float = 0.7                     # 显存占用比例
    log_level: str = "INFO"                  # 控制台日志级别
    gpus: int = 1                            # 卡数（>1 时按起报时间分摊到各卡）
    cuda_devices: str = "0"                  # 物理卡号，逗号分隔，按序对应各 rank
    output_dir: str = ""                     # 输出目录（空 = 只做输入校验，不跑模型）
    # 仅文档 / K8s Job 用：标明该配置应在哪个镜像里跑（自定义算子 .so 是 ABI 绑定）
    image: str = "registry.bingosoft.net/pytorch/pytorch:2.11.0-cuda12.8-mamba3-20260403"

    def model_path_abs(self):
        return _abs(self.model_path)


def _resolve_config_path(value):
    """解析外部配置路径或包内配置名。"""
    direct = os.path.abspath(value)
    if os.path.isfile(direct):
        return direct

    name = value
    if name.startswith("configs/") or name.startswith("configs\\"):
        name = os.path.basename(name)
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
    spec = importlib.util.spec_from_file_location("_xmetai_run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        raise SystemExit(
            f"配置文件 {path} 没有定义 `cfg`（应写 cfg = InferConfig(...)）")
    if not isinstance(cfg, InferConfig):
        raise SystemExit(
            f"配置文件 {path} 的 cfg 类型是 {type(cfg).__name__}，应为 InferConfig")
    missing = [
        name for name in (
            "pre_processors",
            "recurrent_processors",
            "output_processors",
        )
        if getattr(cfg, name) is None
    ]
    if missing:
        raise SystemExit(
            f"配置文件 {path} 没有完整声明 Processor：{', '.join(missing)}")
    return cfg
