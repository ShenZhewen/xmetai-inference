# -*- coding: utf-8 -*-
"""config 基础设施：InferConfig / EvalConfig 两个 dataclass + load_config()。

配置是普通 dataclass，不是 detectron2 的 LazyConfig/omegaconf —— 推理项目只要
「运行配方」，用不上训练框架那一堆重依赖（yacs/omegaconf/hydra/iopath）。

职责边界：
  * spec JSON（模型契约：78 通道、单位、网格）仍独立放在 specs/，被 build_input
    与 evaluate.py 共享 —— 那是「模型是什么」，别和「这次怎么跑」混在一起。
  * config 是「运行配方」：跑哪份 spec、用什么权重、什么起报时间、几张卡、输出到哪。

约定：
  * 每个 configs/<name>.py 定义全局变量 `cfg`（InferConfig 或 EvalConfig 实例）；
  * config 里的相对路径（spec / model_path）相对仓库根 ROOT 解析；
  * 支持环境变量插值：config 里可用 os.environ.get(...) 覆盖默认（如模型路径）。
  * 起报时间统一用单个 `times` 字段（见 parse_times），推理与评测同一格式。
"""
import importlib.util
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

# 仓库根目录（绝对）。优先环境变量，其次按本文件位置推导（configs/ 的上一级），
# 本地与容器路径不同也能自动对上。
ROOT = os.environ.get("XMETAI_INFERENCE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))


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

    语法（推理与评测共用同一个字段/格式）：
      ""                                 → []（评测=扫描全部；推理=报错，必须指定）
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
    """一次推理的运行配方（对应 runner.py 的 --* 参数 + 多卡编排）。"""
    name: str
    model_path: str                          # 模型文件路径（.onnx/.pt2/.ckpt）
    spec: str = "specs/fuxi_ens.json"        # 模型契约 spec JSON（相对 ROOT 或绝对）
    ops_library: str = ""                    # 自定义算子库 .so（空 = 不注册，标准模型）
    backend: str = ""                        # 逃生舱：覆盖 spec 的 model.class（通常留空）
    loader: str = "era5_store"               # 输入数据源 era/zarr/era5_store
    data_root: str = ""                      # 可选覆盖数据源根目录（对应 --zarr）
    times: str = ""                          # 起报时间（统一格式，见 parse_times）
    steps: int = 60                          # 预报步数
    members: int = 0                         # 集合成员数（0 = 读 spec 的 model.members）
    vars: str = ""                           # 输出变量，逗号分隔（空 = 保存全部通道）
    gpu_mem: float = 0.7                     # 显存占用比例
    gpus: int = 1                            # 卡数（>1 时按起报时间分摊到各卡）
    cuda_devices: str = "0"                  # 物理卡号，逗号分隔，按序对应各 rank
    output_dir: str = ""                     # 输出目录（空 = 只做输入校验，不跑模型）
    # 仅文档 / K8s Job 用：标明该配置应在哪个镜像里跑（自定义算子 .so 是 ABI 绑定）
    image: str = "registry.bingosoft.net/pytorch/pytorch:2.11.0-cuda12.8-mamba3-20260403"

    def model_path_abs(self):
        return _abs(self.model_path)

    def spec_abs(self):
        return _abs(self.spec)


@dataclass
class EvalConfig:
    """一次评测的运行配方（对应 evaluate.py 的 --* 参数）。"""
    name: str
    fcst: str                                # 预测输出根目录（runner 的 output_dir）
    spec: str = "specs/fuxi_ens.json"        # 模型契约 spec JSON
    loader: str = "era5_store"               # 实况数据源
    times: str = ""                          # 起报时间（统一格式；空 = 扫描 fcst 全部）
    init_hour: int = 0                       # 扫描模式下日期目录缺的起报小时（0=00UTC）
    steps: int = 60                          # 预报步数
    members: int = 0                         # 成员数（0 = 读 spec 的 model.members）
    vars: str = "z500,u200,v200,msl,tp"      # 要检验的变量
    output_dir: str = ""                     # CSV 输出目录

    def spec_abs(self):
        return _abs(self.spec)


def load_config(path):
    """加载 configs/<name>.py，返回其中的 `cfg`（InferConfig / EvalConfig）。"""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"配置文件不存在：{path}")
    spec = importlib.util.spec_from_file_location("_xmetai_run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "cfg", None)
    if cfg is None:
        raise SystemExit(
            f"配置文件 {path} 没有定义 `cfg`（应写 cfg = InferConfig(...) 或 EvalConfig(...)）")
    return cfg
