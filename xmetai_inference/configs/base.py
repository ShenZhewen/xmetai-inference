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
  * 每个 xmetai_inference/configs/<name>.py 定义全局变量 `cfg`（InferConfig 实例）；
  * config 里的相对路径（model_path）相对仓库根 ROOT 解析；
  * 支持环境变量插值：config 里可用 os.environ.get(...) 覆盖默认（如模型路径）。
  * 起报时间统一用单个 `times` 字段（见 parse_times）。
"""
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

# 项目根目录（绝对）。优先环境变量，其次从包内 configs/ 向上两级推导。
# 安装为 wheel 后若权重位于外部目录，应显式设置 XMETAI_INFERENCE_ROOT，
# 或在配置中通过模型对应的环境变量指定绝对权重路径。
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
    # 确定性 ONNX -> 集合：权重扰动（SPW）规格。None = 不加扰动；dict 见
    # weights.spw.WeightPerturbSpec（groups / sigma_full / sigma / balance / ...）。
    # 需要 onnx extra（图改写）。详见 xmetai_inference/perturbation/README.md。
    weight_perturb: Any = None
    # 确定性 ONNX -> 集合：潜空间 AR(1) 强迫规格（噪声的**时间相关**那条腿）。None = 不加；
    # dict 见 weights.spa.LatentForcingSpec（targets / sigma / tau_days / rank /
    # length_scale / seed）。它不改权重，只给图加一个 `spw/xi` 输入，所以与 weight_perturb
    # 正交、可叠加；成员之间的差别只来自 xi 的种子 -> 同一份图服务全部成员。
    # 需要 onnx extra（图改写）。详见 xmetai_inference/perturbation/README.md 的 2M 节。
    latent_forcing: Any = None
    # perturb_graphs：**集合模型（图内随机算子出成员）+ weight_perturb** 时生效 ——
    # 把 members 个成员摊到 G 份扰动图上（均分 + 余数给前面的图，如 51/5 = 11+10x4）：
    # 成员 m = 第 g 份冻结权重 x 该图 session 内的第 j 次前向。编舞是「图在外层
    # （逐图 load/release）、成员在内层（一次 Rollout 的多条轨迹）」，多卡按**图**切。
    # 与 m1a（成员 = 权重文件，成员外层）的区别只在编舞，weight_perturb 规格本身不变。
    #   0  = 缺省，不启用：weight_perturb 走 m1a 语义（要求模型类是确定性的）；
    #   1  = 扰动一份图服务全部成员（消融臂：「扰动过的 fgvp_ens」）；
    #   G>1 = 生产形态（如 5 图 x ~10 成员）。要求 G <= members，且模型类必须声明
    #        forecast_type="ensemble"（图内随机）——确定性模型配了它直接报错。
    perturb_graphs: int = 0
    # ONNX Runtime 运行旋钮：
    # intra_op_threads：单 session 的算子并行线程数；0 = 不设置（用 ORT 默认）。
    # cudnn_algo_search："" = 后端默认 EXHAUSTIVE；HEURISTIC 能缩短启动预热
    # （代价是每层卷积未必拿到最优 kernel）。
    intra_op_threads: int = 0
    cudnn_algo_search: str = ""
    # rollout_batch：一次 forward 堆几条轨迹（1 = 逐条前向，缺省，与历史行为一致）。
    # ⚠ 与上面 dataset 的 batch_size **不是一个东西**：那个是 Dataset 一次装几个起报
    # （I/O 批），这个是**前向批**。members>1 时批的是成员，members==1 时批的是起报。
    # 三个前提（不满足不会静默算错：前两个在 runner 里挡掉，第三个图自己会拒）：
    #   * 图导出时输入/输出的 batch 轴都是符号维 —— 后端 load 时实测后置
    #     batch_axis_dynamic，没置位就打 WARNING 退回 1，这一趟照跑完；
    #   * 不能与 gpu_state=True 同用（配置互斥，直接报错，不静默退回）；
    #   * 一批里各轨迹的时刻可以不同（step/hour/doy 按 (k,) 逐行喂），所以既支持
    #     「同起报的多成员」也支持「多起报」。
    # 显存按此倍数涨：0.25° 78 通道 2 帧，一条轨迹的状态约 648 MB，照可用显存给。
    rollout_batch: int = 1
    # resident_members：常驻 session 池的额度。扰动档的每个成员是一份独立权重（= 一个
    # onnx session），逐成员装载建一次 session 不只读一遍权重，还要重新预热 cuDNN。
    #   1  = 一换一放（缺省，小显存口径：任何时刻只驻留 1 份）；
    #   N  = 最多驻留 N 份，第 N+1 份进来时按 LRU 换出最久没用过的那份；
    #   -1 = 不限（全常驻）。
    # 只在配了 weight_perturb 时有意义（latent_forcing 是一份 session 服务全部成员，
    # 没有第二份可驻留）。实现见 backends/base.py 的 SessionPool。
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
