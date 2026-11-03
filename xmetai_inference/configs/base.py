# -*- coding: utf-8 -*-
"""Inference config machinery: InferConfig + ModelProcessingConfig + load_config().

Each ``configs/<name>.py`` = model class + module-level ``cfg``. Shared grid/channel
names come from ``utils/config_util.py`` and are re-exported here for legacy callers;
per-model channel lists live in the model's own config.
"""
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from xmetai_inference.utils.config_util import (
    FUXI_ENS_CHANNELS,
    FUXI_ENS_SFC_VARS,
    GRID_025,
    GRID_025_S,
    LEVELS,
    PL_VARS,
    grid_coords,
    parse_times,
)


# Project root (absolute); set XMETAI_INFERENCE_ROOT explicitly when the wheel is installed elsewhere.
ROOT = os.environ.get("XMETAI_INFERENCE_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))

# In-package recipe directory (both positional config.py and --config <name> resolve here).
PACKAGE_RECIPES = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ModelProcessingConfig:
    """Model-side three-stage processing recipe: input / recurrent / output."""

    input: Any = field(default_factory=list)
    recurrent: Any = field(default_factory=list)
    output: Any = field(default_factory=list)


@dataclass
class InferConfig:
    """Complete run recipe for a single inference job (loaded and executed by the CLI)."""
    name: str
    model_path: str                          # Model file path (.onnx/.pt2)
    model_class: Any = "fuxi_ens_onnx"       # Registered name, or a BaseInferModel subclass directly
    # Processors are part of the model run recipe, not of Dataset/DataLoader.
    model_processing: Any = None
    pre_processors: Any = None               # Legacy compatibility; new configs use model_processing
    recurrent_processors: Any = None
    output_processors: Any = None
    ops_library: str = ""                    # Custom operator library .so (empty = standard model)
    gpu_state: Any = None                    # None = use the model class default; True/False overrides
    intra_op_threads: int = 0                # 0 = ORT default
    cudnn_algo_search: str = ""              # "" = EXHAUSTIVE
    # Number of trajectories stacked per forward pass (forward batch, not the dataset batch_size).
    # The graph's batch axis must be symbolic, and this must not be combined with
    # gpu_state=True; memory scales by this factor.
    model_batch: int = 1
    dataset: Any = None
    dataloader: Any = field(default_factory=dict)
    data_root: str = ""                      # Optional data source root override (reserved; current Datasets use absolute paths)
    times: str = ""                          # Initialization times (unified format; see parse_times)
    steps: int = 60                          # Forecast steps
    hour_interval: int = 0                  # Time step in hours (0 = read hour_interval from the model class)
    members: int = 0                         # Ensemble member count (0 = read members from the model class)
    vars: str = ""                           # Output variables, comma-separated (empty = save all channels)
    gpu_mem: float = 0.7                     # GPU memory fraction
    log_level: str = "INFO"                  # Console log level
    gpus: int = 1                            # GPU count (>1 distributes initialization times across GPUs)
    cuda_devices: str = "0"                  # Physical device IDs, comma-separated, mapped to ranks in order
    output_dir: str = ""                     # Output directory (empty = input validation only, no model run)
    # Documentation / K8s Job only: the image this config should run in (custom operator .so is ABI-bound)
    image: str = "registry.bingosoft.net/pytorch/pytorch:2.11.0-cuda12.8-mamba3-20260403"
    _source_path: str = field(init=False, repr=False, default="")

    def resolve_path(self, value):
        """Resolve a local path relative to the external config's directory; URLs and absolute paths are unchanged."""
        if not value or os.path.isabs(value) or "://" in value:
            return value
        base = os.path.dirname(self._source_path) if self._source_path else ROOT
        return os.path.abspath(os.path.join(base, value))

    def model_path_abs(self, value=None):
        return self.resolve_path(self.model_path if value is None else value)

    def ops_library_abs(self):
        return self.resolve_path(self.ops_library)

    def processor_specs(self):
        """Return the (input, recurrent, output) triple; also supports the legacy flat fields."""
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
    """Resolve an external config path or an in-package recipe name (in-package recipe = ``configs/<name>.py``)."""
    value = os.fspath(value)
    direct = os.path.abspath(value)
    if os.path.isfile(direct):
        return direct

    name = value.replace("\\", "/")
    # Paths with a directory prefix (including absolute fragments assembled in job yamls) remain accepted.
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
    """Load an external config file or in-package recipe name and return its InferConfig.

    The file is exec'd by path each time (rather than import_module): reloading yields a fresh
    set of config objects, and both file edits and module-level ``os.environ`` reads take effect.
    Recipes must therefore use absolute imports.
    """
    path = _resolve_config_path(path)
    if not os.path.isfile(path):
        raise SystemExit(f"配置文件不存在：{path}")
    config_dir = os.path.dirname(path)
    if config_dir not in sys.path:
        # Do not remove: local models imported by the config may later load other modules from the same directory.
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
    "FUXI_ENS_CHANNELS",
    "FUXI_ENS_SFC_VARS",
    "GRID_025",
    "GRID_025_S",
    "InferConfig",
    "LEVELS",
    "ModelProcessingConfig",
    "PACKAGE_RECIPES",
    "PL_VARS",
    "ROOT",
    "grid_coords",
    "load_config",
    "parse_times",
]