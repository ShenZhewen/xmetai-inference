# -*- coding: utf-8 -*-
"""ONNX Runtime backend: generic loading plus single-step forward.

Differences in time scalar encoding are declared via the class attribute
``time_features``; forward need not be overridden.
"""
import logging
import os

import numpy as np
import onnxruntime as ort

from .base import BaseInferModel, _default_gpu_mem_limit

log = logging.getLogger(__name__)


def _as_numpy_onnx(value):
    """torch -> contiguous numpy; the ONNX Runtime input boundary permits this one conversion."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    return np.ascontiguousarray(value)


def _preload_onnxruntime_lib():
    """Preload the versioned libonnxruntime with RTLD_GLOBAL so custom-op .so files can resolve it.

    Failures are silent; the real error is left to register_custom_ops_library to raise.
    """
    import ctypes
    import glob

    try:
        import onnxruntime as _ort

        capi = os.path.join(os.path.dirname(_ort.__file__), "capi")
        # Prefer the unversioned symlink (if the environment already created one), otherwise the versioned file
        for pattern in ("libonnxruntime.so.1", "libonnxruntime.so.1.*"):
            matches = sorted(glob.glob(os.path.join(capi, pattern)))
            if matches:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                return matches[0]
    except Exception:  # noqa: BLE001
        pass
    return None


# Default encoding for time scalar inputs (Tongyun-CRA convention, bit-identical to the
# previous implementation). Model classes override any entry via ``time_features``; omitted
# keys fall back to this table and are filtered by the graph signature at the call site.
DEFAULT_TIME_FEATURES = {
    "step": {"divisor": 1.0},
    "hour": {"divisor": 24.0},
    "doy": {"divisor": 365.0, "clip_max": 365},
    # Amplitude knob for in-graph perturbation (present in the CRA Tongyun version, absent
    # from ERA5): 1.0 = nominal noise at training time.
    "perturb_std_scale": {"value": 1.0},
}


def _day_of_year(value):
    """day-of-year: pandas Timestamp has day_of_year / dayofyear; datetime goes through timetuple."""
    for attr in ("day_of_year", "dayofyear"):
        day = getattr(value, attr, None)
        if day is not None:
            return float(day)
    return float(value.timetuple().tm_yday)


def _scalar_feeds(valid_time, step, features=None):
    """Build in-graph scalar inputs (keys absent from the model signature are dropped).

    Each entry may provide ``divisor`` / ``clip_max`` / ``value``; hour/doy use the
    timestamp of the last frame in the input window. Encoding is done in float64 and
    cast to float32 at the end, bit-identical to the previous implementation.
    """
    table = DEFAULT_TIME_FEATURES if features is None else features
    times = [valid_time] if np.ndim(valid_time) == 0 else list(valid_time)
    feeds = {}
    for name, spec in table.items():
        if "value" in spec:
            feeds[name] = np.full(
                (len(times),), float(spec["value"]), dtype=np.float32)
            continue
        if name == "step":
            raw = np.full((len(times),), step, dtype=np.float64)
        elif name == "hour":
            raw = np.asarray(
                [getattr(value, "hour", 0.0) for value in times], dtype=np.float64)
        elif name == "doy":
            raw = np.asarray(
                [_day_of_year(value) for value in times], dtype=np.float64)
        else:
            raise KeyError(
                f"未知的时间标量输入 {name!r}（只认 step/hour/doy）")
        clip = spec.get("clip_max")
        if clip is not None:
            raw = np.minimum(raw, float(clip))
        feeds[name] = (raw / float(spec.get("divisor", 1.0))).astype(np.float32)
    return feeds


class OnnxInferModel(BaseInferModel):
    """ONNX Runtime backend (CUDAExecutionProvider)."""
    backend = "onnx"
    # Overrides for time scalar encoding; only keys to change need to be written (graph interface contract)
    time_features = {}
    use_cpu_initializers = True
    ops_library = None                   # Custom operator library .so; None = not needed

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        super().__init__(device_id=device_id, gpu_mem_fraction=gpu_mem_fraction)
        overrides = dict(self.time_features or {})
        unknown = set(overrides) - set(DEFAULT_TIME_FEATURES)
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.time_features 里有未知的时间标量输入 "
                f"{sorted(unknown)}（可选 {sorted(DEFAULT_TIME_FEATURES)}）")
        allowed_keys = {"divisor", "clip_max", "value"}
        for name, spec in overrides.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"time_features[{name!r}] 应为 dict（如 {{\"divisor\": 1.0}}），"
                    f"实际 {type(spec).__name__}")
            bad = set(spec) - allowed_keys
            if bad:
                raise ValueError(
                    f"time_features[{name!r}] 有未知字段 {sorted(bad)}"
                    f"（可选 {sorted(allowed_keys)}）")
        # Merge into an instance table: subclasses override only selected entries; behavior equals copying the whole table
        self._time_features = {**DEFAULT_TIME_FEATURES, **overrides}

    def _make_session(self, path):
        """Create an ONNX InferenceSession (path validation + options + provider + custom ops)."""
        # Give a clear message when a JSON file is passed instead of a model, rather than ORT's Protobuf error
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在：{path}")
        if path.lower().endswith(".json"):
            raise ValueError(
                f"拿到的是 JSON 文件（{path}），不是 ONNX 模型。"
                f"模型应是 .onnx 文件。")
        with open(path, "rb") as fh:
            head = fh.read(1)
        if head in (b"{", b"["):
            raise ValueError(
                f"指向的文件是 JSON（{path}），不是 ONNX 模型。"
                f"请改成 .onnx 模型路径。")
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Without intra_op set, ORT saturates physical cores and oversubscribes with multiple
        # workers; inter_op is ineffective under ORT_SEQUENTIAL and is always 1.
        if int(getattr(self, "intra_op_threads", 0) or 0) > 0:
            options.intra_op_num_threads = int(self.intra_op_threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.use_cpu_initializers:
            # Keep weights in host memory and transfer to GPU on demand (key to saving GPU memory for large models)
            options.add_session_config_entry("session.use_device_allocator_for_initializers", "0")
            options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
        options.log_severity_level = 3

        cuda_opts = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
            # CUDA EP option: the first inference is slow; afterwards the best kernel per layer is cached
            "cudnn_conv_algo_search": str(
                getattr(self, "cudnn_algo_search", "") or "EXHAUSTIVE"),
        }
        # Omit gpu_mem_limit when the real memory limit cannot be queried; invalid values are treated by ORT as a 0-byte limit.
        limit = _default_gpu_mem_limit(self.device_id, self.gpu_mem_fraction)
        if limit is not None:
            cuda_opts["gpu_mem_limit"] = str(limit)

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(("CUDAExecutionProvider", cuda_opts))
        providers.append(("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        # The custom operator .so must match the onnxruntime version ABI
        if self.ops_library:
            if not os.path.exists(self.ops_library):
                raise FileNotFoundError(f"自定义算子库不存在：{self.ops_library}")
            _preload_onnxruntime_lib()
            options.register_custom_ops_library(self.ops_library)
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    def load(self, path):
        """Load the ONNX session."""
        self.session = self._make_session(path)
        active_providers = self.session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            log.warning(
                "onnxruntime 未启用 CUDAExecutionProvider（当前 %s），推理将退回 CPU；"
                "GPU 环境请安装 onnxruntime-gpu", active_providers)
        elif active_providers[0] != "CUDAExecutionProvider":
            log.warning(
                "CUDAExecutionProvider 不是首选 provider（当前顺序 %s）", active_providers)
        self._pred_name = self.session.get_outputs()[0].name
        self._out_shape = self.session.get_outputs()[0].shape
        self._input_names = set(i.name for i in self.session.get_inputs())
        # Running multiple trajectories at once requires the batch axis of both inputs and outputs to be symbolic
        _inputs = self.session.get_inputs()
        _state_input = next((i for i in _inputs if i.name == "input"), _inputs[0])
        _in_shape = list(getattr(_state_input, "shape", None) or [])
        _out_dim = list(getattr(self.session.get_outputs()[0], "shape", None) or [])
        self.batch_axis_dynamic = (
            bool(_in_shape) and bool(_out_dim)
            and isinstance(_in_shape[0], str) and isinstance(_out_dim[0], str))
        self._check_gpu_state_frames()
        self._spare = None    # Output buffer reuse to avoid reallocating GPU memory every step

    def _check_gpu_state_frames(self):
        """When gpu_state=True, require "graph output frames == input frames".

        The GPU-resident path bypasses the frame padding and frame-count validation in
        the forward override, and would silently compute wrong results.
        """
        if not self.gpu_state:
            return
        expected = getattr(self, "history_steps", None)
        if not expected or expected <= 1:
            return
        shape = self._out_shape
        # Only reject the unambiguous pattern "5D output with frame dim == 1"; let other layouts pass to avoid false positives
        if not shape or len(shape) != 5 or shape[1] != 1:
            return
        raise ValueError(
            f"{type(self).__name__} 设了 gpu_state=True，但图输出只有 1 帧，"
            f"而输入需要 history_steps={expected} 帧。"
            "该模型靠 forward() override 手工补齐帧数，而 GPU 常驻分支走的是 "
            "forward_gpu()，不经过 override（补齐被跳过），也不做帧数校验；"
            "forward_gpu 的 _spare buffer 复用还假设输入输出同形。开启会静默"
            "算错。请设 gpu_state=False，或为该模型实现 forward_gpu() 并在其中"
            "补齐帧数。"
        )

    def forward(self, x, step, valid_time):
        feeds = {"input": _as_numpy_onnx(x)}
        if valid_time is not None:
            for k, v in _scalar_feeds(
                    valid_time, step, self._time_features).items():
                if k in self._input_names:
                    feeds[k] = v
        # Return the full sliding window; the last frame is this step's forecast; backfill is handled by Rollout
        result = self.session.run([self._pred_name], feeds)[0]
        return result

    # GPU-resident (IOBinding): state stays in GPU memory throughout and only moves back to CPU when written
    def to_gpu(self, x):
        return ort.OrtValue.ortvalue_from_numpy(
            _as_numpy_onnx(x), "cuda", self.device_id)

    def forward_gpu(self, state, step, valid_time):
        # Reuse the buffer reclaimed from the previous step as this step's output; the read and written OrtValues are different
        out = self._spare if self._spare is not None else \
            ort.OrtValue.ortvalue_from_shape_and_type(
                self._out_shape, np.float32, "cuda", self.device_id)
        self._spare = None
        binding = self.session.io_binding()
        binding.bind_ortvalue_input("input", state)
        if valid_time is not None:
            for k, v in _scalar_feeds(
                    valid_time, step, self._time_features).items():
                if k in self._input_names:
                    binding.bind_cpu_input(k, v)
        binding.bind_ortvalue_output(self._pred_name, out)
        self.session.run_with_iobinding(binding)
        # The input has been consumed; reclaim it as the next output buffer
        self._spare = state
        return out

    def release(self):
        """Drop the current session."""
        self.session = None
        self._spare = None
        super().release()

    def to_numpy(self, state):
        return state.numpy()

    def describe(self):
        return f"providers={self.session.get_providers()}"
