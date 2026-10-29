# -*- coding: utf-8 -*-
"""ONNX Runtime 后端：通用加载 + 一步前向。

时间标量的编码差异用类属性 ``time_features`` 声明，不必重写 forward。
"""
import gc
import logging
import os

import numpy as np
import onnxruntime as ort

from .base import (BaseInferModel, SessionPool, _default_gpu_mem_limit,
                   normalize_budget)

log = logging.getLogger(__name__)


def _as_numpy_onnx(value):
    """torch → contiguous numpy；ONNX Runtime 输入边界允许这一次转换。"""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    return np.ascontiguousarray(value)


def _preload_onnxruntime_lib():
    """用 RTLD_GLOBAL 预加载版本化的 libonnxruntime，供自定义算子 .so 解析。

    失败静默；真正的错误留给 register_custom_ops_library 抛。
    """
    import ctypes
    import glob

    try:
        import onnxruntime as _ort

        capi = os.path.join(os.path.dirname(_ort.__file__), "capi")
        # 优先无版本号的软链（若环境已经补过），否则取版本化文件
        for pattern in ("libonnxruntime.so.1", "libonnxruntime.so.1.*"):
            matches = sorted(glob.glob(os.path.join(capi, pattern)))
            if matches:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                return matches[0]
    except Exception:  # noqa: BLE001
        pass
    return None


# 时间标量输入的默认编码（Tongyun-CRA 口径，逐位等于旧版实现）。
# 模型类用 ``time_features`` 覆盖任意一项，没写的键沿用这里；调用点按图签名过滤。
DEFAULT_TIME_FEATURES = {
    "step": {"divisor": 1.0},
    "hour": {"divisor": 24.0},
    "doy": {"divisor": 365.0, "clip_max": 365},
    # 图内扰动的幅度旋钮（CRA 版 Tongyun 有，ERA5 版没有）：1.0 = 训练时的标称噪声。
    "perturb_std_scale": {"value": 1.0},
}


def _day_of_year(value):
    """day-of-year：pandas Timestamp 有 day_of_year / dayofyear，datetime 走 timetuple。"""
    for attr in ("day_of_year", "dayofyear"):
        day = getattr(value, attr, None)
        if day is not None:
            return float(day)
    return float(value.timetuple().tm_yday)


def _scalar_feeds(valid_time, step, features=None):
    """构造图内标量输入（模型签名里没有的键会被 drop）。

    每项可给 ``divisor`` / ``clip_max`` / ``value``；hour/doy 取输入窗末帧的时刻。
    编码在 float64 上做、最后转 float32，与旧实现逐位相同。
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
    """ONNX Runtime 后端（CUDAExecutionProvider）。"""
    backend = "onnx"
    supports_weight_perturb = True
    supports_latent_forcing = True
    # 时间标量编码的覆盖，只写要改的键（图接口契约）
    time_features = {}
    use_cpu_initializers = True
    resident_members = 1                 # 常驻 session 额度：1 = 一换一，-1 = 全常驻
    _pool = None
    ops_library = None                   # 自定义算子库 .so，None = 不需要

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
        # 合并成实例表：子类只写要覆盖的项，行为与「照抄整张表」等价
        self._time_features = {**DEFAULT_TIME_FEATURES, **overrides}

    def _make_session(self, path):
        """建 ONNX InferenceSession（路径校验 + 选项 + provider + 自定义算子）。"""
        # JSON 误传成模型时给明确提示，而不是 ORT 的 Protobuf 报错
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
        # 不设 intra_op 时 ORT 吃满物理核，多 worker 会超订；inter_op 在
        # ORT_SEQUENTIAL 下无效，恒为 1。
        if int(getattr(self, "intra_op_threads", 0) or 0) > 0:
            options.intra_op_num_threads = int(self.intra_op_threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.use_cpu_initializers:
            # 权重留在主机内存，按需送显存（大模型省显存的关键）
            options.add_session_config_entry("session.use_device_allocator_for_initializers", "0")
            options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
        options.log_severity_level = 3

        cuda_opts = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
            # CUDA EP 选项：首次推理慢，之后缓存每层最优 kernel
            "cudnn_conv_algo_search": str(
                getattr(self, "cudnn_algo_search", "") or "EXHAUSTIVE"),
        }
        # 拿不到真实显存上限时不写 gpu_mem_limit；非法值会被 ORT 当成 0 字节上限。
        limit = _default_gpu_mem_limit(self.device_id, self.gpu_mem_fraction)
        if limit is not None:
            cuda_opts["gpu_mem_limit"] = str(limit)

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(("CUDAExecutionProvider", cuda_opts))
        providers.append(("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        # 自定义算子 .so 必须与 onnxruntime 版本 ABI 匹配
        if self.ops_library:
            if not os.path.exists(self.ops_library):
                raise FileNotFoundError(f"自定义算子库不存在：{self.ops_library}")
            _preload_onnxruntime_lib()
            options.register_custom_ops_library(self.ops_library)
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    def load(self, path):
        """加载 ONNX 会话；按 weight_perturb / latent_forcing 解析实际路径并自检。

        自检只读 metadata、不跑前向，抓「看着注入了、其实没生效」的静默失败。
        """
        # 重新 load = 新的一批轨迹，丢掉旧的驱动状态（与旧版每次新建等价）
        self._latent_stream.release()
        spec = getattr(self, "weight_perturb", None)
        forcing = getattr(self, "latent_forcing", None)
        if spec or forcing:
            from xmetai_inference.weights.spa import resolve_stack_path

            # 先解析权重（SPW 成员图），再在其上叠加强迫（只改图结构）
            path = resolve_stack_path(
                path, spec, forcing, getattr(self, "member_index", 0))
            if spec:
                log.info("权重扰动已启用：%s member=%d -> %s",
                         spec.describe(), getattr(self, "member_index", 0), path)
            if forcing:
                log.info("潜空间 AR(1) 强迫已启用：%s member=%d -> %s",
                         forcing.describe(), getattr(self, "member_index", 0), path)
        self.session = self._acquire_session(path)
        self._verify_perturbation(spec)
        self._pred_name = self.session.get_outputs()[0].name
        self._out_shape = self.session.get_outputs()[0].shape
        self._input_names = set(i.name for i in self.session.get_inputs())
        # 一次吃多条轨迹要求输入与输出的 batch 轴都是符号维
        _inputs = self.session.get_inputs()
        _state_input = next((i for i in _inputs if i.name == "input"), _inputs[0])
        _in_shape = list(getattr(_state_input, "shape", None) or [])
        _out_dim = list(getattr(self.session.get_outputs()[0], "shape", None) or [])
        self.batch_axis_dynamic = (
            bool(_in_shape) and bool(_out_dim)
            and isinstance(_in_shape[0], str) and isinstance(_out_dim[0], str))
        self._check_gpu_state_frames()
        self._verify_forcing(forcing)
        self._spare = None    # 输出 buffer 复用，避免每步重新分配显存

    def _verify_perturbation(self, spec):
        """加载后的扰动自检：确认图就是 self.member_index 那个成员。"""
        if not spec:
            return
        self._verify_member_identity()

    def _verify_member_identity(self):
        """只读 metadata，抓「成员号没接上」的静默失败（多成员加载同一文件）。"""
        want = int(getattr(self, "member_index", 0))
        meta = dict(self.session.get_modelmeta().custom_metadata_map)
        got = meta.get("spw.member")
        if meta.get("spw.mechanism") != "m1a" or got is None:
            raise RuntimeError(
                f"启用了 m1a，但加载进来的图没有 m1a 的成员元数据"
                f"（spw.mechanism={meta.get('spw.mechanism')!r}, spw.member={got!r}）。"
                "多半是路径指向了源模型或 m3 的扰动图，而不是成员模型。"
                "成员模型路径应由 weights.spw.member_model_path 给出。")
        if got != str(want):
            raise RuntimeError(
                f"加载到的成员图编号是 {got}，但本实例要跑的是 member "
                f"{want}。同一份成员被加载多次会产出逐位相同的成员——"
                "请检查 cli 是否在每次 load 前给 model.member_index 赋了当前成员号。")
        log.info("权重扰动自检：member %d，被扰动张量 %s 个，sigma_per_group=%s",
                 want, meta.get("spw.tensors"), meta.get("spw.sigma_per_group"))

    def _verify_forcing(self, forcing):
        """加载期自检：图里真的有 spw/xi 且 metadata 对得上。

        ORT 会静默忽略多余输入，图没改写成集合时没有任何报错。
        """
        self._latent_n_targets = 0
        if not forcing:
            return
        meta = dict(self.session.get_modelmeta().custom_metadata_map)
        targets = [name for name in meta.get("spw.ar1.targets", "").split(" | ") if name]
        if not getattr(forcing, "verify_on_load", True):
            self._latent_n_targets = len(targets)
            return
        if meta.get("spw.ar1.method") != "latent-ar1-multiplicative":
            raise RuntimeError(
                f"启用了 latent_forcing，但加载进来的图没有潜空间强迫的元数据"
                f"（spw.ar1.method={meta.get('spw.ar1.method')!r}）。"
                "多半是路径指向了源模型 / m3 的扰动图 / m1a 的成员图，而不是叠加了 "
                "spw/xi 输入的那份。路径应由 weights.spa.resolve_stack_path 给出。")
        if meta.get("spw.ar1.input") != forcing.input_name:
            raise RuntimeError(
                f"图里的强迫输入名是 {meta.get('spw.ar1.input')!r}，但本例要喂的是 "
                f"{forcing.input_name!r}；两边对不上，喂进去只会被静默忽略。")
        if forcing.input_name not in self._input_names:
            raise RuntimeError(
                f"latent_forcing 要喂 {forcing.input_name!r}，但它不在图的输入里"
                f"（{sorted(self._input_names)}）—— 图没被真正改写，集合会退化成逐位相同的"
                "确定性预报。请删掉那份缓存的 .onnx 让它重建。")
        if not targets:
            raise RuntimeError(
                "潜空间强迫的 metadata 里没有任何目标张量（spw.ar1.targets 为空），"
                "说明图是在注入之前就被写出来的，请删掉重新构建。")
        self._latent_n_targets = len(targets)
        log.info("潜空间强迫自检：输入 %s，目标 %d 个，sigma=%s tau=%s 天 rank=%s",
                 forcing.input_name, len(targets), meta.get("spw.ar1.sigma"),
                 meta.get("spw.ar1.tau_days"), meta.get("spw.ar1.rank"))

    def _check_gpu_state_frames(self):
        """gpu_state=True 时要求「图输出帧数 == 输入帧数」。

        GPU 常驻分支绕过 forward override 里的补帧和帧数校验，会静默算错。
        """
        if not self.gpu_state:
            return
        expected = getattr(self, "history_steps", None)
        if not expected or expected <= 1:
            return
        shape = self._out_shape
        # 只拦「5D 输出且帧维为 1」这一确凿特征，其余布局放过，避免误伤
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
        # batch 维 = 本次堆了几条轨迹；块内 k 条各吃一行独立 ξ
        width = int(x.shape[0])
        xi = self._latent_row(step, width=width)
        if xi is not None:
            feeds[self.latent_forcing.input_name] = xi
        # 返回完整滑动窗口，末帧是本步预报；回填由 Rollout 负责
        result = self.session.run([self._pred_name], feeds)[0]
        return result

    # GPU 常驻（IOBinding）：state 全程在显存，只在落盘时搬回 CPU
    def to_gpu(self, x):
        return ort.OrtValue.ortvalue_from_numpy(
            _as_numpy_onnx(x), "cuda", self.device_id)

    def forward_gpu(self, state, step, valid_time):
        # 复用上一拍回收的 buffer 作为本步输出；读写的不是同一个 OrtValue
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
        # 输入已被读完，回收为下一个输出 buffer
        self._spare = state
        return out

    def release(self):
        """丢掉当前 session；额度为 1 时连常驻池一起清空（全清用 release_all）。"""
        self.session = None
        self._spare = None
        super().release()
        if self._pool is not None and self._pool.budget == 1:
            self._pool.purge()

    def release_all(self):
        """把当前 session 与整个常驻池一起放开（进程收尾，或想立刻把显存还回去）。"""
        self.session = None
        self._spare = None
        super().release()
        if self._pool is not None:
            self._pool.purge()
        gc.collect()

    def pool_stats(self):
        """常驻池计数（budget / size / builds / hits / evictions）；日志与自检用。"""
        if self._pool is None:
            return {"budget": normalize_budget(self.resident_members), "size": 0,
                    "builds": 0, "hits": 0, "evictions": 0}
        return self._pool.stats()

    def _acquire_session(self, path):
        """按解析后的路径取 session：命中复用，否则新建并入池。"""
        if self._pool is None:
            self._pool = SessionPool(self.resident_members, logger=log)
        session, hit = self._pool.acquire(path, lambda: self._make_session(path))
        if hit:
            log.info("session 命中共驻池（驻留 %d 份，额度 %d）：%s",
                     len(self._pool), self._pool.budget, os.path.basename(path))
        return session

    def to_numpy(self, state):
        return state.numpy()

    def describe(self):
        return f"providers={self.session.get_providers()}"
