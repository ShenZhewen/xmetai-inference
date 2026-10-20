# -*- coding: utf-8 -*-
"""ONNX Runtime 后端（CUDAExecutionProvider）。

通用 ONNX 加载 + 一步前向。这里只放后端机制（session 建立、显存调优、run），
模型特定的东西（喂料、取帧、归一化）由各模型子类覆盖。
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


def _scalar_feeds(valid_time, step):
    """构造 step/hour/doy 标量输入（模型签名里没有的键会被 drop）。"""
    times = [valid_time] if np.ndim(valid_time) == 0 else list(valid_time)
    return {
        "step": np.full((len(times),), step, dtype=np.float32),
        "hour": np.asarray([value.hour / 24.0 for value in times], dtype=np.float32),
        "doy": np.asarray(
            [min(365, value.day_of_year) / 365.0 for value in times],
            dtype=np.float32,
        ),
    }


class OnnxInferModel(BaseInferModel):
    """ONNX Runtime 后端（CUDAExecutionProvider）。"""
    backend = "onnx"
    supports_weight_perturb = True
    supports_latent_forcing = True
    # config 的 latent_forcing；与 weight_perturb 正交、可叠加
    latent_forcing = None
    use_cpu_initializers = True
    # 常驻 session 额度：1 = 一换一，N = 最多 N 份，-1 = 全常驻
    resident_members = 1
    _pool = None
    _pool_hit = False
    # 自定义算子库路径（.so）；None = 不需要
    ops_library = None

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
        # 压掉 ScatterND 等模型内噪音提示
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # 不设 intra_op 时 ORT 吃满物理核，多个 worker 会超订；inter_op 在
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
        """加载 ONNX 会话，并按 weight_perturb / latent_forcing 解析实际路径。

        加载后做一次自检（只读 metadata、不跑前向），抓「看着注入了、其实没生效」
        的静默失败。
        """
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
        # 输出 buffer 复用池，避免每步重新分配显存
        self._spare = None
        # 潜空间强迫的驱动状态，每个成员一份
        # ⚠ 不要在这里清 _latent_n_targets：它刚由 _verify_forcing 按 metadata 设好，
        # 清零会让 forward 喂不上 spw/xi。
        from xmetai_inference.backends.spa import LatentStepStream

        self._latent_stream = LatentStepStream()

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

    def _latent_step_stream(self, step, width=1):
        """本步本块的起始 ξ 行号（None = 不强迫）；实现见 backends.spa。"""
        forcing = getattr(self, "latent_forcing", None)
        if not forcing or self._latent_n_targets <= 0:
            return None
        return self._latent_stream(
            forcing, step,
            n_targets=self._latent_n_targets,
            member_index=getattr(self, "member_index", 0),
            hour_interval=getattr(self, "hour_interval", None),
            width=width)

    def forward(self, x, step, valid_time):
        feeds = {"input": _as_numpy_onnx(x)}
        if valid_time is not None:
            for k, v in _scalar_feeds(valid_time, step).items():
                if k in self._input_names:
                    feeds[k] = v
        # batch 维 = 本次堆了几条轨迹；块内 k 条要同一 index 起的 k 行独立 ξ
        width = int(x.shape[0])
        row = self._latent_step_stream(step, width=width)
        if row is not None:
            feeds[self.latent_forcing.input_name] = self._latent_stream.ar1.row(row, width)
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
            for k, v in _scalar_feeds(valid_time, step).items():
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
        if getattr(self, "_latent_stream", None) is not None:
            self._latent_stream.release()
        if self._pool is not None and self._pool.budget == 1:
            self._pool.purge()

    def release_all(self):
        """把当前 session 与整个常驻池一起放开（进程收尾，或想立刻把显存还回去）。"""
        self.session = None
        self._spare = None
        if getattr(self, "_latent_stream", None) is not None:
            self._latent_stream.release()
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
        self._pool_hit = hit
        if hit:
            log.info("session 命中共驻池（驻留 %d 份，额度 %d）：%s",
                     len(self._pool), self._pool.budget, os.path.basename(path))
        return session

    def to_numpy(self, state):
        return state.numpy()

    def describe(self):
        return f"providers={self.session.get_providers()}"
