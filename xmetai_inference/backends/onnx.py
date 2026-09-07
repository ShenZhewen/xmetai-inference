# -*- coding: utf-8 -*-
"""ONNX Runtime 后端（CUDAExecutionProvider）。

通用 ONNX 加载 + 一步前向。这里只放后端机制（session 建立、显存调优、run），
模型特定的东西（喂料、取帧、归一化）由各模型子类覆盖。
"""
import os

import numpy as np
import onnxruntime as ort

from .base import BaseInferModel, _default_gpu_mem_limit


def _as_numpy_onnx(value):
    """torch → contiguous numpy；ONNX Runtime 输入边界允许这一次转换。"""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    return np.ascontiguousarray(value)


def _preload_onnxruntime_lib():
    """预加载 libonnxruntime.so.1，让自定义算子 .so 能解析到它。

    自定义算子库（xmetai_onnx_plugins.so）编译时链接的是 SONAME
    ``libonnxruntime.so.1``，但 pip 装的 onnxruntime 在 ``capi/`` 下只有版本化文件
    （如 ``libonnxruntime.so.1.24.4``），既没有 ``libonnxruntime.so.1`` 软链、该目录
    也不在 ``LD_LIBRARY_PATH`` 上。于是 ``register_custom_ops_library`` 报
    「libonnxruntime.so.1: cannot open shared object file」。

    ``LD_LIBRARY_PATH`` 只在进程启动时被链接器读取，运行中改 ``os.environ`` 无效，
    所以那条路必须靠外部脚本（见 ``scripts/run.sh``）。这里换个办法：用
    ``RTLD_GLOBAL`` 先把版本化的库 dlopen 进本进程。dlopen 解析依赖时会先按 SONAME
    查已加载的库，命中就不再走文件系统搜索 —— 于是直接
    ``python -m xmetai_inference`` 也能跑，不必套 run.sh。

    失败一律静默：这只是让「不设 LD_LIBRARY_PATH 也能跑」，真正的报错留给
    register_custom_ops_library 抛（它的信息更准确）。
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
    use_cpu_initializers = True
    # 自定义算子库路径（.so）。某些模型（如 GatedDeltaNet 骨干）用了非标准算子，
    # 必须在建 session 前注册对应 plugin；None = 标准模型无需注册。子类覆盖此属性。
    ops_library = None

    def load(self, path):
        """加载 ONNX 会话，绑定到 self.device_id 这张卡，并做显存/性能调优。"""
        # 防御：把 JSON 误当成模型传进来时，直接给明确提示，而不是让 ORT 报晦涩的
        # "Protobuf parsing failed"。--model 应指向 .onnx 文件。
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在：{path}")
        if path.lower().endswith(".json"):
            raise ValueError(
                f"--model 拿到的是 JSON 文件（{path}），不是 ONNX 模型。"
                f"模型应是 .onnx 文件（--model）。")
        with open(path, "rb") as fh:
            head = fh.read(1)
        if head in (b"{", b"["):
            raise ValueError(
                f"--model 指向的文件是 JSON（{path}），不是 ONNX 模型。"
                f"请把 --model 改成 .onnx 模型路径。")
        # 关掉 ORT 默认 logger 的 WARNING：模型里那些 ScatterND 提示纯属噪音。
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # ALL 在 EXTENDED 基础上再加 layout（NHWC）优化与更多算子融合，数值等价、零精度损失。
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.use_cpu_initializers:
            # 模型权重留在主机内存，按需送显存 —— 大模型省显存的关键
            options.add_session_config_entry("session.use_device_allocator_for_initializers", "0")
            options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
        options.log_severity_level = 3

        cuda_opts = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
            # 卷积算法穷举搜索：首次推理慢一点，换来每层最优 kernel（之后缓存）。零精度损失。
            # 注意这是 CUDA EP 的 provider option，之前误放成 session config entry 是无效的。
            "cudnn_conv_algo_search": "EXHAUSTIVE",
        }
        # 只在能拿到真实显存上限时才限制；拿不到（无 torch / 无 CUDA）就干脆不写 gpu_mem_limit。
        # 否则 fallback 成 "0.7" 这种非法值会被 ORT 当成 0 字节上限，把 CUDA 分配器直接压崩。
        limit = _default_gpu_mem_limit(self.device_id, self.gpu_mem_fraction)
        if limit is not None:
            cuda_opts["gpu_mem_limit"] = str(limit)

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(("CUDAExecutionProvider", cuda_opts))
        providers.append(("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        # 自定义算子：注册 .so 后才能建 session（否则报「is not a registered function/op」）。
        # .so 必须与当前 onnxruntime 版本 ABI 匹配，跨环境（尤其容器）运行要先确认。
        if self.ops_library:
            if not os.path.exists(self.ops_library):
                raise FileNotFoundError(f"自定义算子库不存在：{self.ops_library}")
            _preload_onnxruntime_lib()
            options.register_custom_ops_library(self.ops_library)
        self.session = ort.InferenceSession(path, sess_options=options, providers=providers)
        self._pred_name = self.session.get_outputs()[0].name
        self._out_shape = self.session.get_outputs()[0].shape
        self._input_names = set(i.name for i in self.session.get_inputs())
        self._check_gpu_state_frames()
        # 输出 OrtValue 复用池：forward_gpu 把用完的输入 state 回收为下一个输出 buffer，
        # 避免每步 ortvalue_from_shape_and_type 分配/释放 618MB 显存。首个调用时才新建。
        self._spare = None

    def _check_gpu_state_frames(self):
        """gpu_state=True 的前提是「图输出帧数 == 输入帧数」，加载时就查。

        GPU 常驻分支（base.run 的 forward_gpu）绕过子类的 forward override，
        也绕过 numpy 分支里的帧数校验（base.py 的 state.shape[1] != in_frames）。
        对「图只出 1 帧、靠 forward override 手工补成 in_frames 帧」的模型
        （如 iwc_fgvp_gdn2），开 GPU 常驻会把 1 帧当完整 state 回填给需要
        in_frames 帧的图，且 forward_gpu 里的 _spare buffer 复用还假设输入输出
        同形。两者都不会报错，只会算错。所以在加载期直接拦下。
        """
        if not self.gpu_state:
            return
        expected = getattr(self, "history_steps", None)
        if not expected or expected <= 1:
            return
        shape = self._out_shape
        # 只在能确定「图输出 5D 且帧维为 1」时才拦——这是 forward override 补齐
        # 帧数的确凿特征（iwc_fgvp_gdn2）。其余情况（动态轴是字符串、维数不是 5、
        # 帧维是别的值）一律放过：本函数的目的是拦已知的静默错误，不是猜测未知
        # 图的布局，误伤一个能跑的模型比漏拦更糟。
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
            for k, v in _scalar_feeds(valid_time, step).items():
                if k in self._input_names:
                    feeds[k] = v
        # 输出完整滑动窗口 (1, in_frames, C, H, W)：输入 [t-1,t] → 输出 [t,t+1]，
        # 末帧才是本步预报。基类 run 会取 state[:, -1] 当预报、用完整 state
        # 回填（state = result），与官方 inference.py 的 `new_input = model.run(...)`
        # 一致。这里原样返回完整输出，不再只切末帧。
        return self.session.run([self._pred_name], feeds)[0]

    # ---- GPU 常驻（IOBinding）：state 全程待在 GPU，只在落盘时搬回 CPU ----
    # 相比 forward（每步 numpy H2D + D2H），forward_gpu 用 OrtValue 绑定输入/输出到
    # 显存，省掉每步的 H2D/D2H 与 ORT 反复分配显存的开销。仅当模型前后处理/回填都
    # 是恒等（如 FuXi-Ens）时启用，见 gpu_state。
    def to_gpu(self, x):
        return ort.OrtValue.ortvalue_from_numpy(
            _as_numpy_onnx(x), "cuda", self.device_id)

    def forward_gpu(self, state, step, valid_time):
        # 输出 buffer 复用：上一步 run 用完的输入 state 已在 _spare 里，直接拿它当输出。
        # 输入 state 与本步输出是不同 OrtValue（spare 是上一拍回收的另一个 buffer），
        # run 读输入、写输出不会重叠。shape 与 _out_shape 一致（模型输入输出同形）。
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
        # 输入 state 已被 run 读完，回收为下一个输出 buffer。
        self._spare = state
        return out

    def to_numpy(self, state):
        return state.numpy()

    def describe(self):
        return f"providers={self.session.get_providers()}"
