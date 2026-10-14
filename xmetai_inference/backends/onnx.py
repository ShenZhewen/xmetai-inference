# -*- coding: utf-8 -*-
"""ONNX Runtime 后端（CUDAExecutionProvider）。

只放后端机制（session 建立、显存调优、run）；模型特定的喂料、取帧、归一化由子类覆盖。
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

    算子库的 SONAME 是 ``libonnxruntime.so.1``，而 pip 装的 onnxruntime 在 ``capi/``
    下只有版本化文件、该目录也不在 ``LD_LIBRARY_PATH`` 上；且 LD_LIBRARY_PATH 只在
    进程启动时被链接器读取，运行中改 os.environ 无效，所以这里用 RTLD_GLOBAL 先把库
    dlopen 进本进程。失败一律静默，真正的报错留给 register_custom_ops_library。
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
    # 自定义算子库路径（.so），必须在建 session 前注册；None = 标准模型无需注册（子类覆盖）。
    ops_library = None

    def _make_session(self, path):
        """建一个 ONNX InferenceSession（路径校验 + 选项 + provider + 自定义算子）。

        只返回会话，不改派生属性（_pred_name/_out_shape/_input_names）。
        """
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
            # 卷积算法穷举搜索：首次推理慢、之后走 kernel 缓存；零精度损失。必须作 CUDA EP
            # 的 provider option，放成 session config entry 无效。
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
        # .so 必须与当前 onnxruntime 版本 ABI 匹配，跨环境（尤其容器）运行前先确认。
        if self.ops_library:
            if not os.path.exists(self.ops_library):
                raise FileNotFoundError(f"自定义算子库不存在：{self.ops_library}")
            _preload_onnxruntime_lib()
            options.register_custom_ops_library(self.ops_library)
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    def load(self, path):
        """加载 ONNX 会话，绑定到 self.device_id 这张卡，并做显存/性能调优。"""
        self.session = self._make_session(path)
        self._pred_name = self.session.get_outputs()[0].name
        self._out_shape = self.session.get_outputs()[0].shape
        self._input_names = set(i.name for i in self.session.get_inputs())
        # 能否一次吃多条轨迹：state 输入（键名固定 "input"）与输出的 batch 轴都得是
        # 符号维（任一静态则 k>1 不成立）；标量（shape 空）一律当静态。
        _inputs = self.session.get_inputs()
        _state_input = next((i for i in _inputs if i.name == "input"), _inputs[0])
        _in_shape = list(getattr(_state_input, "shape", None) or [])
        _out_dim = list(getattr(self.session.get_outputs()[0], "shape", None) or [])
        self.batch_axis_dynamic = (
            bool(_in_shape) and bool(_out_dim)
            and isinstance(_in_shape[0], str) and isinstance(_out_dim[0], str))
        self._check_gpu_state_frames()
        # 输出 OrtValue 复用池：forward_gpu 把用完的输入 state 回收为下一个输出 buffer，
        # 避免每步 ortvalue_from_shape_and_type 分配/释放 618MB 显存。首个调用时才新建。
        self._spare = None

    def _check_gpu_state_frames(self):
        """gpu_state=True 的前提是「图输出帧数 == 输入帧数」，加载期就查。

        GPU 常驻分支绕过子类的 forward override（也绕过 numpy 分支的帧数校验），对靠
        override 补帧的模型（如 iwc_fgvp_gdn2）会把 1 帧当完整 state 回填，且 _spare
        复用假设输入输出同形 —— 不报错、只算错。
        """
        if not self.gpu_state:
            return
        expected = getattr(self, "history_steps", None)
        if not expected or expected <= 1:
            return
        shape = self._out_shape
        # 只在能确定「图输出 5D 且帧维为 1」（靠 override 补帧的确凿特征）时拦截；
        # 其余一律放过 —— 宁可漏拦也不误伤能跑的模型。
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
        # 末帧才是本步预报；这里原样返回完整输出，不切末帧。
        return self.session.run([self._pred_name], feeds)[0]

    # ---- GPU 常驻（IOBinding）：state 全程待在 GPU，只在落盘时搬回 CPU ----
    # 仅当模型前后处理/回填都是恒等（如 FuXi-Ens）时启用，见 gpu_state。
    def to_gpu(self, x):
        return ort.OrtValue.ortvalue_from_numpy(
            _as_numpy_onnx(x), "cuda", self.device_id)

    def forward_gpu(self, state, step, valid_time):
        # 输出 buffer 复用：_spare 是上一拍回收的另一个 buffer，与本步输入的 state 不是
        # 同一个 OrtValue，run 读写不重叠。shape 与 _out_shape 一致（输入输出同形）。
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
        self._spare = state
        return out

    def to_numpy(self, state):
        return state.numpy()

    def describe(self):
        return f"providers={self.session.get_providers()}"
