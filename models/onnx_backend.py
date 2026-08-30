# -*- coding: utf-8 -*-
"""ONNX Runtime 后端（CUDAExecutionProvider）。

通用 ONNX 加载 + 一步前向。这里只放后端机制（session 建立、显存调优、run），
模型特定的东西（喂料、取帧、归一化）由各模型子类覆盖。
"""
import os

import numpy as np
import onnxruntime as ort

from .base import BaseInferModel, _default_gpu_mem_limit


def _scalar_feeds(valid_time, step):
    """构造 step/hour/doy 标量输入（模型签名里没有的键会被 drop）。"""
    return {
        "step": np.asarray([step], dtype=np.float32),
        "hour": np.asarray([valid_time.hour / 24.0], dtype=np.float32),
        "doy": np.asarray([min(365, valid_time.day_of_year) / 365.0], dtype=np.float32),
    }


class OnnxInferModel(BaseInferModel):
    """ONNX Runtime 后端（CUDAExecutionProvider）。"""
    backend = "onnx"
    use_cpu_initializers = True

    def load(self, path):
        """加载 ONNX 会话，绑定到 self.device_id 这张卡，并做显存/性能调优。"""
        # 防御：把 spec JSON 误当成模型传进来时，直接给明确提示，而不是让 ORT 报晦涩的
        # "Protobuf parsing failed"。--model 应指向 .onnx，spec 才是 .json（--spec）。
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在：{path}")
        if path.lower().endswith(".json"):
            raise ValueError(
                f"--model 拿到的是 spec JSON（{path}），不是 ONNX 模型。"
                f"模型应是 .onnx 文件（--model），spec 才是 .json（--spec）。")
        with open(path, "rb") as fh:
            head = fh.read(1)
        if head in (b"{", b"["):
            raise ValueError(
                f"--model 指向的文件是 JSON（{path}），不是 ONNX 模型。"
                f"请把 --model 改成 .onnx 模型路径，spec 用 --spec 单独传。")
        # 关掉 ORT 默认 logger 的 WARNING：模型里那些 ScatterND 提示纯属噪音。
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.add_session_config_entry("cudnn_conv_algo_search", "HEURISTIC")
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
        self.session = ort.InferenceSession(path, sess_options=options, providers=providers)
        self._pred_name = self.session.get_outputs()[0].name
        self._input_names = set(i.name for i in self.session.get_inputs())

    def forward(self, x, step, valid_time):
        feeds = {"input": x}
        if valid_time is not None:
            for k, v in _scalar_feeds(valid_time, step).items():
                if k in self._input_names:
                    feeds[k] = v
        # 输出完整滑动窗口 (1, in_frames, C, H, W)：输入 [t-1,t] → 输出 [t,t+1]，
        # 末帧才是本步预报。基类 rollout 会取 state[:, -1] 当预报、用完整 state
        # 回填（state = result），与官方 inference.py 的 `new_input = model.run(...)`
        # 一致。这里原样返回完整输出，不再只切末帧。
        return self.session.run([self._pred_name], feeds)[0]

    def describe(self):
        return f"providers={self.session.get_providers()}"
