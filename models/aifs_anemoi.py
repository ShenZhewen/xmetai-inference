# -*- coding: utf-8 -*-
"""AIFS 1.1 推理后端封装：包一层 anemoi-inference 的 SimpleRunner。

框架里其它模型（fuxi_ens/fuxi2.1）走 BaseInferModel（onnx/pt2 后端 + z-score 反归一化），
AIFS 不同：它是 GNN，归一化/反归一化都烘焙在 checkpoint 里，推理用官方 SimpleRunner
（anemoi-inference 的底层 API），喂它「已插值到 N320 的输入 state」即可，模型内部自己
imputer/normalizer/bounding。所以这里不继承 BaseInferModel，只做两件小事：

  1. 兼容 anemoi-inference 版本差异：老版(0.6.x) run(input_state=...)、
     新版 run(input_states=...)；
  2. 把每条返回的 state（dict：fields/date/latitudes/longitudes/step）原样交给调用方。

依赖（服务器环境已有）：anemoi-inference、anemoi-models、torch-geometric、flash_attn。
"""
import inspect


class AifsAnemoiModel:
    """SimpleRunner 的薄封装。checkpoint: 本地 .ckpt 路径（或 HF dict）。"""

    def __init__(self, checkpoint, device="cuda", **runner_kwargs):
        from anemoi.inference.runners.simple import SimpleRunner

        self._runner = SimpleRunner(checkpoint, device=device, **runner_kwargs)
        self.device = device

    def run(self, input_state, lead_time=6, return_numpy=True):
        """逐 step 产出 state dict。lead_time 单位小时（int）或 timedelta。

        返回的 state：{date, fields:{变量名: (N320,) 或 (time,N320)}, latitudes,
        longitudes, step, previous_step}。单数据集 checkpoint 下直接是单个 state
        （不是 {dataset: state}）。
        """
        sig = inspect.signature(self._runner.run)
        if "input_states" in sig.parameters:
            gen = self._runner.run(input_states=input_state, lead_time=lead_time,
                                   return_numpy=return_numpy)
        else:
            gen = self._runner.run(input_state=input_state, lead_time=lead_time)
        yield from gen

    @property
    def runner(self):
        return self._runner
