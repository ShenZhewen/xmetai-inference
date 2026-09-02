# -*- coding: utf-8 -*-
"""checkpoint (.ckpt) 后端：包一层 anemoi-inference 的 SimpleRunner。

AIFS 这类 GNN 把归一化 / 插值 / 边界处理都烘焙进 checkpoint，推理用官方
SimpleRunner（anemoi-inference 底层 API），它本身就是一个自回归 generator（内部含
imputer / normalizer / bounding）。所以本后端**整体覆盖 run()**，不实现单步 tensor
forward：forward 留成显式 raise（框架约定见 base.py：自带循环的后端覆盖 run、
不碰 forward）。

本模型使用「命名 field 字典（N320 非结构化节点）」。状态表示由模型类契约选择，
而不是由 ckpt 后端类型隐式决定；to_dataset 再把结果统一成 Dataset。
"""
import inspect

import numpy as np

from . import BaseInferModel


class CkptInferModel(BaseInferModel):
    """checkpoint 后端：SimpleRunner 薄封装。load = 建 runner；run = 逐 step yield state。"""
    backend = "ckpt"

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        super().__init__(device_id, gpu_mem_fraction)
        self._runner = None

    def load(self, path):
        """path：本地 .ckpt 路径（或 HF dict）。"""
        from anemoi.inference.runners.simple import SimpleRunner

        self._runner = SimpleRunner(path, device=f"cuda:{self.device_id}")
        return self

    def forward(self, x, step, valid_time):
        raise NotImplementedError(
            "ckpt 后端自带自回归循环（SimpleRunner），走 run()，不支持单步 forward")

    def run(self, state, steps, members=1, hour_interval=6, init_time=None,
            member_start=0, member_stride=1, on_step=None, log_step=False,
            progress=True, progress_label="", recurrent_transform=None,
            output_transform=None):
        """逐 step 产出 state dict。与基类 run() 同参名、但状态表示不同：

        state = 统一 adapter 管线产出的 field 字典 {"date", "fields"}；
        lead_time = steps * hour_interval（总预报时长，小时），SimpleRunner 每
        hour_interval 小时 yield 一个 state，共 steps 个（hour_interval 必须与
        checkpoint 原生步长一致，AIFS 1.1 = 6h）；
        确定性模型 members 恒 1，member_start / member_stride 无意义（忽略）。

        返回 (outs, [0])：outs 为 state dict 列表（非流式）或 None（传 on_step 时
        流式回调 on_step(i, state_dict)）。
        """
        lead = steps * hour_interval
        outs = [] if on_step is None else None
        for i, s in enumerate(self._iter_states(state, lead_time=lead)):
            if recurrent_transform is not None:
                s = recurrent_transform(s)
            output = output_transform(s) if output_transform is not None else s
            if on_step is not None:
                on_step(i, output)
            else:
                outs.append(output)
        return outs, [0]

    def _iter_states(self, input_state, lead_time):
        """兼容 anemoi-inference 版本差异：老版(0.6.x) input_state= / 新版 input_states=。"""
        sig = inspect.signature(self._runner.run)
        if "input_states" in sig.parameters:
            gen = self._runner.run(input_states=input_state, lead_time=lead_time,
                                   return_numpy=True)
        else:
            gen = self._runner.run(input_state=input_state, lead_time=lead_time)
        yield from gen

    def to_dataset(self, step_state, save_names=None, lat=None, lon=None):
        """field dict（N320 节点）→ Dataset。节点用 "node" 维 + lat/lon 坐标变量表示。

        AIFS 输出已是物理单位（归一化烘焙在 checkpoint，SimpleRunner 内部反算），
        这里不做单位换算，原样落盘。N320 是非结构化节点，用一维 "node" 维 + lat/lon
        作为坐标变量（非规整经纬网格），与官方 N320 输出一致；若要规整 0.25° 网格
        需再插值（后续按需加）。
        """
        import xarray as xr

        fields = step_state["fields"]
        if save_names is None:
            names = list(fields.keys())
        else:
            names = [n for n in save_names if n in fields]
        if lat is None:
            lat = np.asarray(step_state.get("latitudes"))
        if lon is None:
            lon = np.asarray(step_state.get("longitudes"))
        data_vars = {}
        for n in names:
            data_vars[n] = (("node",), np.asarray(fields[n], dtype=np.float32).reshape(-1))
        coords = {}
        if lat is not None:
            coords["lat"] = ("node", np.asarray(lat).reshape(-1))
        if lon is not None:
            coords["lon"] = ("node", np.asarray(lon).reshape(-1))
        return xr.Dataset(data_vars, coords=coords)

    def describe(self):
        return f"ckpt (anemoi SimpleRunner, cuda:{self.device_id})"
