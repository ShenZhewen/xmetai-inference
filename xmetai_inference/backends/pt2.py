# -*- coding: utf-8 -*-
"""torch.export (.pt2) 后端：加载、dtype/输入名推断、一步前向（torch in / torch out）。

latent_forcing 在加载后改写 fx 图（weights/spa_pt2.py），ξ 走 GraphModule 的
``spw_xi`` 属性（export 图签名固定，塞不进新输入）。
"""
import logging
import os

import numpy as np

from .base import BaseInferModel

log = logging.getLogger(__name__)


class Pt2InferModel(BaseInferModel):
    """torch.export (.pt2) 后端。"""
    backend = "pt2"
    supports_latent_forcing = True       # 走 fx 图改写，不支持权重扰动

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        super().__init__(device_id, gpu_mem_fraction)
        self._module = None
        self._input_names = []
        self._dtype = None
        self._device = None
        # 真正在跑的 module（ep.module() 的返回值），forward 用 copy_ 写它的 spw_xi
        self._forcing_gm = None

    def load(self, path):
        import inspect
        from contextlib import nullcontext

        import torch

        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在：{path}")
        # 重新 load = 新的一批轨迹，丢掉旧的驱动状态（与旧版每次新建等价）
        self._latent_stream.release()
        self._device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")

        forcing = self.latent_forcing

        # 模型 device 已烘焙进导出图，必须在目标卡上加载
        ctx = torch.cuda.device(self._device) if self._device.type == "cuda" else nullcontext()
        with ctx:
            ep = torch.export.load(path)
            # 必须在 module() 之前改图
            if forcing is not None:
                from xmetai_inference.weights.spa_pt2 import attach_forcing

                report = attach_forcing(ep, forcing, src_path=path)
                self._latent_n_targets = int(report.get("n_targets", 0))
                log.info("潜空间 AR(1) 强迫已启用（pt2）：%s member=%d -> %s",
                         forcing.describe(), getattr(self, "member_index", 0),
                         report.get("fingerprint", "-"))
            self._module = ep.module()
            # ⚠ 必须用 copy_ 就地改写 spw_xi：ep.module() 新建的 GraphModule 与
            # graph_module 共享同一批张量对象，setattr 换新张量只改到其中一个，
            # 前向读到的仍是全 0，静默退化成确定性预报。
            self._forcing_gm = self._module if self._latent_n_targets > 0 else None
            del ep

        # torch.export 把参数名变成 args_0/args_1，按规范顺序回填
        sig_names = list(inspect.signature(self._module.forward).parameters.keys())
        if sig_names and sig_names[0].startswith("args_"):
            self._input_names = ["input", "step", "hour", "doy"][:len(sig_names)]
        else:
            self._input_names = sig_names

        for p in self._module.parameters():
            self._dtype = p.dtype
            break
        if self._dtype is None:
            self._dtype = torch.float32

        torch.cuda.empty_cache()
        return self

    def forward(self, x, step, valid_time):
        import torch

        if valid_time is None:
            raise ValueError("PT2 模型需要起报时间 init_time 来计算 hour/doy 时间条件")

        if isinstance(x, torch.Tensor):
            state = x
        else:
            state = torch.as_tensor(x)
        state = state.to(device=self._device, dtype=self._dtype)

        # 分钟精度归一化（与官方 prepare_features 的 tod 一致）
        times = [valid_time] if np.ndim(valid_time) == 0 else list(valid_time)
        tod = [(value.hour * 60 + value.minute) / 1440.0 for value in times]
        feats = {
            "step": torch.full((len(times),), step, device=self._device, dtype=self._dtype),
            "hour": torch.tensor(tod, device=self._device, dtype=self._dtype),
            "doy": torch.tensor(
                [min(365, value.day_of_year) / 365.0 for value in times],
                device=self._device,
                dtype=self._dtype,
            ),
        }
        args = [state]
        for name in ("step", "hour", "doy"):
            if name in self._input_names:
                args.append(feats[name])

        xi = self._latent_row(step)
        if xi is not None:
            if self._forcing_gm is None:
                # 注入成功却丢了 gm 引用：喂不上 ξ 会静默退化，直接拦下
                raise RuntimeError(
                    "latent_forcing 已注入但 GraphModule 引用丢失（spw_xi 喂不上）。")
            if int(state.shape[0]) != 1:
                # spw_xi 是 [T,R] 缓冲、没有 batch 轴，塞不进 k 条轨迹的 ξ；
                # 硬跑会让块内成员共用同一行、集合静默欠散。
                raise ValueError(
                    "pt2 后端的潜空间强迫（latent_forcing）不支持 model_batch>1："
                    "ξ 走 spw_xi 缓冲（[T, R]，无 batch 轴），一次 forward 只能喂一条轨迹。"
                    "请把 config 的 model_batch 设为 1（onnx 后端不受此限）。")
            # 就地 copy_（不能 setattr，理由见 load）
            self._forcing_gm.spw_xi.copy_(
                torch.from_numpy(np.ascontiguousarray(xi[0])).to(
                    device=self._device, dtype=self._dtype))

        with torch.no_grad():
            result = self._module(*args)

        return result.float()    # 保持 device 上的 torch tensor，落盘时才 D2H

    def release(self):
        """丢掉 module 与潜空间驱动状态（换模型/进程收尾时调）。"""
        self._module = None
        self._forcing_gm = None
        super().release()

    def describe(self):
        text = f"pt2 (dtype={self._dtype}, inputs={self._input_names})"
        if self._latent_n_targets > 0:
            text += f" + latent-ar1({self._latent_n_targets} targets)"
        return text
