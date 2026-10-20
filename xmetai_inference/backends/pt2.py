# -*- coding: utf-8 -*-
"""torch.export (.pt2) 后端。

通用 PT2 加载 + 一步前向。这里只做后端机制（export 加载、dtype/输入名推断、跑 module），
**不含任何模型特定归一化**——mean/std、tp 的 log1p/expm1 等全部由统一
Processor 管线处理。

热路径为 torch in / torch out：forward 直接接收并返回同一 device 上的 torch.Tensor，
不再每步做 numpy↔torch 转换。

潜空间 AR(1) 强迫（config 的 latent_forcing）：加载后对 fx 图做**内存改写**
（weights/spa_pt2.py，与 ONNX 版语义同构：σ 折进基、τ 在驱动侧、不落盘）。
xi 走 GraphModule 属性（spw_xi）而不是图输入——torch.export 的图签名固定，
塞不进新输入。驱动状态机与 onnx 后端共享 `backends/spa.py` 的
LatentStepStream。
"""
import logging
import os

import numpy as np

from .base import BaseInferModel

log = logging.getLogger(__name__)


class Pt2InferModel(BaseInferModel):
    """torch.export (.pt2) 后端。"""
    backend = "pt2"
    # 潜空间 AR(1) 强迫规格（config 的 latent_forcing）；None = 不加。
    # m1a 权重扰动（supports_weight_perturb）在本后端**不支持**——那条腿依赖
    # ONNX 图的逐张量改写；latent_forcing 走 fx 图改写，两者能力位分开声明。
    latent_forcing = None
    supports_latent_forcing = True

    def __init__(self, device_id=0, gpu_mem_fraction=0.7):
        super().__init__(device_id, gpu_mem_fraction)
        self._module = None
        self._input_names = []
        self._dtype = None
        self._device = None
        # 潜空间强迫：注入目标数（0 = 未启用；load() 里按 spec 设定）与驱动状态机。
        self._latent_n_targets = 0
        self._latent_stream = None
        # **真正在跑的那个** module 的引用（= ep.module() 的返回值，不是 ep.graph_module）
        # —— forward 每步用 copy_ 把 ξ 写进它的 spw_xi 缓冲。两者的关系见 load()。
        self._forcing_gm = None

    def load(self, path):
        import inspect
        from contextlib import nullcontext

        import torch

        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在：{path}")
        self._device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu")

        forcing = getattr(self, "latent_forcing", None)
        if forcing is not None:
            from xmetai_inference.backends.spa import LatentStepStream

            # load() 里重建（同 onnx 后端的纪律）：每个成员一份、从不共享。
            self._latent_stream = LatentStepStream()

        # 模型 device 已烘焙进导出图，要在目标卡上加载（否则可能落到 cuda:0）
        ctx = torch.cuda.device(self._device) if self._device.type == "cuda" else nullcontext()
        with ctx:
            ep = torch.export.load(path)
            # 潜空间 AR(1)：必须在 module() 之前改图（unlift 拿到的是注入后的图）。
            # tau 只在驱动侧，不进图 ⇒ 换 τ 复用同一份改写逻辑（同 ONNX 版的共图设计）。
            if forcing is not None:
                from xmetai_inference.weights.spa_pt2 import attach_forcing

                report = attach_forcing(ep, forcing, src_path=path)
                self._latent_n_targets = int(report.get("n_targets", 0))
                log.info("潜空间 AR(1) 强迫已启用（pt2）：%s member=%d -> %s",
                         forcing.describe(), getattr(self, "member_index", 0),
                         report.get("fingerprint", "-"))
            self._module = ep.module()
            # ⚠ xi 必须喂到 **self._module**（真正在跑的那个）上，而不是 ep.graph_module：
            # ep.module() 会走 unlift，用 deepcopy 出来的图**新建**一个 GraphModule。
            # spw_V/U/xi 因为是 get_attr 的目标会被一起带过去，但那是同一批张量对象的
            # **引用** —— 所以每步必须 copy_ 就地改写，用 setattr 换成新张量只会改到其中
            # 一个副本，另一个（真正被前向读的那个）还是全 0 ⇒ **静默退化成确定性预报**。
            # 2026-09-30 探针实测：graph_module 与 module() 是两个对象、但共享同一批张量。
            self._forcing_gm = self._module if self._latent_n_targets > 0 else None
            del ep

        # 输入名：torch.export 会把参数名变成 args_0/args_1，这里按规范顺序回填
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

    def _latent_step(self, step):
        """潜空间 AR(1) 的驱动状态机（与 onnx 后端共享 LatentStepStream）。"""
        forcing = getattr(self, "latent_forcing", None)
        if not forcing or self._latent_n_targets <= 0:
            return None
        if self._forcing_gm is None:
            # sigma=0 空转对照走不到这里（n_targets=0 早退）；走到这里说明注入了图
            # 却丢了 gm 引用 —— 喂不上 ξ 会静默退化成确定性预报，直接拦下。
            raise RuntimeError("latent_forcing 已注入但 GraphModule 引用丢失（spw_xi 喂不上）。")
        return self._latent_stream(
            forcing, step,
            n_targets=self._latent_n_targets,
            member_index=getattr(self, "member_index", 0),
            hour_interval=getattr(self, "hour_interval", None))

    def forward(self, x, step, valid_time):
        import torch

        if valid_time is None:
            raise ValueError("PT2 模型需要起报时间 init_time 来计算 hour/doy 时间条件")

        if isinstance(x, torch.Tensor):
            state = x
        else:
            state = torch.as_tensor(x)
        state = state.to(device=self._device, dtype=self._dtype)

        # hour 用官方 FuXi-2.1 的分钟精度归一化：(hour*60+minute)/1440，与官方
        # data_util.prepare_features 的 tod 一致；doy 同理。
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

        row = self._latent_step(step)
        if row is not None:
            if int(state.shape[0]) != 1:
                # pt2 的 ξ 走 GraphModule 的 **spw_xi 缓冲**（图属性，不是图输入），
                # 缓冲形状是 [T, R]、没有 batch 轴，塞不进 k 条轨迹的 ξ。这里必须炸：
                # 按旧接线（width 恒 1）跑下去块内 k 条轨迹会共用同一行，集合静默欠散
                # 且没有任何报错 —— 正是 LatentStepStream docstring 里那个危险方向。
                # onnx 后端已支持 width>1（喂 [k,T,R] 图输入），pt2 要支持得先给注入侧
                # 加 batch 轴。改之前：pt2 + latent_forcing 只能 model_batch=1。
                raise ValueError(
                    "pt2 后端的潜空间强迫（latent_forcing）不支持 model_batch>1："
                    "ξ 走 spw_xi 缓冲（[T, R]，无 batch 轴），一次 forward 只能喂一条轨迹。"
                    "请把 config 的 model_batch 设为 1（onnx 后端不受此限）。")
            # 本步第 row 条轨迹的 ξ（[1, T, R] float32）→ **就地**写进 [T, R] 的 spw_xi。
            # get_attr 在执行时按名字取当前张量，所以改的是值不是属性；就地改写（而不是
            # setattr 换张量）是必须的 —— 见 load() 里那段（unlift 之后有两个 GraphModule
            # 共享同一批张量对象，换张量只会改到其中一个）。
            xi = self._latent_stream.ar1.row(row)
            self._forcing_gm.spw_xi.copy_(
                torch.from_numpy(np.ascontiguousarray(xi[0])).to(
                    device=self._device, dtype=self._dtype))

        with torch.no_grad():
            result = self._module(*args)

        # 完整输出 (1,in_frames,C,H,W) → 保持 device 上的 torch tensor；
        # 末帧是本步预报，完整 state 用于 state=result 回填（与官方一致）。
        # 落盘时统一由 BaseInferModel.to_dataset 再做唯一一次 D2H。
        return result.float()

    def release(self):
        """丢掉 module 与潜空间驱动状态（换模型/进程收尾时调）。

        pt2 没有 onnx 那个逐成员换 session 的场景（latent-only 集合 = 一个 module +
        一批轨迹），这里主要是对称的清理口：module 释放后图/基/ξ 属性一起被 GC。
        """
        self._module = None
        self._forcing_gm = None
        if self._latent_stream is not None:
            self._latent_stream.release()

    def describe(self):
        text = f"pt2 (dtype={self._dtype}, inputs={self._input_names})"
        if self._latent_n_targets > 0:
            text += f" + latent-ar1({self._latent_n_targets} targets)"
        return text
