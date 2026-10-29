# -*- coding: utf-8 -*-
"""fengshun2.0 集合 ONNX 模型（87 通道、日步长、图内自带归一化）。

运行契约（2026-10-28 静态反推 + store 对表 + 10 步 rollout 复现定案）：

  * **87 通道**，顺序见 ``models.FENGSHUN_CHANNELS``：z/t/u/v/q × 13 层
    （50→1000 升序）+ 地面 ``[msl,t2m,t2m_min,t2m_max,d2m,sst,ws10m,ws100m,
    u10m,v10m,u100m,v100m,lcc,mcc,hcc,tcc,ssr,ssrd,fdir,ttr,tcw,tp]``。
    与输入盘 ``cra.1979_2024.c87.p25.h24.v1`` 的 channel 数组逐位一致
    （ERA5 的 ``era5.1959_2025.c87.p25.h24.v1`` 同序同统计量，见 config 注释）。
  * **物理进 / 物理出**：图头 ``Sub(mean)→Div(std)``，图尾 ``Mul(std)→Add(mean)``，
    所以 config 侧**不配 normalize / denormalize**。输入盘的 q 已是 g/kg、
    辐射已是 W/m²（不 ×6）、tp 是 log1p(累积 mm)，故数据侧只配
    ``denormalize(expm1_channels=["tp"])`` 把归一化态还原成物理量。
  * **tp 是纯输出通道**：模型内部把输入 tp 自然置 0（2026-10-28 用户确认），
    输入喂什么值都不改变结果。输出 tp 的量级按 1 小时累积读（≈0.1 mm）。
  * **一次前向只前进 1 天**，输入两帧 ``[t-1, t]``（每日 00Z），输出单帧
    ``(1,1,87,721,1440)``，滑窗归 ``Rollout``（``windowed_output=False``）。
    ``step`` 基本惰性；第三方的 ``member/10`` = 第 10 步，不是「一步到第 10 天」。
  * **时间编码喂原始值**：``step``/``hour``/``doy`` 走同一个 transformer 式正弦
    位置编码，图里没有 /24、/365。框架 ``_scalar_feeds`` 的 ``hour/24``、
    ``doy/365`` 是 Tongyun CRA 口径，**不能复用**，所以本类自己实现 forward。
  * **网格** = ``GRID_025``（lat 90→−90、lon 0→359.75），与 store 原生朝向一致，
    不要配 ``geometry`` 处理器。
  * **集合成员由图内 ``RandomNormalLike`` 自带**：一次前向 = 一个成员，51 成员 =
    51 次独立前向。成员之间没有可复现的种子，两次运行不会逐位相同。
  * 自定义算子 ``xmetai_plugins::Mamba3SisoFn`` ×6 只有 CUDA 实现，config 的
    ``ops_library`` 必须指向与 onnxruntime 1.24 ABI 匹配的 .so。
"""
from __future__ import annotations

import numpy as np

from xmetai_inference.backends.onnx import OnnxInferModel, _as_numpy_onnx

from . import FENGSHUN_CHANNELS, GRID_025


def _hour_of(value):
    return float(getattr(value, "hour", 0.0))


def _doy_of(value):
    """day-of-year：pandas Timestamp 有 day_of_year / dayofyear，datetime 走 timetuple。"""
    for attr in ("day_of_year", "dayofyear"):
        day = getattr(value, attr, None)
        if day is not None:
            return float(day)
    return float(value.timetuple().tm_yday)


class Fengshun20OnnxModel(OnnxInferModel):
    model_name = "fengshun20_onnx"
    input_channels = FENGSHUN_CHANNELS
    output_channels = FENGSHUN_CHANNELS
    grid = GRID_025
    history_steps = 2
    hour_interval = 24        # 日步长：一次前进一步 1 天
    forecast_type = "ensemble"
    members = 11
    # 图只出预报那一帧，滑窗归 Rollout；与 gpu_state=False 绑定。
    windowed_output = False
    gpu_state = False
    # 时间条件按 valid_time 原值喂（offset=0 ⇒ valid_time = init + step*24h）。
    valid_time_step_offset = 0
    # forward 自己拼 feeds、不走 backends/onnx.py 的 _scalar_feeds，所以潜空间强迫
    # 要在这里自己接（见 forward 末尾）。打开能力位后，配了 latent_forcing 的 config
    # 才会把 ``spw/xi`` 真的喂进图；不配时行为与原来逐位相同。
    supports_latent_forcing = True

    def forward(self, x, step, valid_time):
        """一步前向：物理量进 / 物理量出，只出预报那一帧。

        x: (1, 2, 87, H, W) float32；返回 (1, 1, 87, H, W)。
        """
        feeds = {"input": _as_numpy_onnx(x)}
        if valid_time is not None:
            times = [valid_time] if np.ndim(valid_time) == 0 else list(valid_time)
            feeds["step"] = np.full((len(times),), step, dtype=np.float32)
            feeds["hour"] = np.asarray(
                [_hour_of(value) for value in times], dtype=np.float32)
            feeds["doy"] = np.asarray(
                [_doy_of(value) for value in times], dtype=np.float32)
        # 图签名里没有的键会被 drop（本模型只有 input/step/hour/doy；
        # 改写过的强迫图多一个 spw/xi）。
        feeds = {key: value for key, value in feeds.items() if key in self._input_names}
        # 潜空间 AR(1) 强迫（SPA）：未配 latent_forcing 时 row 为 None，行为不变。
        # width = 本次堆了几条轨迹；fengshun 的 dataloader batch_size=1 ⇒ 恒 1。
        width = int(x.shape[0])
        row = self._latent_step_stream(step, width=width)
        if row is not None:
            feeds[self.latent_forcing.input_name] = self._latent_stream.ar1.row(row, width)
        return self.session.run([self._pred_name], feeds)[0]
