# -*- coding: utf-8 -*-
"""风顺 ERA 1.5° 集合 ONNX 模型（源图 ``s2s_gssp_gdn2.onnx``，76 通道、日步长）。

与 Tongyun-CRA 同族（同为 GatedDeltaNet2 骨干 + 图内先验采样），区别在通道契约、
网格与步长。运行契约（2026-10-29 静态扫图 + store 对表定案）：

  * **76 通道**，顺序见 ``models.S2S_CHANNELS``：z/t/u/v/q × 13 层
    （**1000→50 降序**，与 fengshun2.0 的 50→1000 升序相反）+ 地面
    ``[t2m,d2m,sst,ttr,u10m,v10m,u100m,v100m,msl,tcwv,tp]``。与训练盘
    ``s2s.19500101_20260417.c76.1p5.h24.v1`` 的 channel 数组逐位一致。
  * **物理进 / 物理出**：图头 ``Sub(mean)→Div(std)``、图尾 ``Mul(std)→Add(mean)``，
    config 侧不配 normalize / denormalize。输入盘存的是归一化值，数据侧要配
    ``denormalize(stats_path=该 store)``。
  * **tp 不取 log1p**：该 store 里 tp 的 mean/std = 0/1，存的就是物理 mm，
    所以 ``denormalize`` **不要**配 ``expm1_channels``（与 fengshun2.0 相反）。
  * **网格** = ``GRID_1P5``（lat 90→−90、lon 0→358.5），与 store 原生朝向一致，
    不要配 ``geometry`` 处理器。
  * **时间编码喂归一化值**：``hour`` = 小时/24、``doy`` = 年积日/365（与 Tongyun-CRA
    同口径，走 backends/onnx.py 的 ``_scalar_feeds``；fengshun2.0 才喂原始值）。
    扫图佐证：图内把 hour 经 ``frac(hour+1)`` 还原成「一天内的比例」，把
    ``doy + floor(hour+1)/365`` 当作「年内的比例」——喂原始 0…23 会让前者恒 0、
    后者偏移最多 24 天。floor(hour+1) 恒为 1 也说明喂的是**输入窗末帧**的时刻
    （图内部自己 +1 天），故 ``valid_time_step_offset=0``。
  * **扰动旋钮**：图有 ``perturb_std_scale`` 标量输入（1.0
   = 训练时的标称噪声），
    由 ``_scalar_feeds`` 统一喂 1.0。图内的先验采样是逐帧 iid 的；SPA 是在
    decoder 入口 ``add_18518/523/528``（latent_mixer 三组 group_projs 出口 + 残差，
    各 [B,7200,512]）上再叠一条时间相关的 AR(1) 强迫。``cat_15``（三分支后验的
    Concat，先验头 prior_perturb 的入口）在 Tongyun 位置消融里效应差 ≥300 倍，不用。
  * 集合成员由图内 ``RandomNormalLike`` 自带，51 成员 = 51 次独立前向。

一次前向只前进 1 天：输入两帧 ``[t-1, t]``（每日 00Z），输出单帧
``(B,1,76,121,240)``，滑窗归 ``Rollout``（``windowed_output=False``）；
``gpu_state`` 必须 False，原因同 Tongyun-CRA —— GPU 常驻分支不经过
``_next_state`` 的滑窗，单帧输出会被当成完整状态回填。

自定义算子 ``GatedDeltaNet2Fn``（×42）走 config 的 ``ops_library``；cuDNN 卷积算法
搜索必须 EXHAUSTIVE。
"""
from xmetai_inference.backends.onnx import OnnxInferModel

from . import GRID_1P5, S2S_CHANNELS


class FengshunEra1p5Model(OnnxInferModel):
    model_name = "fengshun_era_1p5_onnx"
    input_channels = S2S_CHANNELS
    output_channels = S2S_CHANNELS
    # 北→南：与 s2s store 原生朝向一致（不是 Tongyun-CRA 的南→北）。
    grid = GRID_1P5
    history_steps = 2
    hour_interval = 24        # 日步长：一次前进一步 1 天
    forecast_type = "ensemble"
    members = 51
    # 图只出 1 帧，滑窗归 Rollout；与 gpu_state=False 绑定，见模块 docstring。
    windowed_output = False
    gpu_state = False
    # 时间条件喂输入窗末帧时刻；图内部自己 +1 天（见模块 docstring）。
    valid_time_step_offset = 0
