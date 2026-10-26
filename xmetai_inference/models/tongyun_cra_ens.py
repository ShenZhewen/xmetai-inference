# -*- coding: utf-8 -*-
"""Tongyun-Ens（CRA 版）ONNX 模型。

与 ERA5 版 tongyun_ens（models/iwc_fgvp_mamba_ens.py + configs/fgvp_ens_spa.py）
同族、同集合语义，区别只在通道契约与数据源：

  * **71 通道**：z/t/u/v/q × 13 层（50→1000 升序）+ 地面 [t2m,q2m,u10m,v10m,msl,tp]，
    见 models.CRA_CHANNELS。CRA store 的 channel 顺序即此顺序。
  * **图内自带归一化与反归一化**：图头 `Div(Sub(x, mean), std)`，图尾
    `Add(Mul(x, std), mean)`，常量与 store 的 mean/std 逐位相同。所以：
      - 输入必须**反归一化**成物理量（store 存的是归一化值）→ config 的
        processors 里配 `denormalize`（stats_path = 该 store）；
      - 输出**已经是物理量**，落盘不用再处理。
    tp 在图内走 log1p 空间，反归一化后得到的是 ``log1p(物理值)``，正是图期望的
    输入表示，不要再 expm1。
  * **纬度朝向是 store 原生的南→北**：图就照着 CRA store 练的，输入与输出都
    保持 −90→90。config 的 processors 里**不要配 geometry**——那个处理器看到
    store 的 lat 在南→北就无条件翻成北→南（FuXi/ERA5 系的约定），对这张图会
    把场喂反。2026-07-01 00Z step0 实测 +6h：不翻转 z500 RMSE 90，翻转 486。
    输出 .nc 的 lat 坐标跟着本类的 `grid`（GRID_025_S）写，天然一致。
  * **通道重排在图内**：图内部顺序是 [tp, 其余]，两侧各有 `Gather` 还原成 store
    顺序；框架按 store 顺序喂、按 store 顺序取即可。
  * **扰动旋钮**：图有 `perturb_std_scale` 标量输入（1.0 = 训练时的标称噪声），
    由 backends/onnx.py 的 _scalar_feeds 统一喂 1.0。ERA5 版没有这个输入。
  * 集合成员由图内 RandomNormalLike 自带，51 成员 = 51 次独立前向。

单步输出（图只出预报那一帧），故 windowed_output=False，由 Rollout._next_state
补滑窗；gpu_state 必须 False，原因同 iwc_fgvp_mamba_ens —— GPU 常驻分支不经过
_next_state 的滑窗，单帧输出会被当成完整状态回填。

自定义算子（GatedDeltaNet2Fn）走 config 的 ops_library；cuDNN 卷积算法搜索必须
EXHAUSTIVE（HEURISTIC 在 stem 的 stride-8 卷积上会失败）。
"""
from xmetai_inference.backends.onnx import OnnxInferModel

from . import CRA_CHANNELS, GRID_025_S


class TongyunCraEnsModel(OnnxInferModel):
    model_name = "tongyun_cra_ens_onnx"
    input_channels = CRA_CHANNELS
    output_channels = CRA_CHANNELS
    # 南→北：图就在这个朝向上训练（图内 mean/std 逐位取自 CRA store），
    # 不是 FuXi 系的北→南。见 GRID_025_S 的注释与模块 docstring。
    grid = GRID_025_S
    history_steps = 2
    hour_interval = 6
    forecast_type = "ensemble"
    members = 51
    # 图只出 1 帧，滑窗归 Rollout；与 gpu_state=False 绑定，见模块 docstring。
    windowed_output = False
    gpu_state = False