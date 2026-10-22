"""AR(1) a(t) 臂 A：温和线性包络，1.0x → 3.0x（第 14 轮）。

与 s04 的**唯一**差别是 ``latent_forcing["amp_profile"]``（时变幅度 a(t)）。
图和 s04 **逐字节共用同一份 .onnx**：amp_profile 不进 budgets()、不参与图指纹，
理由与 tau_days 完全相同（见 weights/spa.py 的 budgets 文档）。所以本臂
**不需要重新导出 85 MB 的图**，只是驱动侧多乘一个标量。

---- a(t) 要解决什么 ----

σ 是一个常数，但"让集合校准所需的有效 σ"随时效单调上涨。07-02 实测（脚本
tmp/at_profile.py，用 s00/s04/s08 三条档的 rmse/spread 反解 σ*(t)）：

    lead_h     6     24     48     72    120    168    240    360
    σ*/0.04  1.76   2.62   2.34   2.36   2.68   3.21   4.74   6.13

中段平坦、尾端上翘，**跨时效要涨 3.5 倍**，而 σ 只给得出一个数 ——
这不是调参没调好，是**缺一个自由度**。

另外 σ 的边际收益在长时效递减：响应指数 p 在 6–120h ≈ 1.0，到 360h 掉到 0.75
（每单位 σ 换到的离散度越来越少），所以尾端的补偿必须比线性更陡。

---- 本臂取保守版，为什么 ----

上面那张表是**从 07-02 自己的 rmse 反推的**，拿它去评 07-02 是循环论证 ⇒
它只用来定**形状的量级**，绝对水平不追（aB 才去追表本身）。

aA 从 1.0 线性升到 3.0：step 0 处就是 s04 本身，所以**短时效与 s04 逐位相同**
（s04 在 6~48h 是当前九条臂里最好的），只有长时效才逐渐加力。
step 是 0 起的模式步、一步 6h，60 步 = 360h。
"""
from copy import deepcopy

from xmetai_inference.tmp_config.fgvp_ens_pert import OUT_ROOT
from xmetai_inference.tmp_config.fgvp_ens_ar1state_s04 import cfg as _main

cfg = deepcopy(_main)
cfg.name = "fgvp_ens_ar1state_aA"
cfg.output_dir = f"{OUT_ROOT}/{cfg.name}"
# (step, 倍数) 折线，分段线性、两端外推取端点值。改这里就是改包络。
cfg.latent_forcing["amp_profile"] = ((0, 1.0), (60, 3.0))
