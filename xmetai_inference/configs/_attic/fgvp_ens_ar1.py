# -*- coding: utf-8 -*-
r"""臂 C：只用 AR(1) 潜空间强迫（时间相关），图内噪声保持原幅度 —— 共用体见 fgvp_ens_pert.py。

这是**单变量对照**臂，用来把「时间相关」从「幅度」里剥出来：与 `fgvp_ens_pert`（裸跑对照）
比，唯一的差别是 ξ。当初的另一个对照 `fgvp_ens_k20`（图内噪声 σ×2）已随 noise-scale
能力于 2026-10-20 移除（该臂判负，见 docs/PAPER.md §3.3.10）。

与 gdn2 上的 A32 主臂同配方（sigma 0.04 / tau 3d / rank 16 / iscale 32），只换了锚点：
`cat_9`（本地已实测能解析，tokens=16200, grid=(90,180)）。**锚点不换就白跑** ——
这张图的节点输出名被导出器拍平，`latent_state` 别名解析不到。

跑之前确认一件事（config 里已设，改代码时别动）：`perturb_graphs=0` 且无 weight_perturb。
图外层编舞会把图号写进 member_index，而 member_index 参与 ξ 的种子 —— 51 个成员会塌成
8 条 ξ 流。第一次跑必须绕开它。

    cd /workspace/szwCode/xmetai-inference2
    xmetai-infer --model fgvp_ens_ar1 --cuda-devices 0,1 --out /tmp/smoke_ar1   # 先 1 起报看审计
    xmetai-infer --model fgvp_ens_ar1

跑完用审计确认 ξ 真的注进去了（**这条不能省**：注不进去的图和裸跑逐位相同，指标上看着
像「AR(1) 没用」，实际是没生效）：

    改 xmetai_inference/weights/spa.py 末尾：ACTION="audit"、
    MODEL="/workspace/data/shenzw/model_onnx/<加载时打印的那张 .ar1-*.onnx>"，
    然后 python -m xmetai_inference.weights.spa

审计要看到：输入里有 spw/xi、目标节点确实是 cat_9、以及 xi 的 lag-1 自相关 ≈ exp(-6/72)=0.92
（若给的是白噪声，自相关会掉到 0，说明驱动侧没接上）。
"""
from xmetai_inference.tmp_config.fgvp_ens_pert import build

# rollout_batch=1 是**显式锁死**的：第 8 轮那条归档就是在 rb=1 下出的（当时 build 里写死了
# "开强迫必须降 1"）。宽度接通后缺省已变成 6，两者逐位等价（fgvp_ens_pert.build docstring /
# tmp/ar1_row_replay.py 第 4 条），但已出的归档要能按原设置复现，所以这里钉住 1。
cfg = build("fgvp_ens_ar1", ar1=True, rollout_batch=1)
