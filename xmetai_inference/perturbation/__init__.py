# -*- coding: utf-8 -*-
"""确定性 ONNX -> 集合模型的权重扰动（SPW）能力。

一条公式 `W~ = W ⊙ (1 + σξ)`（Adamov et al. 2026, arXiv:2609.08412），两条互不相同的腿：

  * **weight_perturb — 逐成员冻结权重**（论文主配置，当前唯一机制）。按
    `W~ = W ⊙ (1 + σξ)` 直接算出 N 套权重落到磁盘，成员内整条轨迹**冻结**同一组权重；
    每个成员就是一份普通的确定性 onnx，推理零开销、可单独审计。见 weights.py。
    （历史上还有一条「图内随机算子」的路——在每个权重张量后插 RandomNormalLike -> Mul，
    每步重抽。它 2026-09-23 实测淘汰：spread/RMSE 随时效下降、CRPS 最差，只有 m1a 的
    1/3~1/8 离散度；代码已于 2026-09-24 删除，结论与教训留在 README 的「M3 定格」。）
  * **latent — 潜空间 AR(1) 强迫**（第二条腿：噪声的**时间相关**）。不改权重，只在图上加
    一个 `spw/xi` 输入，把 `z (x) (1 + sigma*U diag(xi_t) V^T)` 接到低分辨率的潜状态上；
    xi_t 由驱动侧的 AR(1) 过程给出（`xi_{t+1} = rho*xi_t + sqrt(1-rho^2)*eps_t`，
    rho = exp(-dt/tau)）。与 weight_perturb **正交、可叠加**；成员之间的差别只来自 xi 的
    种子，所以同一份图可以服务全部成员（不必落盘 51 份权重）。见 latent.py。
  * **noise_scale — 图内自带噪声的幅度**（第三条腿，只对**集合图**有意义：成员由
    图内随机算子给出的那种图）。`σ × k` 等价于把 `prior_perturb.logvar.bias` 整幅平移
    `2·ln k`（σ = exp(logvar/2)，图上实测），权重只动一个 initializer。与上两条正交。见
    noise_scale.py。

入口：
  * 配置面：configs/*.py 里的 `weight_perturb={...}`（见 spec.WeightPerturbSpec）
  * 命令行：
      python -m xmetai_inference.perturbation groups  <onnx>   # 看候选张量组
      python -m xmetai_inference.perturbation members <onnx>   # 落盘成员集
      python -m xmetai_inference.perturbation probe   <onnx>   # 探针：量离散度随风向
      python -m xmetai_inference.perturbation latent build <onnx> --tau-days 3 --length-scale 32
      python -m xmetai_inference.perturbation latent audit <onnx.ar1-xxxxxxxxxx.onnx>
      python -m xmetai_inference.perturbation check|self-test|e2e
  * 代码面：spec.WeightPerturbSpec / graph.resolve_effective_path / weights.build_member_set
    / segments.parse_segment_plan（分段 rollout：段边界换权重，config 的 spw_segments）
    / latent.LatentForcingSpec + latent.resolve_stack_path（潜空间 AR(1)，config 的
      latent_forcing）
  * 端到端预检：e2e.run_e2e —— 惰性导入（它会拉 runner/models/torch，而本包其余部分不需要）
  * 产物体检：enscheck.run_check（读真实的 member_XXX/ 输出目录）

详见包内 README.md。
"""
from .spec import DEFAULT_EXCLUDE, NOISE_STRUCTS, WeightPerturbSpec
from .graph import (resolve_effective_path, select_initializers)
from .latent import (
    Ar1Driver,
    LatentForcingSpec,
    LatentStepStream,
    audit_forcing,
    build_basis,
    build_forcing_model,
    default_forcing_path,
    format_forcing_audit,
    inject_forcing,
    lag1_autocorr,
    resolve_forcing_path,
    resolve_stack_path,
    resolve_targets,
)
from .noise_scale import (
    DEFAULT_FACTOR,
    DEFAULT_TENSOR,
    default_scaled_path,
    format_noise_scale_report,
    scale_builtin_noise,
    verify_structure as verify_noise_scale,
)
from .segments import Segment, SegmentPlan, parse_segment_plan
from .weights import (
    build_member_set,
    format_member_report,
    member_base_path,
    member_dir_for,
    member_exists,
    member_model_path,
    member_perturbed_path,
    member_tag,
    resolve_member_path,
    set_dir_for,
    verify_member_set,
)

__all__ = [
    "Ar1Driver",
    "DEFAULT_EXCLUDE",
    "DEFAULT_FACTOR",
    "DEFAULT_TENSOR",
    "LatentForcingSpec",
    "LatentStepStream",
    "NOISE_STRUCTS",
    "Segment",
    "SegmentPlan",
    "WeightPerturbSpec",
    "audit_forcing",
    "build_basis",
    "build_forcing_model",
    "build_member_set",
    "default_forcing_path",
    "default_scaled_path",
    "format_forcing_audit",
    "format_member_report",
    "format_noise_scale_report",
    "inject_forcing",
    "lag1_autocorr",
    "member_base_path",
    "member_dir_for",
    "member_exists",
    "member_model_path",
    "member_perturbed_path",
    "member_tag",
    "parse_segment_plan",
    "resolve_effective_path",
    "resolve_forcing_path",
    "resolve_member_path",
    "resolve_stack_path",
    "resolve_targets",
    "scale_builtin_noise",
    "select_initializers",
    "set_dir_for",
    "verify_member_set",
    "verify_noise_scale",
    "run_e2e",
    "run_check",
]


def __getattr__(name):
    """惰性入口：支持 from xmetai_inference.perturbation import run_e2e。

    e2e 会 import runner（torch）与 models，放在模块顶层会让「只想用图改写」的
    调用方也背上这些依赖，所以这里按需解析。
    """
    if name == "run_e2e":
        from .e2e import run_e2e

        return run_e2e
    if name == "run_check":
        from .enscheck import run_check

        return run_check
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
