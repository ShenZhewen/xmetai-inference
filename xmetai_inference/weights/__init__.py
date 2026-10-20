# -*- coding: utf-8 -*-
"""模型权重与扰动代码（SPW / SPA）。

2026-10-20 起重构为：

  * `spa.py`      潜空间 AR(1) 强迫（SPA）的 ONNX 版：规格 / 图改写 / 生成 / 审计；
  * `spa_pt2.py`  同一套 SPA 语义的 PT2（torch.export）版图改写；
  * `spw.py`      权重扰动（SPW）：规格 / 选张量 / 逐成员冻结权重落盘。

SPA 的**推理驱动**（Ar1Driver / LatentStepStream）不在本包：它在
`backends/spa.py`，由 ONNX 与 PT2 后端共用。

图内噪声幅度（σ × k，原 `noise_scale.py`）已于 2026-10-20 移除——实测无收益
（`docs/PAPER.md` §3.3.10 的 `ens_k20`）。

造产物没有命令行：改 `spw.py` / `spa.py` 末尾的 `if __name__ == "__main__"`
里的字面量，直接 `python -m xmetai_inference.weights.spw`（或 `.spa`）。
推理、探针、产物检查与评测分析仍在 `perturbation/`。
"""
from .spa import (
    DEFAULT_LENGTH_SCALE,
    FORCING_FORMAT,
    FORCING_INPUT,
    LatentForcingSpec,
    NODE_PREFIX,
    TARGET_ALIASES,
    audit_forcing,
    build_basis,
    build_forcing_model,
    default_forcing_path,
    detect_layout,
    format_forcing_audit,
    inject_forcing,
    resolve_forcing_path,
    resolve_stack_path,
    resolve_targets,
)
from .spa_pt2 import (
    attach_forcing,
    audit_fx_forcing,
    detect_fx_layout,
    resolve_fx_targets,
)
from .spw import (
    DEFAULT_EXCLUDE,
    NOISE_STRUCTS,
    WeightPerturbSpec,
    build_member_set,
    format_member_report,
    member_base_path,
    member_dir_for,
    member_exists,
    member_model_path,
    member_perturbed_path,
    member_tag,
    print_group_report,
    resolve_effective_path,
    resolve_member_path,
    select_initializers,
    set_dir_for,
    verify_member_set,
)

__all__ = [
    "DEFAULT_EXCLUDE",
    "DEFAULT_LENGTH_SCALE",
    "FORCING_FORMAT",
    "FORCING_INPUT",
    "LatentForcingSpec",
    "NODE_PREFIX",
    "NOISE_STRUCTS",
    "TARGET_ALIASES",
    "WeightPerturbSpec",
    "attach_forcing",
    "audit_forcing",
    "audit_fx_forcing",
    "build_basis",
    "build_forcing_model",
    "build_member_set",
    "default_forcing_path",
    "detect_fx_layout",
    "detect_layout",
    "format_forcing_audit",
    "format_member_report",
    "inject_forcing",
    "member_base_path",
    "member_dir_for",
    "member_exists",
    "member_model_path",
    "member_perturbed_path",
    "member_tag",
    "print_group_report",
    "resolve_effective_path",
    "resolve_forcing_path",
    "resolve_fx_targets",
    "resolve_member_path",
    "resolve_stack_path",
    "resolve_targets",
    "select_initializers",
    "set_dir_for",
    "verify_member_set",
]
