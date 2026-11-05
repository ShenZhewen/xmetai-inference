# 模型权重、部署产物与 SPA 代码

本目录保存模型权重的说明文档，并随 `xmetai-inference` Python 包一起发布。

实际模型权重和部署产物统一放在项目根目录的 `model_artifacts/` 中。该目录整体被
`.gitignore` 忽略，其中的任何文件和子目录都不会上传 Git。

## 扰动与图手术代码（2026-10-20 起）

所有「改模型图 / 生成模型产物」的代码都收在本目录；`perturbation/` 只留推理运行时
（自检、端到端验证），离线分析（probe / check / spread / verdict / a(t) 审计 /
逐尺度谱）在 `analysis/`。

| 文件 | 内容 |
|------|------|
| `spa.py` | SPA（潜空间 AR(1)）ONNX 版：`LatentForcingSpec`（σ / τ / rank / length_scale / seed / amp_profile）、空间基 `build_basis`、ONNX 手术 `inject_forcing`、生成 `build_forcing_model`、审计 `audit_forcing` |
| `spa_pt2.py` | 同一套 SPA 语义的 PT2 版：在加载后的 fx 图上插节点，`attach_forcing` / `audit_fx_forcing` |
| `spw.py` | 权重扰动（SPW）：`WeightPerturbSpec`、选张量底座（`select_initializers` / `resolve_sigmas`）、逐成员冻结权重落盘（`build_member_set`）、候选组报告 `print_group_report`。图内噪声幅度（σ × k）能力已于 2026-10-20 移除（实测无收益，`docs/PAPER.md` §3.3.10） |
| `__init__.py` | 公开 API 的 re-export |

SPA 的**推理驱动**（ξ 递推 `Ar1Driver`、发号 `LatentStepStream`、诊断
`lag1_autocorr`）不在本目录：在 `backends/spa.py`，由 ONNX 与 PT2
后端共用。模型侧直接复用 `configs/weather_era_0p25_fgvp_ens.py` 的契约，不再有单独的
SPA 选择器类。

造产物没有命令行。改 `spw.py` / `spa.py` 末尾 `if __name__ == "__main__"` 里的字面量
（`ACTION`、`MODEL` 等），然后直接运行：

```bash
python -m xmetai_inference.weights.spw   # ACTION = groups / members
python -m xmetai_inference.weights.spa   # ACTION = build / audit
```

SPW 与 SPA 正交、可叠加（`spa.resolve_stack_path`）。

## 可以存放的内容

- 原作者发布的 `.pt2`、`.ckpt`、`.onnx` 等原始权重。
- 从原始权重导出的项目运行格式。
- 模型需要的 `mean.nc`、`std.nc` 等统计量。
- ONNX external data 和自定义算子库。
- 输入输出通道、形状、数据类型等模型契约说明。
- 权重校验和、导出环境和数值一致性验证结果。
- 小型测试输入输出；不建议保存几百 MB 的完整示例数据。

如果需要同时保留原始权重和部署权重，可以按下面的方式组织：

```text
model_artifacts/
└── <model_name>/
    ├── source/       # 原作者发布的权重，保持不修改
    └── runtime/      # 项目实际加载的导出模型及其配套文件
```

当前没有导出需求时，可以直接沿用已有模型子目录，不必强制增加 `source/` 和
`runtime/`。

## 导出模型

导出不是修改模型参数，而是在原模型外包装固定的 Tensor 处理，再保存为项目支持的
PT2、ONNX 或其他部署格式。例如：

```text
物理量 Tensor
→ 通道排序
→ normalize / log1p
→ 原始模型
→ denormalize / expm1
→ 物理量 Tensor
```

适合在导出时固化的处理包括：

- 固定通道顺序；
- 归一化和反归一化；
- TP 的 `log1p` 和 `expm1`；
- 固定的 dtype 转换；
- 输出非负截断等确定性 Tensor 操作。

Loader、NetCDF/Zarr 读取、网格适配、日期调度、多卡分配和结果写盘仍应由推理框架负责，
不应放入模型权重。

## 当前模型

| 模型 | 运行权重 | 配套文件 |
|------|----------|----------|
| FuXi-Ens | `fuxi_ens.onnx` | ONNX external data |
| FuXi-2.1 | `fuxi-2.1.pt2` | `mean.nc`、`std.nc` |
| FGVP | ONNX 模型 | 与运行环境匹配的自定义算子库 |
| FengQing V1.5Beta | `fengqing_pre.onnx` | `mean_std/`、`utils/constant_masks.npy` |
| Pangu-Weather | `pangu_weather_6.onnx` + `pangu_weather_24.onnx` | 无（归一化烘焙进图）；BY-NC-SA 4.0，商用禁止 |

FuXi-2.1 当前仍由统一 Processor 管线完成归一化、反归一化和 TP 变换，因此
`mean.nc`、`std.nc` 必须与模型一起保留。

## 使用原则

1. 原始权重尽量保持不修改，导出结果使用新的文件名或目录。
2. 配置中的 `model_path` 应指向实际运行的模型文件。
3. 删除统计量或 external data 前，必须确认处理已经固化进模型且模型不再引用这些文件。
4. 权重格式转换后，应使用同一输入比较原模型和导出模型的输出误差。
5. 扰动与图改写代码（`weights/spa.py`、`spa_pt2.py`、`spw.py`）刻意和权重说明放在同一目录，方便随部署产物一起发布；其他导出脚本仍建议放在 `tools/` 或 `exporters/`。
