# 模型权重与部署产物

本目录只保存模型权重的说明文档，并随 `xmetai` Python 包一起发布。

实际模型权重和部署产物统一放在项目根目录的 `model_artifacts/` 中。该目录整体被
`.gitignore` 忽略，其中的任何文件和子目录都不会上传 Git。

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
| AIFS 1.1 | `aifs-single-mse-1.1.ckpt` | Anemoi 运行环境 |

FuXi-2.1 当前仍由统一 Processor 管线完成归一化、反归一化和 TP 变换，因此
`mean.nc`、`std.nc` 必须与模型一起保留。AIFS checkpoint 自带 Anemoi 的部分
normalizer、imputer 和自回归逻辑，不能在未验证数值一致性的情况下直接替换格式。

## 使用原则

1. 原始权重尽量保持不修改，导出结果使用新的文件名或目录。
2. 配置中的 `model_path` 应指向实际运行的模型文件。
3. 删除统计量或 external data 前，必须确认处理已经固化进模型且模型不再引用这些文件。
4. 权重格式转换后，应使用同一输入比较原模型和导出模型的输出误差。
5. 导出脚本应放在 `tools/` 或 `exporters/`，不要与二进制权重混放。
