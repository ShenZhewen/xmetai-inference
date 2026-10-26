---
name: xmetai-inference
description: 往 xmetai-inference 推理框架里接入新模型或新数据源，或排查「跑起来了但输出像垃圾 / 单位不对 / 归一化错位 / 集合没 spread」时的路由总章。SKILL.md 只负责分类与路由，具体内容按任务加载对应 reference。接入新模型、写新 config、加 data-side/模型侧 processor、接一组新 zarr store 时使用。
metadata:
  short-description: xmetai-inference 模型/数据源接入指南
---

# xmetai-inference：新增模型 / 数据源

这是**路由总章**，不写细节。先分类，再按下表加载对应的 reference。

## 用法三步

1. 定位仓库并读本地规则：`README.md`、`AGENTS.md`（若有）。
2. 分类任务（见 Reference Routing）。
3. **只加载匹配的那一份 reference 跟着做**，不把多份同时灌进来。

## 铁律（每次开工必读）

- 动笔前先把要跑的确切命令列给用户、由用户决定何时在哪个环境跑**自己不改状态**，
  尤其不擅自跑 GPU / 训练 / 长耗时脚本（远程服务器环境，我够不着）。
- **Zarr 写入/改动需用户明确批准**，优先只读检查与 dry-run。
- 非平凡任务先拆步骤对齐再动手；能为小需求做的事，别做成架构。
- 配置项用字面量默认值，不引入环境变量条件表达式（可调项写进 config 文件）。
- 不 claim 跑过 train/export/deploy，除非真的跑过。
- 上报最窄但有意义的验证：config 加载 / 契约检查 / 数据 schema 抽样 / ONNX IO 检查。

## Reference Routing

| 要做什么 | 加载 |
|---|---|
| 动笔写模型前，向模型方确认契约（通道序/单位/归一化位置/集合噪声/时间口径） | `references/model-contracts.md` |
| 接新模型的**落地步骤**（模型类/注册表/配方/单测/集成档/文档表/processor 扩展） | `references/model-landing.md` |
| 接**新数据源**(store)，判断原始态 vs 归一化态、配 processor、南→北网格 | `references/datasets.md` |

## 分文件与分工

- `references/model-contracts.md` —— **先问后写**：接口契约清单 A–J + 可直接复制发出的提问问卷。
  顺序上它排第一：契约没问清就写代码，多半写完才发现通道顺序或单位错了，这两类错误
  **不抛异常**，表现为「跑通了但输出是垃圾」。
- `references/model-landing.md` —— **怎么写**：路径选择、权重落位、模型类、`MODEL_REGISTRY`、
  `configs` 配方、单测、集成档、文档表格、新增 processor（数据侧 `TensorProcessor` / 模型侧 `Processor`）。
- `references/datasets.md` —— **数据源本身**：两种 store 形态（原始态 vs 归一化态）、
  `denormalize` 数据侧还原、南→北存储配 `GRID_025_S` 且不配 `geometry`、link 到一个新 store 的小问卷。

> 数据侧 `denormalize`、`GRID_025_S`、`tongyun_cra.py` 是本 skill 在 2026-10
> 接 CRA 归一化态 store 时新增的产物；`references/datasets.md` 是它们对应的心得归档。