# 气象模型推理框架

把 ERA5 再分析数据转换成模型输入，执行自回归天气预报并将结果保存为 NetCDF。

整套框架遵循「数据源可插拔 + Processing 管线可配置 + 后端可插拔」：模型类只声明
固定输入契约（通道、网格、时间窗口），每份运行配置完整声明数据集、输入前处理、
自回归回填和输出处理。

## 目录

- [支持的模型](#支持的模型)
- [核心特性](#核心特性)
- [目录结构](#目录结构)
- [架构与数据流](#架构与数据流)
- [快速开始](#快速开始)
- [配置参考](#配置参考)
- [扩展自己的模型和数据集](#扩展自己的模型和数据集)
- [多卡运行](#多卡运行)
- [集合预报（权重扰动）](#集合预报权重扰动)
- [源码评测工具](#源码评测工具)
- [模型契约与单位](#模型契约与单位)
- [权重管理](#权重管理)
- [已知限制与注意事项](#已知限制与注意事项)
- [License](#license)

## 支持的模型

| 模型 | 后端 | 类型 | 成员 | 通道 | 归一化 |
|------|------|------|------|------|--------|
| FuXi-Ens | `onnx`（ONNX Runtime） | 集合 `ensemble` | 51 | 78 | 已烘焙进图，输入/输出均为物理量 |
| FuXi-2.1 | `pt2`（torch.export） | 确定性 `deterministic` | 1 | 85 | 输入侧 `normalize`、输出侧 `denormalize`（mean.nc/std.nc） |
| IWC FGVP GDN2 | `onnx`（ONNX Runtime） | 确定性 `deterministic` | 1 | 78 | 已烘焙进图；骨干用自定义算子，需注册 `.so` |
| FengQing V1.5Beta | `onnx`（ONNX Runtime） | 确定性 `deterministic` | 1 | 70 | 输入 `normalize_fengqing`、输出 `denormalize_fengqing`（图出残差） |
| Pangu-Weather | `onnx`（ONNX Runtime） | 确定性 `deterministic` | 1 | 69 | 已烘焙进图，输入/输出均为物理量 |

五种模型共用 0.25° 全球规则网格（721×1440，纬度北→南、经度 0–360°）。除 Pangu 外
时间窗口均为 2 帧历史；Pangu 单帧历史（`history_steps=1`）、6 小时步长，但每 4 步
（lead 24h）改用 24h 模型从锚点直接跳 24h（官方「方案B」）。

> 模型名有两套，别混：**config 名**（`--model` / `--config` 用的，如 `fuxi_ens`、
> `fuxi21`、`fgvp`、`fengqing`、`pangu`）对应 `configs/` 下同名配方文件；**模型类
> 注册名**（config 里 `model_class=` 引用的，如 `fuxi_ens_onnx`、`fuxi21_pt2`、
> `iwc_fgvp_gdn2_onnx`、`fengqing_pre_onnx`、`pangu_onnx`）对应
> `models/MODEL_REGISTRY`。

## 核心特性

- **数据源统一走 `processed_multi_zarr`**：底层是 `xmetai-core` 的
  `MultiZarrDataset`，按起报时间读取历史帧、按模型通道序取变量。输入前处理
  （单位换算 / 几何对齐 / 归一化 / 填缺测）由 `dataset.processors` 声明，在
  DataLoader worker 里经 `tensor_processors.py` 执行。
- **后端可插拔**：`onnx` / `pt2` 两种执行引擎与模型语义分离，引擎只负责加载和跑。
- **模型契约驱动**：通道、网格、历史窗口、成员语义和状态表示由模型类声明。
- **完整运行配方**：config 同时声明 `dataset`（含 `processors`）、`dataloader` 和
  `model_processing`（`recurrent`、`output` 两阶段）。
- **统一自回归编排**：`runner.py` 的 `Rollout`/`Trajectory` 把单步 `forward` 串成
  多轨迹循环，`state = result` 回填与 `windowed_output` 两种输出形状在此汇合。
- **异步落盘**：后台单线程写 NetCDF，GPU 不等磁盘写；写失败时兜底存
  `raw_step_XXX.npy`。
- **任务级动态认领调度**：一次运行切成 `(起报, 成员块)` 任务，worker 自己去共享盘认领，
  快的多拿、慢的少拿（没有拖尾）；完成标记落在共享盘上，重启跳过已完成的，崩掉的 worker
  手里的任务会被别人回收。每进程经 `CUDA_VISIBLE_DEVICES` 隔离单卡，进程数不必等于卡数
  （`scheduling.procs_per_gpu`）。超参数全在 config 的 `scheduling={...}`，见
  [多卡运行](#多卡运行)。
- **确定性模型转集合（SPW）**：config 里 `weight_perturb={...}` 一开，就让一个确定性检查点
  直接产出 N 成员集合（`W~ = W ⊙ (1 + σξ)`），**零重训**。两种注入机制：`m3` 图内随机算子
  （每次前向重抽，**已淘汰**）与 `m1a` 逐成员冻结权重（论文主配置，主线，推理零开销）。
  详见 [perturbation/README.md](docs/README.md)。

## 目录结构

```
.
├── pyproject.toml           # PyPI 构建元数据和 CLI 入口
├── xmetai_inference/        # 可安装的 Python 包（包名为 xmetai-inference）
│   ├── cli.py               # 配置加载、多卡调度、自回归主循环与异步落盘
│   ├── runner.py            # Rollout / Trajectory 自回归编排层
│   ├── data/                # 数据源工厂 + processed_multi_zarr（包 xmetai-core）
│   ├── processing/          # 模型侧管线（recurrent/output）+ 数据侧 tensor processor
│   ├── backends/            # BaseInferModel + onnx / pt2 执行引擎
│   ├── models/              # 具体模型与按需加载注册表 MODEL_REGISTRY
│   ├── scheduling/          # 多进程/多卡调度：任务表、共享盘认领账本、worker 心跳
│   ├── configs/             # 内置完整运行配方
│   └── logging_util.py      # 推理日志配置
├── util/                    # 源码附带评测工具，不进入 PyPI wheel
├── scripts/                 # Bash 推理启动脚本（旧部署方式）
└── model_artifacts/         # 本地模型产物，整个目录不上传 Git
```

## 架构与数据流

一条预报从数据到落盘的完整链路：

```
zarr store  →  xmetai-core MultiZarrDataset          （读历史帧，按模型通道取变量）
            →  dataset.processors                    （DataLoader worker 里，torch）
                 unit_convert → geometry → normalize → fill_missing
            →  prepare_dataset_batch                 （[B,T,C,H,W] tensor）
            →  Rollout 自回归                        （state=result 回填）
                 recurrent processors（zero_channels）
            →  output processors（denormalize）       （工作空间 → 物理量）
            →  to_dataset                            （xarray，tp 非负兜底）
            →  _AsyncWriter                          （后台单线程写 NetCDF）
```

| 步骤 | 谁负责 | 干什么 |
|------|--------|--------|
| 数据读取 | `xmetai-core` | 打开 Zarr、选历史帧、按模型通道名映射变量 |
| 输入前处理 | `dataset.processors` | 物理量 → 模型工作空间：单位换算、几何对齐、归一化、填缺测 |
| 自回归 | `Rollout` | `state = result` 回填；`windowed_output` 决定滑窗在哪做 |
| 回填 Processor | `model_processing.recurrent` | 回填前原地改（如诊断通道 `zero_channels` 置归一化 0） |
| 输出 Processor | `model_processing.output` | 模型工作空间 → 物理量（`denormalize` / `denormalize_fengqing`） |
| 落盘 | `_AsyncWriter` / `to_dataset` | 物理量 → xarray/NetCDF；`tp` 统一在此做非负兜底 |

> 输入前处理已迁到 `dataset.processors`，`model_processing.input` 必须为空（误配会
> 直接报错）。模型文件内部已融合的归一化不在框架重复执行；输出反变换与回填规则也由
> 同一套 config 管理。

## 快速开始

### 前置条件

- Linux 环境（驱动脚本为 bash；Windows 建议 WSL2 或 Git Bash）
- NVIDIA GPU + CUDA（ONNX / PT2 推理）
- 模型权重文件与 ERA5 输入数据（两者均不进 git，见 [权重管理](#权重管理)）

### 安装依赖

```bash
pip install -e .
```

可选依赖按运行内容安装：

```bash
pip install -e ".[netcdf,zarr,onnx]"   # ONNX 模型（FuXi-Ens / FGVP / FengQing）需 onnxruntime
pip install -e ".[netcdf,zarr]"        # 仅需 NetCDF 读/写引擎（FuXi-2.1 / 评测）
```

> `torch` 是核心依赖（DataLoader 推理路径全程用它：`data/factory.py` 建
> `DataLoader`，`tensor_processors`、`runner`、`backends/base.py` 都跑 torch tensor），
> 随 `pip install -e .` 一并装好。写输出 NetCDF 和读 `mean.nc`/`std.nc` 需要
> `netCDF4` 或 `h5netcdf`（`netcdf` extra）；底层 Zarr 存储由 `xmetai-core` 读取
> （`zarr` extra）；三个 ONNX 模型运行时需要 `onnxruntime`（`onnx` extra）。

### 一条命令推理

```bash
xmetai-infer --model fuxi_ens
xmetai-infer --model fuxi21
xmetai-infer --model fgvp
xmetai-infer --model fengqing
xmetai-infer --model pangu
```

`--model` / `--config` 都选择一份内置运行配方（`configs/<name>.py`），配方里已声明
权重路径、数据集、前处理链、起报时间、步数、卡数和输出目录。起报时间、步数、成员数、
GPU 和输出目录都可用命令行参数覆盖，见 [配置参考](#配置参考)。

未安装、直接从源码根目录运行时：

```bash
python -m xmetai_inference --model fuxi21
```

`xmetai-infer --worker ...` 是多卡调度的内部子进程入口，不应手工调用。

### 覆盖参数

```bash
xmetai-infer --model fuxi_ens \
  --times "2025010600..2025011200:24" \
  --steps 60 --members 51 --gpus 4 --cuda-devices 0,1,2,3 \
  --vars z500,u200,v200,msl,tp \
  --out /workspace/data/shenzw/fuxi_ens_output
```

- `--model-path` 可替换配方中的权重文件，`--model` 选择配方。
- `--times` 语法见 [configs/base.py `parse_times`](xmetai_inference/configs/base.py)：单次
  `2025010600`、逗号列表、闭区间 `A..B`（默认 24h 步长）、`:N` 覆盖步长。
- 外部 `.py` 配方用 `xmetai-infer /path/to/config.py`（或 `--config`）加载，见
  [扩展自己的模型和数据集](#扩展自己的模型和数据集)。

## 配置参考

### `xmetai-infer` 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `config_path` | — | 位置参数，`--config` 的外部配置文件写法 |
| `--model` | 与 `--config` 二选一 | 内置配方名，如 `fuxi_ens`、`fuxi21`、`fgvp`、`fengqing` |
| `--config` | 与 `--model` 二选一 | 内置配置名或外部 `.py` 配置路径 |
| `--model-path` | 配方值 | 覆盖模型权重路径 |
| `--times` | 配方值 | 覆盖起报时间（`parse_times` 语法） |
| `--members` | 配方值 | 覆盖集合成员数 |
| `--steps` | 配方值 | 覆盖预报步数 |
| `--vars` | 配方值 | 覆盖输出变量，逗号分隔 |
| `--out` | 配方值 | 覆盖输出目录 |
| `--gpus` | 配方值 | 覆盖使用的 GPU 数量 |
| `--cuda-devices` | 配方值 | 覆盖物理卡号，逗号分隔 |
| `--batch-size` | 配方值 | 覆盖 DataLoader batch size |
| `--num-workers` | 配方值 | 覆盖 DataLoader worker 数 |
| `--log-level` | 配方值 | 覆盖控制台日志级别 |
| `--schedule` | 配方 `scheduling.mode` | 调度模式：`claim` = 任务级动态认领（默认）、`static` = 旧静态分片回退 |
| `--procs-per-gpu` | 配方 `scheduling.procs_per_gpu` | 每张卡起几个 worker 进程（>1 时显存按份数分） |
| `--resume` | 配方 `scheduling.resume` | 断点续跑口径：`outputs` 产物齐全即跳过（默认）/ `marker` 只认完成标记 / `off` 整份重算 |

> 其余调度超参数（`members_per_task`、`claim_order`、`task_retries`、`max_passes`、
> `heartbeat_interval`、`heartbeat_timeout`、`task_ttl_seconds`、`prefetch_tasks`、
> `input_cache`、`poll_interval`、`monitor_interval`、`atomic_output`、`plan_dir`）
> **一律在 config 的 `scheduling={...}` 里配** —— 它们属于「这份配方怎么跑」，要跟着配方
> 进版本控制。命令行只留上表那三个日常覆盖项。字段含义见
> [多卡运行](#多卡运行)，键名写错会直接报错而不是被静默忽略。

### 环境变量

| 变量 | 作用于 | 说明 |
|------|--------|------|
| `XMETAI_INFERENCE_ROOT` | 全部 | 项目根目录（config 相对路径/内置权重查找基准） |
| `ERA5_STORE_ROOT` | fuxi21 / fuxi_ens / fgvp | foundation store2 数据根目录 |
| `FUXI21_MODEL_PATH` | fuxi21 | FuXi-2.1 权重 `.pt2` 路径 |
| `FUXI_ENS_ONNX` | fuxi_ens | FuXi-Ens 权重 `.onnx` 路径 |
| `FGVP_ONNX` | fgvp | IWC FGVP GDN2 权重 `.onnx` 路径 |
| `FGVP_OPS_LIBRARY` | fgvp | 自定义算子库 `.so` 路径 |
| `XMETAI_GPU_STATE` | fuxi_ens / fgvp | `1`/`0` 覆盖 GPU 常驻开关（见 [多卡运行](#多卡运行)） |
| `FENGQING_MEAN_STD_DIR` | fengqing | FengQing `mean_std/` 目录 |
| `FENGQING_MASKS_PATH` | fengqing | FengQing `constant_masks.npy` 路径 |
| `PANGU_ONNX_24` | pangu | Pangu 24h 跳步模型 `.onnx` 路径（默认取 6h 同目录 `pangu_weather_24.onnx`） |

## 扩展自己的模型和数据集

用户安装主框架后，在自己的项目中编写 Model 和 config 即可，无需修改 `xmetai_inference`
源码。

#### 1. 编写 Model

Model 声明输入/输出通道、网格、历史窗口、成员语义，并继承合适的后端。PT2、ONNX 模型
分别继承 `Pt2InferModel`、`OnnxInferModel`；其他执行方式继承 `BaseInferModel` 并实现
`load()` / `forward()`。

```python
# models/my_model.py
from xmetai_inference.backends.pt2 import Pt2InferModel


class MyModel(Pt2InferModel):
    input_channels = ("z500", "t850", "msl", "tp")
    output_channels = input_channels
    grid = {
        "lat": {"start": 90.0, "step": -0.25, "size": 721},
        "lon": {"start": 0.0, "step": 0.25, "size": 1440},
    }
    history_steps = 2
    hour_interval = 6
    forecast_type = "deterministic"
    members = 1
```

图内已烘焙归一化的模型（如 FuXi-Ens）可只配 `unit_convert` + `geometry`；图只出预报
那一帧的模型要加 `windowed_output = False`，滑窗由 `Rollout` 完成。

#### 2. 编写 config

config 直接引用自定义 Model，声明数据集、`dataset.processors` 输入前处理链和
`model_processing` 回填/输出链。相对路径按 `config.py` 所在目录解析。

```python
# config.py
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig
from models.my_model import MyModel


cfg = InferConfig(
    name="my_model",
    model_path="./weights/my_model.pt2",
    model_class=MyModel,                     # 直接传模型类，绕过内置注册表
    dataset={
        "type": "processed_multi_zarr",
        "paths": ["./data/era5.zarr"],
        "processors": [
            {"name": "unit_convert", "scales": {"tp": 6000.0}},
            {"name": "geometry"},
            {"name": "normalize", "mean_file": "mean.nc", "std_file": "std.nc",
             "log1p_channels": ["tp"]},
        ],
    },
    dataloader={"batch_size": 1, "num_workers": 0},
    model_processing=ModelProcessingConfig(
        input=[],                             # 必须为空（前处理在 dataset.processors）
        recurrent=[],
        output=[{"name": "denormalize", "expm1_channels": ["tp"],
                 "nonnegative": ["tp"]}],
    ),
    times="2025010200",
    steps=10,
    members=1,
    gpus=1,
    output_dir="./output",
)
```

`dataset.processors` 可选名字：`unit_convert`、`geometry`、`normalize`、
`normalize_fengqing`、`fill_missing`；`model_processing` 的 `recurrent`/`output` 可选
`zero_channels`、`denormalize`、`denormalize_fengqing`。两组是相互独立的注册表。

#### 3. 启动推理

```bash
cd /workspace/my_forecast_project
xmetai-infer config.py
xmetai-infer /workspace/my_forecast_project/config.py   # 绝对路径
xmetai-infer config.py --times 2025010300 --steps 10 --gpus 4
```

多卡模式会让每个 worker 重新加载同一份 `config.py`，因此外部 Model、Loader 以及 config
导入的自定义 Processor 在各进程中都能正常创建。

## 多卡运行

单次 `forward` 不跨卡；加速来自把一次运行切成可独立认领、独立落盘的任务分摊到多张卡。
任务单元是 **`(起报, 成员块)`**：确定性模型每个起报恰好一个任务，集合模型把成员切成若干
块（`scheduling.members_per_task`，默认一块一个成员）。框架按任务表起 worker 子进程，每个
worker 经 `CUDA_VISIBLE_DEVICES` 隔离物理卡。

**任务级动态认领**（`scheduling.mode` 取 `claim`，默认）：任务表与指纹落在
`{output_dir}/.xmetai_plan/`，worker 用 `mkdir` 的原子性去认领，跑完把完成标记写回共享盘。

- **没有拖尾**：谁先干完谁接着认领，慢的 worker 少拿；不再出现「一个 rank 分到最慢的
  起报、其余卡空等」。
- **断了能续**：完成标记在共享盘上，重启后已完成的任务直接跳过。`resume` 三种口径：
  `outputs`（产物齐全即算完成，默认）、`marker`（只认完成标记）、`off`（整份重算，只认
  本次运行写的标记，不必手删 `.xmetai_plan`）。
- **崩溃重派**：worker 边跑边写心跳，心跳超时（`heartbeat_timeout`）或租约到点
  （`task_ttl_seconds`）就判该 worker 已死，它手里的任务会被别人捡走重跑；单个任务失败
  超过 `task_retries` 次才记 FAILED 放弃。
- **进程数不必等于卡数**：`procs_per_gpu` 可以让两张卡跑三个进程，也可以让一张卡跑两个
  进程（显存按份数分，配合 `gpu_mem`）。GPU 常驻（IOBinding）时显存按 session 计，一卡
  多进程容易 OOM，这种模型建议保持 `procs_per_gpu=1`。

调度超参数全部在 config 的 `scheduling={...}`（解析成 `SchedulingConfig`）；键名写错会
直接报错而不是被静默忽略：

| 键 | 默认 | 含义 |
|----|------|------|
| `mode` | `claim` | `claim` = 任务级动态认领；`static` = 旧静态分片（回退开关） |
| `procs_per_gpu` | `1` | 每张卡起几个 worker 进程 |
| `members_per_task` | `0` | 一个任务装几个成员；`0` = 自动（1 个成员一个任务） |
| `claim_order` | `auto` | 认领偏好：`auto` 下 m1a 按成员优先（省 session 换入）、其余按起报优先（省输入重读）；也可写死 `member` / `init` |
| `resume` | `outputs` | 断点续跑口径：`outputs` / `marker` / `off` |
| `task_retries` | `1` | 单个任务最多重跑几次，超过判失败不再重派 |
| `max_passes` | `3` | 主控最多重派几轮（只有某轮零进展才重派） |
| `heartbeat_interval` | `15.0` | worker 心跳写盘间隔（秒） |
| `heartbeat_timeout` | `120.0` | 心跳超过这么久判 worker 已死、任务可回收（秒） |
| `task_ttl_seconds` | `0.0` | 认领的硬租约上限（秒）；`0` = 关，只靠心跳判死 |
| `prefetch_tasks` | `2` | 预取队列深度（任务数）：认领与读帧在后台线程做，GPU 不等 IO |
| `input_cache` | `2` | 输入帧缓存条数（同一起报的多个成员任务复用同一份输入）；`0` = 关 |
| `poll_interval` | `5.0` | 无任务可认领时的轮询间隔（秒） |
| `monitor_interval` | `30.0` | 主控进度汇报间隔（秒） |
| `atomic_output` | `True` | 落盘先写隐藏临时名再 `os.replace`，下游读不到半成品 |
| `plan_dir` | 空 | 计划/认领目录；空 = `{output_dir}/.xmetai_plan` |

例（4 卡、每卡 1 进程、崩溃最多重跑 2 次）：

```python
cfg = InferConfig(
    ...,
    scheduling={
        "procs_per_gpu": 1,
        "members_per_task": 1,
        "task_retries": 2,
    },
)
```

与 ONNX Runtime 相关、同属「怎么跑」的两个旋钮也在 config 里：`intra_op_threads`（单
session 的算子并行线程数，`0` = 核数 / worker 数，避免多进程把 CPU 线程超订）与
`cudnn_algo_search`（空 = 后端默认 EXHAUSTIVE；`HEURISTIC` 让每卡多进程的启动预热快很多）。

**m1a 的常驻 session 池**（`resident_members`，只对 `weight_perturb.mechanism="m1a"`
有意义）：m1a 的每个成员是一份独立权重 = 一个 onnx session，逐成员装载时建一次 session
不只读一遍权重，还要重新预热 cuDNN。这个键决定同时驻留几份：

| 值 | 含义 |
|----|------|
| `1` | 缺省，一换一放（小显存口径：任何时刻只驻留 1 份，资源行为与改造前逐位相同） |
| `N` | 最多驻留 N 份，第 N+1 份进来时按 LRU 换出最久没用过的那份 |
| `-1` | 不限（全常驻） |

`m3` / `latent_forcing` 是一份 session 服务全部成员，这个键对它们没有意义。显存够大就
调大：切回已驻留的成员时不再重建 session、不再重预热 cuDNN（`claim_order` 只保证「连续
任务是同一成员」，池子补的是「被别的 worker 插队后切回来」与「失败重派」那部分）。
收尾日志会打 `session 池：新建 X 份、命中 Y 次、换出 Z 份（额度 N）`，命中率是决定给
多大额度的依据。注意池子是**每个 worker 进程一份**（ORT session 不能跨进程共享）——
跨进程的并行来自任务级认领，不是共享 session。

ONNX Runtime 单 session 独占一张卡；支持 GPU 常驻（IOBinding，state 全程待在显存，
只在落盘时搬回 CPU）。仅当模型前后处理/回填是恒等（图输入输出同形）时才能开，由
`config` 的 `gpu_state` 或环境变量 `XMETAI_GPU_STATE` 控制；`windowed_output=False`
的图（只出一帧）不具备开快路径的条件（启动期会拦下）。

## 集合预报（权重扰动）

拿一个**确定性检查点**直接做集合：不动训练、不改模型代码，只对权重张量乘一层噪声
`W~ = W ⊙ (1 + σξ)`（Adamov et al. 2026, arXiv:2609.08412 的 SPW）。框架里有两种注入
机制，由 config 的 `weight_perturb.mechanism` 选：

| 机制 | 做法 | 状态 |
| --- | --- | --- |
| `m3` 图内随机算子 | 图里插 `RandomNormalLike -> Add(1) -> Mul`，**每次前向重抽** | **已淘汰（2026-09-23）**：两轮独立评测都把它判负——Spread/RMSE 全程 0.09–0.23（m1a 是 0.71–1.24）、CRPS 234.2 比 m1a 差 23%、51 成员均值与确定性预报谱相关 0.99987。代码保留（磁盘受限形态 + 轨迹内刷新轴端点 + 回归测试），但不要再拿它做实验 |
| `m1a` 逐成员冻结权重 | 生成阶段算出 N 套 `W ⊙ (1+σξ)` 落盘，成员内**整条轨迹冻结**（论文主配置） | **主线**：推理零开销，每个成员就是一份可单独跑、单独审计的普通确定性 onnx |

```bash
SPW_AB=A xmetai-infer --model fgvp_spw_ab   # M1a 主线（原 configs/fgvp_spw_m1a.py 已删，A 变体同配方）
xmetai-infer --model fgvp_spw_ab    # 注入点/幅度/结构消融（configs/fgvp_spw_ab.py：SPW_AB=A|B|C|D|E|F|G|H|T|S|V|E3）。
                                    #   A-D 第一轮 4-6 成员、E-H 第二轮 6 成员（都已跑完）
                                    #   T/S/V 第三轮 6 成员（同预算、只换注入结构）；E3 第四轮（分段 rollout）
                                    #   I/J/K/L/M 已按 2L 的证据删除（见 fgvp_spw_ab.py 文件头）
                                    #   M3 对照的 configs/fgvp_spw.py 亦已于 2026-09-24 删除
                                    #   （机制已淘汰，判据见 perturbation/README 的 2A 节）
xmetai-infer --model fgvp_spw_prod_s2b  # 生产跑：51 成员 / 9 起报 / 60 步 / z500
                                    #   （S2B 配方：latent_mixer + channel + σ_full=0.027，见该文件头；
                                    #    原 fgvp_spw_prod.py（F 配方，σ=0.04）已删，协议存档见 2F 节）
```

配了 `weight_perturb` 后 `cli.py` 自动按集合语义处理（`members>1` 不再告警）。加载期的兜底
自检按机制分流：m3 是「两次同输入前向必须不同」；m1a 是「图里的成员号必须等于本次要跑的
成员号」（抓「51 次加载读了同一个文件」这个 m1a 专属的静默失败）。

**实测结论（2026-09-23，FGVP + `decoder_group` + σ_full=0.01）**：链路完全跑通，但**论文的结论没有复现**
——论文在 AIFS 上是长时效**过**离散（240 h 空间平均 SSR 2.5），我们在同一时效是**欠**离散
（σ̂/rmse 0.202 → 0.157，参考带 0.536 → 0.881，+240 h 差 5.6 倍）：离散度的增长率跑不过误差
增长。诊断是注入点落在了 0.25° **输出渲染那一级**（FGVP 的 `decoder_group`），而论文 AIFS 的
decoder 读的是 O96（≈1.5°）处理器网格。修复走 A/B：`configs/fgvp_spw_ab.py` 用 `SPW_AB` 选变体（当时是 `A|B|C|D` —— B 换到含 `dt_bias` 的 recurrent 核、
C 排除 `dt_bias`、D 把 σ 翻倍），基线曲线与四条判据见 perturbation/README 的 2C 节；跑完用 `$P spread <目录...>` 直接出对比表。

**第一轮 A/B 结果（2026-09-23，B/C/D 跑完）**：`latent_mixer`（B）用 A 的 **45%** 的 σ 拿到
1.7x 离散度与 1.6x 增长率，+240 h 涡动相关 0.983 → 0.945，成员域均值偏移只有 3.3%（A 是
7.9%、C 是 14% 且一路持续）——**B 完胜**。两条硬结论：**增长率只由注入点决定，σ 只决定幅度**
（D/A 在 ≤120 h 精确 ×2.00）；**`dt_bias` 假说被否**（占该组 60% 能量，排掉后 σ̂ 只差 ≤10%）。
离论文结论仍有距离：我们仍是**欠**离散（论文长时效是**过**离散），1-2 天已打平参考集合
（+24..48 h 是 HOU 的 0.81-1.08 倍），缺口集中在 **+168..300 h**（+240 h 只有 0.30 倍）。
完整归档（元数据、口径、留一法采样噪声、全 60 步数据、局限、下一轮 E/F/G/H）见
perturbation/README 的 **2D 节**。

**第二轮 A/B 结果（2026-09-23，E/F/G/H 跑完）**：把 `latent_mixer` 的 σ 从 0.01 加到 0.04（F），第一次把**标定比**推到 1 附近（σ̂/确定性 rmse 0.65→1.47，参考集合 fgvp_hou 自己带是 0.38→1.23），**+240 h 达到参考集合的 0.72**（B 只有 0.30）—— 目前最好的长时效结果。代价两个：**1-3 天过冲到参考集合的 2.8-3.2 倍**，以及论文点名的「整场一致偏移」同步放大（成员级域均值离散 ±6.7 → ±30.6）。
三条硬结论：

- **单一 σ 补不了两头**：要打平参考集合所需的 σ 倍数随时效变 **2.3-4.0 倍**（A 3.1x / B 2.7x / F 4.0x），而 `latent_mixer` 的 σ 响应**次线性且长时效饱和**（σ×4 在 +6 h 换来 3.65 倍、到 +360 h 只剩 1.67 倍）—— 长时效的缺口不是幅度问题。只有 `decoder_group` 的 σ 是干净的线性旋钮（σ×2 → 精确 ×2.00）。
- **多组注入没有收益（幅度口径）**：G（latent+decoder）、H（全权重 `all`）相对 B 都是 0.83-1.14；H 的扰动元素/磁盘翻倍（731M vs 373M 元素、2.93 vs 1.49 GB/成员）买不到东西 —— `latent_mixer` 占候选能量 **76.3%**，`all` 只是把同一份方差预算换个分配。
  **2026-09-24 补测（`$P verdict F B C D G H`，产物已归档）**：上面这条是**幅度口径**的结论，不是形状口径的。用 `k(lead)` 判据量，**跨组的臂不平**：`decoder_group`（D）**−31%**、`latent_mixer+decoder_group`（G）**−24%**、`all`（H）**+38%**（H 的等效 σ 倍数随时效上升 = 本项目最强的形状信号）。受控对照 **B vs D**：同为 4 成员、同为低 σ，**B 平而 D 不平** ⇒ 差别只能归给注入组。所以「换注入点只改幅度」只在 `latent_mixer` **组内**成立；跨组时注入点**确实改形状**，第一轮的「增长率由注入点决定」与论文 Phase 2 因此成立、不是幅度伪影。**D/G/H 仍全部判负**（H 的 k=0.35 vs 门槛 3.41，差 9.7 倍）：形状改对了方向、改错了量级。
- **4-6 成员的 σ̂ 由单个成员主导**：拿掉第 3 号成员 σ̂ 掉 22-34%（B/F），别的成员只动 ±3-7% → **比较 10% 级的差异至少要 6 成员**；正式跑 51 成员不存在这个问题。

第三轮（变体 **I..M**，已写进 `configs/fgvp_spw_ab.py`，全部 8 成员）：I/J = `latent_mixer` 的 σ_full 0.025/0.03 工作点；K/L = 按**块类型**换注入点（K 只扰 6 个 GatedDeltaNet 块、占候选能量 61.4%，L 只扰 6 个 attn+mlp 块、14.9%，M 是 B 的 8 成员对照）—— 三者同 σ_full=0.01、同成员数，所以差别**只有注入点**。为此新增两件工具：`groups --under/--depth`（子组扫描）和 group token 的**并集写法**（`"a,b,c"` 算一个组；写成三个 token 会逐组配平成 3 倍预算 —— G 踩过的坑）。
完整归档（每张表的独立复核、留一采样噪声、丢帧事故、下一轮设计）见 perturbation/README 的 **2E 节**。

**生产跑（第四轮，2026-09-23 定档；2026-09-24 换配方）**：**51 成员 / 9 起报（20250702-0710，与 `fgvp_hou`、M3 同窗口）/ 60 步 / z500**。原定复刻第二轮的 F（`latent_mixer` + σ_full=**0.04**）—— 理由是 F 是唯一把长时效推上去的配置，而 **CRPS 的均值由长时效主导**（+360 h 的 CRPS 有 300-400、+6 h 只有 ~12；M3 在 +24..144 h 反而赢 hou，输的全在 +156 h 之后），所以 2E.8 当时按「spread 形状」建议的 0.025-0.03 在这里是反方向。**但随后 A/B 的判决把 S2B 判成五轮里唯一「两头都不输」的配置**（长时效 CRPS 比 F 好 6.8%、短时效只差 1.9%，且全项目唯一过尺度选择性判据），**2026-09-24 改成跑 S2B**。

配置 `configs/fgvp_spw_prod_s2b.py`（`latent_mixer` + `noise_struct=channel` + σ_full=**0.027**）；原 F 配方 `configs/fgvp_spw_prod.py` 已删，其协议存档在 perturbation/README 的 **2F 节**。S2B 与 F 只差 `sigma_full` 与 `noise_struct` → **指纹不同**，走自己的成员集目录（与 F 互不串味），而该目录里第五轮留下的 `member_000..005` 可复用、只补 45 个新的（67 GB）；成员集根已改到 `/workspace/data/shenzw/model_onnx`，**搬旧目录**那一步见该 config 的文件头（不搬会整份重造 51 个 = 76 GB）。复用的 6 个成员与 A/B 归档 `spw_ab/S2B` 在 0702/0703 上可做**逐帧对账** —— 整条链路唯一的端到端正确性校验。
验收三条：**CRPS 均值 < 234.2**（M3 同窗口 51 成员）且力争 **≤ 215**（hou 204.3）、**σ̂@51 ÷ σ̂@6 落在 1.0-1.2 倍**（新加的 `spread --members 0-5` 在同一份产物上抽子集）、**SSR 不冲到 1.5 以上**。预算：成员集 67 GB、输出 114 GB、4 卡约 7 h。详见 perturbation/README 的 **2F 节**。

**超参数敏感性复盘（2G 节，2026-09-23 用已有产物重算，没跑新推理）**：把 2C-2E 的数据按旋钮重排，量级排序是 —— **注入点**（`latent_mixer` 是 `decoder_group` 的 2.4-4.0 倍效率）> **机制**（m3 每步重抽的 spread/rmse 只有 0.09-0.23、集合均值与确定性预报谱相关 0.99987，冻结是前提）> **σ 幅度**（`decoder_group` 线性 p≈1.0，但 `latent_mixer` 长时效饱和到 **p=0.37**，即 σ 翻倍只换 1.29 倍后期离散度）> **配平口径**（energy vs 论文的 count 原式，同一组差 **0.57-1.22 倍** σ）> **成员数**（隐藏超参数：4 vs 6 差 1-14%、2 成员 20-30% **且符号不定**）。`dt_bias`（占候选能量 45.8%）、`A_log`（144 元素、0.13%）与 `min_elements` 都不是能调出收益的旋钮。**结论：以后任何「换 X 有 10% 收益」的说法，都必须先用 `spread --members` 排除成员数的影响。**

**外部独立评测：变体 F 上了裁判桌（2H 节，2026-09-23）**：`D:\xmetai-evaluation-v2\report\spw_F_vs_ens_ref` 是 v2 框架 `weather_ens_field_scores` 的独立裁判（CRPS / Spread-RMSE / 球谐带 / 纬向谱），评的是我们的 **F**（`latent_mixer` + σ=0.04、6 成员、3 起报、z500）。三条要点：

- **同族口径下 SPW 赢了，而且这条结论对成员数免疫**：对同底模的 51 成员参考集合 `fgvp_hou`，F 的 CRPS **原始** 207.17 vs 209.77（F 只有 6 成员、原始值里含约 **30 点**的成员数罚分），折算 51 成员后 177.21 vs 205.08（**赢 13.6%**），6–72 / 78–168 / 174–360 h 三个时段折算后全胜。对 51 成员的 `fuxi` 折算后 6–168 h 反超、只在 174–360 h 仍输 9.4%。
- **m1a 相对 m3 被裁判侧确认**：姊妹报告评的 M3（51 成员、图内每步重抽）CRPS **234.2**、成员均值与确定性预报谱相关 0.99987、5–20 带（0.806）与确定性单成员（0.821）几乎相同 —— 成员没有真正分道扬镳。F 用 **6** 个成员就把 CRPS 压到 ≤207.2，且 5–20 带掉到 0.639（比确定性暗 22%）→ **成员真的发散**。2G.7 的「冻结 > 刷新」成立。
- **缺口照旧是长时效**：标定比 +24..72 h **1.04–1.27**（四个集合档里唯一贴线），然后 +120 h 0.774 → +240 h 0.658 → +360 h **0.608**，与 2E.2 的「前重后轻」同形。

两处口径修正：**① σ=0.04 的代价第一次被量化** —— 集合均值在 **+48..120 h 比未扰动的 `fgvp_det` 单成员差 13%**（ACC 也掉 2.2%），而 `aifs` 以标定比 1.70（全程过离散）拿到最好的 CRPS ⇒ **「把标定比调到 1」不是正确目标，σ 应往下走（0.025–0.03）、判据换成 CRPS**；**② k>74 的谱噪声底不是 SPW 的问题** —— 未做任何扰动的 `fgvp_det` 在谱尾与 `spw_F` 逐点重合（±6%），两个报告独立证实。生产跑的三条判据据此重定：**CRPS 及格 < 204.3、目标 ≤ 185**（由成员数偏差公式推出的预期值 ≈180.7），**比旧判据 <234.2 严格得多**（旧的会被构造性地通过），**SSR 升级为一级判据**。完整复盘见 perturbation/README 的 **2H 节**。

**自由度地图与综合结论（2I 节，2026-09-23 把 A-M 十二个变体按「旋钮」重排）**：论文只给两个自由度（选择集 S、幅度 σ），我们在 FGVP 上实际碰到 **七个**（论文两个 + 口径一个 + 实现一个 + 一个论文提了但用错端点的）—— 量级排序是 **机制**（m3 vs m1a，3-8 倍）> **注入点**（`latent_mixer` 是 `decoder_group` 的 2.4-4.0 倍）> **σ**（1.7-4 倍）> **配平口径**（0.57-1.22 倍）> **成员数**（5-30%）。三条最该记住的：

- **两个自由度不可分离**：σ 的幂律**是注入点的函数** —— `decoder_group` 上 p≈1.00（干净线性），`latent_mixer` 上从 0.93(+6 h) 掉到 **0.37(+360 h)**。所以在 latent 上「把 σ 再调大一点」只对短时效有效。
- **现有七个旋钮里没有一个能改「形状」**：σ 只改幅度、注入点只改增长率、口径只改记账、成员数只改采样噪声，而我们的缺口恰恰是形状（中短期过离散 + 长时效欠离散）⇒ **下一轮只能换一类东西**（扰动初值 / 轨迹内刷新的中间值 / 粗尺度约束）。
- **两个设计耦合**：① 同组变体的 ξ 逐元素相同（`SeedSequence([seed=0, 成员号])` + 张量顺序固定）⇒ σ 扫描是**配对设计**，所以 p 测得很干净，但**「换一次噪声抽样」的不确定度从没估过**；② 成员数在变体间不一致（6/4/2）⇒ 只有 ≥2 倍的差别能当结论。

2I 节还附了**F 的两个自由度的精确复算**（210 张量 / 373.02M 元素 / 占候选能量 76.3% / `sigma_group = 0.0457845`，含 12 个 block 的构成与「6 个 `dt_bias` 占该组 60% ΣW² 却只有 0.0025% 元素」的反差）、**空白清单**（初值扰动 0 次、噪声实现只抽了一次、SSR 从未测…），以及 **ONNX 到底支不支持扰动**（不支持也不需要 —— 扰动是离线生成阶段，所以 m1a 推理零开销）。

**下一轮实验定档：N 轴对比（2J 节，2026-09-23，方案，未跑）**：七个自由度里唯一「论文有、我们只测了两端」的就是**噪声时间结构 = 轨迹内刷新步数 N** —— N=1 的 m3 已否、N=∞ 的 F 是主线，**中间从没试过**，而它恰好是论文用来压区域平均过离散（SSR）的那味药（SFNO 0.66→0.83 且不付 CRPS，Aurora/AIFS 几乎无效）。定档：**N=20（60 步 = 3 段）**、**8 成员 / 3 起报（与 F 同窗口）/ 60 步 / z500**、注入点 `latent_mixer`，σ 取**两档** —— `0.04`（冻结口径，量刷新削掉多少 spread）与 `0.04·√3 = 0.0693`（论文式 `σ_N = σ_frozen·√(T/N)`，量补偿公式成不成立）⇒ 单卡 3 h。代码改动约 10 行、全在 `cli.py`（成员外层再加一层**段**循环，每段重 load + 上段末态作下段初态，`Rollout`/`Trajectory` 不动，权重取 `SeedSequence([seed, member_id×K + seg])` ⇒ 逐 (成员,段) 独立、零新增噪声代码，`on_step` 输出名要加段偏移）。白送校验：K=1 时索引退化成 `member_id`，所以 **N=20 臂的前 20 步应与 F 逐帧吻合**。**前置条件：必须新加区域平均版 SSR**（v2 的 `spread_rmse_ratio` 是逐点面积加权，没有空间平均版），否则这个臂跑了也判不出来。三种结局与排序见 2J.5/2J.6。

**创新候选：13 条风评风险的根因与反攻路线（2K 节，2026-09-23，方案）**：13 条里**只有 3 条是方法问题**（#1 长时效精度掉队、#2 长时效欠离散、#4+#12 中小尺度能量不足），而它们**同源**：σ 是一个标量，同时决定「离散度幅度」与「集合均值退化」，且逐元素 iid 噪声**没有尺度选择性** —— 大尺度过冲、中小尺度被平滑变暗、长时效又不够散，三个诉求互相冲突。其余 10 条是**测量口径**（成员数/3 天窗口/CRPS 口径翻转/源目录/只有 z500…）与**工程归因**（#5 网格尺度噪声底，已收敛到产出链），**不先修就判不出创新实验的好坏**；尤其 #5 必须先查，因为 #4/#12 的判读带（5–64）就在它旁边。四个可证伪的创新方向：**① 噪声相关化**（通道级/低秩/块常数 ξ，只改生成阶段、推理零改动，预测 5–20 带不暗且 +360 h Spread/RMSE ≥ 0.70）> **② Fisher 配平**（按输出敏感度而不是 ΣW² 分预算，论文点名的未来方向，预测集合均值退化从 +2.6% 压到 <1%）> **③ 分段 σ**（把「冻结 vs 刷新」从二值变连续，与 2J 同批）> **④ 潜空间 AR(1) 随机强迫**（补完论文的 SPPT 类比 —— 神经网络的「倾向项」就是每步的潜变量更新，扰它才有时间相关性，直击 #2）。执行顺序 S1 → S6：S1 #5 归因 + 口径钉死（0 推理） → S2 通道级 ξ（只改生成 + 单卡 3 h） → S3 2J 的 N=20 → S4 分段 σ → S5 Fisher 配平 → S6 潜空间 AR(1)。统一验收四条（均在 51 成员/9 起报上）：中小尺度不暗、CRPS 折算 51 成员 < 180.7（同时报原始值）、+360 h Spread/RMSE ≥ 0.70、集合均值 RMSE 退化 < 1%（详见 2K.5）。

**口径修正：「中小尺度变暗」是平均效应，不是损伤（2L 节，2026-09-23，零推理，已算完）**：本地复算确认 v2 框架对集合产品报的带功率是**成员均值场**的（F @2025-07-01 +240 h 带 5-20：复算 4 500 922.5 vs `scores.csv` 4 500 922.00，而成员带功率的平均是 5 909 007.5，**高 31%**）。拆成两个乘子（以未扰动的确定性档为 1）：**成员自己几乎没变钝**（`memavg/det` 0.997 → 0.846 @5-20；41-64 带甚至 **1.18-1.25，比确定性档还亮**），**全部变暗来自集合平均**（`ensmean/memavg` 跌到 0.593 / 0.268 / 0.446，且**尺度越小抵消越狠**，与 spread 同步加深）。所以 **#4/#12 是评测口径 + 过离散的后果，不是模型的病**；#10（确定性单成员优于 6 成员均值）与 #1 的一半（集合均值 RMSE +2.6%）同源。**13 条里方法层面只剩一条：#2 形状病**。另一个硬结果是 σ 的响应指数实测（B 0.01 vs F 0.04，**成员数对齐到 4v4**）：`p` 从 +6 h 的 **1.007**（严格线性）单调衰到 +342 h 的 **0.466**（严重饱和）；要把长时效 spread 翻倍需 σ 翻 **4.4 倍**，而那时短时效 spread 也翻 4.4 倍 —— **单一标量 σ 数学上不可能同时标定短/长时效**。同时修正了 2I.3 的 `p(+6 h)=0.93`（那是 6 vs 4 成员数伪影），并给出机制统一读法：**逐元素 iid 的大部分能量注入了被抑制的方向，只有少数领导不稳定模态上的分歧能持续增长**（这也终于解释了论文「四个股干的优胜点都紧邻低分辨率潜空间」的假说 —— 扰的是潜模的长时间演化）。FGVP 上对应的不是卷积核，而是 **DeltaNet 的时变/门控参数**（`dt_bias`/`A_log`/`f_proj`/`g_proj`）与 `group_projs`。下一步定档：**E0**（零推理）把分解与 `p(lead)` 变成常规诊断、修评测口径（集合产品的谱报两列）；**E2**（**2 个新臂，单卡 3 h**）同预算下只改注入结构 —— 臂 `T`（只扰 DeltaNet 的时变参数 `dt_bias`+`A_log`，12 张量 / **9 360** 元素，配平后 σ=0.0590658） vs 臂 `S`（`latent_mixer` + 通道级 ξ，配平后 σ=0.0457845，**与 F 逐位相同** ⇒ 唯一只差结构的对照），对照已有的 iid-F；两臂都已写进 `configs/fgvp_spw_ab.py`（I/J/K/L/M 已删），判据四条：长时效更散（+240..360 h 比值 > 1）、短时效不更散（+6..48 h ≤ 1）、**增益落在 1–4/5–20 带**、不把域均值离散推到 F 的 ±32.8；**E3** = 2J 的 N=20（同样用 `p(lead)` 判）。详见 2L.7。

**响应指数的尺度分解（2L.8 节，2026-09-23）**：同样是 B(0.01) vs F(0.04)、成员 4v4、σ 比 4，但量的是**偏差场（成员 − 集合平均）的带功率**（功率口径：除 2 才是 rms：+6 h 十分链合 2L.4 的 1.007）。**+6 h 五个带全是 2.02 ⇒ sigma 在任何尺度上都是严格线性的幅度旋钮**；到 +342 h 强烈分化（rms 口径）：**1–4 行星 0.669 / 5–20 0.427 / 21–40 0.163 / 41–64 0.241 / 65–128 0.622**（日常尺度约 0.6，中尺度只剩 0.16）⇒ **σ 的边际收益按尺度递减，行星尺度比中尺度耐用 4 倍**。注意 65–128 那一带看着最耐用，但它已被 **#5 的网格尺度噪声底**污染（k>74 不衰减），不可用。这是 2L.2「集合平均抵消」的另一面，也直接给出了 E2 的靶心：**增益必须落在 1–4/5–20，落在 21–40/41–64 就只是把中尺度注得更脏**。复算命令见 2L.9。

**第三轮（E2）+ 第四轮（E3）（2L.7 / 2L.10 节，2026-09-23，已落 config、等卡）**：同一份能量预算（`sigma_full=0.04`）下只换**注入结构**，与已有的 iid-F 对照 —— 臂 `T`（只扰 DeltaNet 的时变参数 `dt_bias`+`A_log`，12 张量 / 9 360 元素，σ=0.0590658）、臂 `S`（`latent_mixer` + **通道级** ξ，σ 与 F **逐位相同** 0.0457845）、臂 `V`（`latent_mixer` + **张量级** ξ：一个成员只有 210 个随机数，σ 同上）。三点把「相关长度」画成一条曲线（iid → channel → scalar），判据是 2L.7 的四条（长时效更散 / 短时效不更散 / 增益落在 1–4 & 5–20 带 / 不把域均值离散推到 ±32.8）。`E3` 是**时间结构**那一轴（= 2J 的 N 轴）：config 的 `spw_segments` 把一条轨迹切成 K 段、每段换一份冻结权重（默认 3 段 × 20 步、三段各摊 1/3 预算 ⇒ 每段 σ=0.0264337；换段只重开 session、不动运行态，每成员多 K−1 次 load）。它**可证伪**：`spread²(lead)=Σ_k p_k·增长(lead−k)²`，增长随时效单调递增 ⇒ 预算全放 t=0 是极值，刷新只会**降**长时效离散度；若 E3 实测长时效反而 > F，说明得换成真正时间相关的强迫（潜空间 AR(1)，2J 第三条）。验收：`self-test` 179 条（含分段表的预算 / 继承 / 段边界 / 错例拦截）、`e2e` 第 9 节（边界前逐位相同、边界后全不同、每成员 load K 次）。

**第三轮 S/T 实测判决（2026-09-24，见 2L.11）**：S 与 T 都跑完了，**「长时效更散 + 短时效不更散」两条判据同时成立的臂一个也没有**。S（通道级 ξ）同预算下 spread ×1.21–1.51（3/3 天同向），但把它换算成「等效 σ 倍数」`k(lead)` 后**平在 1.41–1.52** ⇒ 相关化只是**幅度放大器**：短时效被一起放大成过离散（标定比 1.08→1.63，CRPS 3/3 天差 8–23%），长时效 0.59→0.77 但 CRPS 的改善只有 1/3 天同向（不可当结论），21–40 带反而更暗 17%。T（只扰 `dt_bias`+`A_log`，12 张量/9 360 元素）的离散度只有 F 的 0.37–0.55 倍（等效 σ×0.37）——作为不确定度旋钮低效；**但它的集合均值是三档里最准的**（RMSE 415.4 vs F 434.6、外部确定性档 423.4；ACC 全程 3/3 天更优；activity +11–28%；5–20 / 21–40 带功率 ×1.47 / ×1.62）⇒ 它是**偏差 / 锐度杠杆，不是不确定度杠杆**。机制读数：2L.4 的饱和来自「随机 δW 被投影到大量衰减方向」，相关化只放大同一投影的增益、没换方向 ⇒ **要治形状必须换方向（时间相关的强迫），不是同一方向的幅度**，这也把 2L.6 的假说否掉了。复算：`python D:\xmetai-evaluation-v2\report\spw_arms_vs_F\compare.py F S T`（V/E3 见下一段）。

**第四轮 V/E3 实测判决（2026-09-24，见 2L.12）**：`V`（张量级 ξ，一个成员只有 210 个随机数）与 `E3`（3 段 × 20 步的分段 rollout，每段 σ=0.0264337）都跑完了，结论是**两条轴一起关掉**。V 的 `k(lead)` 平在 **2.33–2.74**（不是预测的 1.4–1.5）⇒ `iid → channel → scalar` 是**同一条单调幅度旋钮**（等效 σ 倍数 1.0 → 1.47 → 2.4），三个点没有一个改形状；把秩降到 1 等价于把 σ 乘 2.4，于是短时效过离散到标定比 2.04、CRPS 全时段 ×1.21–1.72、RMSE ×1.256、ACC ×0.814，中小尺度功率反而暴涨（41–64 带 ×5.6、65–128 带 ×7.5）⇒ **V 是把模型推坏，不是修好**。E3 则在**所有时效**都比 F 更欠离散（+240..360 h 只有 0.60 倍，`k` 平在 0.37–0.49），**2L.7 写下的可证伪预测被证实**；但段边界（+120 h 局部极大 0.503、+240 h 之后斜率由平转升）证明**换段确实注入了新鲜方差**，只是幅度太小 ⇒ **"iid 重抽"换不来时间相关性，时间轴也关闭，唯一还开着的门是潜空间 AR(1) 强迫**。意外收获：**不去扩大离散度的臂，集合均值反而更准** —— E3 的 RMSE / ACC / activity 是五档最好（412.7 / 0.7905 / 0.8574，对 F 是 ×0.950 / ×1.040 / ×1.116，且逐箱 3/3 天同向），T 紧随其后，而放大离散度的 S / V 都更差 ⇒ 扰动本身在替模型做"多成员平均"、抹平了一点系统性偏差，**这给 `D0`（σ≈0，同源确定性基准）升级了动机**。复算：`python D:\xmetai-evaluation-v2\report\spw_arms_vs_F\compare.py F S T V E3`。

**综合结论与「通用扰动方案」的判据（2M 节，2026-09-24）**：把 A–M 与 S/T/V/E3 的全部实测收成**一条统一律 + 一个判据 + 一个靶心**。**统一律**：`spread(lead) ≈ f(σ)^p(lead)`，`p` 只由「注入点在骨干上的位置」决定，与噪声秩 / 注入时刻 / 张量组都无关（`p(lead)` 从 +6 h 的 1.007 衰到 +342 h 的 0.466；按带 rms 口径 1–4 = 0.669 / 21–40 = 0.163 ⇒ 行星尺度耐用约 4 倍），机制是「乘性 δW 被投影到一大片衰减方向，σ 只改总增益、不改方向分布」。**判据**：若某族配置在目标时效上 `p(lead) ≤ 1` 且 `k(lead)` 跨配置持平（极差远小于补到 1 所需的倍数），则该族**无论怎么调 σ / 秩 / 注入时刻都改不了形状**，直接判负 —— 不必再扫，也不必等 CRPSS，把论文「每个新检查点都得重扫一遍」的穷举降成一个可解析判据（论文自己把这件事列为 future work）。量化门槛：+240..360 h 的标定比要从 0.594 补到 1 需要 `k ≈ 3.2`，五臂实际给到 0.37 / 0.40 / 1.47 / 2.4，**没有一个够** —— 所以结论不是"某配置不好"而是"**这一族不够**"。**靶心**：要抬 `p(lead)` 必须同时换**方向分布**与**时间相关**，唯一没被碰过的候选是**潜空间 AR(1) 强迫**（`z_{t+1} = ρ z_t + √(1−ρ²) ε_t`，ρ = exp(−Δt/τ)、τ ≈ 1–3 天，注在 `latent_mixer` 的残差更新上、只扰映射到行星尺度的分量），它与权重扰动正交、可直接叠在现有配置上，并已写下可证伪预测（τ = 3 天时 `k(+240..360 h) ≥ 3.2` 且 `k(+6..48 h) ≤ 1`）。其余六条与论文的差异（失败的镜像方向、秩轴是单调幅度轴、扰动同时是**偏差杠杆**、论文「刷新只对 SFNO 有效」的机制解释、集合平均口径会误判"中小尺度变暗"、「冻结是前提」）都记在 **2M.4**。

**Step 0 判决页已就位（2026-09-24，`analysis/verdict.py`）**：2M.3 的判据现在是一条**零推理、零 GPU** 的命令 —— `$P verdict F S T V E3`。只读 v2 评测归档的 `scores.csv` + `manifest.json`，打印六节：`[0]` 可比性核对（起报 / 变量 / 协议 / 步长 / 输入文件数，不一致的行前打 ⚠，**先看这一节**）、`[1]` 逐时效 `r(lead)` / `k(lead) = r^(1/p(lead))`、`[1b]` 逐日起报的符号一致性（3 个起报日里唯一能看显著性的地方）、`[2]` 基准臂自己的标定缺口 ↔ 需要的 `k`（长时效行标 `<==`）、`[3]` 四条判据表（含带功率相对值）、`[4]` 一句话判决、`[5]` 集合均值质量（`rmse` / `acc` / `act`，并提示「均值变好而 spread 没变 = 偏差杠杆，不是集合收益」）。**读法只有一句**：`k` 平 ⇒ 这族只是幅度，换个 σ 重跑同一个臂即可，别的自由度不必再扫。四臂实测原样复现 2L.11 / 2L.12 的判决（S 1.53 / T 0.38 / V 2.82 / E3 0.44，门槛 `k >= 3.41`）——**这条命令的目的就是让「要不要再跑一组」不再靠讨论**。顺带修掉判据 3 一条口径 bug：原先用 `min(1-4, 5-20)` 当选择性门槛，会把「只抬 5-20（靶心带）」的正解误判成判负，现改成只看 5-20 是否过 8% 的可辨识门槛。变体：`--dir 臂=路径`（某臂换目录）、`--p-from <σ小> <σ大>`（换骨干时现场重测 p）、`--csv`（落盘）、`--base` / `--prefix` / `--var` / `--region`。

**第二条腿：AR(1) 潜空间强迫已就位（2026-09-24，`weights/spa.py` + `configs/fgvp_ar1.py`，见 2N 节）**：前四轮把「秩轴（S/V）」「时变参数轴（T）」「注入时刻轴（E3）」依次关掉 —— 三条轴的 `k(lead)` 全平，说明它们都只是**幅度等价物**；而 F 的缺口要求长时效等效 `k >= 3.41`，五个臂最大只给到 2.8。按 2M.1 的统一律 `spread(lead) ≈ f(σ)^p(lead)`、`p` 只由注入点位置决定 ⇒ 要改形状必须换**方向分布**与**时间相关**。AR(1) 强迫就是这两样一起换：**不给权重加噪声**，而是每步给潜状态一个时间上相关的乘性强迫 `z' = z ⊙ (1 + σ·n)`、`n = Σ_r ξ_t[r]·u_r[c]·v_r[tok]`，其中 `ξ_{t+1} = ρξ_t + √(1−ρ²)ε_t`、`ρ = exp(−Δt/τ)`（τ ≈ 1–3 天，对齐 SPPT 的相关时间）。它对应论文的 SPPT 类比（扰「倾向项」= 潜变量残差更新），但论文只用「沿轨迹刷 / 不刷」这个**二值**，我们把它做成**连续**的 τ；空间尺度用 `length_scale`（高斯低通相关长度）控制，对应论文 Phase 3 的「只扰最粗谱模」，且**空间基强制去 DC**（避免论文的主失效模式「区域平均过离散」）。实现上噪声走**驱动侧的额外图输入** `spw/xi`（`[batch, T, R]`，每步 48 个 float32）而不是图内随机算子 —— 因为 ONNX 单步图没有可靠的跨步状态，图内 iid 噪声**给不了时间相关**（这正是 m3 失败的原因：每步白噪声被 Δt 平滑掉）。好处是**不用落盘 51 份成员权重**（同一份图，成员差别只来自 ξ 的种子，含**进程 rank**）且**与权重扰动正交可叠加**；代价是图被改写（权重一字节未动、无训练）。5 个臂：`A32`（主臂，行星尺度 + τ=3 天）/ `A1`（尺度轴）/ `A32T1`（τ=1 天）/ `A32T0`（iid 端点，与 `A32` **共用同一份图**、只差驱动侧的 τ ⇒ 唯一变量对照）/ `W32`（叠加 F 的 m1a 权重，复用 F 已落盘的成员集）。本地验收：`self-test` 233 条全过（含 σ=0 与源图**逐位相同**、同 ξ 两次前向**逐位相同**、基正交/无 DC、ρ 与实测 lag-1 一致）；真实 FGVP 图 `latent build` → 19 节点 / +1 输入 / 87.4 MB / 2.9 s，`latent audit` 结构自洽。判决：`$P verdict F A32 A1 A32T1 A32T0 W32`（读法不变：`k` 平 ⇒ 只是幅度；`k` 上升 ⇒ 真改了形状）。

**判决页第一次跑出结论（2M.7 / 2M.8 节，2026-09-24）**：在 WSL 里实跑 `verdict F S2 T02 T08`（归档根要显式给：`SPW_EVAL_BASE=/mnt/d/xmetai-evaluation-result` —— `verdict.py` 的默认值是 Windows 路径，WSL/Linux 下解析不了），拿到三条新事实。**① `S2`（`latent_mixer` + 通道级 ξ + σ_full=0.0341）是至今唯一过判据 3 的臂**：带功率 `1_4=1.012 / 5_20=1.111 / 21_40=0.860 / 41_64=0.984 / 65_128=0.959` —— 唯一涨的是天气尺度 5-20（+11.1%），行星带不动、中尺度反而降 14%、谱尾不脏；而它的 `k(lead)` 仍平（1.15 → 1.32，极差 1.15x）⇒ **「注对了尺度」与「幅度够不够」是两条独立判据，`k` 判负只淘汰「换幅度」、不能淘汰「换尺度」**，这是 2M.3 判据的一个必要限定，也是能写进论文的一条。**② `T02` 口径不可比**：`[0]` 打 ⚠540（F/T08/S2 都是 1080），manifest 的路径抽样只见 `member_000/002/004` ⇒ 只有 **3 成员**（540 = 3×3×60），spread 被系统性压低（k=0.18 正好约为 T 的一半），要进结论必须 6 成员重跑。**③ `T` 的 σ 轴有最优点**：`T`(σ_full=0.04) rmse −4.4% / act +11.2%，`T08`(σ_full=0.08) rmse **+2.7%** / act **−0.3%** ⇒ 2L.11 的「T 是偏差/锐度杠杆」**不是单调的**，0.04 附近就是 optimum。**④ 零成本追加：`S2` 的 CRPS 读数（2M.7 ⑤）** —— CRPS 本来就在同一批 `scores.csv` 里，按时效分段读`臂/F` 的 CRPS 比：`S2` 短时效 **+11~15%**（3/3 天同向变差）、长时效 **−6%**（3/3 天同向变好，幅度逐天从 −10.7% 衰减到 −1.3%），**交叉点在 +240 h**；全时效均值 `S2` = −0.15%（打平 F），而 `E3` = **−1.43%**（最好）—— 但 `E3` 的方向是反的（短时效好、长时效差）⇒ **CRPS 必须带时效分段报，否则会把 E3 误排成最优配置**（风险表 #6 再次被证实）。`S2` 在每一段都严格优于 `S`（1.108<1.196、1.065<1.138、0.938<0.961）⇒ **降幅档是正确方向**。下一步（2M.8，按性价比）：**第 1 顺位 = `S2B`（`sigma_full` 0.0341 → ≈0.027-0.030）** —— 按 `S`→`S2` 两点外推，短时效 CRPS 代价对 σ 的斜率 ≈ 14.9/单位，把 1.108 压回 1.000 只要 σ_full ≈ 0.027，而长时效只从 0.938 走到 ≈0.910（仍净赚），这是目前**唯一有定量靶心**的配置；第 2 顺位 = `T02` 6 成员重跑（修可比性）；第 3 顺位 = 把 `S2`/`S2B` 扩到 51 成员 × 9 起报钉死量级；第 5 顺位 = AR(1) 潜空间强迫 5 臂（`S2` 管尺度、AR(1) 管时间，两者正交可叠）。

m1a 的成员由框架在建 session 前按 rank 分片生成（`perturbed.mNNN.data` 是直接 `wb` 打开的，
两个 rank 同时造同一个成员会写出交错的字节），并且缺省**一次只驻留一个成员的 session**——
51 个同时挂住会按 ~3 GB/份吃光显存。所以 m1a 的循环是成员外层、数据集内层。显存够大就把
`resident_members` 调大（见「多卡运行」）：池子按 LRU 留 N 份，换回来时不重建 session、
不重预热 cuDNN。

```bash
P="python -m xmetai_inference.perturbation"     # 分析 / 测试命令
$P self-test          # 合成图上的 233 条断言：改写无损 / 噪声重抽 / 种子纪律 / 外部数据不复制 /
                      # m1a 成员冻结与分片生成
$P e2e --members 51   # 端到端预检：真实框架路径下 m3/m1a 的成员是否互不相同、m1a 的 cli
                      # 编排是否落对目录、noise_op 开销实测、分段 rollout 的段边界（61 条断言）
$P probe  <onnx> --m1a-members 8                 # 探针：量离散度，不必跑完整推理
$P check  <out_dir>   # 成员是否互不相同、离散度是否随时效增长
$P spread <out_dir> ..  # 标定口径曲线：σ̂ / σ̂-rmse 比 / 大尺度占比 / k50 / 涡动相关，多目录给对比表
$P verdict F S T V E3   # 判决页（零推理、零 GPU）：读评测归档算 r/k + 带功率，先判死一族再决定跑不跑 CRPSS

# 造产物没有命令行：改 weights/spw.py 或 weights/spa.py 末尾的
# ACTION / MODEL 等字面量，然后直接运行
python -m xmetai_inference.weights.spw   # ACTION = groups / members
python -m xmetai_inference.weights.spa   # ACTION = build / audit（AR(1) 潜空间强迫）

SPW_AB=S2B xmetai-infer --model fgvp_spw_ab   # 生产候选：latent_mixer + 通道级 xi + sigma_full=0.027（见 2M.8 第 1 顺位）
```

挑注入点：改 `weights/spw.py` 主块里的 `UNDER` / `DEPTH` / `TOP`（子组扫描），或直接调 `print_group_report(path, under=..., depth=...)`；一个 group token 里可以用逗号写**多个前缀的并集**（FGVP 的 12 个 block 是交错的，只能用并集），并集算**一个组**、吃一份方差预算。

m3 的三个实测静默错误点：onnxruntime 的随机算子在**不同 session** 里序列逐位相同（多卡
必须按 rank 换种子，否则 51 个成员里真正独立的只剩 13 个）；**同一个 seed** 会让所有张量的
扰动完全相关；σ→0 时扰动图与源图因 kernel/融合变化**不可能逐位相同**（等价性只能按相对
容差判）。m1a 没有第一条问题——成员是各自落盘的权重，与 session 随机流无关。

m3 的噪声开销也实测了：随机算子只在 CPU 上跑、而且是单线程（约 120M 元素/s，把线程数调大
反而更慢），按 FGVP 的 731.45M 元素/前向外推 ≈ **6 s/前向**——m3 的瓶颈是 CPU 生成随机数，
不是显存也不是 PCIe。**m1a 把这一项整个消掉了**（扰动在生成阶段算一次），代价换成磁盘：
每成员 = 85 MB 图 + 被扰动张量字节数（`decoder_group` 710 MB/成员、`all` 2.93 GB/成员），
其余权重靠外部数据**硬链接**共享源 `.data`——所以 51 份权重的磁盘是 36 GB（decoder_group）
而不是 51 × 2.8 GB。成员集是**扁平布局**（`base.data` + `<stem>.mNNN.onnx` +
`perturbed.mNNN.data` 同目录，不逐成员开子目录）：`base.data` 优先硬链接（要求与源 `.data`
同卷、源文件属当前用户），不可用时整份复制**一次**。**不用符号链接**——onnxruntime 会把外部
数据解析成真实路径并要求它落在模型目录内，软链会被直接拒绝（见 perturbation/README 坑 6）。
另外：`cuda_devices` 填的必须是容器里真实可见的卡号，`gpus` 个数要与之相等，否则会报
`CUDA failure 100: no CUDA-capable device is detected`。

第一次上服务器建议分两步：先单卡小规模跑通链路（`configs/fgvp_spw_prod_s2b.py` 的文件头
里写了现成的 smoke 命令，**务必带 `--out`** 指到临时目录，否则会和正式产物混在一起），
`$P check <out> --steps all` 过了再加到 51 成员。

细节、m3 的失败诊断与四处偏离论文的地方（README 2A 节）、m1a 的落盘布局与选组理由（2B 节）、
基线实测曲线与 A/B 协议（2C 节）、第一轮归档（2D 节）、第二轮归档（2E 节）、
生产跑协议（2F 节）、超参数敏感性复盘（2G 节）、外部独立评测复盘（2H 节）、
自由度地图与综合结论（2I 节）、
N 轴对比方案（2J 节）、
创新候选与反攻路线（2K 节）、
口径修正与形状判决（2L 节）、
上服务器前必做的四件事见 `xmetai_inference/perturbation/README.md`。

## 源码评测工具

本项目定位为推理库，PyPI 包只安装 `xmetai-infer`，不安装评测命令。仓库根目录的
`util/` 是开发者内部工具，下载源码后使用，详细参数见 `util/README.md`：

```bash
python util/eval_single_util.py    # 评预置 config（默认 fengqing ）的输出
python util/eval_ens_util.py       # 评 fuxi_ens 的输出
python util/compare_outputs.py run_new run_old   # 两次预测互比，判断结果是否一致
```

评测默认值（`--forecast`、`--vars`、单位换算、实况 store）**全部从推理 config 读取**，
与推理天然同源；换配置用 `XMETAI_EVAL_CONFIG`，覆盖实况 store 用 `--stores`。

## 模型契约与单位

模型固定契约由 `xmetai_inference/models/*.py` 声明，数据源和完整 Processing 流程由
`xmetai_inference/configs/*.py` 声明，不复用重复的 spec JSON。

关键单位约定：`z` 使用 `m2 s-2`（位势）、`q` 使用 `g kg-1`、`tp` 使用 `mm`、
辐射场使用 `Wh m-2`（6h 累积）；规则网格纬度为北→南，经度为 0–360°。

## 权重管理

权重文件（`.onnx` / `.pt2` 及统计量）统一放在根目录 `model_artifacts/`，整个目录
不上传 Git：

- 本地跑：把权重放进 `model_artifacts/`，内置模型配方会从这里读取；
- 服务器跑：用 `--model-path` 或对应模型的环境变量覆盖默认路径；
- FuXi-2.1 的 `mean.nc` / `std.nc` 必须与 `.pt2` 放同一目录；
- FengQing 除 `fengqing_pre.onnx` 外还需 `mean_std/`（逐像素统计量）和
  `utils/constant_masks.npy`，见 `models/fengqing_pre_onnx.py` 的查找顺序；
- Pangu-Weather 需 `pangu_weather_6.onnx` + `pangu_weather_24.onnx`（同目录），
  归一化已烘焙进图、无统计量文件；License BY-NC-SA 4.0，商用禁止；
- IWC FGVP GDN2 需匹配当前 onnxruntime 版本的 `xmetai_onnx_plugins.so`（ABI 绑定）。

## 已知限制与注意事项

- `windowed_output=False` 的图（IWC FGVP GDN2、FengQing）只出预报帧，不能开 GPU 常驻。
- 自定义算子 `.so` 与 onnxruntime 版本 ABI 绑定，跨环境（尤其容器）运行前要确认。
- 权重/数据路径写在 config 里（多为服务器绝对路径），换机器用环境变量覆盖。
- NetCDF4/HDF5 非线程安全，输出写盘为单线程串行；写失败按 step 兜底存 `.npy`。
- 权重扰动（SPW）分两种机制：`m3` 下每个 rank 各持一份 SPW 图（种子按 rank 偏移，这是为
  了不出现重复成员），带外部数据的模型默认把图写在模型同目录，只读目录用
  `XMETAI_SPW_CACHE_DIR` 覆盖；`m1a` 下成员集（扁平布局，含共享的 `base.data`）默认也落在
  模型同目录，且**必须与源 `.data` 同卷、源文件属当前用户**，硬链接才成立（不可用时整份
  复制一次，所有成员共用；**不用符号链接**，会被 onnxruntime 的真实路径校验拒绝）。
  成员集目录名带「落盘格式版本」，换过版本后请手工删掉旧目录。见
  `xmetai_inference/perturbation/README.md`。
- 调度状态（任务表、认领目录、完成标记、worker 心跳）都落在 `{output_dir}/.xmetai_plan/`，
  跟着输出目录走：换 `--out` 就等于换一份断点续跑的账本；要清账就删这个目录。
- **同一个 `output_dir` 不能并发跑两次**：主控启动时扫一遍 `.xmetai_plan/` 里的 worker
  心跳，发现另一个还活着的主控就直接退出（否则两批 worker 会抢同一批产物）。真要并行，
  换 `--out`，或给 `scheduling.plan_dir` 指一个独立目录。
- `resume` 取 `off` 时只认**本次运行**写的完成标记：上一次跑出来的产物会被重算覆盖。这是
  「想整份重跑、又不想手删计划目录」时的口径；跨运行续跑用默认的 `outputs`。

## License

本项目自身尚未声明许可证。`TODO:` 如需开源，请补充 `LICENSE` 文件。

集成的 Pangu-Weather 模型权重遵循 **CC BY-NC-SA 4.0**（署名-非商业-相同方式共享），
仅限非商业用途；详见 `model_artifacts/Pangu-Weather-main/README.md`。
