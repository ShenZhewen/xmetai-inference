# 气象模型推理框架

把 ERA5 再分析数据转换成模型输入，执行自回归天气预报并将结果保存为 NetCDF。

整套框架遵循「数据源可插拔 + Processing 管线可配置 + 后端可插拔」：模型类只声明
固定输入契约，每份运行配置完整声明数据集、输入处理、回填处理和输出处理。

## 目录

- [支持的模型](#支持的模型)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [架构与数据流](#架构与数据流)
- [快速开始](#快速开始)
- [配置参考](#配置参考)
- [多卡运行](#多卡运行)
- [模型契约与单位](#模型契约与单位)
- [权重管理](#权重管理)
- [已知限制与注意事项](#已知限制与注意事项)
- [License](#license)

## 支持的模型

| 模型 | 后端 | 类型 | 成员 | 归一化 |
|------|------|------|------|--------|
| FuXi-Ens | `onnx`（ONNX Runtime） | 集合 `ensemble` | 51 | 归一化已烘焙进图，输入/输出均为物理量 |
| FuXi-2.1 | `pt2`（torch.export） | 确定性 `deterministic` | 1 | Processing 管线使用 `mean.nc` / `std.nc` |
| IWC FGVP GDN2 | `onnx`（ONNX Runtime） | 确定性 `deterministic` | 1 | 归一化及反归一化已烘焙进图 |
| AIFS 1.1 | `ckpt`（anemoi SimpleRunner） | 确定性 `deterministic` | 1 | 归一化已烘焙进 ckpt |

## 核心特性

- **数据源可插拔**：每个 loader 只实现 `load(time) -> xr.Dataset`，支持 `era`（逐变量 .nc）、`zarr`（通用 store）、`era5_store`（ERA5 基础库多组 zarr）。
- **后端可插拔**：`onnx` / `pt2` / `ckpt` 三种执行引擎与模型语义分离，引擎只负责加载和跑。
- **模型契约驱动**：通道、网格、历史窗口和状态表示由模型类声明。
- **完整运行配方**：config 同时声明 loader、`pre_processors`、
  `recurrent_processors` 和 `output_processors`。
- **异步落盘**：后台单线程写 NetCDF，GPU 不等磁盘写；写失败时兜底存 `raw_step_XXX.npy`。
- **多卡数据并行**：按起报时间 / 成员切分到多张卡，每进程经 `CUDA_VISIBLE_DEVICES` 隔离单卡。


## 目录结构

```
.
├── pyproject.toml       # PyPI 构建元数据和 CLI 入口
├── xmetai/              # 可安装的 Python 包
│   ├── inference.py          # 配置、多卡调度、自回归与异步落盘
│   ├── processing/           # 输入、回填、输出与模型装配
│   ├── loaders/              # ERA / Zarr / ERA5 Store 数据源
│   ├── backends/             # ONNX / PT2 / checkpoint 执行引擎
│   ├── models/               # 具体模型及按需加载注册表
│   ├── configs/              # 内置完整运行配方
│   ├── logging_util.py       # 推理日志配置
│   └── weights/README.md     # 模型权重与部署产物说明
├── util/                # 源码附带的内部评测工具，不进入 PyPI wheel
├── scripts/             # Bash 推理启动脚本
└── model_artifacts/     # 本地模型产物，整个目录不上传 Git
```

## 架构与数据流

一条预报从数据到落盘的完整链路：

```
数据层        →   前处理管线       →   输入装配       →   推理        →   保存
loader        Processors       Assembler      model.run     to_dataset
```

| 步骤 | 谁负责 | 干什么 |
|------|--------|--------|
| `Processor` | 数据适配层 | 静态场合并、字段映射、单位转换、网格对齐和量级检查 |
| `Assembler` | 数据适配层 | 按模型声明装成五维张量或 Anemoi field 字典 |
| 输入 Processor | Adapter | 物理量 → 模型工作空间；FuXi-2.1 做 z-score，IWC 填充 SST 陆地缺失值 |
| `forward` 自回归 | `run()` 主循环 | `state = result` 回填；回填 Processor 可清零诊断通道 |
| 输出 Processor | Adapter | 模型工作空间 → 物理量；FuXi-2.1 做反 z-score 和 tp expm1 |
| `to_dataset` / `_transform` | 保存层 | 物理量 → 用户单位 + 落盘；tp→mm 统一在此处理（q 保持 g/kg） |

> 数据进入模型前只经过统一 Processor 管线。模型文件内部已经融合的归一化不在框架
> 重复执行；输出反变换和自回归回填规则也由同一管线对象管理。

## 快速开始

### 前置条件

- Linux 环境（驱动脚本为 bash；Windows 建议 WSL2 或 Git Bash）
- NVIDIA GPU + CUDA（ONNX / PT2 推理）
- 模型权重文件与 ERA5 输入数据（两者均不进 git，见 [权重管理](#权重管理)）

### 安装依赖

核心依赖：

```bash
pip install -e .
```

按运行内容安装可选依赖：

```bash
pip install -e ".[netcdf,zarr,onnx]"  # FuXi-Ens / FGVP
pip install -e ".[netcdf,zarr,pt2]"   # FuXi-2.1
pip install -e ".[netcdf,zarr,aifs]"  # AIFS
```

### 一条命令推理

安装后不需要修改或调用 Bash 脚本，直接选择模型和数据源：

```bash
xmetai-infer --model fuxi_ens --data era5_store
xmetai-infer --model fuxi21 --data era5_store
xmetai-infer --model fgvp --data era5_store
xmetai-infer --model aifs11 --data era5_store
```

起报时间、步数、成员数、GPU 和输出目录都在对应配置文件中声明。
`scripts/run.sh` 仅保留给需要准备 ONNX Runtime 动态库环境的旧部署方式，不是标准入口。

### 任意路径 / 本地运行

安装后使用 `xmetai-infer` 命令。`--model` 选择注册模型配方，`--data` 选择 Loader，
其他参数只覆盖本次任务：

```bash
xmetai-infer --model fuxi_ens --data era5_store \
  --times 2025010600 --steps 10 --members 21 --gpus 1 --out ./output
```

- 模型类和三阶段 Processor 由注册模型配方声明。
- `--model-path` 可以替换配方中的权重文件。
- `--data-root` 可以替换 Loader 的默认数据地址。
- `era5_store` 的数据根目录可用环境变量 `ERA5_STORE_ROOT` 覆盖。
- 复杂场景仍可使用 `--config /path/to/custom.py` 加载外部配方。

### 扩展自己的模型和数据集

用户只安装主框架，在自己的项目中编写 Model、Loader 和 config 即可，不需要修改
`xmetai` 源码，也不需要把扩展代码发布到 PyPI。

#### 1. 安装主框架

正式安装：

```bash
pip install xmetai-inference
```

从源码开发时：

```bash
pip install -e .
```

#### 2. 创建扩展项目

```text
my_forecast_project/
├── config.py
├── weights/
│   └── my_model.pt2
├── models/
│   ├── __init__.py
│   └── my_model.py
└── loaders/
    ├── __init__.py
    └── my_loader.py
```

用户的模型代码、Loader、权重和数据都保存在自己的项目中，不需要复制到 Python 的
`site-packages` 或 `xmetai` 安装目录。

#### 3. 编写 Model

Model 负责声明输入输出通道、网格、历史窗口和时间步长，并继承合适的 Backend。PT2、
ONNX 和 CKPT 模型分别继承 `Pt2InferModel`、`OnnxInferModel` 和
`CkptInferModel`；其他执行方式可以继承 `BaseInferModel` 并实现 `load()` 和
`forward()`。

```python
# models/my_model.py
from xmetai.backends.pt2 import Pt2InferModel


class MyModel(Pt2InferModel):
    input_channels = ("t2m", "u10m", "v10m")
    output_channels = input_channels
    grid = {
        "lat": {"start": 90.0, "step": -0.25, "size": 721},
        "lon": {"start": 0.0, "step": 0.25, "size": 1440},
    }
    history_steps = 2
    hour_interval = 6
    members = 1
    forecast_type = "deterministic"
    input_assembler = "tensor"
```

#### 4. 编写 Loader

Loader 负责读取用户自己的数据集，并转换成统一 State。必须实现
`load_state(time, channels=None)`；Loader 类或工厂函数可以按需接收 `path`、
`model_cls` 和 `groups` 关键字参数。

```python
# loaders/my_loader.py
import numpy as np


class MyLoader:
    def __init__(self, path=None, model_cls=None, groups=None):
        self.path = path

    def load_state(self, time, channels=None):
        # 实际项目中从 self.path 读取对应时刻和通道。
        return {
            "date": time,
            "fields": {
                "t2m": np.zeros((721, 1440), dtype=np.float32),
                "u10m": np.zeros((721, 1440), dtype=np.float32),
                "v10m": np.zeros((721, 1440), dtype=np.float32),
            },
            "latitudes": np.linspace(90.0, -90.0, 721),
            "longitudes": np.arange(1440) * 0.25,
        }
```

`fields` 中每个值是二维 `(lat, lon)` 数组。Loader 返回的字段名、单位和网格必须满足
Model 及其 Processor 的输入契约。

#### 5. 编写 config

config 直接引用用户自己的 Model 和 Loader，并完整声明三阶段 Processor。相对路径
按照 `config.py` 所在目录解析。

```python
# config.py
from xmetai.configs import InferConfig
from loaders.my_loader import MyLoader
from models.my_model import MyModel


cfg = InferConfig(
    name="my_model",
    model_path="./weights/my_model.pt2",
    model_class=MyModel,
    loader=MyLoader,
    data_root="./data",
    pre_processors=[],
    recurrent_processors=[],
    output_processors=[],
    times="2025010200",
    steps=60,
    members=1,
    gpus=1,
    output_dir="./output",
)
```

#### 6. 启动推理

进入扩展项目，然后把 config 交给安装好的命令：

```bash
cd /workspace/my_forecast_project
xmetai-infer config.py
```

也可以从任意目录传绝对路径：

```bash
xmetai-infer /workspace/my_forecast_project/config.py
```

命令行参数可以临时覆盖 config：

```bash
xmetai-infer config.py \
  --times 2025010300 \
  --steps 10 \
  --gpus 4 \
  --out ./other_output
```

没有安装主框架、直接从 `xmetai-inference` 源码根目录运行时：

```bash
python -m xmetai /workspace/my_forecast_project/config.py
```

喜欢使用 Bash 部署脚本时：

```bash
bash /workspace/szwCode/xmetai-inference/scripts/run.sh \
  /workspace/my_forecast_project/config.py
```

多卡模式会让每个 worker 重新加载同一份 `config.py`，因此外部 Model、Loader 以及
config 导入的自定义 Processor 在各进程中都能正常创建。Python config 会执行代码，
只应运行可信来源的配置文件。

### Python API

相同能力也可以从 Python 调用：

```python
import xmetai

xmetai.infer(
    model="fuxi21",
    data="era5_store",
    times="2025010200",
    steps=10,
    out="/workspace/data/shenzw/fuxi21_output",
)
```

外部 config 也可以通过 Python API 启动：

```python
xmetai.infer(
    config="/workspace/my_forecast_project/config.py",
    times="2025010200",
    steps=10,
)
```

使用外部权重或数据目录时：

```python
xmetai.infer(
    model="fuxi21",
    data="zarr",
    model_path="/models/fuxi-2.1.pt2",
    data_root="/data/era5.zarr",
    times="2025010200",
)
```

## 配置参考

### `xmetai-infer` 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | 与 `--config` 二选一 | 注册模型配方，如 `fuxi21`、`fuxi_ens` |
| `--config` | 与 `--model` 二选一 | 高级用法：内置配置名或外部 `.py` 配置 |
| `--model-path` | 配方值 | 覆盖模型权重路径 |
| `--data` | 配方值 | 覆盖 Loader |
| `--data-root` | Loader 默认值 | 覆盖数据根目录 |
| `--times` | config 值 | 覆盖起报时间或时间范围 |
| `--steps` | config 值 | 覆盖预报步数 |
| `--members` | config 值 | 覆盖集合成员数 |
| `--vars` | config 值 | 覆盖输出变量，逗号分隔 |
| `--out` | config 值 | 覆盖输出目录 |
| `--gpus` | config 值 | 覆盖使用的 GPU 数量 |
| `--cuda-devices` | config 值 | 覆盖物理卡号，逗号分隔 |
| `--log-level` | config 值 | 覆盖控制台日志级别 |

### 临时覆盖配置

```bash
xmetai-infer --model fuxi_ens --data era5_store \
  --times 2025010600..2025011200:24 \
  --steps 60 --members 51 --gpus 4 --cuda-devices 0,1,2,3
```

## 多卡运行

单次 run 中 ONNX session 独占一张卡，没有跨卡并行；多卡加速来自把起报时间或成员
分到不同卡。推理入口为每个 rank 启动独立进程，并通过 `CUDA_VISIBLE_DEVICES`
隔离物理 GPU。

## 源码评测工具

本项目定位为推理库，PyPI 包只安装 `xmetai-infer`，不会安装或暴露评测命令。
仓库根目录的 `util/` 是开发者内部工具，只有下载源码后才能使用：

```bash
python util/eval_single_util.py \
  --loader era5_store \
  --forecast /workspace/data/shenzw/fuxi_single_output_new

python util/eval_ens_util.py \
  --loader era5_store \
  --forecast /workspace/data/shenzw/fuxi_ens_output
```

详细参数、目录格式和输出说明见 `util/README.md`。这些文件不会进入发布 wheel。

## 模型契约与单位

模型固定契约由 `xmetai/models/*.py` 声明，数据源和完整 Processing
流程由 `xmetai/configs/*.py` 声明，不再维护重复的 spec JSON。

关键单位约定：`z` 使用 `m2 s-2`（位势）、`q` 使用 `g kg-1`、`tp` 使用 `mm`、
辐射场使用 `Wh m-2`（6h 累积）；规则网格纬度为北到南，经度为 0–360°。

## 权重管理

权重文件（`.onnx` / `.pt2` / `.ckpt` 及统计量）统一放在根目录
`model_artifacts/`，整个目录不上传 Git。详细约定见
`xmetai/weights/README.md`：

- 本地跑：把权重放进 `model_artifacts/`，内置模型配方会从这里读取；
- 服务器跑：可使用 `--model-path` 或对应模型的环境变量覆盖默认路径；
- FuXi-2.1 的 `mean.nc` / `std.nc` 从模型文件同目录读取，须与 `.pt2` 放一起。


## License

尚未声明许可证。`TODO:` 如需开源，请补充 `LICENSE` 文件。
