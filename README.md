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
- **多卡数据并行**：按「起报时间 × 成员」分摊到多卡，每进程经 `CUDA_VISIBLE_DEVICES`
  隔离单卡。

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

单次 `forward` 不跨卡；加速来自把「起报时间 × 成员」的任务分摊到不同卡。框架为每个
rank 启动独立子进程，经 `CUDA_VISIBLE_DEVICES` 隔离物理卡，按任务数更均衡的方式在
「按起报切」和「按成员切」之间选择。

ONNX Runtime 单 session 独占一张卡；支持 GPU 常驻（IOBinding，state 全程待在显存，
只在落盘时搬回 CPU）。仅当模型前后处理/回填是恒等（图输入输出同形）时才能开，由
`config` 的 `gpu_state` 或环境变量 `XMETAI_GPU_STATE` 控制；`windowed_output=False`
的图（只出一帧）不具备开快路径的条件（启动期会拦下）。

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

## License

本项目自身尚未声明许可证。`TODO:` 如需开源，请补充 `LICENSE` 文件。

集成的 Pangu-Weather 模型权重遵循 **CC BY-NC-SA 4.0**（署名-非商业-相同方式共享），
仅限非商业用途；详见 `model_artifacts/Pangu-Weather-main/README.md`。