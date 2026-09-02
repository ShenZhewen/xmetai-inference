# 气象模型推理与评测框架

把 ERA5 再分析数据喂给气象模型（默认 FuXi 集合预报 ONNX），做自回归滚动集合预报并落盘为 NetCDF，再与实况对比计算 CRPS / RMSE / Spread / BSS / AROC 等检验指标。

整套框架遵循「数据源可插拔 + 后端可插拔 + spec 驱动」：换模型只换一份 spec JSON，换数据源只加一个 loader，换推理后端只加一个模型类，代码主体不用动。

## 目录

- [支持的模型](#支持的模型)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [架构与数据流](#架构与数据流)
- [快速开始](#快速开始)
- [配置参考](#配置参考)
- [评测](#评测)
- [多卡运行](#多卡运行)
- [spec 规范](#spec-规范)
- [权重管理](#权重管理)
- [已知限制与注意事项](#已知限制与注意事项)
- [License](#license)

## 支持的模型

| 模型 | 后端 | 类型 | 成员 | spec | 归一化 |
|------|------|------|------|------|--------|
| FuXi-Ens | `onnx`（ONNX Runtime） | 集合 `ensemble` | 51 | `specs/fuxi_ens.json` | 归一化已烘焙进图，输入/输出均为物理量 |
| FuXi-2.1 | `pt2`（torch.export） | 确定性 `deterministic` | 1 | `specs/fuxi21.json` | z-score 空间，`mean.nc`/`std.nc` 反归一化 |
| AIFS 1.1 | `ckpt`（anemoi SimpleRunner） | 确定性 `deterministic` | 1 | `specs/aifs11.json` | GNN，归一化已烘焙进 ckpt |

## 核心特性

- **数据源可插拔**：每个 loader 只实现 `load(time) -> xr.Dataset`，支持 `era`（逐变量 .nc）、`zarr`（通用 store）、`era5_store`（ERA5 基础库多组 zarr）。
- **后端可插拔**：`onnx` / `pt2` / `ckpt` 三种执行引擎与模型语义分离，引擎只负责加载和跑。
- **spec 驱动**：单位推断、通道选择、网格对齐、量程校验、集合语义全部由 spec JSON 声明。
- **异步落盘**：后台单线程写 NetCDF，GPU 不等磁盘写；写失败时兜底存 `raw_step_XXX.npy`。
- **多卡数据并行**：按起报时间 / 成员切分到多张卡，每进程经 `CUDA_VISIBLE_DEVICES` 隔离单卡。
- **集合检验**：CRPS、RMSE、Spread、BSS、AROC；确定性模型自动只算 RMSE / MAE。


## 目录结构

```
.
├── runner.py            # 主入口：后端加载 → 自回归 run → 单位换算 → 异步落盘
├── evaluate.py          # 评测：预测 vs 实况，算 CRPS/RMSE/Spread/BSS/AROC
├── adapters/            # 数据适配层（数据 → 模型输入）
│   ├── build_input.py        # FuXi：单位推断 / 网格对齐 / 张量装配
│   └── build_input_aifs.py   # AIFS：field 字典 + N320 插值
├── loaders/             # 数据层（load(time) -> xr.Dataset）
│   ├── era.py                # ERA 逐变量 .nc 文件
│   ├── zarr.py               # 通用 zarr store（无默认地址）
│   └── era5_store.py         # ERA5 基础库（多组 zarr，默认地址内置）
├── backends/            # 执行引擎（只懂 load + 跑，不懂模型语义）
│   ├── base.py               # BaseInferModel：run 主循环 + to_dataset
│   ├── onnx.py               # ONNX Runtime 后端
│   ├── pt2.py                # torch.export (.pt2) 后端
│   └── ckpt.py               # checkpoint 后端（anemoi SimpleRunner）
├── models/              # 具体模型 + 模型注册表
│   ├── fuxi_ens_onnx.py      # FuXi-Ens（归一化已烘焙进图）
│   ├── fuxi21_pt2.py         # FuXi-2.1（z-score，mean/std 反归一化）
│   └── aifs11_ckpt.py        # AIFS 1.1（GNN，归一化烘焙进 ckpt）
├── scripts/             # bash 驱动脚本（多卡推理 / 评测）
├── specs/               # 模型 spec JSON（换模型 = 换一份 spec）
│   ├── fuxi_ens.json
│   ├── fuxi21.json
│   └── aifs11.json
└── weights/             # 模型权重（.onnx/.pt2/.ckpt，gitignore 不提交）
```

## 架构与数据流

一条预报从数据到落盘的完整链路：

```
数据层        →   数据适配层     →   推理        →   保存
loader        build_input     model.run     to_dataset
```

| 步骤 | 谁负责 | 干什么 |
|------|--------|--------|
| `build_input` | 数据适配层 | loader 读出的物理量 → 按 spec 做单位推断、网格对齐、量程校验、张量装配 |
| `pre_process` | 模型 | 物理量 → 模型工作空间；FuXi-2.1 做 z-score（tp 先 log1p），FuXi-Ens / AIFS 恒等直通 |
| `forward` 自回归 | `run()` 主循环 | `state = result` 回填；回填前 `zero_recurrent` 清零诊断通道（辐射/降水不反馈） |
| `post_process` | 模型 | 工作空间 → 物理量；FuXi-2.1 做反 z-score（tp expm1） |
| `to_dataset` / `_transform` | 保存层 | 物理量 → 用户单位 + 落盘；tp→mm 统一在此处理（q 保持 g/kg） |

> 边界：`pre_process` / `post_process` 是模型自己的变换（z-score 等）；单位换算（tp→mm）是 spec 驱动的数据适配，统一留在 `to_dataset` 的 `_transform`，不在模型里重复实现；q 官方输出即 g/kg，不做二次换算。

## 快速开始

### 前置条件

- Linux 环境（驱动脚本为 bash；Windows 建议 WSL2 或 Git Bash）
- NVIDIA GPU + CUDA（ONNX / PT2 推理）
- 模型权重文件与 ERA5 输入数据（两者均不进 git，见 [权重管理](#权重管理)）

### 安装依赖

核心依赖：

```bash
pip install numpy pandas xarray netCDF4 onnxruntime zarr
```

按模型补充：

- FuXi-2.1（PT2）：`torch`（见 `weights/fuxi2.1/requirements.txt`）
- AIFS 1.1（ckpt）：`anemoi-inference` + `torch`
- 可选：`torch` 用于自动探测显存上限；未安装时按比例退回

### 一条命令推理（部署服务器）

驱动脚本默认使用部署服务器的绝对路径（`/workspace/szwCode/xmetai-inference`），在这些机器上可直接运行：

```bash
bash scripts/run_fuxi_ens.sh   # FuXi-Ens 集合（默认 4 卡 51 成员）
bash scripts/run_fuxi_pt2.sh   # FuXi-2.1 确定性（单卡单成员）
```

默认跑 `2025010600..2025011200`（间隔 24h，共 7 个起报）× 61 步 × 51 成员，4 卡并行。

### 任意路径 / 本地运行

脚本里的 `ROOT` 是硬编码部署路径，本地运行时更推荐直接调用 `runner.py`：

```bash
python runner.py --model /path/to/fuxi_ens.onnx --spec specs/fuxi_ens.json \
  --loader era5_store --time 2025010600 --steps 10 --members 21 \
  --out ./output
```

- `--model` 的后端由 spec 的 `model.class` 决定，也支持 `--backend` 逃生舱覆盖。
- `--loader era5_store` 的数据根目录默认写在 `loaders/era5_store.py`，可用环境变量 `ERA5_STORE_ROOT` 覆盖；`--loader zarr` 则必须用 `--zarr` 传 store 路径。
- 不写 `--out` 时只做一次输入构建校验，不跑模型。

### 评测

```bash
python evaluate.py --fcst ./output --init 2025010600 --steps 61 \
  --vars z500,u200,v200,msl,tp --loader era5_store \
  --spec specs/fuxi_ens.json --out ./eval_results
```

## 配置参考

### runner.py 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | 必填 | 模型文件路径（.onnx / .pt2 / .ckpt） |
| `--backend` | 无 | 覆盖 spec 的 `model.class`，可传模型名或引擎名 |
| `--time` | 无 | 单次起报时间 `YYYYMMDDHH`（与 `--start`/`--end` 二选一） |
| `--start` / `--end` / `--freq` | 无 | 起报时间范围与间隔小时 |
| `--spec` | `specs/fuxi_ens.json` | 模型 spec JSON |
| `--loader` | `era` | `era` / `zarr` / `era5_store` |
| `--zarr` | 无 | 覆盖数据源默认地址；`zarr` loader 必传 |
| `--steps` | 10 | 预报步数 |
| `--members` | 无 | 集合成员总数（缺省读 spec） |
| `--history` / `--interval` | 无 | 输入历史帧数 / 时间步长小时（默认用 spec） |
| `--device` | 无 | GPU 设备号（多卡时由 `CUDA_VISIBLE_DEVICES` 隔离） |
| `--world-size` | 无 | 卡数（默认读 `WORLD_SIZE`） |
| `--gpu-mem` | 0.7 | 显存占用比例 |
| `--out` | 无 | 输出目录；不写则只做输入校验 |
| `--vars` | 无 | 要保存的输出变量，逗号分隔；不传保存全部通道 |

### run_fuxi_ens.sh 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL` | `$ROOT/weights/fuxiens/fuxi_ens_onnx/fuxi_ens.onnx` | 模型路径 |
| `START` / `END` / `FREQ` | `2025010600` / `2025011200` / `24` | 起报范围与间隔 |
| `TIME` | 空 | 未设 `START` 时的单次起报 |
| `SPEC` | `$ROOT/specs/fuxi_ens.json` | 模型 spec |
| `STEPS` | 61 | 预报步数 |
| `MEMBERS` | 51 | 集合成员数 |
| `OUT` | `$ROOT/output` | 输出目录 |
| `VARS` | `z500,u200,v200,msl,tp` | 输出变量 |
| `GPU_MEM` | 0.7 | 显存占用比例 |
| `GPUS` / `CUDA_DEVICES` | `4` / `0,1,2,3` | 卡数及物理卡号 |
| `LOADER` | `era5_store` | 数据源 |
| `ZARR` | 无 | `LOADER=zarr` 时必填的 store 路径 |

## 评测

`evaluate.py` 把预测结果与 `era5_store` 实况对比：

- 集合模型：`CRPS`、`RMSE`、`Spread`（ddof=1 无偏）、`Spread/RMSE`、降水 `BSS` / `AROC`。
- 确定性模型（`forecast_type=deterministic`）：只算 `RMSE` / `MAE`，集合指标留 NaN。
- 结果输出为 CSV，默认 `./eval_results`。

## 多卡运行

单次 run 中 ONNX session 独占一张卡，没有跨卡并行；多卡加速来自把起报时间 / 成员分到不同卡。`scripts/run_fuxi_ens.sh` 已封装好：每个 rank 一个进程，`CUDA_VISIBLE_DEVICES` 单独隔离一张卡（进程内 device 恒为 0），`LOCAL_RANK` / `WORLD_SIZE` 用于切分——起报时间多于 1 个时按起报时间连续切块，只有单个起报时才按成员拆分。

## spec 规范

spec JSON 描述「一个模型怎么跑」，不含权重。核心字段包括 `model.class`、`history_steps` / `hour_interval`、`forecast_type` / `members`、`grid`、`levels` / `layout`，以及每个变量的 `unit` / `accepts` / `range` / `aliases`。

三份 spec：

| 文件 | 模型 | 类型 | 成员 |
|------|------|------|------|
| `fuxi_ens.json` | FuXi-Ens | 集合 | 51 |
| `fuxi21.json` | FuXi-2.1 | 确定性 | 1 |
| `aifs11.json` | AIFS 1.1 | 确定性 | 1 |

单位与通道约定详见 `specs/README.md`。关键约定：`z` 单位 `m2 s-2`（位势）、`q` 单位 `g kg-1`、`tp` 单位 `mm`、辐射场 `Wh m-2`（6h 累积）；lat 北→南、lon 0→360。

## 权重管理

权重文件（`.onnx` / `.pt2` / `.ckpt` 及统计量）是大文件，不进 git（`.gitignore` 已忽略）。各模型对应权重与用法见 `weights/README.md`：

- 本地跑：把权重拷进 `weights/`，并改脚本的 `MODEL` / `CHECKPOINT` 指向相对路径；
- 服务器跑：脚本默认走 `/workspace/szwCode/xmetai-inference/` 绝对路径；
- FuXi-2.1 的 `mean.nc` / `std.nc` 从模型文件同目录读取，须与 `.pt2` 放一起。


## License

尚未声明许可证。`TODO:` 如需开源，请补充 `LICENSE` 文件。
