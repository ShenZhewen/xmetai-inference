# 气象模型推理 + 评测框架

把再分析数据（ERA5）喂给一个气象模型（默认 FuXi 集合预报 ONNX），做自回归滚动
集合预报，落盘成 NetCDF；再拿预测跟实况对比算 CRPS / RMSE / Spread / BSS / AROC。

整套是「数据源可插拔 + 后端可插拔 + spec 驱动」的结构：换模型只换一份 spec JSON，
换数据源只加一个 loader，换推理后端只加一个模型类，代码主体不用动。

---

## 目录结构

```
.
├── runner.py            # 主入口：后端加载 → 自回归 run → 单位换算 → 异步落盘
├── evaluate.py         # 评测：预测 vs era5_store 实况，算 CRPS/RMSE/Spread/BSS/AROC
├── adapters/           # 数据适配层（数据 → 模型输入）
│   ├── build_input.py      #   FuXi：单位推断 / 网格对齐 / 张量装配
│   └── build_input_aifs.py #   AIFS：field 字典 + N320 插值
├── loaders/            # 数据层（每个实现 load(time) -> xr.Dataset）
│   ├── era.py          #   ERA 逐变量 .nc 文件
│   ├── zarr.py         #   打包好的 zarr store（通用，无默认地址）
│   └── era5_store.py   #   ERA5 基础库（多组 zarr，默认地址内置）
├── backends/           # 执行引擎（只懂 load + 跑，不懂模型语义）
│   ├── base.py         #   BaseInferModel：run 主循环（含进度条）+ to_dataset
│   ├── onnx.py         #   ONNX Runtime 后端（CUDAExecutionProvider）
│   ├── pt2.py          #   TorchScript/PT2 后端
│   └── ckpt.py         #   checkpoint 后端（anemoi SimpleRunner，覆盖 run）
├── models/             # 具体模型（继承某引擎 + 覆盖钩子）+ 模型注册表
│   ├── fuxi_ens_onnx.py#   FuXi-Ens（归一化已烘焙进图）
│   ├── fuxi21_pt2.py   #   FuXi-2.1（z-score 空间，mean/std 反归一化）
│   └── aifs11_ckpt.py  #   AIFS 1.1（GNN，归一化烘焙进 ckpt）
├── scripts/            # 驱动脚本（bash 封装：多卡推理 / 评测）
│   ├── run_fuxi_ens.sh #   FuXi-Ens 集合推理（多卡）
│   ├── run_fuxi_pt2.sh #   FuXi-2.1 确定性推理（单卡）
│   ├── run_eval.sh     #   评测的 bash 封装
│   └── run_aifs_minimal.sh # AIFS 最小闭环冒烟
├── specs/              # 模型 spec JSON（换模型 = 换一份 spec，见 specs/README.md）
│   ├── fuxi_ens.json   #   FuXi-Ens 模型 spec（通道、单位、量程、网格）
│   ├── fuxi21.json     #   FuXi-2.1 模型 spec
│   └── aifs11.json     #   AIFS 1.1 模型 spec
└── weights/            # 模型权重（.onnx/.pt2/.ckpt + mean/std，gitignore 不提交）
```

---

## 推理数据流（四层框架）

```
数据层 → 数据适配层 → 推理 → 保存
loader   build_input   model.run   to_dataset
```

一条预报从头到尾的数据变换，前处理 / 后处理都挂在模型上、由 `run()` 统一调用：

```
build_input（数据适配）  →  pre_process（模型，z-score）  →  forward 自回归
  →  post_process（模型，反 z-score）  →  to_dataset/_transform（单位换算 tp→mm、q→kg/kg）
```

| 步骤 | 谁负责 | 干什么 |
|------|--------|--------|
| `build_input` | 数据适配层 | loader 读出的物理量 → 按 spec 做单位推断、网格对齐、量程校验、张量装配 |
| `pre_process` | 模型 | 物理量 → 模型工作空间。FuXi-2.1 是 z-score（tp 先 log1p）；FuXi-Ens / AIFS 归一化烘焙进图 / ckpt，恒等直通 |
| `forward` 自回归 | `run()` 主循环 | `state = result` 回填；回填前 `zero_recurrent` 清零诊断通道（辐射/降水不反馈） |
| `post_process` | 模型 | 工作空间 → 物理量。FuXi-2.1 反 z-score（tp expm1） |
| `to_dataset` / `_transform` | 保存层 | 物理量 → 用户单位 + 落盘。tp→mm、q g/kg→kg/kg 这步单位换算在框架里统一做（spec 驱动），所有模型共用 |

> 边界：`pre_process` / `post_process` 是模型自己的变换（z-score 等，由模型写）；
> 单位换算（tp→mm、q→kg/kg）是 spec 驱动的数据适配，统一留在 `to_dataset` 的 `_transform`，
> 不在模型里重复实现。

---

## 依赖

```bash
pip install numpy pandas xarray netCDF4 onnxruntime zarr
# 可选：torch（用于自动探测显存上限；无 torch 时按比例退回）
```

---

## 快速开始

### 1. 完整推理（多卡，推荐）

```bash
bash scripts/run_fuxi_ens.sh   # FuXi-Ens 集合（默认 4 卡 51 成员）
bash scripts/run_fuxi_pt2.sh   # FuXi-2.1 确定性（单卡单成员）
```

默认跑 `2025010600..2025011200`（间隔 24h，共 7 个起报）× 61 步 × 51 成员，
4 卡并行。所有可配置项都能用环境变量覆盖：

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `MODEL` | `fuxi_onnx/fuxi_ens.onnx` | 模型路径 |
| `BACKEND` | `onnx` | 推理后端：`onnx`/`pt2` 或模型名 |
| `LOADER` | `era5_store` | 数据源：`era`/`zarr`/`era5_store` |
| `START` / `END` / `FREQ` | `2025010600` / `2025011200` / `24` | 起报时间范围与间隔 |
| `STEPS` | `61` | 预报步数 |
| `MEMBERS` | `51` | 集合成员数 |
| `GPUS` / `CUDA_DEVICES` | `4` / `0,1,2,3` | 卡数及物理卡号 |
| `VARS` | `z500,u200,v200,msl,tp` | 要保存的输出变量 |
| `OUT` | `output` | 输出目录 |

### 2. 单次起报（直接调 runner.py）

```bash
python runner.py --model fuxi_ens.onnx --time 2025010600 \
  --loader era5_store --steps 10 --members 21 --out ./output
```

`--model` 缺省后端时按扩展名自动识别（`.onnx`→onnx，`.pt2`→pt2）。
`--loader` 的数据地址写死在各 loader 的 py 里，一般不用传；`--zarr` 可临时覆盖
（zarr 这个通用 loader 无默认地址，必须用 `--zarr` 传路径）。

不写 `--out` 时只做一次输入构建校验（不加载模型、不推理），用来先确认数据没问题。

### 3. 评测预测 vs 实况

```bash
bash scripts/run_eval.sh          # 默认评测 20250106 起报，60 步 x 51 成员
```

结果写到 `eval_results/eval_2025010600.csv`，每个变量每步一行，字段：

| 指标 | 含义 | 适用变量 |
|------|------|----------|
| `rmse` / `mae` | 集合平均 vs 实况的 RMSE / MAE（cos(lat) 加权） | 全部 |
| `crps` | 连续分级概率评分（0=完美） | `z*` 位势高度 |
| `spread` / `ssr` | 集合离散度 / 离散度-误差比（≈1 为校准良好） | `z*` |
| `bss` | Brier 技巧评分（1=完美） | `tp` 降水 |
| `aroc` | ROC 曲线下面积（1=完美，0.5=随机） | `tp` |

> 单位已对齐：位势高度保持 m²/s²（论文口径），降水实况 ×1000（m→mm）。
> 网格已对齐：lat 翻到北→南、lon 滚到 0→360。

---

## 推理参数一览（`runner.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | 必填 | 模型路径（.onnx/.pt2） |
| `--backend` | 按扩展名 | `onnx`/`pt2` 或 `fuxi_ens_onnx`/`fuxi21_pt2` |
| `--time` | — | 单次起报 `YYYYMMDDHH`（与 `--start/--end` 二选一） |
| `--start` / `--end` / `--freq` | — | 一段时期的起报范围与间隔（小时） |
| `--spec` | `specs/fuxi_ens.json` | 模型 spec JSON |
| `--loader` | `era` | `era`/`zarr`/`era5_store` |
| `--zarr` | 无 | 覆盖数据源默认地址（zarr 必传） |
| `--steps` | `10` | 预报步数 |
| `--members` | `1` | 集合成员总数（多卡时指全体成员数） |
| `--history` / `--interval` | 用 spec | 输入历史帧数 / 步长（小时） |
| `--device` | `0` | GPU 设备号（多卡时由 CUDA_VISIBLE_DEVICES 隔离） |
| `--gpu-mem` | `0.7` | 显存占用比例 |
| `--out` | 空 | 输出目录；不写只做输入校验 |
| `--vars` | 全部通道 | 要保存的输出变量（逗号分隔），如 `z500,u200,v200,msl,tp` |
| `--verbose` | 关 | 打印逐步耗时明细、输入通道统计 |

---

## 数据源（loaders）

每种数据源只实现一个接口 `load(time) -> xr.Dataset`，返回自描述的数据集，
`build_input` 对它的单位、层级、网格一视同仁。**默认地址写死在各自 py 里**，
需要临时换目录时用环境变量覆盖，不用改代码：

| loader | 默认地址 | 覆盖方式 |
|--------|----------|----------|
| `era5_store` | `/workspace/data/liujunjie/era5_foundation_store` | `ERA5_STORE_ROOT` 或 `--zarr` |
| `era` | `/workspace/data/xmetai-data/ERA/nc/0p25` | `DATADIR` |
| `zarr` | 无（通用） | `--zarr` 必传 |

`era5_store` 是「基础库」，把数据按物理组拆成多个 zarr store（pl 84 通道 /
sfc 15 通道 / cldrad 8 通道 / soil / wave / static），loader 打开按需的组、
通道名归一化（`z_50`→`z50`）、多组沿 channel 维 concat、1h 累积场（ssr/ssrd/fdir/ttr/tp）
×6 归一化到 6h 步长。哪些通道该用、什么顺序、什么单位，全部由 spec 决定，
loader 不认识模型。

---

## Spec JSON

| 字段 | 含义 |
|------|------|
| `model.history_steps` / `model.hour_interval` | 输入历史帧数、步长（未传参时用） |
| `grid` | 输出网格（lat 北→南、lon 0→360） |
| `levels` | 气压层级列表（FuXi 标准 13 层：50…1000 hPa） |
| `layout` | 通道排列：`@levels` 的变量按层级展开成 `var{level}`，否则是地面变量 |
| `variables` | 每个变量的 `unit`（规范单位）、`accepts`（可换算单位及 scale/offset）、`range`（量程，支持 `per_level`）、`aliases` |

`build_input` **不信任 units 属性，也不瞎猜**：每个通道的数值都拿 spec 声明的
物理量程做假设检验，单位标错会被拦住（`OUT_OF_RANGE`/`CONFLICT`）而不是悄悄写坏；
只会自动做单位换算、纬度翻转、经度滚动；分辨率对不上直接报错，不做插值。

---

## 输出格式

集合预报（members>1）：

```
{out}/{起报日 yyyymmdd}/member_{成员3位}/{预测步3位}.nc
```

确定性（members=1）：

```
{out}/{起报日 yyyymmdd}/{预测步3位}.nc
```

- 变量名大写：`Z500/U200/V200/MSL/TP`（层级编码在名字里）；
- lat 保持模型自身方向（北→南，90→-90），lon 0→360；
- 单位换算：`tp` 输入侧 m→mm（build_input 按 spec 换算）、输出侧仅 clamp≥0；`q` g/kg→kg/kg（×0.001）；
- 落盘走**后台单线程 writer**（GPU 不等磁盘写，netCDF4 非线程安全所以只串行写）；
  写失败时兜底存 `raw_step_XXX.npy`。

---

## 多卡

单次 run 是 ONNX session 独占一张卡，没有跨卡并行；多卡加速来自**把起报时间 /
成员分到不同卡**。`scripts/run_fuxi_ens.sh` 已经封装好：给每个 rank 一个进程，`CUDA_VISIBLE_DEVICES`
单独隔离一张卡（进程内 device 恒为 0），`LOCAL_RANK`/`WORLD_SIZE` 用来切分：
起报时间多于 1 个就按起报时间连续切块，只有单个起报才按成员拆。

---

## 进度条

推理的 run 循环**默认就有一条进度条**（不用 `--verbose`）：

```
[rank 0/4] 0106 [████████░░░░░░░░░░░░░░░░] 20/61 步 ( 32.8%) 平均0.82s/步 ETA  33.6s
```

`--verbose` 才回到逐步耗时明细。多卡时每个 rank 刷自己那一行（前缀带 rank 和起报）。
