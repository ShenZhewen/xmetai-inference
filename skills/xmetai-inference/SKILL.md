---
name: xmetai-inference
description: 往 xmetai-inference 框架里新增模型。两部分——一是落地步骤（模型类 / MODEL_REGISTRY / configs 配方 / 单测 / 集成档 / 文档表格各要改什么，含最小模板指引），二是动笔前必须向模型方确认的契约细节清单（通道顺序、单位、归一化位置、NaN 填充、时间特征口径、集合噪声、batch 轴等），末尾附一份可直接复制发出的提问问卷。接入新模型、写新 config，或排查「跑起来了但输出像垃圾 / 单位不对 / 集合没 spread」时使用。
---

# xmetai-inference：新增模型接入

两部分的顺序是**先二后一**：契约没问清就写代码，多半写完才发现通道顺序或单位错了，
而这两类错误**不抛异常**，表现为「跑通了但输出是垃圾」，返工成本远高于多问一句。

---

# 第一部分：落地步骤

## 0. 先选路径

| 路径 | 适用 | 要改的东西 |
|---|---|---|
| **A. 加内置模型** | 模型要进本仓库、长期维护 | 注册表 + `models/` + `configs/` + 单测 + 文档表格 |
| **B. 外部扩展** | 模型只在自己项目里用 | 只写一个 `.py`，`model_class` 直接给**类对象**，不碰框架源码 |

路径 B 的完整说明在 `README.md` 的「扩展自己的模型和数据集」一节。
本 skill 的**第二部分对两条路都适用**，第一部分主要讲路径 A。

路径 B 的关键点：`get_model_class()` 接受类对象（`models/__init__.py:107`），
所以外部 config 可以 `from my_model import MyModel` 后直接 `model_class=MyModel`，
不必注册。

## 1. 权重落位

`model_artifacts/<name>/source|runtime/`（整个目录被 `.gitignore:13` 忽略）：

- `source/` —— 原作者发布的原始权重，保持不修改
- `runtime/` —— 项目实际加载的导出件 + 配套文件（`mean.nc`/`std.nc`/external data/`.so`）

注意 `.gitignore` 同时忽略 `*.nc`、`*.npy`、`*.zarr/`，所以统计量文件和切片数据都不进 git。

## 2. 模型类 `xmetai_inference/models/<name>.py`

继承 `OnnxInferModel` / `Pt2InferModel` / `BaseInferModel`。

**必须声明**（框架用 `getattr` 读，缺了不报错但下游会出错）：

| 属性 | 谁读 | 说明 |
|---|---|---|
| `input_channels` | dataset 选通道、normalize 校验 | 通道名列表，**顺序即 tensor 顺序** |
| `output_channels` | 落盘变量名、zero_channels / denormalize 的索引基准 | |
| `grid` | `grid_coords()`、expected_shape | 用 `GRID_025` 或自定义 dict |

**有默认值、但建议显式写出来**（`.py` 里写死比让读者去 CLI 里翻默认值强）：

| 属性 | 默认 | 位置 |
|---|---|---|
| `history_steps` | 2 | `cli.py:471` |
| `hour_interval` | 6 | `cli.py:470` |
| `members` | 1 | `cli.py:474` |
| `forecast_type` | `"deterministic"` | `cli.py:473` |
| `windowed_output` | `True` | `runner.py:174` |
| `valid_time_step_offset` | 0 | `runner.py:173` |
| `nonnegative_channels` | `("tp",)` | `base.py:74` |
| `gpu_state` | `False` | `base.py:127` |

**必须实现**：`load(path)` 与 `forward(x, step, valid_time)`（`base.py:105/109`）。

`forward` 的契约：`x` 是 `(1, in_frames, C, H, W)` float32，返回同形完整输出，
**末帧 = 本步预报**。图只出预报帧（不出完整窗口）时声明 `windowed_output=False`，
滑窗交给 `Rollout` 补 —— 这两个必须成对，配错要么静默算错、要么在
`OnnxInferModel._check_gpu_state_frames`（`onnx.py:163`）加载期被拦下。

**跨批有内部状态的模型**（如 Pangu 每 4 步从锚点跳 24h）要覆盖
`reset_runtime_state()`（`base.py:117`），否则上一个起报的状态会漏到下一个。

**最小模板**：`models/iwc_fgvp_gdn2.py`（34 行）+ `configs/fgvp.py`（80 行）——
五个现有模型里唯一的「确定性 ONNX **纯声明、零方法覆盖**」样板，`model_processing`
三段全空（恒等）。PT2 模型抄 `models/fuxi21_pt2.py`（18 行，更短）。
Pangu（115 行双 session）和 FengQing（174 行残差反演）的模型类逻辑重，不要整份照抄。

## 3. 注册到 `MODEL_REGISTRY`

`xmetai_inference/models/__init__.py:77`，**普通 dict，不是装饰器**：

```python
MODEL_REGISTRY = {
    ...
    "my_model_onnx": (
        "xmetai_inference.models.my_model",
        "MyModel",
    ),
}
```

值是 `(模块路径, 类名字符串)`，按需 `importlib` 导入。config 里用
`model_class="my_model_onnx"` 引用；未命中会抛
`ValueError: 未知模型 ...`（`models/__init__.py:116`）。

通道布局若与现有模型不同，在 `models/__init__.py` 里加通道常量
（参考 `FUXI_ENS_CHANNELS` / `PANGU_CHANNELS` 的 `_expand_channels` 写法）。

## 4. 配方 `xmetai_inference/configs/<name>.py`

必须定义名为 `cfg` 的 `InferConfig` 实例（`configs/base.py:190` 校验）。

```python
cfg = InferConfig(
    name="my_model",                 # 必填
    model_path="...",                # 必填
    model_class="my_model_onnx",
    ops_library="",                   # 自定义算子 .so（需要时才填）
    model_processing=ModelProcessingConfig(
        input=[],                     # 必须为空，见下
        recurrent=[...],
        output=[...],
    ),
    dataset={
        "type": "processed_multi_zarr",
        "paths": [...],               # zarr store 列表
        "processors": [...],          # 前处理链，顺序有意义
    },
    dataloader={"batch_size": 4, "num_workers": 4,
                "pin_memory": True, "shuffle": False, "drop_last": False},
    times="2025010200..2025010500:24",
    steps=60,
    hour_interval=6,
    members=1,
    vars="z500,t850,u10m,v10m,t2m,msl,tp",
    gpus=2,
    cuda_devices="0,1",
    output_dir="...",
)
```

**`model_processing.input` 必须为空**（`pipeline.py:263` 硬报错）——
输入侧前处理一律写在 `dataset.processors`，那里跑在 DataLoader worker 里。

**processor 链的顺序决定语义**，典型顺序：

```
unit_convert  →  geometry  →  normalize  →  fill_missing
（单位换算）     （几何对齐）  （归一化）     （填缺测）
```

`fill_missing` 放 `normalize` **前**还是**后**，决定填进去的是物理量还是归一化值。
`configs/fuxi21.py:57-61` 放在 normalize 之后，所以填的是**归一化空间的 0.0**。

可用 processor：

| 位置 | 名字 | 参数 |
|---|---|---|
| `dataset.processors` | `unit_convert` | `scales`（通道基名 → 系数，未列出默认 1.0） |
| | `geometry` | `expected_shape`（可选） |
| | `normalize` | `mean_file`、`std_file`、`log1p_channels` |
| | `normalize_fengqing` | `mean_std_dir` |
| | `fill_missing` | `rules`、`unconfigured`（`error`/`keep`） |
| `model_processing.recurrent` | `zero_channels` | `channels`（通道名列表） |
| `model_processing.output` | `denormalize` | `mean_file`、`std_file`、`expm1_channels`、`nonnegative` |
| | `denormalize_fengqing` | `mean_std_dir` |

## 5. 单测 `tests/unit/test_<name>_config.py`

**先决定抄哪一份**：

| 新模型长什么样 | 抄 |
|---|---|
| 归一化烘焙进图（ONNX 出物理量） | `test_fgvp_config.py` |
| config 侧显式 normalize / denormalize | `test_fuxi21_config.py` |
| 集合模型 | `test_fgvp_ens_config.py` |

抄过来后要换的是：`load_config("<name>")`、`EXPECTED_VARS`、`model_class` 断言、store 名单。

这些档断言的是**声明之间的一致性**，不是数值：

- 加载与身份（`isinstance(cfg, InferConfig)`、`cfg.name`）
- `model_class` 解析到**正确的那个类**（防「同通道布局的另一支」，
  如 `fgvp` 必须解析到 `iwc_fgvp_gdn2_onnx` 而非集合档 `mamba_ens`）
- 通道契约：通道数、通道名、`input_channels == output_channels` 的对称性
- `cfg.vars` 的每一项都 ∈ `model_cls.output_channels`
- `unit_convert.scales` 的键、`fill_missing.rules` 的变量名都是**真通道**
  （去层号后用基名比对）
- 网格 / `history_steps` / `hour_interval` / `members` / `forecast_type` 两侧一致
- 前处理链的顺序（图内烘焙 → 断言 `model_processing` 三段全空；
  config 侧归一化 → 断言 `log1p` ↔ `expm1` 配对、进出用同一对 mean/std）

## 6. 集成档（可选）`tests/integration/test_<name>_realdata.py`

判据原语在 `realdata_common.py`（`classify` / `field_metrics` / `compare_dirs` /
`expected_vars`），复用它们，不要重写。

**fixture 目录**（脚本里硬编码绝对路径，如
`/workspace/data/shenzw/inference_test_data/fgvp_test_data`）需要：

- 被检模型文件（`.onnx` + external `.data`）
- 配套的 `xmetai_onnx_plugins*.so`（ABI 绑定，必须随模型）
- 切片 `dataset/*.zarr`（文件名沿用源 store 名，靠 glob 对齐）
- `golden/<INIT_DIR>/`（标准答案，`{YYYYMMDD}/{NNN}.nc`；集合档是 `member_NNN/NNN.nc`）

**配方不是新增文件，而是运行时生成**：用 `string.Template` 从生产配方
`deepcopy` 派生，只覆盖 fixture 特有项（`dataset.paths` / `model_path` /
`times` / `steps` / `members` / `gpus`），落成临时 `.py` 喂 `main(["--config", ...])`。

**golden 不在仓库里**（`*.nc` 被 gitignore），由作者在正式环境跑一遍后手工归档，
并写 `golden/PROVENANCE.md` 记录出处。

**fixture 不齐就整模块 skip**：模块顶层收集 `_missing` 列表，非空则
`pytest.skip(..., allow_module_level=True)`。集成档默认**不跑**——
`pyproject.toml:64` 的 `addopts` 带 `-m 'not realdata'`，要显式
`pytest tests/integration -m realdata` 才跑（真起 GPU，代价大，故做成 opt-in）。

## 7. 文档三处表格

- `README.md` 「支持的模型」表：加一行（模型 | 后端 | 类型 | 成员 | 通道数 | 归一化），
  并同步表下的网格/时间窗口补充说明
- `README.md` 的 `xmetai-infer --model ...` 枚举：加一行
- `README.md` 的环境变量表：**仅当**模型读 env 才加
- `xmetai_inference/weights/README.md` 「当前模型」表：加一行（模型 | 运行权重 | 配套文件）

## 8. 现有 processor 不够用时

两条路，都**不用改 schema**（config 是 Python 文件，没有 JSON schema）：

**数据侧**（`processing/tensor_processors.py`，跑在 DataLoader worker）：

1. 定义 `class Xxx(TensorProcessor)`，实现 `process(self, value)`，
   输入输出都是 `[T,C,H,W]`
2. 加 `@register_tensor_processor("xxx")`
3. config 里 `dataset.processors` 挂 `{"name": "xxx", ...kwargs}`

**模型侧**（`processing/pipeline.py`，回填前 / 落盘前）：

1. 定义 `class Xxx(Processor)`，声明类属性 `stage = "recurrent"|"output"`，
   实现 `process_array(self, value)`
2. 加 `@register("xxx")`
3. config 里挂到 `model_processing.recurrent` 或 `.output`

模型侧 processor 要**原地改**时注意：`Rollout` 是先取预报帧副本再
`recurrent_transform` 回填，所以 `zero_channels` 原地改不会污染已取出的预报
（`runner.py:204-214`）。自己写 recurrent processor 时保持同样的假设。

---

# 第二部分：契约确认清单

每条给：**问什么** / **为什么问** / **答错或不问的症状** / **落在哪个字段**。

## A. 输入通道

| # | 问什么 | 答错的症状 | 落在 |
|---|---|---|---|
| A1 | 通道**清单**：变量名 + 气压层 | 缺通道直接报错，好办 | `input_channels` |
| A2 | 通道**顺序** | **不报错，输出是垃圾**。框架按 `input_channels` 顺序从 store 取变量拼 tensor | `input_channels` |
| A3 | 层序方向：1000→50 还是 50→1000 | 不报错，精度崩。实测两派都有：FuXi 系是**升序** 50→1000，FengQing / Pangu 是**降序** 1000→50 | `input_channels` |
| A4 | 地面变量顺序与集合；有没有 `ws10m` 这类派生量 | 同上 | `input_channels` |
| A5 | 通道是单帧快照还是累积量 | 累积量的时间窗见 D3/D4 | |

## B. 网格与几何

| # | 问什么 | 为什么 / 症状 | 落在 |
|---|---|---|---|
| B1 | 分辨率与形状（如 0.25° / 721×1440） | shape 不符会报错 | `grid` |
| B2 | 纬度方向：北→南还是南→北 | 不报错，南北镜像 | `geometry` processor |
| B3 | 经度起止：0–360 还是 -180–180，起始格点在哪 | 经向错位 | `geometry` processor |
| B4 | store 里 lon 的实际起始值 | `geometry` 做「纬向 north→south 翻转 + 经向 0:360 重排」；**这套 roll 方向曾经写反过，因为实测的 store lon 都从 0 起、从未触发过那条分支**。换新 store 前务必确认 | `geometry` |
| B5 | 是否规则等距网格 | 不规则网格框架不支持，要自己加 processor | |

## C. 时间

| # | 问什么 | 为什么 | 落在 |
|---|---|---|---|
| C1 | 吃几帧历史 | | `history_steps` |
| C2 | 步长（通常 6h） | | `hour_interval` |
| C3 | 图输出**完整滑窗**还是**只出预报帧** | 前者 `windowed_output=True`（如 fuxi_ens），后者 `False`（如 fgvp_gdn2），配错要么静默算错要么加载期被拦 | `windowed_output` |
| C4 | `hour`/`doy` 取**输入窗末帧**还是**目标时刻** | 实测 FuXi-2.1 官方取**末帧**；FengQing 是唯一例外（取目标时刻）。取错不报错，但时间特征整体偏移 | 后端 `forward` |
| C5 | `step` 是 0-based 还是 1-based | 实测 FuXi-2.1 是 **0-based** | 后端 `forward` |
| C6 | `valid_time` 口径：`init + step*interval` 还是目标预报时刻 | | `valid_time_step_offset`（0 / 1） |
| C7 | 有没有**跳步**结构（如每 4 步用 24h 模型从锚点跳） | 有则需要内部锚点状态 + 覆盖 `reset_runtime_state()` | 模型类 |

## D. 单位（最贵的坑）

| # | 问什么 | 实测结论 | 落在 |
|---|---|---|---|
| D1 | `q`：g/kg 还是 kg/kg | FuXi 系要 g/kg，store 是 kg/kg，**×1000**；Pangu 落盘 q 已是 kg/kg，**不换算**。判据：只有 q 异常 + RMSE 不随时效增长 | `unit_convert.scales` |
| D2 | `z`：位势 (m²/s²) 还是位势高度 (m) | 差一个 g ≈ 9.80665 | 同上 |
| D3 | `tp`：m 还是 mm；**几小时累积** | store 是 **1h 累积**，喂 6h 步长模型要 **×6**；再 ×1000 转 mm → 系数 6000 | 同上 |
| D4 | 辐射 `ssr`/`ssrd`/`fdir`/`ttr`：J/m² 还是 Wh/m²；几小时累积 | store 是 1h 累积 J/m² → Wh/m² 且 6 步 → 系数 `6/3600` | 同上 |
| D5 | 温度 K 还是 °C | | 同上 |
| D6 | 气压 Pa 还是 hPa | | 同上 |
| D7 | 风速还是 u/v 分量 | | 同上 |

`scales` 的键是**通道基名**：`_base_name` 把 `z500`→`z`、`t850`→`t` 后再匹配，
未列出的通道系数默认 1.0。

## E. 归一化

| # | 问什么 | 为什么 | 落在 |
|---|---|---|---|
| E1 | 归一化在**哪**做：图内烘焙 / 框架侧 mean.nc+std.nc / 逐像素 | 决定 `model_processing` 是恒等还是配 processor | `model_processing.output` |
| E2 | 统计量文件能不能删（图内烘焙的能否不保留） | 删错了模型直接加载失败 | `model_path` 同目录 |
| E3 | 统计量是**逐通道标量**还是**逐像素** | 逐像素走 `normalize_fengqing` / `denormalize_fengqing` | |
| E4 | `tp` 有没有 `log1p` ↔ `expm1` | **必须配对**，只配一侧数值全错 | `log1p_channels` / `expm1_channels` |
| E5 | 输出是**残差**还是**绝对值**？残差的反演公式是什么 | 残差要在**模型类 forward 里**反演，processor 管线不管。FengQing 是 `prev_norm + out * (res_std/std)` | 模型类 |
| E6 | 有没有物理约束（如 exp 前要 clamp 防溢出） | | `exp_clip_max` / `nonnegative` |

## F. 模型不吃的输入通道

| # | 问什么 | 为什么 | 落在 |
|---|---|---|---|
| F1 | 官方预处理后的 `input.nc` 里，模型会读但**不参与预报**的通道（诊断量 `ssr`/`ssrd`/`fdir`/`ttr`/`tp`）值是多少 | 实测 FuXi-2.1 在**归一化空间精确为 0**。回填时要用 `zero_channels` 清零，否则误差累积 | `model_processing.recurrent` |
| F2 | 反向：模型输出里有没有框架**不该落盘**的通道 | | `output_channels` / `vars` |

> **`zero_channels` 的坑**：它清的是**归一化空间的 0**，不是物理 0。
> 若某天 `tp` 的 mean/std 不再是 0/1，字面 `0.0` 的物理含义会随之改变。

## G. 缺测 / NaN

| # | 问什么 | 为什么 | 落在 |
|---|---|---|---|
| G1 | 哪些通道会出 NaN / Inf（`sst` 最常见） | | `fill_missing.rules` |
| G2 | 训练 / 官方推理时怎么处理的，填什么值 | | 同上 |
| G3 | 填充值该填**物理量**还是**归一化值** | 决定 `fill_missing` 在链中放 `normalize` **前**还是**后** | processor 顺序 |
| G4 | 没配规则的通道：报错还是透传 | `unconfigured="error"`（严格，如 fgvp 只给 sst 配规则）vs `"keep"`（透传，如 fuxi21 配 `{"*": 0.0}`） | `unconfigured` |
| G5 | 需要插值 / 气候态填充吗 | 框架的 `fill_missing` **只有 constant 一种策略**，其他策略得自己加 processor | |

## H. 集合语义

| # | 问什么 | 为什么 | 落在 |
|---|---|---|---|
| H1 | `members` 是「同一初值 + N 个扰动」还是「N 个初值」 | 框架里集合 = N 条 `Trajectory` **共享一个初始状态**；确定性档的 N 是 N 个起报 | `members` / `forecast_type` |
| H2 | 图内有没有随机算子、有没有 seed、能不能复现 | 没 seed 就不能逐成员对拍，集成档只能做集合统计 | 集成档判据 |
| H3 | **噪声张量形状**：`(1,...)` 广播还是 `(N,...)` | 若是 `(1,...)` 广播，开 `model_batch` 后一个 forward 里的 k 条轨迹拿到**同一份**噪声 → **集合静默坍缩**，日志无异常 | `model_batch` 能不能开 |
| H4 | spread 应该多大 | 实测 FGVP 集合短中时效 spread 只有 AIFS 的 1/4，扰动侧存疑 | 集成档判据 |

> H1 的冲突**只 warning 不报错**（`cli.py:475`）：确定性模型给 `members>1`、
> 或集合模型 `members<=1`，只打一条日志继续跑。

## I. 输出

| # | 问什么 | 落在 |
|---|---|---|
| I1 | 输出通道集合与**顺序**（含不含诊断量） | `output_channels` |
| I2 | 非负约束（`tp`） | `nonnegative_channels` |
| I3 | dtype（通常 float32） | |
| I4 | 输出形状 | `windowed_output` 配对 |

## J. 数值与工程

| # | 问什么 | 为什么 | 落在 |
|---|---|---|---|
| J1 | GPU 上能否**逐位复现** | 实测 FuXi-2.1 **不能**：高空通道每次差 ~0.2% 场变率，地面精确为 0。**不是 bug，别当回归查**。决定集成档容差怎么设（分档 + 灾难线） | 集成档判据 |
| J2 | 图输入/输出的 **batch 轴是否符号维** | 静态 batch 轴则 `model_batch>1` 会**静默退回 1**（打一条 WARNING），不报错。ONNX 后端在 `load` 时实测 | `model_batch` |
| J3 | 有没有自定义算子 `.so`，绑哪个 onnxruntime 版本 | ABI 不匹配时 `register_custom_ops_library` 报错 | `ops_library` |
| J4 | 权重许可证 | Pangu 是 BY-NC-SA 4.0，**商用禁止** | 文档 |
| J5 | 配套文件清单（`mean.nc`/`std.nc`/external data/`constant_masks.npy`） | 缺一个就加载失败 | `weights/README.md` |
| J6 | 是不是多 session（如 Pangu 6h + 24h 双图） | | 模型类 |
| J7 | 显存需求 | | `gpu_mem` |

---

# 附录：可直接复制发给模型方的提问问卷

> 以下整段可原样发出。带「（默认）」的是我方推测，请对方确认或纠正。

```
为把模型接入推理框架，需要确认以下契约。带（默认）的是我方推测，请逐条确认或纠正。

【输入】
1. 输入通道清单（变量名 + 气压层），以及**顺序**（顺序写错不会报错，只会算错）。
2. 气压层的排列方向：1000→50 还是 50→1000？
3. 地面变量有哪些、什么顺序？有没有风速这类派生量？
4. 模型吃几帧历史？步长几小时？
5. 网格：分辨率、形状（纬度×经度）？
6. 纬度是北→南还是南→北？经度起止是 0–360 还是 -180–180？起始格点在哪？

【单位】原始 store（ERA5 再分析）的单位是：q=kg/kg、z=位势 m²/s²、
        tp=1 小时累积 m、辐射=1 小时累积 J/m²。
7. 模型期望的 q 单位？g/kg 还是 kg/kg？
8. 模型期望的 z 是位势（m²/s²）还是位势高度（m）？
9. 模型期望的 tp 单位与累积窗口？（默认：mm，6 小时累积）
10. 模型期望的辐射单位与累积窗口？（默认：Wh/m²，6 小时累积）
11. 温度单位是 K 还是 °C？气压是 Pa 还是 hPa？

【归一化】
12. 归一化是在模型图内完成，还是需要外部提供 mean/std？如果是外部，统计量文件是哪些、
    逐通道标量还是逐像素？
13. 这些统计量文件是否必须与模型一起保留（可否删除）？
14. tp 是否做了 log1p 变换？输出侧是否对应 expm1？
15. 模型输出是**残差**还是**绝对值**？若是残差，反演公式是什么？

【输入中被忽略的通道】
16. 官方预处理后的输入里，诊断量通道（ssr / ssrd / fdir / ttr / tp 等）的取值是多少？
    是否为 0、且在归一化空间还是物理空间为 0？

【缺测】
17. 哪些通道可能出现 NaN / Inf？
18. 训练或官方推理时如何处理？填充成什么值？
19. 该填充值是在归一化**之前**还是**之后**填的？

【时间特征】
20. hour / day-of-year 取自输入窗口的**末帧**还是**目标预报时刻**？
21. 模型的 step 参数是 0-based 还是 1-based？
22. 有没有跳步结构（例如每若干步切换到一个更长步长的模型）？

【集合】（仅集合模型）
23. 成员是「同一初值 + N 个扰动」还是「N 个不同初值」？
24. 图内有随机算子吗？有固定 seed 吗？同一输入能否复现同一输出？
25. 随机噪声张量的形状是 (1,...) 广播到所有成员，还是 (N,...) 逐成员独立？

【输出与工程】
26. 输出通道集合与顺序？是否包含诊断量？
27. 有没有非负约束（如 tp）？
28. 在 GPU 上能否逐位复现？若不能，噪声量级大概多少？
29. 自定义算子库绑定的 onnxruntime 版本是？
30. 权重许可证是什么？
31. 除权重外还必须提供哪些配套文件？
32. 显存需求大概多少？
```
