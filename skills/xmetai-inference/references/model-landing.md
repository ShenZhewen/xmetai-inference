# 模型落地步骤（新增模型 / 扩展 config / 写测试 / processor）

接新模型的**写法**在这一份；动笔前先确认契约，**契约清单和发问问卷在
`model-contracts.md`**，别跳过。顺序是**先契约后落地**：契约没问清就写代码，
多半写完才发现通道顺序或单位错了，而这两类错误**不抛异常**，表现为「跑通了但输出
是垃圾」，返工成本远高于多问一句。

---

## 0. 先选路径

| 路径 | 适用 | 要改的东西 |
|---|---|---|
| **A. 加内置模型** | 模型要进本仓库、长期维护 | 注册表 + `models/` + `configs/` + 单测 + 文档表格 |
| **B. 外部扩展** | 模型只在自己项目里用 | 只写一个 `.py`，`model_class` 直接给**类对象**，不碰框架源码 |

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
确定性 ONNX 里唯一的「**纯声明、零方法覆盖**」样板，`model_processing`
三段全空（恒等）。PT2 模型抄 `models/fuxi21_pt2.py`（18 行，更短）。
Pangu（115 行双 session）和 FengQing（174 行残差反演）的模型类逻辑重，不要整份照抄。
CRA 归一化态数据源接的图抄 `models/tongyun_cra_ens.py`（头/尾各带一次归一化、`GRID_025_S`
南→北、`windowed_output=False`），配套数据侧处理见 `datasets.md`。

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
（参考 `FUXI_ENS_CHANNELS` / `PANGU_CHANNELS` / `CRA_CHANNELS` 的 `_expand_channels` 写法）。

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

**processor 链的顺序决定语义**，典型顺序（原始态 store）：

```
unit_convert  →  geometry  →  normalize  →  fill_missing
（单位换算）     （几何对齐）  （归一化）     （填缺测）
```

`fill_missing` 放 `normalize` **前**还是**后**，决定填进去的是物理量还是归一化值。
`configs/fuxi21.py:57-61` 放在 normalize 之后，所以填的是**归一化空间的 0.0**。
**归一化态 store 的处理器链不一样，见 `datasets.md`（CRA）。

可用 processor：

| 位置 | 名字 | 参数 |
|---|---|---|
| `dataset.processors` | `unit_convert` | `scales`（通道基名 → 系数，未列出默认 1.0） |
| | `geometry` | `expected_shape`（可选） |
| | `normalize` | `mean_file`、`std_file`、`log1p_channels` |
| | `normalize_fengqing` | `mean_std_dir` |
| | `denormalize` | `stats_path`（归一化态 store 用它还原物理量，见 `datasets.md`） |
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
- 数据源能对齐到「支持的模型」就一行，形态特殊（归一化态）才需要动「支持的数据集」表

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

**新增数据侧 processor 的完整动机/样板**（`denormalize` 那次，归一化态 store）：
见 `datasets.md`。

模型侧 processor 要**原地改**时注意：`Rollout` 是先取预报帧副本再
`recurrent_transform` 回填，所以 `zero_channels` 原地改不会污染已取出的预报
（`runner.py:204-214`）。自己写 recurrent processor 时保持同样的假设。