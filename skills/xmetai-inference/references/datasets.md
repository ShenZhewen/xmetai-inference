# 数据源（新增数据集）

本 skill 是「接模型 + 接数据源」。**新模型怎么接**见 `model-contracts.md`（先问契约）
和 `model-landing.md`（落地步骤）。这份讲**数据源本身**：store 存的是什么、怎么喂图、
面向不同形态要配哪些 processor。

核心要区分**两种 store 形态**——这是接对数据源的第一问，问错了归一化链路会整体错位，
且**不抛异常**，只表现为数值不对。

## 1. 两种 store 形态

| 形态 | 存的是什么 | 图怎么归一化 | 喂图前要做什么 | 典型 |
|---|---|---|---|---|
| **原始态** | 物理单位（q=kg/kg、z=m²/s²、tp/辐射=1h 累积，需 `unit_convert`） | 图内已烘焙，或框架侧 mean.nc/std.nc | 物理量直接喂（配 `unit_convert`+`geometry`，或再配 `normalize`/`denormalize`） | ERA5 foundation store（`era5_pl/sfc/cldrad/soil/wave`） |
| **归一化态** | **已经归一化过的值**，mean/std 数组就存在 store 里 | 图头自带 `Div(Sub(x, mean), std)`、图尾 `Add(Mul(x,std),mean)`，常量与 store 的 mean/std 逐位一致 | **数据侧 `denormalize`**（`stats_path`=该 store），还原成物理量再喂图 | CRA c71 store（`cra...c71.p25.h6.v1`）；zarr_refresh_v3 的 ERA5 盘（`era5.202501_202607.c116...` / `era5.1979_2024.c78...`） |

一句话：**原始态 store 的归一化在「图 / 框架」侧；归一化态 store 的归一化在「store」侧**。
前者框架负责 normalize 或图内烘焙；后者 store 已经把值压到归一化空间（通常为无损存储），
喂图前必须 `denormalize` 还原物理量，图头再收一遍。

## 2. 怎么判断一个 store 是哪种

不开箱验证就会踩坑，按序确认：

1. **store 里有没有 `mean`/`std` 数组**？有 → 大概率归一化态（如 CRA）。
2. **图头是不是 `Div(Sub(x, mean), std)`**（图尾 `Add(Mul(x,std),mean)`）？
   是 → 图**期望物理量输入**，而 store 给的是归一化值 → 配数据侧 `denormalize`。
3. **lat 朝向**：北→南（FuXi 系惯例）还是南→北（CRA 的 store 原生朝向）？
   南→北 → 用 `GRID_025_S`，**别配 `geometry`**（`geometry` 会无条件把南→北翻成北→南，
   图照着南→北练的话场就被喂反，+6h z500 RMSE 90→486）。
4. **log1p 通道**（tp）：反归一化得到的是 `log1p(物理值)`，**不是物理值本身**。
   要不要 expm1 看图的输入表示，两套盘相反：
   - CRA 的图要的就是 log1p 空间 → **不要** expm1；
   - zarr_refresh_v3 的 ERA5 盘是 `log1p(6h 累积 mm)` 归一化，而 FuXi / FGVP /
     风清要物理量 mm/6h → 配 `expm1_channels=["tp"]` 还原。

## 3. 归一化态 store 怎么配

数据侧 `denormalize`（`utils/dataset_processors.py` 的 `TensorDenormalize`，
`@register_tensor_processor("denormalize")`）：

```python
cfg = InferConfig(
    ...
    dataset={
        "type": "processed_multi_zarr",
        "paths": [CRA_STORE],
        "processors": [
            # store(归一化) -> 物理量（tp 仍是 log1p 空间，图要的就是这个）。
            # 图头自带归一化；不配 geometry（图就在南→北朝向上训练/导出）。
            {"name": "denormalize", "stats_path": CRA_STORE},
        ],
    },
    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),
)
```

要点：

- `stats_path` 指向含 `mean`/`std` 数组的 zarr store；均值化常量必须与图头烘焙的**逐位一致**。
  `__init__` 按**名字**对齐通道（`_` 归一化后比对，兼容 `z_50` vs `z50`）：store 的
  通道数与顺序**不必**等于模型，超集盘也能用（模型要的通道缺一个就报错）。
- 输出侧 `model_processing.output` **恒等**：图尾已把结果还原成物理量，落盘不用再处理。
- `model_processing.input` 恒为空（前处理在 `dataset.processors`）。

### 3.1 归一化态的 ERA5 新盘（zarr_refresh_v3）

`/workspace/data/zarr_refresh_v3/` 两套，单 store 全通道、zscore float16：

| store | 通道 | 时段 |
|---|---|---|
| `era5.202501_202607.c116.p25.h6.v3` | 116（五家 ERA5 模型输入的超集） | 2025-01-01 … 2026-07-31 |
| `era5.1979_2024.c78.p25.h6.v1` | 78（= FuXi-Ens / FGVP 的输入集） | 1979-01-01 … 2024-12-31（回算） |

反归一化后与 store2 的单位约定**不同**，必须按模型期望再换算：

| 通道 | 新盘（还原后） | 模型期望 → 处理 |
|---|---|---|
| `q` | g/kg | FuXi / FGVP 直接用；风清 / Pangu 要 kg/kg ⇒ `unit_convert {"q": 1e-3}`（风清的单位烧在逐像素 `mean_std/*.npy` 里，漏乘**不报错**，只会整场偏 1000×） |
| `ssr/ssrd/fdir/ttr` | W/m²（时段平均通量） | 6h Wh/m² ⇒ `unit_convert` ×6 |
| `tp` | log1p(6h 累积 mm) | mm/6h ⇒ `denormalize` 的 `expm1_channels` |
| 其余 | 物理量 | 直接用 |

切新盘的链就**直接写在配方的 `dataset.processors` 里**，没有共享的封装层。
`configs/weather_era_0p25_fgvp_v3.py` 是直接可跑的 A/B 示例（与 `weather_era_0p25_fgvp.py` 只差 `dataset` / `output_dir`）；
其余四家照上表 + 下表各写各的：

| 模型 | 与 FGVP 的差别 |
|---|---|
| FuXi-Ens | 链同 FGVP（sst 的 `fill_missing` 同样要补） |
| FuXi-2.1 | 再挂 `normalize`（`mean.nc`/`std.nc`，`log1p_channels=["tp"]`），`fill_missing` 填归一化 `0.0` |
| 风清 | `q` ×1e-3、不吃辐射；新盘 tp 本身就是 6h 累积，不用再拼 tp store |
| Pangu | 只 `q` ×1e-3，不碰辐射 / tp |

现用盘（store2）的 config **一个都没改**，新盘只在 `weather_era_0p25_fgvp_v3.py` 里启用。

## 4. 三种数据源↔处理器链总表

| store 形态 | 归一化落在哪 | `dataset.processors` 链 | `model_processing` | 配方 |
|---|---|---|---|---|
| 原始态 | 图内烘焙 | `unit_convert` → `geometry` →（可选）`fill_missing` | 三阶段恒等 | `weather_era_0p25_fgvp.py` / `weather_era_0p25_fengqing.py` |
| 原始态 | 框架 mean.nc/std.nc | `unit_convert` → `geometry` → `normalize` → `fill_missing` | `recurrent`=`zero_channels`；`output`=`denormalize` | `weather_era_0p25_fuxi21.py` |
| **归一化态** | store（图头再收一遍） | `denormalize`（`stats_path`=store），**不配 geometry** | 三阶段恒等 | `weather_cra_0p25_tongyun_ens.py` |
| **归一化态（超集）** | store | `denormalize`（`stats_path`=store，`expm1_channels` 视模型）→ `unit_convert` → `geometry` →（可选）`normalize`/`fill_missing` | 三阶段恒等 | `weather_era_0p25_fgvp_v3.py` |

## 5. `fill_missing` 位置语义（原始态常见坑）

`fill_missing` 放 `normalize` **前**还是**后**，决定填的是物理量还是归一化值：

- normalize 之前填 → 填物理值
- normalize 之后填 → 填归一化值（如 fuxi21 填 `0.0`，见 `configs/weather_era_0p25_fuxi21.py` 的 `fill_missing`）

归一化态 store 一般不需要 `fill_missing`（值已标准化，mean/std 兜住）。

## 6. 单元 / 累积窗口（原始态 store 专属）

原始态 store 存 1h 累积物理量，喂 6h 步长的模型要归一到模型期望：

| 量 | store 单位 | 模型期望 → 系数 |
|---|---|---|
| `q` | kg/kg | g/kg → ×1000（部分模型，见 model-contracts D1） |
| `tp` | 1h m | 6h mm → ×6000 |
| 辐射 `ssr`/`ssrd`/`fdir`/`ttr` | 1h J/m² | 6h Wh/m² → ×6/3600 |

归一化态 store（CRA）**不做 `unit_convert`**——值已在归一化空间，没有物理单位换算。

## 7. 接新数据源前的小问卷

```
为把新数据源接入框架，需要确认 store 形态与契约：
01. store 存的是原始物理单位，还是已归一化的值？（有没有 mean/std 数组在 store 里）
02. 若是归一化值：归一化用的 mean/std 与图头烘焙的是否逐位一致？tp 这类 log1p 通道，
    反归一化后应保持 log1p(物理值) 还是还原到底？
03. store 的 lat 朝向：北→南还是南→北？经度起止与起始格点？
04. 通道清单与顺序？是否 === 模型 input_channels（能否直接按序取）？
05. 累积量（tp/辐射）的时间窗口是多少（1h？6h？）？模型期望多少？
06. 有没有需要清零的诊断通道（模型读但不参与预报）？
07. 哪些通道会出 NaN / Inf？官方怎么填？
```