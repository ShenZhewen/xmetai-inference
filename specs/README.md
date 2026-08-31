# specs —— 模型规格（spec JSON）

本目录放所有模型的 spec JSON。整套框架是「spec 驱动」：**换模型 = 换一份 spec，代码主体不动**。

## spec 是什么

一份 spec JSON 描述「一个模型怎么跑」。它不含权重，只声明：

| 字段 | 含义 |
|------|------|
| `name` | 模型名（展示用） |
| `model.class` | 用哪个模型类（`fuxi_ens_onnx` / `fuxi21_pt2` / `aifs11_ckpt`） |
| `model.history_steps` / `hour_interval` | 输入历史帧数、时间步长（小时） |
| `model.forecast_type` / `members` | 确定性 `deterministic` / 集合 `ensemble`，集合成员数 |
| `grid` | 输出网格（lat/lon 起点、步长、点数） |
| `levels` / `layout` | 气压层级 + 通道排列（`@levels` 的变量按层展开成 `var{level}`，否则是地面变量） |
| `variables` | 每个变量的 `unit`（规范单位）、`accepts`（可接受单位及 scale/offset）、`range`（量程，做输入假设检验）、`aliases` |

`build_input` 拿 `unit`/`accepts`/`range` 做「输入单位推断 + 网格对齐 + 量程校验」；
`runner.py`/`evaluate.py` 拿 `model.class` 选模型类、拿 `members`/`forecast_type` 决定集合语义。

## 三份 spec

| 文件 | 模型 | 后端 | 类型 | 成员 | 说明 |
|------|------|------|------|------|------|
| `fuxi_ens.json` | FuXi-Ens | ONNX | ensemble | 51 | 集合预报；归一化烘焙进图，输入/输出都是物理量 |
| `fuxi21.json` | FuXi-2.1 | PT2 | deterministic | 1 | 确定性；z-score 空间，`pre_process`/`post_process` 做归一化/反归一化 |
| `aifs11.json` | AIFS 1.1 | ckpt (GNN) | deterministic | 1 | anemoi GNN；输入 0.25° 插值到 N320；归一化烘焙进 ckpt |

## 通道差异

`fuxi_ens` 与 `fuxi21` 都是 13 层 × 0.25°，但地面通道不同：

- **fuxi_ens**：5 层变量（z/t/u/v/q）× 13 层 + 13 地面变量 = **78 通道**；
  地面含 `u100m/v100m`，**没有** 风速 `ws10m/ws100m`、云量、总水汽 `tcw`。
- **fuxi21**：5 层变量 × 13 层 + 20 地面变量 = **85 通道**（C85）；
  比 fuxi_ens 多 `ws10m/ws100m`、云量 `lcc/mcc/hcc/tcc`、`tcw`。
- **aifs11**：不是通道张量，是 field 字典（`pl_vars` + `surface` + `soil` + `static`，
  把 ERA5 变量名映射到 AIFS 内部变量名），`pl_vars` 里多一个 `w`（垂直速度）。

## 怎么用

```bash
# 推理：--spec 指定（bash 脚本里已用 SPEC 环境变量封装好）
python runner.py --model fuxi_ens.onnx --spec specs/fuxi_ens.json ...

# 评测
python evaluate.py --fcst ... --spec specs/fuxi_ens.json ...

# 脚本里换模型：改 SPEC 环境变量即可
SPEC=/path/to/specs/fuxi21.json bash scripts/run_fuxi_pt2.sh
```

`--spec` 缺省时各入口默认读 `specs/fuxi_ens.json`（runner.py / evaluate.py），
`adapters/build_input_aifs.py` 的 `load_spec` 默认读 `specs/aifs11.json`。

## 关键约定（和官方对齐，别乱改）

- `z` 规范单位 `m2 s-2`（位势，不是 gpm；`gh→z` 是 ×9.80665）；
- `q` 规范单位 `g kg-1`（输出侧 `_transform` ×0.001 → kg/kg）；
- `tp` 规范单位 `mm`（输入侧数据若是 `m`，build_input 按 `accepts` ×1000 换到 mm）；
- 辐射 `ssr/ssrd/fdir/ttr` 单位 `Wh m-2`（6h 累积；era5_store 的 1h 累积 ×6 在 loader 里做）；
- lat 北→南、lon 0→360。
