# util

`util` 提供推理日志配置以及独立的确定性、集合预报评测工具。评测工具不依赖推理
config，只需要指定实况 Loader 和预测结果目录。

请在项目根目录运行下面的命令：

安装项目及需要的评测依赖：

```bash
cd /workspace/szwCode/xmetai-inference
pip install -e ".[netcdf,zarr]"
```

## 工具说明

| 文件 | 用途 |
|------|------|
| `eval_single_util.py` | 确定性预报评测 |
| `eval_ens_util.py` | 集合预报评测 |
| `eval_common.py` | 两种评测共用的数据发现、读取、网格对齐、指标和 CSV 汇总 |
| `logging_util.py` | 推理和评测共用的日志配置，不需要单独运行 |

## 预测目录格式

确定性预报：

```text
<forecast>/
├── 20250102/
│   ├── 001.nc
│   ├── 002.nc
│   └── ...
└── 20250103/
    └── ...
```

集合预报：

```text
<forecast>/
└── 20250102/
    ├── member_000/
    │   ├── 001.nc
    │   └── ...
    ├── member_001/
    └── ...
```

- 起报目录支持 `YYYYMMDD` 和 `YYYYMMDDHH`。
- 预测文件名必须为三位数字，例如 `001.nc`、`060.nc`。
- NetCDF 中的评测变量必须是带 `lat`、`lon` 坐标的二维规则网格。
- 集合成员取所有已选起报目录共同存在的 `member_*`；每个起报只评测这些成员共同存在
  的预测步骤。

## 确定性评测

评测预测目录中自动发现的全部起报、步骤和变量：

```bash
xmetai eval-single \
  --loader era5_store \
  --forecast /workspace/data/shenzw/fuxi_single_output_new
```

只评测指定的多个起报日期：

```bash
xmetai eval-single \
  --loader era5_store \
  --forecast /workspace/data/shenzw/fuxi_single_output_new \
  --inits 20250102,20250105,20250108
```

只评测前 10 步和部分变量：

```bash
xmetai eval-single \
  --loader era5_store \
  --forecast /workspace/data/shenzw/fuxi_single_output_new \
  --steps 10 \
  --vars z500,t850,u10m,v10m,msl,tp
```

确定性评测输出纬度加权的：

- `RMSE`
- `MAE`
- `Bias`

## 集合评测

自动发现全部共同成员：

```bash
xmetai eval-ens \
  --loader era5_store \
  --forecast /workspace/data/shenzw/fuxi_ens_output
```

只使用前 20 个成员，并选择多个起报日期：

```bash
xmetai eval-ens \
  --loader era5_store \
  --forecast /workspace/data/shenzw/fuxi_ens_output \
  --members 20 \
  --inits 20250102,20250105,20250108
```

所有变量都会根据集合平均计算：

- `RMSE`
- `MAE`
- `Bias`

除 `tp` 外的连续变量还会计算：

- `CRPS`
- `Spread`
- `SSR`，即 `Spread / RMSE`

`tp` 在中国区域 `15–55°N, 70–140°E` 内，使用
`0.1 / 4 / 13 / 25 mm` 阈值计算平均 `BSS` 和 `AROC`。

## Loader 选择

通过 `--loader` 明确指定实况数据类型：

| Loader | `--data-root` | 说明 |
|--------|---------------|------|
| `era5_store` | 可选 | 默认使用 Loader 内置的 ERA5 Store 地址 |
| `zarr` | 必填 | 普通物理量 Zarr |
| `zarr_normalized` | 必填 | 使用配套统计量恢复物理量的标准化 Zarr |

普通 Zarr 示例：

```bash
xmetai eval-single \
  --loader zarr \
  --data-root /workspace/data/example.zarr \
  --forecast /workspace/data/shenzw/fuxi_single_output_new
```

`--input` 不是评测参数。实况来源由 `--loader` 选择；只有需要覆盖 Loader 默认地址或
使用 Zarr Loader 时才传 `--data-root`。

## 通用参数

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `--forecast` | 是 | 预测输出根目录 |
| `--loader` | 是 | `era5_store`、`zarr` 或 `zarr_normalized` |
| `--data-root` | 否 | 覆盖实况 Loader 的数据地址 |
| `--inits` | 否 | 逗号分隔的起报日期或时次 |
| `--steps` | 否 | 只评测编号不大于该值的预测步骤 |
| `--vars` | 否 | 逗号分隔的变量；默认使用预测文件中的全部二维变量 |
| `--interval` | 否 | 相邻预测步骤的小时数，默认 `6` |
| `--out` | 否 | CSV 和日志目录 |
| `--log-level` | 否 | `DEBUG`、`INFO`、`WARNING` 或 `ERROR` |
| `--members` | 否 | 集合工具专用，只使用前 N 个共同成员，至少为 2 |

例如预测结果为逐 3 小时输出时：

```bash
xmetai eval-single \
  --loader era5_store \
  --forecast /workspace/data/example_output \
  --interval 3
```

## 输出文件

默认输出到：

```text
<forecast>/evaluation/
```

确定性评测生成：

```text
eval_single_detail.csv
eval_single_by_lead.csv
eval_single_summary.csv
eval_single.log
```

集合评测生成：

```text
eval_ens_detail.csv
eval_ens_by_lead.csv
eval_ens_summary.csv
eval_ens.log
```

三类 CSV 的含义：

| 文件 | 聚合方式 |
|------|----------|
| `*_detail.csv` | 每个起报、每个预报时效、每个变量一行 |
| `*_by_lead.csv` | 相同变量和预报时效在不同起报日期之间取平均 |
| `*_summary.csv` | 每个变量在所有起报和预报时效上取整体平均 |

`*_by_lead.csv` 中的 `n_inits` 表示该变量和预报时效实际参与平均的起报数量。

可以使用 `--out` 指定其他输出目录：

```bash
xmetai eval-single \
  --loader era5_store \
  --forecast /workspace/data/shenzw/fuxi_single_output_new \
  --out /workspace/data/shenzw/evaluation/fuxi21
```

## 注意事项

1. 预测和实况必须具有一致的规则经纬度网格；工具会统一纬度方向和 `0–360°` 经度顺序，
   但不会进行空间插值。
2. 预测文件应保存物理量，单位需要与 Loader 转换后的实况一致。
3. 评测会自动忽略预测与实况中不能共同参与计算的 NaN/Inf 网格点。
4. `--inits` 中的日期必须已经存在于预测目录。
5. `--steps` 不要求所有起报拥有相同步数；逐时效平均中的 `n_inits` 会反映实际样本数。
6. 查看完整命令参数可运行：

```bash
xmetai eval-single --help
xmetai eval-ens --help
```

未安装 CLI 时，也可以从项目根目录使用模块形式：

```bash
python -m xmetai eval-single --help
python -m xmetai eval-ens --help
```
