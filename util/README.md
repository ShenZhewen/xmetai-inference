# util

`util` 是源码仓库附带的内部评测工具，不属于公开的 `xmetai-inference` 推理包，
不会进入 PyPI wheel，也不会安装命令行入口。下载源码后可直接通过 Python 使用。

请在项目根目录运行下面的命令。安装项目及需要的评测依赖：

```bash
cd /workspace/szwCode/xmetai-inference2
pip install -e ".[netcdf,zarr]"
```

## 工具说明

| 文件 | 用途 |
|------|------|
| `eval_single_util.py` | 确定性预报评测（对实况） |
| `eval_ens_util.py` | 集合预报评测（对实况） |
| `compare_outputs.py` | 两个输出目录互比，判断两次跑的结果是否一致 |
| `eval_common.py` | 上述工具共用的发现、实况读取、网格对齐、指标和 CSV 汇总 |

## 可复现性检查（compare_outputs.py）

改了代码之后再跑一遍，用这个确认「结果没变」。**不需要实况数据**，纯粹两份预测互比。

```bash
python util/compare_outputs.py /path/run_new /path/run_old
```

按「起报目录 + 步序号 + 变量名」三者都相同才比；任一侧缺的部分跳过并在末尾列出，
所以**谁的日期/步数少就以少的为准**，不用手工对齐。

判定四档：

| 判定 | 含义 |
|---|---|
| `IDENTICAL` | 逐位相同 |
| `CLOSE` | 随机舍入级别的差异，可认为结果没变 |
| `DRIFT` | 系统性偏移但未波及全场，浮点路径变了 |
| `DIFFER` | 多数点都变了，需要查 |

**主判据是超容差点的占比，不是最大差值。** 因为 float32 的舍入与场的量级成正比 ——
z500 在 54000 量级上单次运算的 eps 就有 6.5e-3，rollout 60 步累积到 0.1 量级完全
正常，所以单看 `max|diff|` 分不开「累积舍入」和「真算出了不同的场」。能分开的是
分布：舍入是随机散布的少数点，真实改动会让绝大多数点都偏离。

退出码可直接用在脚本里（`0` = 一致，`1` = 有 DRIFT/DIFFER，`2` = 无可比对象）：

```bash
python util/compare_outputs.py A B && echo "结果一致"
```

常用参数：

```bash
python util/compare_outputs.py A B --vars z500,t2m    # 只比指定变量
python util/compare_outputs.py A B --steps 20         # 只比前 20 步
python util/compare_outputs.py A B --rtol 1e-4        # 放宽相对容差
python util/compare_outputs.py A B --quiet            # 只打汇总
python util/compare_outputs.py A B --out /tmp/cmp     # 明细存 CSV
```

## 零参数即可运行

评测的默认值**全部从推理 config 读取**，与推理配置天然同源：

| 项 | 来源 |
|---|---|
| `--forecast` | `cfg.output_dir` |
| `--vars` | `cfg.vars` |
| `--interval` | `cfg.hour_interval` |
| 实况 store | `cfg.dataset["paths"]` |
| 单位换算 | `cfg.dataset` 里 `unit_convert` 的 `scales` |

```bash
python util/eval_single_util.py          # 评 configs/fengqing.py 的输出
python util/eval_ens_util.py             # 评 configs/fuxi_ens.py 的输出
```

换 config 用 `XMETAI_EVAL_CONFIG`：

```bash
XMETAI_EVAL_CONFIG=fuxi21 python util/eval_single_util.py
XMETAI_EVAL_CONFIG=fgvp   python util/eval_single_util.py
```

**为什么默认值从 config 读而不是各自写死**：写死过，踩过。config 的 `output_dir`
写 `fengqing_single_output2`、评测默认值写 `fengqing_single_output`（少个 2），
不显式传 `--forecast` 就静默评到另一个目录里的旧结果，指标看着「没变化」。
单位换算写错更危险：推理按 `tp×6000` 跑、评测按别的读，预测与实况不在同一单位上，
RMSE 全错且不报错。

## 实况数据源

实况走 `xmetai_inference.data` 的 `processed_multi_zarr`，与推理**同一条数据路径**
（同一套 `unit_convert` + `geometry` 处理链，只是不做 normalize）。旧的
`xmetai_inference.loaders` 与 `--loader` 参数已删除。

覆盖 store 的优先级：

| 参数 | 说明 |
|---|---|
| `--stores A.zarr B.zarr` | 显式指定，最高优先级 |
| `--data-root DIR` | 用该根目录拼默认的 pl/sfc store 名 |
| （不传） | 用 config 的 `dataset["paths"]` |

```bash
python util/eval_single_util.py \
  --stores /data/era5_pl_2025.zarr /data/era5_sfc_2025.zarr
```

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

- 起报目录支持 `YYYYMMDD`（00Z 起报）和 `YYYYMMDDHH`（其他时次）。
- 预测文件名必须为三位数字，例如 `001.nc`、`060.nc`。
- NetCDF 中的评测变量必须是带 `lat`、`lon` 坐标的二维规则网格。
- 集合成员取所有已选起报目录共同存在的 `member_*`；每个起报只评测这些成员共同存在
  的预测步骤。

## 常用用法

只评指定起报：

```bash
python util/eval_single_util.py --inits 20250102,20250103
```

只评前 10 步和部分变量：

```bash
python util/eval_single_util.py --steps 10 --vars z500,t850,u10m,v10m,msl,tp
```

集合只用前 20 个成员：

```bash
python util/eval_ens_util.py --members 20 --inits 20250102,20250105
```

预测是逐 3 小时输出时（覆盖 config 的步长）：

```bash
python util/eval_single_util.py --interval 3
```

## 指标

确定性评测输出纬度加权的 `RMSE`、`MAE`、`Bias`。

集合评测对集合平均计算上述三项；除 `tp` 外的连续变量还会计算 `CRPS`、`Spread`、
`SSR`（= `Spread / RMSE`）。`tp` 在中国区域 `15–55°N, 70–140°E` 内，使用
`0.1 / 4 / 13 / 25 mm` 阈值计算平均 `BSS` 和 `AROC`。

## 通用参数

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `--forecast` | 否 | 预测输出根目录；缺省读 config 的 `output_dir` |
| `--stores` | 否 | 显式指定实况 zarr store 路径（可多个） |
| `--data-root` | 否 | 实况 store 根目录（与 `--stores` 二选一） |
| `--inits` | 否 | 逗号分隔的起报日期或时次 |
| `--steps` | 否 | 只评测编号不大于该值的预测步骤 |
| `--vars` | 否 | 逗号分隔的变量；缺省读 config 的 `vars` |
| `--interval` | 否 | 相邻预测步骤的小时数；缺省读 config 的 `hour_interval` |
| `--out` | 否 | CSV 和日志目录 |
| `--log-level` | 否 | `DEBUG`、`INFO`、`WARNING` 或 `ERROR` |
| `--members` | 否 | 集合工具专用，只使用前 N 个共同成员，至少为 2 |

## 输出文件

默认输出到 `<forecast>/evaluation/`：

```text
eval_single_detail.csv / eval_single_by_lead.csv / eval_single_summary.csv / eval_single.log
eval_ens_detail.csv    / eval_ens_by_lead.csv    / eval_ens_summary.csv    / eval_ens.log
```

| 文件 | 聚合方式 |
|------|----------|
| `*_detail.csv` | 每个起报、每个预报时效、每个变量一行 |
| `*_by_lead.csv` | 相同变量和预报时效在不同起报日期之间取平均 |
| `*_summary.csv` | 每个变量在所有起报和预报时效上取整体平均 |

`*_by_lead.csv` 中的 `n_inits` 表示该变量和预报时效实际参与平均的起报数量 ——
**它应该等于你的起报总数**；小于说明部分起报缺文件（推理没跑完），此时指标是
不完整样本的平均。

用 `--out` 指定其他输出目录：

```bash
python util/eval_single_util.py --out /workspace/data/shenzw/evaluation/fuxi21
```

## 注意事项

1. 预测和实况必须具有一致的规则经纬度网格；工具会统一纬度方向和 `0–360°` 经度顺序，
   但不会进行空间插值。网格坐标比对容差 `1e-4`（预测坐标经 NetCDF float32 往返，
   实况坐标由 Dataset 直接算出，两者来源不同）。
2. 预测文件应保存物理量，单位需与 `unit_convert` 换算后的实况一致 —— 默认从同一份
   config 读，不必手工对齐。
3. 评测会自动忽略预测与实况中不能共同参与计算的 NaN/Inf 网格点。
4. `--inits` 中的日期必须已经存在于预测目录。
5. `--steps` 不要求所有起报拥有相同步数；逐时效平均中的 `n_inits` 会反映实际样本数。
6. 查看完整命令参数：

```bash
python util/eval_single_util.py --help
python util/eval_ens_util.py --help
```
