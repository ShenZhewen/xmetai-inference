# weights —— 模型权重 / 参数

本目录存放模型权重文件（`.onnx` / `.pt2` / `.ckpt` 及配套统计量）。这些是大文件，
**不进 git**：目录内的 `.gitignore` 会忽略除本 README 外的所有内容，仓库里只留这个占位。

> `models/` 放的是模型**代码**（Python 类），`weights/` 放的是模型**参数**（二进制），别混。

## 各模型对应的权重文件

| 模型 | 权重 | 配套文件 |
|------|------|----------|
| FuXi-Ens | `fuxi_ens.onnx` | 无（归一化已烘焙进图） |
| FuXi-2.1 | `fuxi-2.1.pt2` | `mean.nc`、`std.nc`（z-score 反归一化统计量，**必须和 .pt2 同目录**） |
| AIFS 1.1 | `aifs-single-mse-1.1.ckpt` | 无（归一化已烘焙进 ckpt） |

## 用法

- **本地跑**：把权重文件拷进本目录，再把 `scripts/*.sh` 里的 `MODEL` / `CHECKPOINT`
  环境变量改成指向这里的相对路径；
- **服务器跑**：脚本默认走 `/workspace/szwCode/xmetai-inference/` 下的绝对路径
  （见各脚本的 `MODEL` / `CHECKPOINT` 默认值），本目录只是本地镜像；
- FuXi-2.1 的 `mean.nc` / `std.nc` 由 `models/fuxi21_pt2.py` 的 `load()` 从模型文件
  **同目录**读取，所以三个文件要放一起。
