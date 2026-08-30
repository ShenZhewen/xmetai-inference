#!/usr/bin/env bash
#
# FuXi-Ens 集合推理：一条命令跑通「构建输入 → ONNX 自回归推理 → NetCDF 落盘」。
# 支持多卡并行（默认 4 卡）：每个 rank 一个进程，按起报时间跳着分摊到各卡。
#
# 用法：
#   ./run_fuxi_ens.sh                                            # 全部用默认值（4 卡）
#   START=2024010200 END=2024010500 FREQ=6 ./run_fuxi_ens.sh     # 一段时期，间隔起报
#   GPUS=4 MEMBERS=21 ./run_fuxi_ens.sh                          # 4 卡并行
#   GPUS=1 ./run_fuxi_ens.sh                                     # 单卡
#   ERA5_STORE_ROOT=/别的/路径 ./run_fuxi_ens.sh                # 临时换 era5_store 数据目录
#
set -euo pipefail

# ---------------------------------------------------------------------------
# 可配置参数（都能用环境变量覆盖）
# ---------------------------------------------------------------------------
MODEL="${MODEL:-/workspace/szwCode/xmetai-inference/fuxi_onnx/fuxi_ens.onnx}"        # ONNX 模型路径
BACKEND="${BACKEND:-onnx}"         # 推理后端：onnx/pt/pt2/chpt 或模型名（fuxi_ens_onnx/fuxi21_pt2 等）
START="${START:-2025010600}"       # 起始起报时间 YYYYMMDDHH（默认一周：2025-01-06 00 时）
END="${END:-2025011200}"           # 结束起报时间 YYYYMMDDHH（含，默认 2025-01-12 00 时）
FREQ="${FREQ:-24}"                 # 起报间隔小时（默认每天 1 次，一周共 7 个起报）
TIME="${TIME:-2024010200}"         # 单次起报时间 YYYYMMDDHH（未设 START 时用）
SPEC="${SPEC:-/workspace/szwCode/xmetai-inference/fuxi_ens.json}"      # 模型 spec JSON
STEPS="${STEPS:-61}"               # 预报步数（61×6h ≈ 15 天）
MEMBERS="${MEMBERS:-51}"           # 集合成员总数
OUT="${OUT:-/workspace/szwCode/xmetai-inference/output}"             # 输出目录
VARS="${VARS:-z500,u200,v200,msl,tp}"   # 要保存的输出变量（逗号分隔）
GPU_MEM="${GPU_MEM:-0.7}"          # 显存占用比例
GPUS="${GPUS:-4}"                  # 用几张卡并行（1=单卡）
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"   # 物理卡号列表，按顺序对应各 rank

# 输入数据源：era=ERA 逐变量文件，zarr=打包好的 zarr store，
# era5_store=新 ERA5 基础库。各数据源地址都写死在对应 loader 的 py 里
# （可用环境变量覆盖），这里只选 loader，不设地址；zarr 是通用 loader，
# 无默认地址，用 ZARR 环境变量传。
LOADER="${LOADER:-era5_store}"

# 不在全局 export CUDA_VISIBLE_DEVICES；下面启动循环里给每个进程单独隔离一张卡。

# ---------------------------------------------------------------------------
# 打印本次运行配置
# ---------------------------------------------------------------------------
echo "== 模型=$MODEL | 后端=$BACKEND | 数据源=$LOADER | 卡=$GPUS (CUDA $CUDA_DEVICES) =="
if [ -n "$START" ]; then
  echo "起报 $START .. ${END:-$START} (间隔 ${FREQ:-步长}h) | $STEPS 步 x $MEMBERS 成员"
else
  echo "起报 $TIME | $STEPS 步 x $MEMBERS 成员"
fi
echo "输出 $OUT | 变量 $VARS"

# ---------------------------------------------------------------------------
# 组装起报参数：设了 START 就跑一段（间隔 FREQ），否则单次 TIME
# ---------------------------------------------------------------------------
if [ -n "$START" ]; then
  TIME_ARGS="--start $START"
  [ -n "$END" ]  && TIME_ARGS="$TIME_ARGS --end $END"
  [ -n "$FREQ" ] && TIME_ARGS="$TIME_ARGS --freq $FREQ"
else
  TIME_ARGS="--time $TIME"
fi

# loader 参数：地址都在各 loader 的 py 里，只有通用 loader zarr 需要传路径
LOADER_ARGS="--loader $LOADER"
if [ "$LOADER" = "zarr" ]; then
  LOADER_ARGS="$LOADER_ARGS --zarr ${ZARR:?zarr loader 需要设 ZARR（store 路径）}"
fi

# ---------------------------------------------------------------------------
# 每个 rank 一个进程并行推理；等所有卡跑完，任一失败则退出非零
# ---------------------------------------------------------------------------
pids=()
for r in $(seq 0 $((GPUS - 1))); do
  # 从 CUDA_DEVICES 取第 r 张物理卡，单独隔离给这个进程：每个进程只看到一张卡，
  # device 恒为 0；LOCAL_RANK 只负责把起报时间连续切块分给各卡。
  gpu=$(echo "$CUDA_DEVICES" | tr ',' '\n' | sed -n "$((r + 1))p")
  echo "启动 rank $r/$GPUS (GPU $gpu) ..."
  # shellcheck disable=SC2086  # TIME_ARGS / LOADER_ARGS 需要按空格拆分
  CUDA_VISIBLE_DEVICES="$gpu" LOCAL_RANK=$r WORLD_SIZE=$GPUS \
    python -u /workspace/szwCode/xmetai-inference/infer.py \
      --model "$MODEL" \
      --backend "$BACKEND" \
      $TIME_ARGS \
      --spec "$SPEC" \
      $LOADER_ARGS \
      --steps "$STEPS" \
      --members "$MEMBERS" \
      --gpu-mem "$GPU_MEM" \
      --vars "$VARS" \
      --out "$OUT" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if [ "$status" -ne 0 ]; then
  echo "有 rank 失败，退出码 $status"
  exit "$status"
fi
echo "完成，输出在 $OUT"
