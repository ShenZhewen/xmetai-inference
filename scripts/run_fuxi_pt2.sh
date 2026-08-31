#!/usr/bin/env bash
#
# FuXi-2.1 确定性推理：一条命令跑通「构建输入 → PT2 自回归推理 → NetCDF 落盘」。
# 确定性模型（无集合，members=1）。默认跑 2025 全年（1/3..12/29，跳过年初/年末各 2 天
# 以防边界数据缺失），用于验证 pre_process / post_process / zero_recurrent 与官方一致。
#
# 用法（全绝对路径，任意目录下直接运行）：
#   bash /workspace/szwCode/xmetai-inference/scripts/run_fuxi_pt2.sh          # 2025 全年
#   START=2024010200 END=2024010500 FREQ=6 bash .../run_fuxi_pt2.sh          # 一段时期
#   GPUS=4 CUDA_DEVICES=0,1,2,3 bash .../run_fuxi_pt2.sh                     # 多卡整段
#   ERA5_STORE_ROOT=/别的/路径 bash .../run_fuxi_pt2.sh                       # 换数据目录
#
set -euo pipefail

# 仓库根目录（绝对路径）。runner.py 顶部 `from adapters/loaders/backends/models import ...`
# 依赖「脚本所在目录在 sys.path」：python 用绝对路径启动 runner.py 时，sys.path[0] 自动
# 就是脚本目录，所以下面 cd 不是必需，只为让下游可能的相对行为有统一 cwd；本脚本所有
# 路径一律写绝对，不依赖 cwd。
ROOT="/workspace/szwCode/xmetai-inference"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 可配置参数（都能用环境变量覆盖）
# ---------------------------------------------------------------------------
MODEL="${MODEL:-$ROOT/weights/fuxi2.1/fuxi-2.1.pt2}"       # PT2 模型路径（绝对；mean.nc/std.nc 同目录）
START="${START:-2025010300}"       # 起始起报时间 YYYYMMDDHH（2025 全年，跳过年初 1/1、1/2 两天）
END="${END:-2025122900}"           # 结束起报时间 YYYYMMDDHH（含；跳过年末 12/30、12/31 两天）
FREQ="${FREQ:-24}"                 # 连续起报间隔小时（每天 1 次）
TIME="${TIME:-}"                   # 单次起报时间 YYYYMMDDHH（未设 START 时用；默认走 START/END 全年）
SPEC="${SPEC:-$ROOT/specs/fuxi21.json}"     # 模型 spec JSON（绝对）
STEPS="${STEPS:-60}"               # 预报步数（60×6h=15 天）
MEMBERS="${MEMBERS:-1}"            # 确定性模型：1 个成员（不是集合）
OUT="${OUT:-/workspace/data/szw_output_pt2_test}"   # 输出目录（确定性：{date}/{step}.nc）
VARS="${VARS:-z500,q700,t700,t850,u850,v850,u10m,v10m,t2m,d2m,msl,tp}"   # 输出变量（fuxi2.1 还可加 tcw,ssr,ssrd,fdir,ttr,tcc,lcc,mcc,hcc）
GPU_MEM="${GPU_MEM:-0.7}"          # 显存占用比例
GPUS="${GPUS:-4}"                  # 用几张卡并行（确定性按起报时间分摊到各卡）
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"  # 物理卡号列表，按顺序对应各 rank

# 输入数据源：era=ERA 逐变量文件，zarr=打包好的 zarr store，
# era5_store=新 ERA5 基础库。各数据源地址都写死在对应 loader 的 py 里
# （可用环境变量覆盖），这里只选 loader，不设地址；zarr 是通用 loader，
# 无默认地址，用 ZARR 环境变量传。
LOADER="${LOADER:-era5_store}"

# 不在全局 export CUDA_VISIBLE_DEVICES；下面启动循环里给每个进程单独隔离一张卡。

# ---------------------------------------------------------------------------
# 打印本次运行配置
# ---------------------------------------------------------------------------
echo "== 模型=$MODEL | 数据源=$LOADER | 卡=$GPUS (CUDA $CUDA_DEVICES) =="
if [ -n "$START" ]; then
  echo "起报 $START .. ${END:-$START} (间隔 ${FREQ}h) | $STEPS 步 x $MEMBERS 成员（确定性）"
else
  echo "起报 $TIME | $STEPS 步 x $MEMBERS 成员（确定性）"
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
mkdir -p "$OUT"   # 先建输出目录，否则下面重定向 rank_$r.log 会失败
pids=()
for r in $(seq 0 $((GPUS - 1))); do
  # 从 CUDA_DEVICES 取第 r 张物理卡，单独隔离给这个进程：每个进程只看到一张卡，
  # device 恒为 0；LOCAL_RANK 只负责把起报时间连续切块分给各卡。
  gpu=$(echo "$CUDA_DEVICES" | tr ',' '\n' | sed -n "$((r + 1))p")
  echo "启动 rank $r/$GPUS (GPU $gpu) ..."
  # shellcheck disable=SC2086  # TIME_ARGS / LOADER_ARGS 需要按空格拆分
  CUDA_VISIBLE_DEVICES="$gpu" LOCAL_RANK=$r WORLD_SIZE=$GPUS \
    python -u "$ROOT/runner.py" \
      --model "$MODEL" \
      $TIME_ARGS \
      --spec "$SPEC" \
      $LOADER_ARGS \
      --steps "$STEPS" \
      --members "$MEMBERS" \
      --gpu-mem "$GPU_MEM" \
      --vars "$VARS" \
      --out "$OUT" > "$OUT/rank_$r.log" 2>&1 &
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
