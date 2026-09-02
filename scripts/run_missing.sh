#!/usr/bin/env bash
#
# 补齐缺失起报（断点续跑）：扫描旧输出目录 → 检查每个日期的完整性 →
# 算出「目标范围内还没跑完」的日期 → 只跑这些缺失日期，结果进新目录。
# 旧目录一字不动，新结果落在独立文件夹，方便回退 / A/B 对比。
#
# 用法：
#   ./scripts/run_missing.sh                                          # 全部默认
#   START=2025010600 END=2025120600 ./scripts/run_missing.sh         # 指定目标范围
#   OLD=/旧/目录 NEW=/新/目录 ./scripts/run_missing.sh               # 指定新旧目录
#   DRY_RUN=1 ./scripts/run_missing.sh                                # 只预览缺失区间，不推理
#   GPUS=4 MEMBERS=51 STEPS=60 ./scripts/run_missing.sh               # 按实际配置
#
set -euo pipefail

ROOT="/workspace/szwCode/xmetai-inference"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 可配置参数（都能用环境变量覆盖）
# ---------------------------------------------------------------------------
START="${START:-2025010600}"       # 目标范围起点 YYYYMMDDHH
END="${END:-2025120600}"           # 目标范围终点 YYYYMMDDHH（含）
FREQ="${FREQ:-24}"                 # 起报间隔小时（默认每天 1 次）
OLD="${OLD:-/workspace/data/szw_output_fuxiens}"           # 已跑结果的旧目录
NEW="${NEW:-/workspace/data/szw_output_fuxiens_new}"       # 新结果目录（不覆盖旧）
MEMBERS="${MEMBERS:-51}"           # 集合成员数
STEPS="${STEPS:-60}"               # 预报步数（60×6h=15 天）
VARS="${VARS:-z500,u200,v200,msl,tp}"   # 输出变量
MODEL="${MODEL:-$ROOT/weights/fuxiens/fuxi_ens_onnx/fuxi_ens.onnx}"
SPEC="${SPEC:-$ROOT/specs/fuxi_ens.json}"
LOADER="${LOADER:-era5_store}"
ZARR="${ZARR:-}"
GPU_MEM="${GPU_MEM:-0.7}"
GPUS="${GPUS:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
DRY_RUN="${DRY_RUN:-}"
PYTHON="${PYTHON:-python}"          # 解释器（与 run_fuxi_ens.sh 一致，默认 python）
LOGDIR="${LOGDIR:-$NEW/logs}"       # 每 rank 日志目录（每段 × 每卡一个文件）

LOADER_ARGS="--loader $LOADER"
if [ "$LOADER" = "zarr" ]; then
  LOADER_ARGS="$LOADER_ARGS --zarr ${ZARR:?zarr loader 需要设 ZARR}"
fi

# ---------------------------------------------------------------------------
# 扫描旧目录 + 算缺失区间（内嵌 Python 做日期运算与完整性检查）
# ---------------------------------------------------------------------------
BLOCKS_FILE=$(mktemp)
trap 'rm -f "$BLOCKS_FILE"' EXIT

"$PYTHON" - "$START" "$END" "$FREQ" "$OLD" "$MEMBERS" "$STEPS" > "$BLOCKS_FILE" <<'PY'
import sys, os, glob
from datetime import datetime, timedelta

start   = datetime.strptime(sys.argv[1], "%Y%m%d%H")
end     = datetime.strptime(sys.argv[2], "%Y%m%d%H")
freq    = int(sys.argv[3])
old     = sys.argv[4]
members = int(sys.argv[5])
steps   = int(sys.argv[6])

def n_nc(p):
    return len(glob.glob(os.path.join(p, "*.nc")))

def check(dt):
    d = os.path.join(old, dt.strftime("%Y%m%d"))
    if not os.path.isdir(d):
        return "缺目录", None
    if members > 1:
        mdirs = sorted(x for x in os.listdir(d)
                       if os.path.isdir(os.path.join(d, x)) and x.startswith("member_"))
        if len(mdirs) != members:
            return f"成员 {len(mdirs)}/{members}", None
        # 完成 = 至少 steps 个 .nc；旧数据若多存了超出 15 天的步（如 61 vs 60），
        # 多出来那步无害，忽略即可，别因此把已完成的日期误判为残档重跑。
        bad = [m for m in mdirs if n_nc(os.path.join(d, m)) < steps]
        return (f"{len(bad)} 个成员步数不全", None) if bad else ("ok", None)
    return ("ok", None) if n_nc(d) >= steps else (f"文件 {n_nc(d)}/{steps}", None)

full = []
t = start
while t <= end:
    full.append(t)
    t += timedelta(hours=freq)

complete, incomplete = [], []
for t in full:
    st, _ = check(t)
    if st == "ok":
        complete.append(t)
    elif st != "缺目录":
        incomplete.append(t)

missing = [t for t in full if t not in complete]

blocks = []
for t in missing:
    if blocks and (t - blocks[-1][1]) == timedelta(hours=freq):
        blocks[-1][1] = t
    else:
        blocks.append([t, t])

print(f"目标 {start:%Y%m%d%H}..{end:%Y%m%d%H} 每 {freq}h = {len(full)} 个起报", file=sys.stderr)
print(f"已完成 {len(complete)} | 缺失 {len(missing)}（其中残档 {len(incomplete)}）", file=sys.stderr)
for t in incomplete:
    st, _ = check(t)
    print(f"  不完整 {t:%Y%m%d}（{st}），将重跑", file=sys.stderr)
for b in blocks:
    print(f"{b[0]:%Y%m%d%H} {b[1]:%Y%m%d%H}")
PY

# ---------------------------------------------------------------------------
# DRY_RUN：只打印缺失区间，不真正推理
# ---------------------------------------------------------------------------
if [ -n "$DRY_RUN" ]; then
  echo "（DRY_RUN=1：仅预览缺失区间，不推理）"
  n=0
  while read -r bs be; do
    [ -z "$bs" ] && continue
    n=$((n + 1))
    echo "  区间 $bs .. $be"
  done < "$BLOCKS_FILE"
  [ "$n" -eq 0 ] && echo "  没有缺失日期"
  exit 0
fi

# ---------------------------------------------------------------------------
# 逐段（每个连续缺失区间）多卡并行推理；旧目录不动，全部写进 NEW
# ---------------------------------------------------------------------------
nblocks=0
status=0
mkdir -p "$LOGDIR"
while read -r bs be; do
  [ -z "$bs" ] && continue
  nblocks=$((nblocks + 1))
  echo ""
  echo "== 缺失区间 $bs .. $be | $STEPS 步 x $MEMBERS 成员 | $GPUS 卡 -> $NEW =="
  pids=()
  for r in $(seq 0 $((GPUS - 1))); do
    gpu=$(echo "$CUDA_DEVICES" | tr ',' '\n' | sed -n "$((r + 1))p")
    logf="$LOGDIR/rank_${r}_${bs}.log"
    echo "  启动 rank $r/$GPUS (GPU $gpu) -> $logf"
    CUDA_VISIBLE_DEVICES="$gpu" LOCAL_RANK=$r WORLD_SIZE=$GPUS \
      "$PYTHON" -u "$ROOT/runner.py" \
        --model "$MODEL" --start "$bs" --end "$be" --freq "$FREQ" \
        --spec "$SPEC" $LOADER_ARGS \
        --steps "$STEPS" --members "$MEMBERS" --gpu-mem "$GPU_MEM" \
        --vars "$VARS" --out "$NEW" > "$logf" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
done < "$BLOCKS_FILE"

if [ "$nblocks" -eq 0 ]; then
  echo "没有缺失日期，无需重跑。"
fi

if [ "$status" -ne 0 ]; then
  echo "有 rank 失败，退出码 $status"
  exit "$status"
fi
echo "完成，新结果在 $NEW"
