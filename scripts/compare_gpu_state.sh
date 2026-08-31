#!/usr/bin/env bash
#
# GPU 常驻（IOBinding）vs numpy 版对比：同一批起报各跑一次预测、各 eval 一次，
# 对比 CRPS/RMSE。图内 RandomNormalLike 让两次跑的成员随机数不同，所以差异
# 落在随机扰动波动范围内即可（远小于指标本身），不会完全相等。
#
# 用法：
#   bash scripts/compare_gpu_state.sh
#   TIME=2025010600 STEPS=10 GPUS=4 bash scripts/compare_gpu_state.sh     # 换起报/步数/卡
#   WORK=/workspace/data/xxx bash scripts/compare_gpu_state.sh            # 换中间产物目录
#
set -euo pipefail

ROOT="/workspace/szwCode/xmetai-inference"
cd "$ROOT"

TIME="${TIME:-2025010600}"          # 单起报时间 YYYYMMDDHH
STEPS="${STEPS:-5}"                 # 预报步数（越少越快）
GPUS="${GPUS:-1}"                   # 跑预测用几张卡
WORK="${WORK:-/workspace/data/compare_gpu_state}"   # 中间产物根目录

ON_OUT="$WORK/ens_on"
OFF_OUT="$WORK/ens_off"
ON_EVAL="$WORK/eval_on"
OFF_EVAL="$WORK/eval_off"

echo "== GPU 常驻 vs numpy 对比 =="
echo "起报 $TIME | $STEPS 步 x 51 成员 | $GPUS 卡"
echo "中间产物 $WORK"

echo ""
echo "--- [1/3] 跑 GPU 常驻版（默认 on） ---"
START="$TIME" END="$TIME" STEPS="$STEPS" GPUS="$GPUS" OUT="$ON_OUT" \
  bash "$ROOT/scripts/run_fuxi_ens.sh"

echo ""
echo "--- [2/3] 跑 numpy 版（off） ---"
START="$TIME" END="$TIME" STEPS="$STEPS" GPUS="$GPUS" OUT="$OFF_OUT" \
  XMETAI_DISABLE_GPU_STATE=1 bash "$ROOT/scripts/run_fuxi_ens.sh"

echo ""
echo "--- [3/3] 评测 on / off ---"
INIT="$TIME" STEPS="$STEPS" FCST="$ON_OUT"  OUT="$ON_EVAL"  bash "$ROOT/scripts/run_eval.sh"
INIT="$TIME" STEPS="$STEPS" FCST="$OFF_OUT" OUT="$OFF_EVAL" bash "$ROOT/scripts/run_eval.sh"

echo ""
echo "== 对比结果 =="
python - "$ON_EVAL/eval_${TIME}.csv" "$OFF_EVAL/eval_${TIME}.csv" <<'PY'
import sys
import numpy as np
import pandas as pd

on = pd.read_csv(sys.argv[1])
off = pd.read_csv(sys.argv[2])
m = on.merge(off, on=["var", "lead_hour"], suffixes=("_on", "_off"))

def f(v):
    return f"{v:.4f}" if np.isfinite(v) else "   -"

print(f"{'var':<6}{'lead':>6} | {'RMSE on':>8} {'RMSE off':>8} {'d':>8} | {'CRPS on':>8} {'CRPS off':>8} {'d':>8}")
print("-" * 70)
for _, r in m.iterrows():
    print(f"{r['var']:<6}{int(r['lead_hour']):>6} | "
          f"{f(r['rmse_on']):>8} {f(r['rmse_off']):>8} {f(r['rmse_on']-r['rmse_off']):>8} | "
          f"{f(r['crps_on']):>8} {f(r['crps_off']):>8} {f(r['crps_on']-r['crps_off']):>8}")

d_rmse = (m["rmse_on"] - m["rmse_off"]).abs()
d_crps = (m["crps_on"] - m["crps_off"]).abs()
print("-" * 70)
print(f"RMSE 平均差 {d_rmse.mean():.4f}  最大差 {d_rmse.max():.4f}")
print(f"CRPS 平均差 {d_crps.mean():.4f}  最大差 {d_crps.max():.4f}")
print("差异应远小于指标本身（<1% 量级，来自随机扰动）；若到几个百分点需排查。")
PY
