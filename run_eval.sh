#!/usr/bin/env bash
#
# 评测：预测 (infer.py 输出) vs era5_store 实况，算 CRPS/RMSE/Spread/BSS/AROC。
#
# 用法：
#   bash run_eval.sh                              # 默认 20250106 起报
#   INIT=2025010700 STEPS=61 bash run_eval.sh     # 换起报/步数
#   MEMBERS=21 bash run_eval.sh                   # 换成员数
#
set -euo pipefail

FCST="${FCST:-/workspace/szwCode/xmetai-inference/output}"          # 预测输出根目录
INIT="${INIT:-2025010600}"                                          # 起报时间 YYYYMMDDHH
STEPS="${STEPS:-60}"        # 这次实际跑了多少步（member 目录里 .nc 的个数）
MEMBERS="${MEMBERS:-51}"    # 集合成员数（member_xxx 目录个数，含 member_000）
VARS="${VARS:-z500,u200,v200,msl,tp}"                              # 要检验的变量
LOADER="${LOADER:-era5_store}"                                     # 实况数据源（地址内置）
SPEC="${SPEC:-/workspace/szwCode/xmetai-inference/fuxi_ens.json}"  # 模型 spec
OUT="${OUT:-/workspace/szwCode/xmetai-inference/eval_results}"     # 结果输出目录

echo "== 评测 $INIT 起报 | $STEPS 步 x $MEMBERS 成员 | 变量 $VARS =="
python /workspace/szwCode/xmetai-inference/evaluate.py \
  --fcst "$FCST" \
  --init "$INIT" \
  --steps "$STEPS" \
  --members "$MEMBERS" \
  --vars "$VARS" \
  --loader "$LOADER" \
  --spec "$SPEC" \
  --out "$OUT"
