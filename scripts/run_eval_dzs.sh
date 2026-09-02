#!/usr/bin/env bash
#
# 评测 iwc_fgvp_gdn2（确定性模型）预测 vs era5_store 实况，算 RMSE/MAE。
# 与 run_dzs_single.sh 配套：评测它落盘的 {日期}/{step}.nc（确定性，无 member 维度）。
#
# 确定性模型（members=1，forecast_type=deterministic），集合指标
# （CRPS/Spread/BSS/AROC）为 NaN，只留 RMSE/MAE 有意义。
#
# 用法：
#   bash scripts/run_eval_dzs.sh                                 # 扫描 FCST 下所有起报
#   INIT=2025010600 bash scripts/run_eval_dzs.sh                 # 只评这一个起报
#   INITS=20250106,20250112 bash scripts/run_eval_dzs.sh         # 评指定起报列表
#   STEPS=60 VARS=z500,msl,tp bash scripts/run_eval_dzs.sh       # 换步数/变量
#
set -euo pipefail

FCST="${FCST:-/workspace/data/shenzw/dzs_single_output}"     # 预测输出根目录（run_dzs_single.sh 的 --out）
INIT="${INIT:-}"                                              # 留空=扫描所有；填 YYYYMMDDHH=只评这个
INITS="${INITS:-}"                                            # 起报列表，逗号分隔；非空=只评这些
INIT_HOUR="${INIT_HOUR:-0}"                                   # 扫描模式下日期目录(YYYYMMDD)缺的起报小时（0=00UTC）
STEPS="${STEPS:-60}"                                          # 预报步数（与 run_dzs_single.sh 的 --steps 一致，60×6h=15 天）
VARS="${VARS:-z500,u200,v200,msl,tp}"                         # 要检验的变量（须是 run_dzs_single.sh 落盘时保存过的）
LOADER="${LOADER:-era5_store}"                                # 实况数据源（地址内置）
SPEC="${SPEC:-/workspace/szwCode/xmetai-inference/specs/iwc_fgvp_gdn2.json}"  # 确定性模型 spec
OUT="${OUT:-/workspace/szwCode/xmetai-inference/eval_results_dzs}"          # 结果输出目录

ARGS=(--fcst "$FCST" --steps "$STEPS" --vars "$VARS" --loader "$LOADER" --spec "$SPEC" --out "$OUT")

if [ -n "$INIT" ]; then
  echo "== 评测 $INIT 起报 | $STEPS 步 | 变量 $VARS =="
  ARGS+=(--init "$INIT")
elif [ -n "$INITS" ]; then
  echo "== 评测指定起报 [$INITS] | $STEPS 步 | 变量 $VARS =="
  ARGS+=(--inits "$INITS")
else
  echo "== 评测 $FCST 下所有起报 | $STEPS 步 | 变量 $VARS =="
  ARGS+=(--init-hour "$INIT_HOUR")
fi

python /workspace/szwCode/xmetai-inference/evaluate.py "${ARGS[@]}"
