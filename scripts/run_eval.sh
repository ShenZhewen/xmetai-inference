#!/usr/bin/env bash
#
# 评测：预测 (runner.py 输出) vs era5_store 实况，算 RMSE/MAE/CRPS/Spread/BSS/AROC。
#
# 默认评测 INITS 里列的几个起报（跨季节 5 天）；把 INITS 设成空串即回到「扫描全部」。
#
# 用法：
#   bash scripts/run_eval.sh                              # 评测 INITS 默认的 5 个起报
#   INITS= bash scripts/run_eval.sh                       # 扫描 FCST 下所有起报
#   INIT=2025010600 bash scripts/run_eval.sh              # 只评这一个起报
#   INITS=20250117,20250222,20250708,20251016,20251121 bash scripts/run_eval.sh  # 换要评测的起报列表
#   STEPS=61 bash scripts/run_eval.sh                     # 换预报步数
#   MEMBERS=21 bash scripts/run_eval.sh                   # 换成员数（缺省读 spec 的 members）
#   VARS=z500,u200,v200,msl,tp bash scripts/run_eval.sh   # 换变量
#
set -euo pipefail

FCST="${FCST:-/workspace/data/szw_output_fuxiens}"                # 预测输出根目录
INIT="${INIT:-}"                                                  # 留空=扫描所有起报；填 YYYYMMDDHH=只评这个
INITS="${INITS-20250117,20250222,20250708,20251016,20251121}"     # 默认评测这 5 个起报；INITS= 设空串回到扫描全部
INIT_HOUR="${INIT_HOUR:-0}"                                       # 扫描模式下日期目录(YYYYMMDD)缺的起报小时（0=00UTC）
STEPS="${STEPS:-61}"                                              # 预报步数（member 目录里 .nc 个数）
VARS="${VARS:-z500,u200,v200,msl,tp}"                            # 要检验的变量
LOADER="${LOADER:-era5_store}"                                    # 实况数据源（地址内置）
SPEC="${SPEC:-/workspace/szwCode/xmetai-inference/specs/fuxi_ens.json}" # 模型 spec（含 members=51）
OUT="${OUT:-/workspace/szwCode/xmetai-inference/eval_results}"   # 结果输出目录

# 成员数缺省不传，让 evaluate.py 读 spec 的 model.members（fuxi_ens=51）；设了 MEMBERS 才覆盖。
ARGS=(--fcst "$FCST" --steps "$STEPS" --vars "$VARS" --loader "$LOADER" --spec "$SPEC" --out "$OUT")
if [ -n "${MEMBERS:-}" ]; then
  ARGS+=(--members "$MEMBERS")
fi

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
