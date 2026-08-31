#!/usr/bin/env bash
# AIFS 1.1 最小闭环冒烟测试：读输入 → 插值 N320 → SimpleRunner 跑 1 步(6h) → 打印输出量级。
#
# 目的：先在服务器上验证「输入装配 + N320 插值顺序 + 单位」全对，再铺开成完整 infer_aifs。
# 跑通后重点看两件事：
#   1. 输入自检里 z_500 中位数 ≈ 5.4e4（位势 m²/s²）、q_850 中位数 ≈ 5e-3（kg/kg）；
#   2. 输出 z_500 中位数仍 ≈ 5.4e4，量级与输入一致、无离谱 NaN/爆点。
#
# 用法：
#   ./scripts/run_aifs_minimal.sh                 # 默认 2025010600 / lead 6h
#   TIME=2025010612 LEAD=12 ./scripts/run_aifs_minimal.sh
#   bash scripts/run_aifs_minimal.sh              # 免 chmod 的方式
#
# 注意：GPU 非确定（README 明确说明），无法与官方逐位一致，只能量级对拍。
set -euo pipefail

# —— 路径（环境变量可覆盖）——
CHECKPOINT="${CHECKPOINT:-/workspace/szwCode/xmetai-inference/AIFS_single/aifs-single-mse-1.1.ckpt}"
OUT="${OUT:-/workspace/szwCode/xmetai-inference/AIFS_out}"
TIME="${TIME:-2025010600}"
LEAD="${LEAD:-6}"
SPEC="${SPEC:-specs/aifs11.json}"

# —— GPU 显存/分块（fuxiens notebook 建议，防止 mapper 一次性加载爆显存）——
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ANEMOI_INFERENCE_NUM_CHUNKS="${ANEMOI_INFERENCE_NUM_CHUNKS:-16}"
# 多卡时指定用哪张卡（默认不设，让 torch 自选）
# export CUDA_VISIBLE_DEVICES=0

# 切到仓库根目录（脚本在 scripts/ 下，向上退一级），保证 aifs_minimal.py / specs/aifs11.json 能被 import 到
cd "$(dirname "$0")/.."

echo "== AIFS 1.1 最小闭环 =="
echo "  checkpoint : $CHECKPOINT"
echo "  time / lead : $TIME / ${LEAD}h"
echo "  out         : $OUT"

python aifs_minimal.py \
    --checkpoint "$CHECKPOINT" \
    --time "$TIME" \
    --lead "$LEAD" \
    --spec "$SPEC" \
    --out "$OUT"

echo "== 完成：结果在 $OUT =="
