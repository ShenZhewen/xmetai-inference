#!/bin/bash
# FuXi 2.1 End-to-End Demo
#
# Usage:
#   bash run.sh --model_dir /path/to/model --input /path/to/input.nc --forecast_time 2024092900
#   bash run.sh --model_dir ./model --input ./input.nc --forecast_time 2024092900 --steps 40
#
# Required:
#   --model_dir      directory containing fuxi-2.1.pt2, mean.nc, std.nc
#   --input          path to input .nc file (pre-normalized)
#   --forecast_time  init time in YYYYMMDDHH format

set -e

MODEL_DIR=""
INPUT=""
FORECAST_TIME=""
OUTPUT_DIR="./output"
STEPS=5
DEVICE="cuda"
CHANNELS="t2m z500 tp msl ws10m"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --forecast_time) FORECAST_TIME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_DIR" ]]; then
    echo "Error: --model_dir is required"
    echo "Usage: bash run.sh --model_dir /path/to/model --input /path/to/input.nc --forecast_time YYYYMMDDHH"
    exit 1
fi

if [[ -z "$INPUT" ]]; then
    echo "Error: --input is required"
    echo "Usage: bash run.sh --model_dir /path/to/model --input /path/to/input.nc --forecast_time YYYYMMDDHH"
    exit 1
fi

if [[ -z "$FORECAST_TIME" ]]; then
    echo "Error: --forecast_time is required (format: YYYYMMDDHH, e.g. 2024092900)"
    echo "Usage: bash run.sh --model_dir /path/to/model --input /path/to/input.nc --forecast_time 2024092900"
    exit 1
fi

echo "=== FuXi 2.1 Inference ==="
echo "  Model dir:     ${MODEL_DIR}"
echo "  Input:         ${INPUT}"
echo "  Steps:         ${STEPS}"
echo "  Device:        ${DEVICE}"
echo "  Forecast time: ${FORECAST_TIME}"
echo ""

# Step 1: Run inference
python inference.py \
    --model_dir "${MODEL_DIR}" \
    --input "${INPUT}" \
    --output_dir "${OUTPUT_DIR}" \
    --steps "${STEPS}" \
    --device "${DEVICE}" \
    --forecast_time "${FORECAST_TIME}"

echo ""
echo "=== Generating Plots ==="

# Step 2: Plot results
python plot.py \
    --output_dir "${OUTPUT_DIR}" \
    --channels ${CHANNELS} \
    --discrete

echo ""
echo "=== Done! ==="
echo "  Predictions: ${OUTPUT_DIR}/*.nc"
echo "  Plots:       ${OUTPUT_DIR}/plots/*.png"
