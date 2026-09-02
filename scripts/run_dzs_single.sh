#!/usr/bin/env bash
#
# iwc_fgvp_gdn2（FuXi-Ens 的 Gated DeltaNet 骨干变体）确定性推理。
# 一条命令跑通「构建输入 → ONNX 自回归推理 → NetCDF 落盘」，默认跑一个月（每天 1 次起报）。
#
# ⚠ 必须在镜像里跑：
#     registry.bingosoft.net/pytorch/pytorch:2.11.0-cuda12.8-mamba3-20260403
#   模型的 xmetai_onnx_plugins.so 是按该镜像里的 onnxruntime 编的（ABI 绑定），
#   其他环境的 onnxruntime 版本对不上时 register_custom_ops_library 会报错。
#
# 用法（全绝对路径，任意目录下直接运行）：
#   bash scripts/run_dzs_single.sh                                   # 默认一个月（4 卡）
#   START=2025010600 END=2025020500 bash scripts/run_dzs_single.sh   # 指定月份
#   GPUS=1 bash scripts/run_dzs_single.sh                            # 单卡
#   ERA5_STORE_ROOT=/别的/路径 bash scripts/run_dzs_single.sh         # 换输入数据目录
#
set -euo pipefail

# 仓库根目录（绝对路径）。本脚本所有路径一律写绝对，不依赖 cwd。
ROOT="/workspace/szwCode/xmetai-inference"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 可配置参数（都能用环境变量覆盖）
# ---------------------------------------------------------------------------
MODEL="${MODEL:-/workspace/tmp/douzsh/models/iwc_fgvp_gdn2_260901.onnx}"          # ONNX 模型路径（绝对）
OPS_LIBRARY="${OPS_LIBRARY:-/workspace/tmp/douzsh/models/xmetai_onnx_plugins.so}" # 自定义算子库（.so）
START="${START:-2025010600}"       # 起始起报时间 YYYYMMDDHH
END="${END:-2025020500}"           # 结束起报时间 YYYYMMDDHH（含；默认约一个月）
FREQ="${FREQ:-24}"                 # 起报间隔小时（每天 1 次，一个月约 31 个起报）
TIME="${TIME:-}"                   # 单次起报时间 YYYYMMDDHH（未设 START 时用）
SPEC="${SPEC:-$ROOT/specs/iwc_fgvp_gdn2.json}"     # 模型 spec JSON（绝对）
STEPS="${STEPS:-60}"               # 预报步数（60×6h=15 天）
MEMBERS="${MEMBERS:-1}"            # 确定性模型：1 个成员
OUT="${OUT:-/workspace/data/shenzw/dzs_single_output}"   # 输出目录（确定性：{date}/{step}.nc）
VARS="${VARS:-z500,u200,v200,msl,tp}"   # 要保存的输出变量（逗号分隔）
GPU_MEM="${GPU_MEM:-0.7}"          # 显存占用比例
GPUS="${GPUS:-4}"                  # 用几张卡并行（确定性按起报时间分摊到各卡）
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"  # 物理卡号列表，按顺序对应各 rank

# 输入数据源：era=ERA 逐变量文件，zarr=打包好的 zarr store，
# era5_store=新 ERA5 基础库（地址写死在 loader 里，可用 ERA5_STORE_ROOT 覆盖）。
LOADER="${LOADER:-era5_store}"

# ---------------------------------------------------------------------------
# 打印本次运行配置
# ---------------------------------------------------------------------------
echo "== 模型=$MODEL | 数据源=$LOADER | 卡=$GPUS (CUDA $CUDA_DEVICES) =="
if [ -n "$START" ]; then
  echo "起报 $START .. ${END:-$START} (间隔 ${FREQ}h) | $STEPS 步 x $MEMBERS 成员（确定性）"
else
  echo "起报 $TIME | $STEPS 步 x $MEMBERS 成员（确定性）"
fi
echo "输出 $OUT | 变量 $VARS | 算子库 $OPS_LIBRARY"

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
# 参考 xmetai-core 的 setup_onnxruntime：插件 xmetai_onnx_plugins.so 编译时依赖
# libonnxruntime.so.1，但 pip 装的 onnxruntime 只有版本化的 libonnxruntime.so.1.xx，
# 没有 libonnxruntime.so.1 软链、也不在 LD_LIBRARY_PATH 上。这里补软链 + 把 capi
# 目录加进 LD_LIBRARY_PATH，否则 register_custom_ops_library 会报
# 「libonnxruntime.so.1: cannot open shared object file」。
# ---------------------------------------------------------------------------
setup_onnxruntime() {
    local pkg_dir ort_dir ort_so ort_real_so
    pkg_dir=$(python -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))" 2>/dev/null || true)
    [ -n "$pkg_dir" ] || { echo "[warn] 找不到 onnxruntime 包，跳过"; return 0; }
    ort_dir="${pkg_dir}/capi"
    ort_so="${ort_dir}/libonnxruntime.so.1"
    ort_real_so=$(find "$ort_dir" -maxdepth 1 -type f -name 'libonnxruntime.so.1.*' -print -quit 2>/dev/null || true)
    [ -n "$ort_real_so" ] || { echo "[warn] $ort_dir 下没有 libonnxruntime.so.1.*，跳过"; return 0; }
    [ -e "$ort_so" ] || (cd "$ort_dir" && ln -sf "$(basename "$ort_real_so")" "$(basename "$ort_so")")
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${ort_dir}:"*) ;;
        *) export LD_LIBRARY_PATH="${ort_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
    echo "已设置 ONNX Runtime 库路径: $ort_dir"
}
setup_onnxruntime

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
    XMETAI_OPS_LIBRARY="$OPS_LIBRARY" \
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
