#!/usr/bin/env bash
#
# 统一启动薄壳：setup_onnxruntime + python -m xmetai.inference。
# 配置进包内 configs/，跑法只靠传一个配置名；环境准备（onnxruntime 软链/LD_LIBRARY_PATH）
# 收敛到这一处，不再散在各 run_*.sh 里。
#
# 用法：
#   bash scripts/run.sh fgvp                          # 内置模型
#   bash scripts/run.sh /workspace/my_project/config.py     # 外部模型和 Loader
#   bash scripts/run.sh fgvp --times 2025010700 --gpus 1   # 临时覆盖
#
# K8s Job 里：
#   command: ["bash", "/workspace/szwCode/xmetai-inference/scripts/run.sh", "fgvp"]
#
set -euo pipefail

CALLER_DIR="$PWD"
TARGET="${1:?用法: bash scripts/run.sh <模型名|config.py> [覆盖参数]}"
shift

# 外部 config 的相对路径按调用脚本时的工作目录解析，而不是按仓库根解析。
if [[ "$TARGET" == *.py && "$TARGET" != /* ]]; then
    TARGET="${CALLER_DIR}/${TARGET}"
fi

# 仓库根目录（绝对）。所有路径一律写绝对，不依赖 cwd。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 参考 xmetai-core 的 train.bash：自定义算子库 xmetai_onnx_plugins.so 编译时依赖
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

if [[ "$TARGET" == *.py || -f "$TARGET" ]]; then
    exec python -u -m xmetai.inference "$TARGET" "$@"
fi

exec python -u -m xmetai.inference --model "$TARGET" "$@"
