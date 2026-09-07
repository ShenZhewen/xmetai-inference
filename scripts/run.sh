#!/usr/bin/env bash
#
# 统一启动薄壳：setup_onnxruntime + python -m xmetai_inference。
#
# 注：backends/onnx.py 里已有 _preload_onnxruntime_lib()（用 ctypes RTLD_GLOBAL 预加载
# libonnxruntime.so.1），所以直接 python -m xmetai_inference 通常也能跑 fgvp。
# 本脚本保留作兜底：万一某环境 dlopen 的 SONAME 匹配不奏效，这里改的是
# LD_LIBRARY_PATH（进程启动前生效，更硬）。
#
# 用法：
#   bash scripts/run.sh fgvp                            # 内置模型配方
#   bash scripts/run.sh fuxi21 --steps 8 --out /tmp/x   # 临时覆盖
#   bash scripts/run.sh /workspace/my/config.py         # 外部 config
#
# K8s Job 里：
#   command: ["bash", "/workspace/szwCode/xmetai-inference2/scripts/run.sh", "fgvp"]
#
set -euo pipefail

CALLER_DIR="$PWD"
TARGET="${1:?用法: bash scripts/run.sh <模型名|config.py> [覆盖参数]}"
shift

# 外部 config 的相对路径按调用脚本时的工作目录解析，而不是按仓库根解析。
if [[ "$TARGET" == *.py && "$TARGET" != /* ]]; then
    TARGET="${CALLER_DIR}/${TARGET}"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 自定义算子库 xmetai_onnx_plugins.so 编译时链接 SONAME libonnxruntime.so.1，但 pip
# 装的 onnxruntime 在 capi/ 下只有版本化文件（如 libonnxruntime.so.1.24.4），既没有
# libonnxruntime.so.1 软链、该目录也不在 LD_LIBRARY_PATH 上。于是
# register_custom_ops_library 报「libonnxruntime.so.1: cannot open shared object file」。
#
# 必须 export：多卡时 cli.py fork 子进程用 dict(os.environ) 继承环境，
# 只在命令前临时赋值传不下去。
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

# 外部 config 走位置参数，内置配方走 --model（见 cli.py 的 _config_main）。
if [[ "$TARGET" == *.py || -f "$TARGET" ]]; then
    exec python -u -m xmetai_inference "$TARGET" "$@"
fi

exec python -u -m xmetai_inference --model "$TARGET" "$@"
