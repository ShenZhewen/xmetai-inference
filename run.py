#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一运行入口：python run.py --config configs/<name>.py [覆盖参数]

读 configs/*.py 的运行配方，按类型分发到推理（runner.py）或评测（evaluate.py）。
config 决定「跑什么、用什么权重、什么起报时间、几张卡、输出到哪」；本文件只做
「配置 → 命令行 → （多卡）子进程编排」，真正的推理/评测逻辑仍在 runner.py /
evaluate.py 里不动（它们仍可独立 CLI 调用，向后兼容）。

用法：
    python run.py --config configs/dzs_single.py                 # 推理（卡数由 config.gpus）
    python run.py --config configs/dzs_eval.py                   # 评测
    python run.py --config configs/dzs_single.py --times 2025010700..2025020500:24  # 临时覆盖起报时间
    python run.py --config configs/dzs_single.py --out /tmp/o --gpus 1  # 覆盖输出/卡数

覆盖参数（都可选，缺省用 config）：--times/--members/--steps/--vars/--out/--gpus/--cuda-devices。
"""
import argparse
import os
import subprocess
import sys

# 仓库根目录 = 本文件所在目录（run.py 就在仓库根）。先插进 sys.path，保证 configs/
# 包可被 config 文件里的 `from configs.base import ...` import 到。
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.base import EvalConfig, InferConfig, load_config, parse_times  # noqa: E402


def _pick(ov, key, default):
    """覆盖参数优先，没给（None）则回落 config 的默认。"""
    v = ov.get(key)
    return default if v is None else v


def _build_infer_argv(cfg, ov):
    """InferConfig -> runner.py 的参数列表（多卡编排不在此，由 _run_infer 做）。"""
    argv = [
        "--model", cfg.model_path_abs(),
        "--spec", cfg.spec_abs(),
        "--loader", cfg.loader,
    ]
    if cfg.backend:
        argv += ["--backend", cfg.backend]
    if cfg.data_root:
        argv += ["--zarr", cfg.data_root]

    inits = parse_times(_pick(ov, "times", cfg.times))
    if not inits:
        raise SystemExit(
            f"配置 {cfg.name}：推理必须指定 times（如 \"2025010600..2025020500:24\"）")
    if len(inits) == 1:
        argv += ["--time", inits[0].strftime("%Y%m%d%H")]
    else:
        argv += ["--inits", ",".join(t.strftime("%Y%m%d%H") for t in inits)]

    steps = _pick(ov, "steps", cfg.steps)
    if steps:
        argv += ["--steps", str(steps)]
    members = _pick(ov, "members", cfg.members)
    if members:
        argv += ["--members", str(members)]
    vars_ = _pick(ov, "vars", cfg.vars)
    if vars_:
        argv += ["--vars", vars_]
    out = _pick(ov, "out", cfg.output_dir)
    if out:
        argv += ["--out", out]
    return argv


def _run_infer(cfg, ov):
    argv = _build_infer_argv(cfg, ov)
    gpus = _pick(ov, "gpus", cfg.gpus)
    cuda_devices = _pick(ov, "cuda_devices", cfg.cuda_devices).split(",")
    out = _pick(ov, "out", cfg.output_dir)

    # 自定义算子库通过环境变量传给 runner 的模型类（iwc_fgvp_gdn2 读 XMETAI_OPS_LIBRARY）
    env = dict(os.environ)
    if cfg.ops_library:
        env["XMETAI_OPS_LIBRARY"] = cfg.ops_library

    cmd = [sys.executable, "-u", os.path.join(ROOT, "runner.py")] + argv

    if gpus <= 1:
        # 单卡：前台直接跑，日志进控制台（与旧脚本单卡行为一致）
        return subprocess.call(cmd, env=env)

    # 多卡：父进程 spawn gpus 个子进程，各自隔离一张卡（device 恒 0），LOCAL_RANK
    # 只负责把起报时间切块分摊。日志重定向到输出目录，避免多进程 stdout 交错。
    if not out:
        raise SystemExit(f"配置 {cfg.name}：多卡（gpus={gpus}）需要 output_dir，才能落 rank 日志")
    os.makedirs(out, exist_ok=True)
    procs = []
    logfs = []
    for r in range(gpus):
        gpu = cuda_devices[r] if r < len(cuda_devices) else str(r)
        rank_env = dict(env)
        rank_env["CUDA_VISIBLE_DEVICES"] = gpu
        rank_env["LOCAL_RANK"] = str(r)
        rank_env["WORLD_SIZE"] = str(gpus)
        logf = open(os.path.join(out, f"rank_{r}.log"), "w", encoding="utf-8")
        print(f"启动 rank {r}/{gpus} (GPU {gpu}) ...", flush=True)
        procs.append(subprocess.Popen(
            cmd, env=rank_env, stdout=logf, stderr=subprocess.STDOUT))
        logfs.append(logf)

    status = 0
    for r, (p, logf) in enumerate(zip(procs, logfs)):
        if p.wait() != 0:
            status = 1
            print(f"rank {r} 失败（日志 {out}/rank_{r}.log）")
        logf.close()
    if status:
        raise SystemExit("有 rank 失败，退出非零")


def _run_eval(cfg, ov):
    argv = [
        "--fcst", cfg.fcst,
        "--spec", cfg.spec_abs(),
        "--loader", cfg.loader,
        "--steps", str(cfg.steps),
        "--vars", cfg.vars,
        "--out", cfg.output_dir,
    ]
    inits = parse_times(_pick(ov, "times", cfg.times))
    if inits:
        if len(inits) == 1:
            argv += ["--init", inits[0].strftime("%Y%m%d%H")]
        else:
            # evaluate.py 的 --inits 用 token 原样当 fcst 目录名；runner 落盘目录是
            # YYYYMMDDHH（10 位），故这里给 10 位完整时间戳。
            argv += ["--inits", ",".join(t.strftime("%Y%m%d%H") for t in inits)]
    else:
        argv += ["--init-hour", str(cfg.init_hour)]
    if cfg.members:
        argv += ["--members", str(cfg.members)]

    if not cfg.output_dir:
        raise SystemExit(f"配置 {cfg.name}：评测需要 output_dir（CSV 结果目录）")
    cmd = [sys.executable, "-u", os.path.join(ROOT, "evaluate.py")] + argv
    return subprocess.call(cmd)


def main():
    p = argparse.ArgumentParser(description="统一运行入口：读 configs/*.py 配方，分发到推理/评测")
    p.add_argument("--config", required=True, help="配置文件路径（configs/<name>.py）")
    p.add_argument("--times", default=None,
                   help="覆盖：起报时间（统一格式，见 configs/base.py::parse_times）")
    p.add_argument("--members", type=int, default=None, help="覆盖：成员数")
    p.add_argument("--steps", type=int, default=None, help="覆盖：预报步数")
    p.add_argument("--vars", default=None, help="覆盖：输出/检验变量，逗号分隔")
    p.add_argument("--out", default=None, help="覆盖：输出目录")
    p.add_argument("--gpus", type=int, default=None, help="覆盖：卡数")
    p.add_argument("--cuda-devices", default=None, help="覆盖：物理卡号，逗号分隔")
    args = p.parse_args()

    cfg = load_config(args.config)
    ov = {k: getattr(args, k) for k in (
        "times", "members", "steps", "vars", "out", "gpus", "cuda_devices")}

    print(f"== 配置 {cfg.name}（{type(cfg).__name__}）== {args.config}")
    if isinstance(cfg, InferConfig):
        raise SystemExit(_run_infer(cfg, ov))
    if isinstance(cfg, EvalConfig):
        raise SystemExit(_run_eval(cfg, ov))
    raise SystemExit(f"未知配置类型 {type(cfg).__name__}（应为 InferConfig 或 EvalConfig）")


if __name__ == "__main__":
    main()
