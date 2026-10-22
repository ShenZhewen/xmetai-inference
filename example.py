# -*- coding: utf-8 -*-
"""按 configs/fgvp_ens_ar1state_aBt15.py 的 SPA 配方，在 tongyun SPA 图上用随机初值推理。

SPA 驱动侧全部内联在本文件，不 import xmetai_inference。抽法与
verify_archive_replay.py 用到的 build_exact_xi 完全一致（构造抽 1 行丢掉 →
step 0 的 reset 再抽 1 行 → 逐行加宽 → 每步 advance → 出口乘 a(t)），
world_size=1 时逐位同流；这里把它推广到 config 的 4 卡切法：

    成员 m：slot = m % 4，行号 = m // 4，熵 = (seed=0, 7919, 0, slot)；
    每个 slot 的驱动持 13/13/13/12 行（advance 按 state 宽度抽噪声）。

与那次 config 跑法相比只有两点不同：
  1. 初始场是随机 N(0,1)，51 个成员共用同一份（config 里也全体共享同一份 ERA5）；
  2. 图是 tongyun 71 通道（config 是 FGVP 78 通道），所以预报场不能和 FGVP 归档
     逐格比 —— 逐位一致的是 SPA 驱动侧的 ξ 流（第一个起报 20250702）。

step 是 0 基；落盘文件名 NNN.nc = step NNN-1（与框架归档一致）。
    python example.py
"""

import csv
import ctypes
import gc
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import onnxruntime as ort
import xarray as xr


MODEL = ("/workspace/data/shenzw/model_onnx/20261019/"
         "tongyun_ens_cra_fp16.spa.onnx")
OUT_DIR = "/workspace/data/shenzw/inference_output/tongyun_spa_random"
# 全 71 通道 x 51 成员 x 60 步 ≈ 902 GB；先只跑驱动 + 推理（summary.csv），
# 确定 tongyun 的通道顺序后再改成 True 并填 SAVE_CHANNELS。
SAVE_FIELDS = False
# 要落盘的通道下标；None = 全部。config 只存 z500/u200/v200/msl/tp 五个变量。
SAVE_CHANNELS = None
MAX_OUTPUT_GB = 100.0
DEVICE_ID = 0
# 图里 stem 的 8x8/stride8 卷积在 cuDNN 9.19/sm90 上 HEURISTIC 查不到方案。
CUDNN_SEARCH = "EXHAUSTIVE"
PERTURB_STD_SCALE = 1.0

# ---- configs/fgvp_ens_ar1state_aBt15.py 的 SPA 配方（字面量）----
SPA_CONFIG_NAME = "fgvp_ens_ar1state_aBt15"
SEED = 0
TAU_DAYS = 3.0
HOUR_INTERVAL = 6.0
RANK = 16          # config.latent_forcing["rank"]
N_TARGETS = 3      # 图 metadata 里 3 个注入目标
MEMBERS = 51       # config.members
STEPS = 60         # config.steps
BATCH_SIZE = 6     # config.rollout_batch
WORLD_SIZE = 4     # config.gpus：决定 slot = member % 4
# config 的 amp_profile；step 是 0 基 code step，分段线性，两端取端点值。
AMP_PROFILE = (
    (0, 1.76), (4, 2.62), (8, 2.34), (12, 2.36),
    (20, 2.68), (28, 3.31), (40, 5.10), (60, 7.05),
)

SEED_INPUT = 0
INIT_TIME = datetime(2025, 7, 2, 0, 0, 0)

# 721x1440 全球 0.25° 网格。
LAT_START, LAT_STEP = 90.0, -0.25
LON_START, LON_STEP = 0.0, 0.25
# 71 个通道的变量名；None = channel_000 ... channel_070。
CHANNEL_NAMES = None

# 某些镜像里 cuDNN 9 只随 torch 的依赖提供；先 import torch 再建 session。
PRELOAD_TORCH = True
# 图含 xmetai_plugins 自定义算子，必须注册。
OPS_LIBRARY = (
    "/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so"
)
# 全体成员共享的随机初值缓存（按形状缓存，只抽一次）。
_INIT_CACHE = {}


def _preload_onnxruntime_lib():
    capi = os.path.join(os.path.dirname(ort.__file__), "capi")
    for pattern in ("libonnxruntime.so.1", "libonnxruntime.so.1.*"):
        matches = sorted(glob.glob(os.path.join(capi, pattern)))
        if matches:
            ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
            return


def _make_session(model):
    if PRELOAD_TORCH:
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
    if not os.path.isfile(model):
        raise FileNotFoundError(f"ONNX 不存在：{model}")
    if not os.path.isfile(OPS_LIBRARY):
        raise FileNotFoundError(f"自定义算子库不存在：{OPS_LIBRARY}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime 没有 CUDAExecutionProvider")

    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    opts.add_session_config_entry(
        "session.use_device_allocator_for_initializers", "0")
    opts.add_session_config_entry(
        "session.use_ort_model_bytes_directly", "1")
    _preload_onnxruntime_lib()
    opts.register_custom_ops_library(OPS_LIBRARY)

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": DEVICE_ID,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": CUDNN_SEARCH,
                "do_copy_in_default_stream": "1",
            },
        ),
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(model, sess_options=opts, providers=providers)


def _check_graph(session):
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    for name in ("input", "spw/xi"):
        if name not in inputs:
            raise RuntimeError(f"图里缺输入 {name!r}")
    if "pred" not in outputs:
        raise RuntimeError(f"图里缺输出 'pred'：{list(outputs)}")

    meta = session.get_modelmeta().custom_metadata_map
    if meta.get("spw.ar1.method") != "latent-ar1-multiplicative":
        raise RuntimeError("不是 latent-ar1-multiplicative 图")
    if meta.get("spw.ar1.input") != "spw/xi":
        raise RuntimeError(
            f"强迫输入是 {meta.get('spw.ar1.input')!r}，不是 'spw/xi'")

    state, xi = inputs["input"], inputs["spw/xi"]
    if len(state.shape) != 5 or len(xi.shape) != 3:
        raise RuntimeError(f"输入维度异常：{state.shape} / {xi.shape}")
    for name, item in (("input", state), ("spw/xi", xi)):
        if isinstance(item.shape[0], int):
            raise RuntimeError(f"{name} 的 batch 轴不是动态的（{item.shape}）")

    frames, channels, height, width = state.shape[1:]
    n_targets, rank = xi.shape[1:]
    for label, value in (
        ("input 帧数", frames), ("input 通道数", channels),
        ("input 高度", height), ("input 宽度", width),
        ("spw/xi T", n_targets), ("spw/xi R", rank),
    ):
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"{label} 必须是静态正整数：{value!r}")
    # 图和 config 必须是同一套配方。
    if rank != RANK:
        raise RuntimeError(f"图 rank={rank}，config rank={RANK}")
    if n_targets != N_TARGETS:
        raise RuntimeError(f"图有 {n_targets} 个目标，config 有 {N_TARGETS} 个")

    # 烘焙进图的参数（进 budgets()，换一个就得重导图）也必须与 aBt15 相同。
    # tau / a(t) 是驱动侧，不进图，不在这里查。
    try:
        spec_meta = json.loads(meta.get("spw.ar1.spec_json", ""))
    except (TypeError, ValueError):
        spec_meta = {}
    if not isinstance(spec_meta, dict):
        spec_meta = {}

    def _baked(key, field):
        raw = meta.get(key)
        return raw if raw is not None else spec_meta.get(field)

    for field, key, want in (
        ("sigma", "spw.ar1.sigma", 0.04),
        ("length_scale", "spw.ar1.length_scale", 32.0),
    ):
        raw = _baked(key, field)
        if raw is None:
            raise RuntimeError(
                f"图 metadata 缺 {key}（spec_json 也没有 {field}），"
                f"无法确认与 {SPA_CONFIG_NAME} 同配方")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise RuntimeError(f"图 {key}={raw!r} 不是数值")
        if abs(value - want) > 1e-9:
            raise RuntimeError(
                f"图 {key}={raw!r}，{SPA_CONFIG_NAME} 要求 {want:g}")
    raw_seed = _baked("spw.ar1.seed", "seed")
    if raw_seed is None:
        raise RuntimeError("图 metadata 缺 spw.ar1.seed，无法确认种子")
    if str(raw_seed).strip() not in ("0", "0.0"):
        raise RuntimeError(
            f"图 spw.ar1.seed={raw_seed!r}，{SPA_CONFIG_NAME} 要求 0")
    return frames, channels, height, width, n_targets, rank, set(inputs)


def _amp_at(step):
    """a(t)：分段线性，两端外推取端点值；step 是 0 基 code step。"""
    knots = AMP_PROFILE
    x = float(step)
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


def _ar1_table(n_targets, rank, rho):
    """按 config 的 4 卡切法构造 [member, step, T, R] 的 ξ 表。
    """
    members = int(MEMBERS)
    steps = int(STEPS)
    world = max(1, int(WORLD_SIZE))
    table = np.empty((members, steps, n_targets, rank), dtype=np.float32)
    decay = float(np.sqrt(max(0.0, 1.0 - rho * rho)))

    for slot in range(world):
        rows = list(range(slot, members, world))
        rng = np.random.default_rng(
            np.random.SeedSequence([int(SEED), 7919, 0, slot]))
        # 构造 Ar1Driver 时抽的那行会被 step 0 的 reset 丢掉。
        rng.standard_normal((1, n_targets, rank)).astype(np.float32)
        state = rng.standard_normal((1, n_targets, rank)).astype(np.float32)
        if len(rows) > 1:
            growth = rng.standard_normal(
                (len(rows) - 1, n_targets, rank)).astype(np.float32)
            state = np.concatenate([state, growth], axis=0)

        for step in range(steps):
            amp = np.float32(_amp_at(step))
            table[rows, step] = (state * amp).astype(np.float32)
            if step + 1 < steps:
                noise = rng.standard_normal(state.shape).astype(np.float32)
                state = (np.float32(rho) * state
                         + np.float32(decay) * noise).astype(np.float32)
    return table


def _initial_state(batch, frames, channels, height, width):
    """全体成员共用的随机初值（config 里也全体共享同一份 ERA5 初值）。"""
    key = (frames, channels, height, width)
    base = _INIT_CACHE.get(key)
    if base is None:
        rng = np.random.default_rng(SEED_INPUT)
        base = rng.standard_normal(
            (1, frames, channels, height, width), dtype=np.float32)
        _INIT_CACHE[key] = base
    return np.repeat(base, batch, axis=0)


def _sample_stats(frame):
    height, width = frame.shape[-2:]
    hs = slice(0, height, max(1, height // 32))
    ws = slice(0, width, max(1, width // 32))
    sample = frame[:, 0, hs, ws].astype(np.float64)
    sample = sample.reshape(frame.shape[0], -1)
    return (sample.mean(axis=1), sample.std(axis=1),
            np.isfinite(sample).all(axis=1))


def _to_dataset(frame, names, selected, lat, lon, member, step,
                valid, feature_time):
    data_vars = {
        names[channel].upper(): (("lat", "lon"), frame[channel])
        for channel in selected
    }
    ds = xr.Dataset(data_vars, coords={"lat": lat, "lon": lon})
    ds.attrs.update({
        "model": MODEL,
        "spa_config": SPA_CONFIG_NAME,
        "spa_amp": float(_amp_at(step)),
        "member": member,
        "step": step,
        # 归档口径：NNN.nc = lead NNN*6h = code step NNN-1。
        "lead_hours": float((step + 1) * HOUR_INTERVAL),
        "valid_time": valid.isoformat(),
        # 图内 hour/doy 用的是输入窗末帧 = init + step*6h（fuxi 系口径）。
        "feature_time": feature_time.isoformat(),
    })
    return ds


def main():
    session = _make_session(MODEL)
    frames, channels, height, width, n_targets, rank, input_names = (
        _check_graph(session))
    print(f"model: {MODEL}")
    print(f"providers: {session.get_providers()}")
    names = (list(CHANNEL_NAMES) if CHANNEL_NAMES is not None
             else [f"channel_{index:03d}" for index in range(channels)])
    if len(names) != channels:
        raise RuntimeError(
            f"CHANNEL_NAMES 有 {len(names)} 个，图有 {channels} 个通道")
    lat = LAT_START + LAT_STEP * np.arange(height)
    lon = LON_START + LON_STEP * np.arange(width)

    if SAVE_CHANNELS is None:
        selected = tuple(range(channels))
    else:
        selected = tuple(int(channel) for channel in SAVE_CHANNELS)
        if any(channel < 0 or channel >= channels for channel in selected):
            raise ValueError(f"SAVE_CHANNELS 越界（0..{channels - 1}）：{selected}")
    if SAVE_FIELDS:
        estimate_gb = (MEMBERS * STEPS * len(selected)
                       * height * width * 4 / 1e9)
        if estimate_gb > MAX_OUTPUT_GB:
            raise SystemExit(
                f"预计落盘 {estimate_gb:.0f} GB > MAX_OUTPUT_GB="
                f"{MAX_OUTPUT_GB:g}。把 SAVE_CHANNELS 设成要存的通道下标"
                f"（config 只存 z500/u200/v200/msl/tp），或 SAVE_FIELDS=False。")

    rho = float(np.exp(-(HOUR_INTERVAL / 24.0) / TAU_DAYS))
    print(f"SPA: config={SPA_CONFIG_NAME} seed={SEED} tau={TAU_DAYS:g}d "
          f"rho={rho:.4f} rank={rank} sigma=0.04(烘焙在图里)")
    print(f"members={MEMBERS} steps={STEPS} batch={BATCH_SIZE} "
          f"world_size={WORLD_SIZE}（slot=m%{WORLD_SIZE}, row=m//{WORLD_SIZE}）")
    print(f"input={frames}x{channels}x{height}x{width}; "
          f"xi T={n_targets} R={rank}")
    print(f"grid: lat {lat[0]:.3f}..{lat[-1]:.3f}, "
          f"lon {lon[0]:.3f}..{lon[-1]:.3f}; "
          f"variables {len(selected)}/{channels} 个")

    xi_table = _ar1_table(n_targets, rank, rho)
    print(f"xi: shape={xi_table.shape} std={xi_table.std():.4f} "
          f"finite={np.isfinite(xi_table).all()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_FIELDS:
        for member in range(MEMBERS):
            os.makedirs(os.path.join(
                OUT_DIR, f"member_{member:03d}"), exist_ok=True)
    print(f"output: {OUT_DIR}")

    records = np.full((MEMBERS, STEPS), np.nan, dtype=np.float64)
    rows = []
    started = time.perf_counter()

    for start in range(0, MEMBERS, BATCH_SIZE):
        chunk = list(range(start, min(start + BATCH_SIZE, MEMBERS)))
        state = _initial_state(len(chunk), frames, channels, height, width)

        for step in range(STEPS):
            # 图内时间特征用输入窗末帧（fuxi 系）：init + step*6h。
            feature_time = INIT_TIME + timedelta(hours=step * HOUR_INTERVAL)
            # 本条预报的目标时刻：init + (step+1)*6h。
            valid = feature_time + timedelta(hours=HOUR_INTERVAL)
            feeds = {
                "input": state,
                "spw/xi": xi_table[chunk, step],
            }
            scalars = {
                "step": float(step),
                "hour": feature_time.hour / 24.0,
                "doy": min(365, feature_time.timetuple().tm_yday) / 365.0,
                "perturb_std_scale": PERTURB_STD_SCALE,
            }
            for name, value in scalars.items():
                if name in input_names:
                    feeds[name] = np.full(
                        (len(chunk),), value, dtype=np.float32)
            missing = input_names - set(feeds)
            if missing:
                raise RuntimeError(f"以下输入没有喂：{sorted(missing)}")

            t0 = time.perf_counter()
            out = np.asarray(session.run(["pred"], feeds)[0], dtype=np.float32)
            elapsed = time.perf_counter() - t0
            if out.ndim == 5:
                last = out[:, -1]
            elif out.ndim == 4:
                last = out
            else:
                raise RuntimeError(f"pred 维度异常：{out.shape}")

            means, stds, finite = _sample_stats(last)
            records[chunk, step] = means
            for local, member in enumerate(chunk):
                path = ""
                if SAVE_FIELDS:
                    path = os.path.join(
                        OUT_DIR, f"member_{member:03d}",
                        f"{step + 1:03d}.nc")
                    _to_dataset(
                        last[local], names, selected, lat, lon,
                        member, step, valid, feature_time).to_netcdf(path)
                rows.append((
                    member, step, float(means[local]), float(stds[local]),
                    bool(finite[local]), path))
            print(f"members {chunk[0]:02d}-{chunk[-1]:02d} step {step:02d}: "
                  f"{elapsed:6.2f}s out={out.shape} "
                  f"mean={means.mean(): .6f} "
                  f"member-std={means.std(): .6f} "
                  f"finite={int(finite.sum())}/{len(chunk)}")
            state = np.concatenate([state[:, 1:], last[:, None]], axis=1)
            del out, last

        del state
        gc.collect()

    print_steps = tuple(step for step in (0, 4, 8, 12, 20, 28, 40, STEPS - 1)
                        if step < STEPS)
    print("\nmember x step 抽样均值（末帧通道 0）：")
    print("member " + " ".join(f"step{step:<7d}" for step in print_steps))
    for member in range(MEMBERS):
        values = " ".join(
            f"{records[member, step]: .6f}" for step in print_steps)
        print(f"{member:03d}    {values}")
    print("ens-mean " + " ".join(
        f"{np.nanmean(records[:, step]): .6f}" for step in print_steps))
    print("ens-std  " + " ".join(
        f"{np.nanstd(records[:, step]): .6f}" for step in print_steps))

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("member", "step", "mean_ch0", "std_ch0",
                         "finite", "file"))
        for member, step, value_mean, value_std, finite, path in rows:
            writer.writerow((member, step, f"{value_mean:.9g}",
                             f"{value_std:.9g}", int(finite), path))
    print(f"summary: {summary_path}")
    print(f"完成：{MEMBERS} 成员 x {STEPS} 步，"
          f"耗时 {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
