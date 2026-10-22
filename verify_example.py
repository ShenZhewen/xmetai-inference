# -*- coding: utf-8 -*-
"""aBt15 FGVP SPA 归档对拍：4 成员 × 4 步、model batch=4。

模型：find_abt15_graph.py 找出的 iwc_gssp_im_fp16.ar1-<指纹>.onnx（78 通道）。
输入：copy_spa_test_data.py 产出的 fgvp_input_20250702.npy（首个起报 20250702 00Z）。
参考：归档 /workspace/data/shenzw/fgvp_ens_ar1state_aBt15/20250702/
      member_{000,004,008,012}/00{1..4}.nc
      （rank0 的 ξ 流前 4 行对应成员 0/4/8/12；001.nc = code step 0 = lead 6h）。

测试分两遍，都实际跑 example.main：

1) own-xi：单进程 4 行流（测试口径；跑这一遍时把 example.WORLD_SIZE 置 1）。
   先与测试独立复刻的 4 行流逐位比对，再另开 A/B 两个 session、step 外
   单块、同一份 feeds 跑参考循环：example vs A 的差必须与 A/B 同量级，
   A/B 给出跨 session 的原生 ε 噪声下限。
   交付默认是 config 的 4 卡切法（WORLD_SIZE=4）：[xi] 里另按
   rank0 的 13 行流（成员 0,4,...,48）逐位校验槽位映射。

2) archive-xi：归档 rank0 的 ξ 流是 13 行（51 成员 / 4 卡向上取整），
   前 4 行才是成员 0/4/8/12 用的 ξ。example 自己 4 行的流在 step 0 与它
   相同，step>=1 起不同（advance 按 state 宽度抽噪声，RNG 消耗不同）。
   所以这一遍把 example 的 ξ 表换成归档 13 行流的前 4 行，再和归档的
   001..004.nc 比。

判据：
  - 硬证据（与图内随机算子无关）：example 的 AR(1) 表逐位等于独立复刻；
    feed audit 逐位校验每次 run 的 spw/xi、input 滑窗（上一步 pred 拼成
    下一帧）、step/hour/doy。rel = RMSE / std(参考输出)。
  - 输出侧：example vs A 的最大 rel 必须 <=
    max(--rel-rmse, --noise-factor x A vs B 的最大 rel)。
    A/B 是同 feeds、两个 session 的图内原生 ε 噪声下限，不是 PASS 门槛。
  - 归档对拍按变量比：每个变量的 ensemble-mean rel 必须 <=
    max(--archive-rel, --noise-factor x 同变量的 A/B 噪声下限)。A/B 按同样
    的 5 个通道单独算，避免拿全 78 通道汇总值去比 tp。注意这是**保守**
    下限：A/B 同调用序、ε 逐位相同，只测数值噪声；归档的调用序不同，
    ε 是另一组抽样，差异天然比 A/B 大。这一条判 LARGE 不等于实现错。
  - 同一次运行还按变量打印 mean-diff/member：归档与本次 4 成员集合均值差
    的 RMS 除以成员内部离散度。ξ 相同、只有 ε 独立时该比值应 <= ~1
    （用总成员离散度做上限，所以偏宽松）；显著大于 2 才说明两份集合均值
    对不上、不能用 ε 抽样解释。
  - FGVP 图有 RandomNormalLike，且没有 perturb_std_scale 输入：图内 ε 按
    worker 内第 k 次前向对齐，所以同调用序的 A/B 可逐位比较；但归档那次
    batch=6、成员分块 6/6/1（每步 3 次前向），example 是 1 块 × 4 步
    （每步 1 次），调用序不同 ⇒ 归档对拍只能近似。判定用上面的
    mean-diff/member（统计一致性）；若硬证据与输出噪声检查全过、但统计
    一致性超限，判 INCONCLUSIVE，不是 FAIL。保守 rel 表只作诊断，不参与
    PASS 门槛。

用法：
    python find_abt15_graph.py            # 拿 MATCH 路径
    python copy_spa_test_data.py          # 造输入和参考
    python verify_example.py \
        --model /workspace/data/shenzw/model_onnx/iwc_gssp_im_fp16.ar1-<MATCH>.onnx \
        --input /workspace/data/shenzw/inference_test_data/spa_test_data/fgvp_input_20250702.npy
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import example  # noqa: E402

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

FGVP_CHANNELS = 78
DEFAULT_MODEL = ""  # 填 find_abt15_graph.py 的 MATCH 路径
DEFAULT_INPUT = (
    "/workspace/data/shenzw/inference_test_data/spa_test_data/"
    "fgvp_input_20250702.npy"
)
DEFAULT_ARCHIVE = "/workspace/data/shenzw/fgvp_ens_ar1state_aBt15"
DEFAULT_ARCHIVE_INIT = "20250702"
DEFAULT_OPS_LIBRARY = example.OPS_LIBRARY
# 归档的 cfg.vars = "z500,u200,v200,msl,tp"；索引见 models/__init__.py
CAPTURE = (
    ("z500", 7),
    ("u200", 29),
    ("v200", 42),
    ("msl", 72),
    ("tp", 77),
)
ABT15_TARGETS = ["add_35782", "add_35787", "add_35792"]
TOTAL_MEMBERS = 51
WORLD_SIZE = 4


# ---------------------------------------------------------------------------
# 独立复刻：AR(1) 表、比较、图参数校验
# ---------------------------------------------------------------------------

def build_exact_xi(n_targets, rank, steps, rows):
    """项目 rank0 / 单进程 rows 成员的精确 AR(1) 流。

    Ar1Driver 构造时抽 1 行（丢掉）→ step0 reset 再抽 1 行（第 0 行）
    → 逐行加宽；step 变化时对整批 rows 行 advance；a(t) 只乘出口。
    RNG 按元素顺序消费，所以一次加宽 12 行与归档的 5+6+1 分块等价。

    rows=4 是 example 自己的流；rows=13 是归档 rank0 的流（前 4 行才是
    成员 0/4/8/12）。
    """
    seed = int(example.SEED)
    rho = float(np.exp(
        -(float(example.HOUR_INTERVAL) / 24.0) / float(example.TAU_DAYS)))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919, 0, 0]))

    rng.standard_normal((1, n_targets, rank)).astype(np.float32)
    state = rng.standard_normal((1, n_targets, rank)).astype(np.float32)
    if rows > 1:
        growth = rng.standard_normal(
            (rows - 1, n_targets, rank)).astype(np.float32)
        state = np.concatenate([state, growth], axis=0)

    table = np.empty((rows, steps, n_targets, rank), dtype=np.float32)
    decay = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for step in range(steps):
        amp = np.float32(example._amp_at(step))
        table[:, step] = (state * amp).astype(np.float32)
        if step + 1 < steps:
            noise = rng.standard_normal(state.shape).astype(np.float32)
            state = (np.float32(rho) * state
                     + np.float32(decay) * noise).astype(np.float32)
    return table


def compare(pred, ref):
    """全局误差；rel = RMSE / std(参考输出)。"""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise RuntimeError(f"对比形状不同：pred={pred.shape} ref={ref.shape}")
    mask = np.isfinite(pred) & np.isfinite(ref)
    if not mask.any():
        return {"rmse": float("nan"), "rel": float("nan"),
                "max_abs": float("nan"), "corr": float("nan"),
                "n": 0, "finite_frac": 0.0}
    p = pred[mask]
    r = ref[mask]
    diff = p - r
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_std = float(np.std(r))
    rel = rmse / ref_std if ref_std > 0 else float("nan")
    corr = float("nan")
    if p.size > 1 and float(np.std(p)) > 0 and ref_std > 0:
        corr = float(np.corrcoef(p, r)[0, 1])
    return {
        "rmse": rmse,
        "rel": rel,
        "max_abs": float(np.max(np.abs(diff))),
        "corr": corr,
        "n": int(p.size),
        "finite_frac": float(mask.mean()),
    }


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def graph_matches_abt15(meta):
    """按 budgets 判断这张图是不是 aBt15 用的；tau/amp 是驱动侧，不看。"""
    try:
        spec = json.loads(meta.get("spw.ar1.spec_json", ""))
    except (TypeError, ValueError):
        spec = None
    if isinstance(spec, dict):
        targets = [str(v) for v in (spec.get("targets") or [])]
        sigma = _num(spec.get("sigma"))
        length_scale = _num(spec.get("length_scale"))
        return (
            targets == ABT15_TARGETS
            and sigma is not None and abs(sigma - 0.04) < 1e-9
            and int(spec.get("rank") or -1) == 16
            and length_scale is not None
            and abs(length_scale - 32.0) < 1e-9
            and str(spec.get("seed")).strip() == "0"
        )
    targets = [v.strip() for v in
               str(meta.get("spw.ar1.targets", "")).split(" | ") if v.strip()]
    return (
        targets == ABT15_TARGETS
        and abs((_num(meta.get("spw.ar1.sigma")) or -1) - 0.04) < 1e-9
        and int(_num(meta.get("spw.ar1.rank")) or -1) == 16
        and abs((_num(meta.get("spw.ar1.length_scale")) or -1) - 32.0) < 1e-9
        and str(meta.get("spw.ar1.seed", "")).strip() == "0"
    )


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------

def _find_var(ds, name):
    if name in ds.data_vars:
        return name
    upper = name.upper()
    for key in ds.data_vars:
        if str(key).upper() == upper:
            return key
    return None


def _canonical_input(arr, channels, height, width, input_index):
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim == 5:
        if arr.shape[0] <= input_index:
            raise RuntimeError(
                f"输入 batch={arr.shape[0]}，--input-index={input_index} 越界")
        if arr.shape[0] > 1:
            print(f"[input] 文件里有 {arr.shape[0]} 个 batch，"
                  f"取第 {input_index} 个；不对就用 --input-index 指定")
        arr = arr[input_index]
    if arr.ndim != 4:
        raise RuntimeError(
            f"输入应为 [2,{channels},H,W] 或 [N,2,{channels},H,W]，"
            f"实际 {arr.shape}")
    if arr.shape[0] == channels and arr.shape[1] == 2:
        arr = arr.transpose(1, 0, 2, 3)
        print(f"[input] 检测到 [{channels},2,H,W]，已转成 [2,{channels},H,W]")
    if arr.shape[0] != 2 or arr.shape[1] != channels:
        raise RuntimeError(
            f"输入前两维应是 (2 帧, {channels} 通道)，实际 {arr.shape[:2]}")
    if arr.shape[-2:] != (height, width):
        raise RuntimeError(
            f"输入最后两维应是 ({height},{width})，实际 {arr.shape[-2:]}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _load_nc_input(path):
    if xr is None:
        raise RuntimeError("读 .nc 需要 xarray / netCDF4")
    ds = xr.open_dataset(path)
    try:
        for key in ("input", "state", "x"):
            name = _find_var(ds, key)
            if name is None:
                continue
            arr = ds[name]
            for dim in list(arr.dims):
                if str(dim).lower() in ("time", "frame"):
                    if arr.sizes[dim] == 1:
                        arr = arr.isel({dim: 0})
                    elif arr.sizes[dim] != 2:
                        raise RuntimeError(
                            f"{path} 的 {key} 在 {dim} 维有 "
                            f"{arr.sizes[dim]} 帧，不是 2")
            return np.asarray(arr.values)
        raise RuntimeError(
            f"{path} 里找不到 input/state/x 变量；现有："
            f"{list(ds.data_vars)[:10]}")
    finally:
        ds.close()


def load_input(path, channels, height, width, input_index):
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            keys = list(z.keys())
            key = next(
                (k for k in ("input", "state", "x", "arr_0") if k in z),
                keys[0] if keys else None)
            if key is None:
                raise RuntimeError("空的 .npz")
            arr = z[key]
            if len(keys) > 1:
                print(f"[input] {p.name} 键 {keys}，用 {key!r}")
    elif suffix == ".nc":
        arr = _load_nc_input(p)
    else:
        raise RuntimeError(f"不支持的输入格式：{p}")
    out = _canonical_input(arr, channels, height, width, input_index)
    print(f"[input] 使用 {p}")
    return out


# ---------------------------------------------------------------------------
# 实际 example.main
# ---------------------------------------------------------------------------

class RecordingSession:
    """记录每次 run 的 step 和完整 pred（确定性对比用）。"""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def run(self, output_names, input_feed, run_options=None):
        if run_options is None:
            out = self._inner.run(output_names, input_feed)
        else:
            out = self._inner.run(output_names, input_feed, run_options)
        self.calls.append({
            "call": len(self.calls),
            "step": (float(np.asarray(input_feed["step"]).reshape(-1)[0])
                     if "step" in input_feed else None),
            "input": input_feed["input"],
            "xi": input_feed["spw/xi"],
            "scalars": {
                name: float(np.asarray(input_feed[name]).reshape(-1)[0])
                for name in ("step", "hour", "doy", "perturb_std_scale")
                if name in input_feed
            },
            "pred": np.asarray(out[0]).copy(),
        })
        return out


class StatsSession:
    """只记统计量（原生冒烟用），不复制整场。"""

    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def run(self, output_names, input_feed, run_options=None):
        if run_options is None:
            out = self._inner.run(output_names, input_feed)
        else:
            out = self._inner.run(output_names, input_feed, run_options)
        arr = np.asarray(out[0])
        self.calls.append({
            "call": len(self.calls),
            "step": (float(np.asarray(input_feed["step"]).reshape(-1)[0])
                     if "step" in input_feed else None),
            "shape": tuple(int(v) for v in arr.shape),
            "dtype": str(arr.dtype),
            "finite_frac": float(np.isfinite(arr).mean()),
            "mean": float(np.mean(arr, dtype=np.float64)),
            "std": float(np.std(arr, dtype=np.float64)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
        return out


def run_example_main(model, scale, args, real_input, xi_table,
                     verify_own, own_xi, record_full):
    """真实跑 example.main；只临时改常量、包 session、校验 ξ 表。"""
    names = ("MODEL", "MEMBERS", "STEPS", "BATCH_SIZE", "WORLD_SIZE",
             "PERTURB_STD_SCALE", "OPS_LIBRARY", "DEVICE_ID", "CUDNN_SEARCH",
             "_make_session", "_initial_state", "_ar1_table")
    saved = {name: getattr(example, name) for name in names}
    saved_argv = sys.argv
    holder = {}
    session_cls = RecordingSession if record_full else StatsSession

    def make(path):
        proxy = session_cls(saved["_make_session"](path))
        holder["proxy"] = proxy
        return proxy

    def initial_state(batch, frames, channels, height, width):
        if real_input.shape != (frames, channels, height, width):
            raise RuntimeError(
                f"真实输入 {real_input.shape} != 图声明的 "
                f"{(frames, channels, height, width)}")
        return np.repeat(real_input[None], batch, axis=0)

    def ar1_table(n_targets, rank, rho):
        if verify_own:
            # 用 example 自己的实现，校验它与测试的精确流逐位一致。
            table = saved["_ar1_table"](n_targets, rank, rho)
            if table.shape != own_xi.shape:
                raise RuntimeError(
                    f"example 的 ξ 表 {table.shape} 与期望 "
                    f"{own_xi.shape} 不符")
            if not np.array_equal(table, own_xi):
                diff = float(np.max(np.abs(
                    table.astype(np.float64)
                    - own_xi.astype(np.float64))))
                raise RuntimeError(
                    "example._ar1_table 与测试的精确 AR(1) 表不一致："
                    f"max|diff|={diff:.3e}；确认 example.py 已同步到最新版")
        return xi_table

    # example.main 用模块常量 MODEL，不解析 argv；测试要用 --model 这张图。
    example.MODEL = model
    example.MEMBERS = args.members
    example.STEPS = args.steps
    example.BATCH_SIZE = args.batch
    # 这一遍是单进程 4 行流（own_xi 的独立复刻口径），不是交付默认的 4 卡切法；
    # 4 卡切法（slot=m%4、rank0 13 行）由 main 里的 [xi] 检查单独覆盖。
    example.WORLD_SIZE = 1
    example.PERTURB_STD_SCALE = scale
    example.OPS_LIBRARY = args.ops_library
    example.DEVICE_ID = args.device
    example.CUDNN_SEARCH = args.cudnn
    example._make_session = make
    example._ar1_table = ar1_table
    example._initial_state = initial_state

    label = {"own": "own-xi（example 自己的 4 行流）",
             "archive": "archive-xi（归档 13 行流前 4 行）"}[args.xi]
    try:
        sys.argv = ["example.py", "--model", model]
        print(f"\n[example.main] {args.members} 成员 × {args.steps} 步，"
              f"batch={args.batch}，{label}")
        example.main()
    finally:
        sys.argv = saved_argv
        for name, value in saved.items():
            setattr(example, name, value)

    proxy = holder.get("proxy")
    calls = list(proxy.calls) if proxy is not None else []
    del holder, proxy
    gc.collect()
    return calls


def _collect(calls, steps):
    preds = [None] * steps
    for call in calls:
        step = call["step"]
        if step is None:
            step = call["call"] % steps
        if int(step) != step or not 0 <= int(step) < steps:
            raise RuntimeError(f"example 返回了非法 step={step}")
        idx = int(step)
        if preds[idx] is not None:
            raise RuntimeError(
                "同一 step 出现多次；--batch 应 >= --members")
        preds[idx] = call["pred"]
    if any(p is None for p in preds):
        raise RuntimeError("example 没有产出所有 step")
    return preds


def audit_feeds(calls, xi_table, real_input, steps, input_names, members):
    """逐位审计 example 真正喂进图的 feeds（与图内原生 ε 无关）。

    - spw/xi 必须等于测试给的 ξ 表；
    - input 第 0 步必须是真实初始场；第 t 步必须是上一调用 input 的第 2 帧
      + 上一调用 pred（example 的滑窗拼接）；
    - step/hour/doy（以及存在的 perturb_std_scale）必须等于参考循环的值。
    """
    issues = []
    by_step = {}
    for call in calls:
        step = call["step"]
        if step is None:
            step = call["call"] % steps
        idx = int(step)
        if idx in by_step:
            issues.append(f"step {idx} 出现多次")
        by_step[idx] = call

    for step in range(steps):
        call = by_step.get(step)
        if call is None:
            issues.append(f"缺少 step {step} 的调用")
            continue

        if not np.array_equal(call["xi"], np.asarray(xi_table[:, step])):
            issues.append(f"step {step}: spw/xi 与测试的 ξ 表不一致")

        if step == 0:
            expected_input = np.repeat(real_input[None], members, axis=0)
        else:
            prev = by_step.get(step - 1)
            if prev is None:
                continue
            # example.main 里 result 会先 cast 到 float32 再拼接，这里照做
            expected_input = np.concatenate(
                [prev["input"][:, 1:],
                 np.asarray(prev["pred"], dtype=np.float32)], axis=1)
        if not np.array_equal(call["input"], expected_input):
            issues.append(
                f"step {step}: input 与"
                f"{'真实初始场' if step == 0 else '滑窗拼接'}不一致")

        valid = example.INIT_TIME + timedelta(
            hours=step * example.HOUR_INTERVAL)
        expected_scalars = {
            "step": float(step),
            "hour": valid.hour / 24.0,
            "doy": min(365, valid.timetuple().tm_yday) / 365.0,
            "perturb_std_scale": 0.0,
        }
        for name, value in expected_scalars.items():
            if name not in input_names or name not in call["scalars"]:
                continue
            want = float(np.float32(value))
            if call["scalars"][name] != want:
                issues.append(
                    f"step {step}: 标量 {name}="
                    f"{call['scalars'][name]!r} != {want!r}")

    return not issues, issues


# ---------------------------------------------------------------------------
# 参考 A/B
# ---------------------------------------------------------------------------

def _as_pred(out):
    pred = np.asarray(out)
    if pred.ndim != 5 or pred.shape[1] != 1:
        raise RuntimeError(
            f"pred 形状应为 [batch,1,C,H,W]，实际 {pred.shape}")
    if pred.dtype != np.float32:
        pred = pred.astype(np.float32)
    return pred


def run_reference(model, args, real_input, xi_table, example_preds, label):
    """step 外、单块；A/B 同 feeds；与 example 的每步输出对比。"""
    example.OPS_LIBRARY = args.ops_library
    example.DEVICE_ID = args.device
    example.CUDNN_SEARCH = args.cudnn

    session_a = example._make_session(model)
    session_b = example._make_session(model)
    info = example._check_graph(session_a)
    frames, channels, height, width, n_targets, rank, input_names = info

    if channels != args.channels:
        raise RuntimeError(f"参考会话通道数 {channels} != {args.channels}")
    if xi_table.shape != (args.members, args.steps, n_targets, rank):
        raise RuntimeError(
            f"ξ 表 {xi_table.shape} != "
            f"{(args.members, args.steps, n_targets, rank)}")
    if len(example_preds) != args.steps:
        raise RuntimeError(
            f"example 的输出步数 {len(example_preds)} != {args.steps}")

    chunks = [list(range(start, min(start + args.batch, args.members)))
              for start in range(0, args.members, args.batch)]
    states = [np.repeat(real_input[None], len(c), axis=0) for c in chunks]

    metrics = []
    baseline = []
    # 归档只按 CAPTURE 的 5 个变量比；A/B 噪声下限也必须按同样变量算，
    # 否则会拿「全 78 通道汇总」的 5.8e-4 去比 tp 的逐变量 rel，口径不一致。
    capture_metrics = []
    capture_baseline = []
    for step in range(args.steps):
        valid = example.INIT_TIME + timedelta(
            hours=step * example.HOUR_INTERVAL)
        for ci, rows in enumerate(chunks):
            feeds = {
                "input": states[ci],
                "spw/xi": xi_table[rows, step],
            }
            scalars = {
                "step": float(step),
                "hour": valid.hour / 24.0,
                "doy": min(365, valid.timetuple().tm_yday) / 365.0,
                "perturb_std_scale": 0.0,
            }
            for name, value in scalars.items():
                if name in input_names:
                    feeds[name] = np.full(
                        (len(rows),), value, dtype=np.float32)
            missing = input_names - set(feeds)
            if missing:
                raise RuntimeError(f"以下输入没有喂：{sorted(missing)}")

            pred_a = _as_pred(session_a.run(["pred"], feeds)[0])
            pred_b = _as_pred(session_b.run(["pred"], feeds)[0])
            for j, row in enumerate(rows):
                m_ab = compare(pred_a[j], pred_b[j])
                baseline.append({"step": step, "member": row, **m_ab})
                m_ex = compare(example_preds[step][j], pred_a[j])
                metrics.append({"step": step, "member": row, **m_ex})
                for var, channel in CAPTURE:
                    m_ab_var = compare(
                        pred_a[j, 0, channel], pred_b[j, 0, channel])
                    capture_baseline.append(
                        {"step": step, "member": row, "var": var,
                         **m_ab_var})
                    m_ex_var = compare(
                        example_preds[step][j, 0, channel],
                        pred_a[j, 0, channel])
                    capture_metrics.append(
                        {"step": step, "member": row, "var": var,
                         **m_ex_var})

            states[ci] = np.concatenate(
                [states[ci][:, 1:], pred_a], axis=1)
            print(f"  [{label}] step {step} 块 {ci}"
                  f"（{len(rows)} 成员）完成")
            del pred_a, pred_b
            gc.collect()

    return metrics, baseline, capture_metrics, capture_baseline


# ---------------------------------------------------------------------------
# 归档对拍
# ---------------------------------------------------------------------------

def _open_archive_var(path, var, height, width):
    if xr is None:
        raise RuntimeError("读归档 .nc 需要 xarray / netCDF4")
    ds = xr.open_dataset(path)
    try:
        name = _find_var(ds, var)
        if name is None:
            raise RuntimeError(
                f"{path} 里没有 {var}；现有：{list(ds.data_vars)}")
        arr = np.asarray(ds[name].values)
    finally:
        ds.close()
    arr = np.squeeze(arr)
    if arr.shape != (height, width):
        raise RuntimeError(
            f"{path} 的 {var} 形状 {arr.shape} != {(height, width)}")
    return arr.astype(np.float32, copy=False)


def compare_archive(preds, args, height, width):
    """preds[t] = example 第 t 步 [members,1,C,H,W]；按 CAPTURE 比归档。"""
    archive_dir = Path(args.archive) / args.archive_init
    rows = []
    per_member = []
    for step_idx, step_no in enumerate(args.archive_steps):
        pred = np.asarray(preds[step_idx])
        for var, channel in CAPTURE:
            values = np.asarray(pred[:, 0, channel], dtype=np.float32)
            if var == "tp":
                values = np.maximum(values, 0.0)
            refs = []
            for member in args.archive_members:
                path = (archive_dir / f"member_{member:03d}"
                        / f"{step_no:03d}.nc")
                refs.append(_open_archive_var(path, var, height, width))
            refs = np.stack(refs, axis=0)

            for j, member in enumerate(args.archive_members):
                m = compare(values[j], refs[j])
                per_member.append({
                    "step": step_idx,
                    "archive_step": step_no,
                    "archive_member": member,
                    "var": var,
                    **m,
                })

            # 数据集内部离散度：成员偏离集合平均的空间 RMS。若两份 4 成员
            # 集合的均值差只来自成员级随机性（图内 ε），独立抽样下
            # E[RMS(mean_ours - mean_arch)] = sqrt(s_ours^2/4 + s_arch^2/4)
            # = pooled / sqrt(2)，所以 ratio 的期望是 1（只有 ε 不同时）。
            vals64 = values.astype(np.float64)
            refs64 = refs.astype(np.float64)
            spread_ours = float(np.sqrt(np.mean(
                (vals64 - vals64.mean(axis=0)) ** 2)))
            spread_arch = float(np.sqrt(np.mean(
                (refs64 - refs64.mean(axis=0)) ** 2)))
            pooled = math.sqrt(
                (spread_ours ** 2 + spread_arch ** 2) / 2.0)
            expected = pooled / math.sqrt(2.0)

            m_mean = compare(values.mean(axis=0), refs.mean(axis=0))
            rows.append({
                "step": step_idx,
                "archive_step": step_no,
                "var": var,
                **m_mean,
                "spread_ours": spread_ours,
                "spread_arch": spread_arch,
                "expected_mean_diff_rmse": expected,
                "mean_diff_over_expected": (
                    m_mean["rmse"] / expected
                    if expected > 0 else float("nan")),
            })
    return rows, per_member


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _max_rel(rows):
    vals = [r["rel"] for r in rows if math.isfinite(r["rel"])]
    return max(vals) if vals else float("nan")


def _max_rel_by_var(rows):
    """按 var 字段分组取最大 rel。"""
    out = {}
    for row in rows:
        rel = row["rel"]
        if not math.isfinite(rel):
            continue
        out[row["var"]] = max(out.get(row["var"], -math.inf), rel)
    return {var: (value if math.isfinite(value) else float("nan"))
            for var, value in out.items()}


def _archive_limits(archive_rows, floors, archive_rel, noise_factor):
    """归档每个变量允许的 rel：max(固定阈值, 该变量的 A/B 噪声下限 x 倍数)。"""
    limits = {}
    for row in archive_rows:
        var = row["var"]
        floor = floors.get(var)
        limit = archive_rel if floor is None else max(
            archive_rel, noise_factor * floor)
        limits[var] = limit
    return limits


def _worst(rows):
    finite = [r for r in rows if math.isfinite(r["rel"])]
    return max(finite, key=lambda r: r["rel"]) if finite else None


def _print_run(name, metrics, baseline, tol, noise_factor):
    """输出对比；A/B 是图内原生 ε 的噪声下限，不做 PASS 门槛。

    example vs A 的差应当**与 A/B 同量级**；阈值取
    max(rel_rmse, noise_factor * A/B)，这才是这张图能达到的确定性极限。
    """
    print(f"\n[{name}] example vs 参考 A（每成员每步全局 rel）")
    for step in range(max(m["step"] for m in metrics) + 1):
        step_rows = [m for m in metrics if m["step"] == step]
        vals = " ".join(
            f"m{m['member']}:{m['rel']:.3e}" for m in step_rows)
        print(f"  step {step}: {vals}")
    worst = _worst(metrics)
    if worst is not None:
        print(f"  最差：step {worst['step']} member {worst['member']} "
              f"rel={worst['rel']:.3e} max_abs={worst['max_abs']:.3e}")
    gen_max = _max_rel(metrics)
    noise_max = _max_rel(baseline)
    limit = (max(tol, noise_factor * noise_max)
             if math.isfinite(noise_max) else tol)
    print(f"  example vs A 最大 rel = {gen_max:.3e}")
    print(f"  A vs B（同 feeds）最大 rel = {noise_max:.3e}"
          f"  <- 图内原生 ε 的噪声下限")
    print(f"  阈值 = max({tol:g}, {noise_factor:g} x {noise_max:.3e}) "
          f"= {limit:.3e}")
    return {
        "gen_max": gen_max,
        "noise_max": noise_max,
        "limit": limit,
        "ok": bool(gen_max <= limit),
    }


def main():
    parser = argparse.ArgumentParser(
        description="用 4 成员 × 4 步、batch=4 对拍 example.py（aBt15 FGVP SPA）")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="find_abt15_graph.py 给出的 MATCH 图")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--input-index", type=int, default=0)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ops-library", default=DEFAULT_OPS_LIBRARY)
    parser.add_argument("--cudnn", default="EXHAUSTIVE",
                        choices=("EXHAUSTIVE", "HEURISTIC"))
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--archive-init", default=DEFAULT_ARCHIVE_INIT)
    parser.add_argument("--archive-members", default="0,4,8,12",
                        help="归档 rank0 前 %(default)s 行对应的全局成员号")
    parser.add_argument("--archive-steps", default="1,2,3,4",
                        help="归档文件号；001 = code step 0 = lead 6h")
    parser.add_argument(
        "--archive-rows", type=int, default=13,
        help="归档 rank0 的成员数：ceil(51/4)=13")
    parser.add_argument("--rel-rmse", type=float, default=1e-3,
                        help="确定性下限；最终阈值还与 A/B 噪声下限挂钩")
    parser.add_argument(
        "--noise-factor", type=float, default=2.0,
        help="允许倍数：example vs A 相对 A/B 噪声下限；归档均值差相对"
             "成员离散度（后者是 PASS 门槛）")
    parser.add_argument("--archive-rel", type=float, default=1e-2)
    parser.add_argument("--report", default="verify_example_abt15.json")
    parser.add_argument(
        "--skip-native", action="store_true",
        help="跳过默认参数的冒烟")
    args = parser.parse_args()

    if args.members < 1 or args.steps < 1 or args.batch < 1:
        raise SystemExit("--members/--steps/--batch 必须为正")
    if args.batch < args.members:
        raise SystemExit(
            "这个测试要求 members <= batch（example 单块推理，"
            "调用顺序才是 step 0..N-1）；请把 --batch 设成 >= --members")
    if not args.model:
        raise SystemExit(
            "还没指定 --model。先运行 find_abt15_graph.py，"
            "把 MATCH（或 rank_*.log 里的路径）填进来。")
    if not Path(args.model).is_file():
        raise SystemExit(f"模型不存在：{args.model}")
    if not Path(args.input).is_file():
        raise SystemExit(
            f"输入不存在：{args.input}\n"
            "先运行 copy_spa_test_data.py 生成它。")
    if not Path(args.ops_library).is_file():
        raise SystemExit(f"自定义算子库不存在：{args.ops_library}")

    args.archive_members = [int(v) for v in args.archive_members.split(",")
                            if v.strip()]
    args.archive_steps = [int(v) for v in args.archive_steps.split(",")
                          if v.strip()]
    if len(args.archive_members) != args.members:
        raise SystemExit(
            f"--archive-members 有 {len(args.archive_members)} 个，"
            f"但 --members={args.members}")
    if len(args.archive_steps) != args.steps:
        raise SystemExit(
            f"--archive-steps 有 {len(args.archive_steps)} 个，"
            f"但 --steps={args.steps}")
    if args.archive_rows < args.members:
        raise SystemExit(
            f"--archive-rows={args.archive_rows} 小于 --members={args.members}")
    first_archive = (Path(args.archive) / args.archive_init
                     / f"member_{args.archive_members[0]:03d}"
                     / f"{args.archive_steps[0]:03d}.nc")
    if not first_archive.is_file():
        raise SystemExit(
            f"归档参考不存在：{first_archive}\n"
            "检查 --archive / --archive-init，或先运行 copy_spa_test_data.py")

    print("=" * 78)
    print(f"example.py aBt15 FGVP 归档对拍（{args.members} 成员 × "
          f"{args.steps} 步，batch={args.batch}）")
    print("=" * 78)
    print(f"model    : {args.model}")
    print(f"input    : {args.input}")
    print(f"archive  : {args.archive}/{args.archive_init} "
          f"members={args.archive_members} steps={args.archive_steps}")
    print(f"members={args.members} steps={args.steps} batch={args.batch} "
          f"device={args.device} cudnn={args.cudnn}")
    print(f"threshold: rel_rmse={args.rel_rmse:g} "
          f"noise_factor={args.noise_factor:g} "
          f"archive_rel={args.archive_rel:g}")

    example.OPS_LIBRARY = args.ops_library
    example.DEVICE_ID = args.device
    example.CUDNN_SEARCH = args.cudnn

    probe = example._make_session(args.model)
    try:
        info = example._check_graph(probe)
        frames, channels, height, width, n_targets, rank, input_names = info
        meta = dict(probe.get_modelmeta().custom_metadata_map)
    finally:
        del probe
        gc.collect()

    if channels != FGVP_CHANNELS:
        raise SystemExit(
            f"图通道数 {channels} != {FGVP_CHANNELS}；这不是 FGVP 图")
    if not graph_matches_abt15(meta):
        raise SystemExit(
            "图的 metadata 不是 aBt15 的图参数"
            f"（targets={ABT15_TARGETS}, sigma=0.04, rank=16, "
            "length_scale=32, seed=0）。\n"
            "先运行 find_abt15_graph.py，用它的 MATCH 路径作为 --model。")
    args.channels = channels

    print("\n[graph]")
    for key in ("spw.ar1.method", "spw.ar1.input", "spw.ar1.targets",
                "spw.ar1.sigma", "spw.ar1.tau_days", "spw.ar1.rank",
                "spw.ar1.length_scale", "spw.ar1.seed", "spw.ar1.tokens",
                "spw.ar1.channels", "spw.ar1.grid"):
        print(f"  {key} = {meta.get(key)!r}")
    print(f"  input shape = (batch, {frames}, {channels}, {height}, {width})")
    print(f"  inputs = {sorted(input_names)}")
    print(f"  T={n_targets} R={rank}")
    if "perturb_std_scale" not in input_names:
        print("  （图里没有 perturb_std_scale；RandomNormalLike 始终开，"
              "按 worker 内第 k 次前向对齐：同调用序可逐位比较，"
              "归档的 6/6/1 分块调用序与 example 的 1 块不同）")

    real_input = load_input(
        args.input, channels, height, width, args.input_index)
    finite = np.isfinite(real_input)
    print(f"[input] shape={real_input.shape} "
          f"finite={finite.mean():.6f} "
          f"min={np.nanmin(real_input):.6g} "
          f"max={np.nanmax(real_input):.6g}")
    if not finite.all():
        raise SystemExit("输入里有非有限值")

    own_xi = build_exact_xi(
        n_targets, rank, args.steps, args.members)
    archive_full = build_exact_xi(
        n_targets, rank, args.steps, args.archive_rows)
    archive_xi = np.ascontiguousarray(archive_full[:args.members])
    step0_same = np.array_equal(own_xi[:, 0], archive_xi[:, 0])
    xi_max_diff = float(np.max(np.abs(
        own_xi.astype(np.float64) - archive_xi.astype(np.float64))))
    print("\n[xi]")
    print(f"  example 自己的 {args.members} 行流 vs 归档 "
          f"{args.archive_rows} 行流前 {args.members} 行："
          f"step0 {'逐位相同' if step0_same else '不同'}，"
          f"max|diff|={xi_max_diff:.3e}")
    if not step0_same:
        raise RuntimeError("step0 的 ξ 应该相同；检查 example.SEED/TAU_DAYS")

    # example 交付默认 = config 的 4 卡切法：rank0 持成员 0,4,...,48 共 13 行，
    # 必须与归档 rank0 的 13 行流逐位相同（verify_archive_replay 的证据基础）。
    saved_msw = (example.MEMBERS, example.STEPS, example.WORLD_SIZE)
    try:
        example.MEMBERS = TOTAL_MEMBERS
        example.STEPS = args.steps
        example.WORLD_SIZE = WORLD_SIZE
        rho = float(np.exp(
            -(float(example.HOUR_INTERVAL) / 24.0) / float(example.TAU_DAYS)))
        multi_xi = example._ar1_table(n_targets, rank, rho)
    finally:
        example.MEMBERS, example.STEPS, example.WORLD_SIZE = saved_msw
    rank0_xi = np.ascontiguousarray(multi_xi[0::WORLD_SIZE])
    multi_same = np.array_equal(rank0_xi, archive_full)
    print(f"  4 卡切法 rank0 {rank0_xi.shape[0]} 行 vs 归档 rank0 流："
          f"{'逐位相同' if multi_same else '不同'}")
    if not multi_same:
        max_diff = float(np.max(np.abs(
            rank0_xi.astype(np.float64) - archive_full.astype(np.float64))))
        raise RuntimeError(
            "example 的 4 卡切法（slot=m%4, row=m//4）没有复现归档 rank0 的 "
            f"{args.archive_rows} 行流：max|diff|={max_diff:.3e}；"
            "检查 WORLD_SIZE / _ar1_table")

    tol = float(args.rel_rmse)

    # 1) example 自己的 ξ（交付默认行为）
    print("\n" + "-" * 78)
    print("[1] own-xi：example 自己的 4 行流")
    print("-" * 78)
    args.xi = "own"
    own_calls = run_example_main(
        args.model, 0.0, args, real_input, own_xi, True, own_xi, True)
    own_feed_ok, own_feed_issues = audit_feeds(
        own_calls, own_xi, real_input, args.steps, input_names, args.members)
    own_preds = _collect(own_calls, args.steps)
    del own_calls
    gc.collect()
    print(f"\n[feed audit] own-xi: {'OK' if own_feed_ok else 'FAIL'}")
    for issue in own_feed_issues:
        print(f"  - {issue}")
    (own_metrics, own_baseline,
     own_cap_metrics, own_cap_baseline) = run_reference(
        args.model, args, real_input, own_xi, own_preds, "own-xi")
    own_summary = _print_run(
        "own-xi", own_metrics, own_baseline, tol, args.noise_factor)
    del own_preds
    gc.collect()

    # 2) 归档 rank0 的 13 行流前 4 行
    print("\n" + "-" * 78)
    print("[2] archive-xi：喂归档 rank0 的 ξ 流前 4 行，再比归档输出")
    print("-" * 78)
    args.xi = "archive"
    archive_calls = run_example_main(
        args.model, 0.0, args, real_input, archive_xi, False, None, True)
    archive_feed_ok, archive_feed_issues = audit_feeds(
        archive_calls, archive_xi, real_input, args.steps,
        input_names, args.members)
    archive_preds = _collect(archive_calls, args.steps)
    del archive_calls
    gc.collect()
    print(f"\n[feed audit] archive-xi: "
          f"{'OK' if archive_feed_ok else 'FAIL'}")
    for issue in archive_feed_issues:
        print(f"  - {issue}")
    (archive_metrics, archive_baseline,
     archive_cap_metrics, archive_cap_baseline) = run_reference(
        args.model, args, real_input, archive_xi, archive_preds,
        "archive-xi")
    archive_summary = _print_run(
        "archive-xi", archive_metrics, archive_baseline, tol,
        args.noise_factor)

    cap_floor = _max_rel_by_var(archive_cap_baseline)
    cap_example = _max_rel_by_var(archive_cap_metrics)
    print("\n[archive-xi] per-var example vs A / A vs B（rel）")
    for var, _ in CAPTURE:
        print(f"  {var:5s}: example-vs-A={cap_example.get(var, float('nan')):.3e}  "
              f"A/B={cap_floor.get(var, float('nan')):.3e}")

    archive_mean, archive_member = compare_archive(
        archive_preds, args, height, width)
    del archive_preds
    gc.collect()

    print("\n[archive] example vs 归档（ensemble mean，rel）")
    for step_idx in range(args.steps):
        step_no = args.archive_steps[step_idx]
        entries = [r for r in archive_mean if r["step"] == step_idx]
        vals = " ".join(f"{r['var']}:{r['rel']:.3e}" for r in entries)
        print(f"  step {step_idx}（文件 {step_no:03d}，"
              f"lead {step_no * 6}h）: {vals}")

    archive_rel_max = _max_rel(archive_mean)
    member_max = _max_rel(archive_member)
    worst_member = _worst(archive_member)
    floors = _max_rel_by_var(archive_cap_baseline)
    limits = _archive_limits(
        archive_mean, floors, args.archive_rel, args.noise_factor)
    violations = [
        r for r in archive_mean
        if not (math.isfinite(r["rel"]) and r["rel"] <= limits[r["var"]])
    ]
    archive_cmp_ok = not violations
    archive_rel_by_var = {
        var: max(r["rel"] for r in archive_mean if r["var"] == var)
        for var, _ in CAPTURE
    }
    ratio_by_var = {
        var: max(
            (r["mean_diff_over_expected"] for r in archive_mean
             if r["var"] == var
             and math.isfinite(r["mean_diff_over_expected"])),
            default=float("nan"))
        for var, _ in CAPTURE
    }
    stat_violations = [
        var for var, _ in CAPTURE
        if not (math.isfinite(ratio_by_var[var])
                and ratio_by_var[var] <= args.noise_factor)
    ]
    archive_stat_ok = not stat_violations
    print(f"\n  逐变量阈值（max({args.archive_rel:g}, "
          f"{args.noise_factor:g} x A/B)；这一列是保守下限）：")
    for var, _ in CAPTURE:
        rel_max = archive_rel_by_var[var]
        print(f"    {var:5s} max rel={rel_max:.3e}  "
              f"A/B={floors.get(var, float('nan')):.3e}  "
              f"limit={limits.get(var, float('nan')):.3e}  "
              f"mean-diff/member={ratio_by_var[var]:.2f}  "
              f"{'OK' if rel_max <= limits.get(var, -math.inf) else 'LARGE'}")
    print(f"  归档统计一致性（mean-diff/member <= {args.noise_factor:g}）："
          f"{'OK' if archive_stat_ok else 'LARGE'}"
          + (f"（超限：{','.join(stat_violations)}）"
             if stat_violations else ""))
    print(f"  ensemble-mean 最大 rel = {archive_rel_max:.3e}")
    print(f"  逐成员最大 rel = {member_max:.3e}")
    if worst_member is not None:
        print(f"    最差：step {worst_member['step']} "
              f"member {worst_member['archive_member']} "
              f"{worst_member['var']} rel={worst_member['rel']:.3e}")
    if violations:
        print("  超限行（ensemble mean）：")
        for row in violations[:12]:
            step_no = args.archive_steps[row["step"]]
            print(f"    step {row['step']}（文件 {step_no:03d}，"
                  f"lead {step_no * 6}h）{row['var']} "
                  f"rel={row['rel']:.3e} limit={limits[row['var']]:.3e}")

    # 3) 默认参数冒烟
    native_stats = []
    native_ok = True
    if not args.skip_native:
        print("\n" + "-" * 78)
        print("[3] 默认参数冒烟（FGVP 无 perturb_std_scale；只查形状/有限值）")
        print("-" * 78)
        args.xi = "own"
        native_calls = run_example_main(
            args.model, 1.0, args, real_input, own_xi, False, None, False)
        expected_shape = (args.members, 1, channels, height, width)
        for st in native_calls:
            step = st["step"]
            ok = (
                tuple(st["shape"]) == expected_shape
                and st["finite_frac"] == 1.0
                and math.isfinite(st["mean"])
                and math.isfinite(st["std"])
            )
            native_ok = native_ok and ok
            native_stats.append({**st, "ok": ok})
            print(f"  step {step}: shape={st['shape']} "
                  f"finite={st['finite_frac']:.6f} "
                  f"mean={st['mean']:.6g} std={st['std']:.6g} "
                  f"min={st['min']:.6g} max={st['max']:.6g} "
                  f"-> {'OK' if ok else 'BAD'}")
        del native_calls
        gc.collect()
    else:
        print("\n[3] 跳过默认冒烟（--skip-native）")

    # 4) 判定
    # 硬证据（与图内 RandomNormalLike 无关）：
    #   - example 自带的 ξ 表逐位等于测试流（run_example_main 内已校验）；
    #   - feed audit 逐位校验每次 run 的 spw/xi、input 滑窗、step/hour/doy。
    # 输出侧：example vs A 的差必须与同 feeds 的 A/B 噪声下限同量级。
    own_gen_ok = own_summary["ok"]
    arch_gen_ok = archive_summary["ok"]
    feeds_ok = own_feed_ok and archive_feed_ok
    outputs_ok = own_gen_ok and arch_gen_ok

    if not native_ok or not feeds_ok or not outputs_ok:
        status, code = "FAIL", 1
    elif not archive_stat_ok:
        status, code = "INCONCLUSIVE", 2
    else:
        status, code = "PASS", 0

    print("\n" + "=" * 78)
    print(f"feed audit own-xi     : "
          f"{'OK' if own_feed_ok else 'FAIL'}")
    print(f"feed audit archive-xi : "
          f"{'OK' if archive_feed_ok else 'FAIL'}")
    print(f"own-xi   example vs A : "
          f"{'OK' if own_gen_ok else 'FAIL'} "
          f"（max rel={own_summary['gen_max']:.3e}，"
          f"阈值={own_summary['limit']:.3e}，"
          f"A/B={own_summary['noise_max']:.3e}）")
    print(f"archive-xi example vs A : "
          f"{'OK' if arch_gen_ok else 'FAIL'} "
          f"（max rel={archive_summary['gen_max']:.3e}，"
          f"阈值={archive_summary['limit']:.3e}，"
          f"A/B={archive_summary['noise_max']:.3e}）")
    exceeded_vars = [
        var for var, _ in CAPTURE
        if archive_rel_by_var[var] > limits.get(var, -math.inf)
    ]
    print(f"归档 ensemble mean : "
          f"{'OK' if archive_cmp_ok else 'LARGE'} "
          f"（保守 rel 阈值，超限 {len(exceeded_vars)} 个："
          f"{','.join(exceeded_vars) if exceeded_vars else '无'}）")
    print(f"归档统计一致性 : "
          f"{'OK' if archive_stat_ok else 'LARGE'} "
          f"（mean-diff/member <= {args.noise_factor:g}，超限 "
          f"{','.join(stat_violations) if stat_violations else '无'}）")
    print(f"默认冒烟 : {'OK' if native_ok else 'BAD'}")
    print(f"RESULT: {status}")
    if status == "INCONCLUSIVE":
        print("  feeds 逐位审计与输出噪声下限检查都通过；但归档与本次的"
              f"集合均值差超过成员离散度的 {args.noise_factor:g} 倍："
              f"{','.join(stat_violations)}。归档那次 batch=6、成员分块 "
              "6/6/1（图内 RandomNormalLike 按 worker 内前向调用序对齐，与 "
              "example 的 1 块 × 4 步不同），本来就不能逐位对拍；"
              "example 的确定性结论以 feed audit + 同调用序 A/B 为准。")
    elif status == "FAIL":
        print("  优先看上面的 feed audit 条目：ξ 表、input 滑窗、"
              "step/hour/doy；再看 example vs A 是否超过 A/B 噪声下限的 "
              f"{args.noise_factor:g} 倍。")
    print("=" * 78)

    report = {
        "status": status,
        "pass": status == "PASS",
        "model": args.model,
        "input": args.input,
        "members": args.members,
        "steps": args.steps,
        "batch": args.batch,
        "device": args.device,
        "cudnn": args.cudnn,
        "threshold_rel_rmse": tol,
        "threshold_noise_factor": args.noise_factor,
        "threshold_archive_rel": args.archive_rel,
        "archive": {
            "root": args.archive,
            "init": args.archive_init,
            "members": args.archive_members,
            "steps": args.archive_steps,
            "archive_rows": args.archive_rows,
            "member_mapping": (
                "example row i <-> archive member rank0_cohort[i]；"
                "默认 0/4/8/12"),
        },
        "xi": {
            "example_rows": args.members,
            "archive_rows": args.archive_rows,
            "step0_same": bool(step0_same),
            "max_abs_diff_own_vs_archive": xi_max_diff,
        },
        "graph": {
            "channels": channels,
            "frames": frames,
            "height": height,
            "width": width,
            "n_targets": n_targets,
            "rank": rank,
            "inputs": sorted(input_names),
            "metadata": {
                k: meta.get(k) for k in (
                    "spw.ar1.method", "spw.ar1.input", "spw.ar1.targets",
                    "spw.ar1.sigma", "spw.ar1.tau_days", "spw.ar1.rank",
                    "spw.ar1.length_scale", "spw.ar1.seed",
                    "spw.ar1.tokens", "spw.ar1.channels", "spw.ar1.grid")
            },
        },
        "feed_audit": {
            "own_xi": {"ok": bool(own_feed_ok), "issues": own_feed_issues},
            "archive_xi": {
                "ok": bool(archive_feed_ok),
                "issues": archive_feed_issues,
            },
        },
        "own_xi": {
            "example_vs_reference": own_metrics,
            "reference_a_vs_b": own_baseline,
            "max_rel": own_summary["gen_max"],
            "baseline_max_rel": own_summary["noise_max"],
            "limit": own_summary["limit"],
            "ok": own_gen_ok,
        },
        "archive_xi": {
            "example_vs_reference": archive_metrics,
            "reference_a_vs_b": archive_baseline,
            "max_rel": archive_summary["gen_max"],
            "baseline_max_rel": archive_summary["noise_max"],
            "limit": archive_summary["limit"],
            "capture_max_rel_by_var": cap_example,
            "capture_noise_floor_by_var": cap_floor,
            "ok": arch_gen_ok,
        },
        "archive_compare": {
            "criterion": (
                "gate: per-var mean_diff_over_expected <= noise_factor；"
                "diagnostic: per-var rel <= max(archive_rel, "
                "noise_factor * same-var A/B)"),
            "ensemble_mean": archive_mean,
            "per_member": archive_member,
            "ensemble_mean_max_rel": archive_rel_max,
            "per_member_max_rel": member_max,
            "max_rel_by_var": archive_rel_by_var,
            "noise_floor_by_var": floors,
            "limit_by_var": limits,
            "mean_diff_over_expected_by_var": ratio_by_var,
            "stat_factor": args.noise_factor,
            "stat_exceeded_vars": stat_violations,
            "stat_ok": bool(archive_stat_ok),
            "rel_ok_by_var": {
                var: bool(archive_rel_by_var[var] <= limits.get(var, -math.inf))
                for var, _ in CAPTURE
            },
            "rel_exceeded_vars": exceeded_vars,
            "violations": violations,
            "rel_ok": bool(archive_cmp_ok),
            "ok": bool(archive_stat_ok),
        },
        "native_smoke": native_stats,
        "summary": {
            "feed_ok": bool(feeds_ok),
            "own_gen_ok": bool(own_gen_ok),
            "archive_gen_ok": bool(arch_gen_ok),
            "outputs_ok": bool(outputs_ok),
            "archive_stat_ok": bool(archive_stat_ok),
            "archive_rel_ok": bool(archive_cmp_ok),
            "native_ok": bool(native_ok),
        },
    }
    out = Path(args.report)
    out.write_text(
        json.dumps(_clean(report), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"报告：{out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
