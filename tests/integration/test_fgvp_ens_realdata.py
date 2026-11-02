# -*- coding: utf-8 -*-
"""fgvp_ens realdata suite: run the real 51-member ensemble ONNX on sliced data and compare
ensemble statistics against golden.

Per-member comparison is not viable (the randomness inside the graph has no seed), so the
ensemble mean is compared: thresholds are self-calibrated from the noise of two independent
51-member samples, sqrt(2/51)*sigma, and points beyond 4 sigma are violations. **tp is an
exception**: zero inflation and heavy tails push the observed exceedance rate to 0.6-0.7%, so
its fraction limit is relaxed to 0.05 (a genuine regression exceeds it across entire
precipitation areas, still leaving an order of magnitude of margin). A separate test watches
the rank log of the batching layer to catch model_batch>1 silently falling back to 1.

Usage: pytest -m realdata tests/integration/test_fgvp_ens_realdata.py
"""
import re
import sys
from pathlib import Path
from string import Template

import numpy as np
import pytest

pytestmark = pytest.mark.realdata
pytest.importorskip("onnxruntime")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realdata_common import (ATOL, EXCEED_FRACTION_CLOSE, RTOL,
                             FGVP_ENS_FIXTURE_ROOT, data_root_hint,
                             expected_vars, load_fields, steps_in, variables_in)

# Runner-side log markers for "batching active / fell back". Imported from the runner rather than
# copied as strings, so reworded markers follow automatically instead of degrading into a stale
# marker that is never found and turns the check into a false pass.
from xmetai_inference.utils.runner import BATCH_ACTIVE_MARK, BATCH_FALLBACK_MARK

#: Fixture root. The absolute path is configured at the top of realdata_common.py.
DATA_ROOT = FGVP_ENS_FIXTURE_ROOT

#: This suite's own sliced stores (the fgvp slices cover only 20250102, which is absent from the golden archive).
DATASET_ROOT = DATA_ROOT / "dataset"

#: ONNX under test: iwc_gssp_im, the current fgvp_ens graph (same source as the repository recipe).
ONNX_NAME = "iwc_gssp_im.onnx"

#: ONNX external weights (ORT reads them from the .onnx directory; loading fails if missing, so this is checked too).
ONNX_DATA_NAME = "iwc_gssp_im.data"

#: Initialization time. Must be a complete date from the fgvp_ens_merged new-.so batch (starting 20250721).
TIMES = "2025072100"

#: Step indices compared (1-based, aligned with the written ``001.nc`` files).
STEPS = 3

#: Full 51 members: ensemble sampling noise scales with sqrt(51); this tests the production recipe semantics.
MEMBERS = 51

#: Forward batch: number of member trajectories stacked per forward pass (51 members = 9 chunks: 8x6 + 3).
#: Pinned explicitly rather than inherited: the template deepcopies the production recipe
#: (configs/weather_era_0p25_fgvp_ens.py is 6), so without this a recipe change would silently propagate here.
MODEL_BATCH = 6

#: Single GPU (cuda_devices is inert on one GPU; see the note at the top of the fgvp suite).
GPUS = 1

#: Golden reference root. Provenance is recorded in golden/PROVENANCE.md.
GOLDEN_ROOT = DATA_ROOT / "golden"

#: Hourly initialization -> the CLI writes the directory name as ``%Y%m%d`` (8 digits).
INIT_DIR = "20250721"

#: Noise multiplier for the mean comparison: P(|z|>4) ≈ 6.3e-5, far below the 1e-3 decision line.
Z_MAX = 4.0

#: Golden per-point member spread below this counts as a deterministic cell, falling back to RTOL/ATOL.
#: An absolute rather than relative value: output variables range over O(1)-O(1e5), and anything below 1e-6 is pure float noise.
SIGMA_FLOOR = 1e-6

#: Failure threshold for the violation fraction of the mean comparison, overridable per variable.
#: tp's 0.05 is empirically derived (see "tp is an exception" in the module header).
_FAIL_FRACTION = {"tp": 0.05}

#: Member directory names: start at member_000 (0-based).
_MEMBER_RE = re.compile(r"^member_(\d{3})$")

_hint = data_root_hint(DATA_ROOT)
if _hint:
    pytest.skip(_hint, allow_module_level=True)

_missing = []
if not DATASET_ROOT.is_dir() or len(list(DATASET_ROOT.glob("*.zarr"))) != 3:
    _missing.append(f"{DATASET_ROOT} 的 3 个切片 store")
if not (DATA_ROOT / ONNX_NAME).is_file():
    _missing.append(ONNX_NAME)
if not (DATA_ROOT / ONNX_DATA_NAME).is_file():
    _missing.append(f"{ONNX_DATA_NAME}（ONNX 外部权重，与 .onnx 同目录）")
if not list(DATA_ROOT.glob("xmetai_onnx_plugins*.so")):
    _missing.append("xmetai_onnx_plugins*.so")
if not (GOLDEN_ROOT / INIT_DIR).is_dir():
    _missing.append(f"golden/{INIT_DIR}/")
else:
    _golden_members = sorted(
        p for p in (GOLDEN_ROOT / INIT_DIR).iterdir()
        if p.is_dir() and _MEMBER_RE.fullmatch(p.name))
    if len(_golden_members) != MEMBERS:
        _missing.append(
            f"golden/{INIT_DIR}/ 的成员目录数应为 {MEMBERS}，实得 {len(_golden_members)}")
    elif any(len(steps_in(p)) < STEPS for p in _golden_members):
        _missing.append(f"golden/{INIT_DIR}/ 里有成员目录不足 {STEPS} 步")

if _missing:
    pytest.skip(
        f"fixture 不齐: {DATA_ROOT}（缺 {', '.join(_missing)}）—— "
        "把切片 store、模型、.so、golden 放进该目录后再跑；"
        "DATA_ROOT 在 tests/integration/realdata_common.py 顶部配置",
        allow_module_level=True,
    )


_CONFIG_TEMPLATE = Template('''\
# -*- coding: utf-8 -*-
"""fgvp_ens realdata inference recipe; generated at runtime by test_fgvp_ens_realdata.py. Do not edit."""
import copy
from pathlib import Path

from xmetai_inference.configs.weather_era_0p25_fgvp_ens import cfg as _production

# deepcopy: import returns the same cached object; mutating it directly would point the production recipe at the fixture.
cfg = copy.deepcopy(_production)

# Use all three sliced stores found by glob; channels are selected by name, so order does not matter.
DATASET_ROOT = Path("$dataset_root")
_paths = sorted(DATASET_ROOT.glob("*.zarr"))
if len(_paths) != 3:
    raise SystemExit(
        f"{DATASET_ROOT} 下应有 3 个切片 store，实得 {len(_paths)} 个："
        f"{[p.name for p in _paths]}。"
    )
cfg.dataset["paths"] = [str(p) for p in _paths]

DATA_ROOT = Path("$data_root")
cfg.model_path = str(DATA_ROOT / "$onnx_name")
# The .so is ABI-bound to onnxruntime/CUDA; use the copy shipped with the fixture.
cfg.ops_library = "$ops_library"

# Autoregressive step i depends only on steps < i. For ensembles, the golden data also stores only 3 steps.
cfg.times = "$times"
cfg.steps = $steps
cfg.members = $members
cfg.gpus = $gpus
# Forward batching: this suite verifies the batching path itself (split into chunks -> one
# forward for k members -> fill rows back), so set the value explicitly instead of inheriting
# it from the production recipe (see MODEL_BATCH). If the graph has no dynamic batch axis, the
# runner logs a WARNING and falls back to 1; test_..._model_batch_takes_effect catches that.
cfg.model_batch = $model_batch
# num_workers=0 avoids spawning DataLoader subprocesses.
cfg.dataloader = {
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": True,
    "shuffle": False,
    "drop_last": False,
}
''')


def _ops_library_path() -> str:
    """Path to the .so in the fixture (the same base name may carry a version suffix; found by glob)."""
    matches = sorted(DATA_ROOT.glob("xmetai_onnx_plugins*.so"))
    return str(matches[0]) if matches else "xmetai_onnx_plugins.so"


def _write_config(dest: Path) -> Path:
    """Materialize the template as a real file. The CLI accepts paths, not objects."""
    dest.write_text(
        _CONFIG_TEMPLATE.substitute(
            dataset_root=str(DATASET_ROOT),
            data_root=str(DATA_ROOT),
            onnx_name=ONNX_NAME,
            ops_library=_ops_library_path(),
            times=TIMES,
            steps=STEPS,
            members=MEMBERS,
            model_batch=MODEL_BATCH,
            gpus=GPUS,
        ),
        encoding="utf-8",
    )
    return dest


# The full 51 members x 3 steps x 5 variables field set (~6 GB as float64) does not fit in memory,
# so sum/sumsq/count are accumulated streaming, member by member.
# float64 accumulation is essential: the variance is mean-square minus mean^2, a difference of
# only ~1e-6 relative to the mean square, and float32 rounding would consume the spread.

def _member_dirs(root: Path) -> dict:
    """Member root directory -> ``{member id (3 digits): path}``; only ``member_NNN`` names are recognized."""
    found = {}
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        match = _MEMBER_RE.fullmatch(path.name)
        if match:
            found[match.group(1)] = path
    return found


def _mean_std(sum_, sumsq, count):
    """Streaming accumulators -> (per-point mean, per-point standard deviation). Points with no accumulated samples are NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, sum_ / np.maximum(count, 1), np.nan)
        var = np.where(count > 1,
                       sumsq / np.maximum(count, 1) - mean * mean, np.nan)
    # Clamp the small negative variance of the sum-of-squares method to 0.
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std


def _collect(run_members_root: Path, golden_members_root: Path) -> dict:
    """Stream-compare the 51-member directories on both sides; returns ``{rows, skipped, twins}``.

    ``rows`` holds one entry per (step, variable) (mean comparison plus NaN/spread metrics);
    ``skipped`` records members/steps/variables present on only one side; ``twins`` records
    (member, step) pairs bit-identical to member_000. A grid mismatch raises ValueError.
    """
    run_members = _member_dirs(run_members_root)
    golden_members = _member_dirs(golden_members_root)

    skipped = []
    skipped += [("成员", name, "只在产物侧") for name in sorted(set(run_members) - set(golden_members))]
    skipped += [("成员", name, "只在 golden 侧") for name in sorted(set(golden_members) - set(run_members))]
    common = sorted(set(run_members) & set(golden_members))
    if not common:
        raise ValueError(
            f"两侧没有共同的成员目录。\n  产物: {sorted(run_members)}\n  golden: {sorted(golden_members)}")

    # Probe the variable list from the first file pair (identical across all members/steps within one run).
    first_run_dir = run_members[common[0]]
    first_golden_dir = golden_members[common[0]]
    names = None
    base = {}          # member_000 fields per (step, var), the baseline for member distinctness
    twin_members = []
    acc = {}            # (step, var) -> accumulators; one set each for run and golden

    def _slot(step, var):
        key = (step, var)
        if key not in acc:
            acc[key] = {
                "run_sum": None, "run_sumsq": None, "run_count": None,
                "gold_sum": None, "gold_sumsq": None, "gold_count": None,
                "nan_mismatch": 0, "run_min": None,
            }
        return acc[key]

    def _ensure_arrays(slot, shape):
        for k in ("run_sum", "run_sumsq", "run_count",
                  "gold_sum", "gold_sumsq", "gold_count", "run_min"):
            if slot[k] is None:
                if k in ("run_count", "gold_count"):
                    slot[k] = np.zeros(shape, dtype=np.int32)
                elif k == "run_min":
                    slot[k] = np.full(shape, np.inf)
                else:
                    slot[k] = np.zeros(shape, dtype=np.float64)

    for member_id in common:
        run_dir, gold_dir = run_members[member_id], golden_members[member_id]
        run_steps, gold_steps = set(steps_in(run_dir)), set(steps_in(gold_dir))
        for step in sorted(run_steps - gold_steps):
            skipped.append(("步", f"{member_id}/{step:03d}", "只在产物侧"))
        for step in sorted(gold_steps - run_steps):
            skipped.append(("步", f"{member_id}/{step:03d}", "只在 golden 侧"))
        steps = sorted(run_steps & gold_steps)

        member_is_twin = {}  # step -> whether identical to the baseline member
        for step in steps:
            path_run = run_dir / f"{step:03d}.nc"
            path_gold = gold_dir / f"{step:03d}.nc"
            if names is None:
                vars_run, vars_gold = variables_in(path_run), variables_in(path_gold)
                for name in sorted(vars_run ^ vars_gold):
                    skipped.append(("变量", name, "只在一侧"))
                names = sorted(vars_run & vars_gold)
                if not names:
                    continue

            fields_run, lat_r, lon_r = load_fields(path_run, names)
            fields_gold, lat_g, lon_g = load_fields(path_gold, names)
            # A grid mismatch means the grid itself changed; cell-wise subtraction is meaningless.
            if lat_r.shape != lat_g.shape or lon_r.shape != lon_g.shape:
                raise ValueError(f"网格 shape 不一致：{path_run} 与 {path_gold}")
            if not (np.allclose(lat_r, lat_g, atol=1e-4)
                    and np.allclose(lon_r, lon_g, atol=1e-4)):
                raise ValueError(f"网格坐标不一致：{path_run} 与 {path_gold}")

            twin_here = False
            for var in names:
                a = fields_run[var]
                b = fields_gold[var]
                slot = _slot(step, var)
                _ensure_arrays(slot, a.shape)
                both = np.isfinite(a) & np.isfinite(b)
                slot["nan_mismatch"] += int(np.count_nonzero(
                    np.isfinite(a) != np.isfinite(b)))
                a_m = np.where(both, a, 0.0)
                b_m = np.where(both, b, 0.0)
                slot["run_sum"] += a_m
                slot["run_sumsq"] += a_m * a_m
                slot["run_count"] += both
                slot["gold_sum"] += b_m
                slot["gold_sumsq"] += b_m * b_m
                slot["gold_count"] += both
                slot["run_min"] = np.minimum(slot["run_min"], np.where(both, a, np.inf))

                if member_id == common[0]:
                    base[(step, var)] = a
                elif np.array_equal(a, base[(step, var)]):
                    twin_here = True
            if twin_here:
                twin_members.append((member_id, step))

    noise_scale = np.sqrt(2.0 / MEMBERS)   # sigma multiplier for the mean difference of two independent 51-member samples
    rows = []
    for (step, var), slot in sorted(acc.items()):
        run_mean, run_std = _mean_std(slot["run_sum"], slot["run_sumsq"],
                                      slot["run_count"])
        gold_mean, gold_std = _mean_std(slot["gold_sum"], slot["gold_sumsq"],
                                        slot["gold_count"])
        compared = (slot["run_count"] > 0) & (slot["gold_count"] > 0)
        diff = run_mean - gold_mean
        noise = gold_std * noise_scale
        # Deterministic cells (negligible golden spread) fall back to RTOL/ATOL; stochastic cells use 4x predicted noise.
        deterministic = compared & (gold_std <= SIGMA_FLOOR)
        stochastic = compared & ~deterministic
        fail = np.zeros(diff.shape, dtype=bool)
        fail[deterministic] = np.abs(diff[deterministic]) > (
            ATOL + RTOL * np.abs(gold_mean[deterministic]))
        fail[stochastic] = np.abs(diff[stochastic]) > Z_MAX * noise[stochastic]
        n_compared = int(np.count_nonzero(compared))
        n_fail = int(np.count_nonzero(fail)) if n_compared else 0
        gold_field_std = float(np.nanstd(gold_mean[compared])) if n_compared else float("nan")
        rows.append({
            "step": step,
            "var": var,
            "n_compared": n_compared,
            "fail_fraction": (n_fail / n_compared) if n_compared else float("nan"),
            "max_abs_diff": float(np.nanmax(np.abs(diff[compared]))) if n_compared else float("nan"),
            "diff_over_std": (float(np.nanmax(np.abs(diff[compared]))) / gold_field_std
                             if n_compared and gold_field_std > 0 else float("nan")),
            "run_spread_mean": float(np.nanmean(run_std[compared])) if n_compared else float("nan"),
            "gold_spread_mean": float(np.nanmean(gold_std[compared])) if n_compared else float("nan"),
            "run_min": float(np.nanmin(slot["run_min"][compared])) if n_compared else float("nan"),
            "nan_mismatch": slot["nan_mismatch"],
        })
    return {"rows": rows, "skipped": skipped, "twins": twin_members}


@pytest.fixture(scope="session")
def run(tmp_path_factory):
    """Run the real 51-member inference once and compare, shared across the module. Returns ``(metrics, recipe path)``."""
    from xmetai_inference import main

    config_path = _write_config(
        tmp_path_factory.mktemp("fgvp_ens_config") / "fgvp_ens_realdata_cfg.py")
    out_dir = tmp_path_factory.mktemp("fgvp_ens_realdata")
    print(f"\n[fixture] 本次用的配方：{config_path}")

    provenance = GOLDEN_ROOT / "PROVENANCE.md"
    if provenance.is_file():
        print(f"\n[fixture] golden 出处：\n{provenance.read_text(encoding='utf-8')}")
    else:
        print(f"\n[fixture] 提醒：{provenance} 缺失 —— golden 出处未记录")

    # Catch SystemExit separately, for the same reason as the fgvp suite.
    try:
        rc = main(["--config", str(config_path), "--out", str(out_dir)])
    except SystemExit as exc:
        pytest.fail(f"推理链以 SystemExit 退出：{exc}")
    assert rc == 0, f"推理链返回 {rc}（非 0 见上面日志）"

    run_root = out_dir / INIT_DIR
    assert run_root.is_dir(), (
        f"产物目录里没有 {INIT_DIR}/，实有：{sorted(p.name for p in out_dir.iterdir())}")
    metrics = _collect(run_root, GOLDEN_ROOT / INIT_DIR)

    # Only the rank log reveals whether batching actually entered the graph: a single GPU goes
    # through cli._run_infer's gpus<=1 branch, and the log always lands at <out>/rank_0.log
    # (rank_N.log only with multiple GPUs). The assertion lives in the test so that a change of
    # log path does not also fail the four tests unrelated to batching.
    log_path = out_dir / "rank_0.log"
    log_text = (log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.is_file() else "")
    metrics["model_batch"] = {
        "requested": MODEL_BATCH,
        "log_exists": log_path.is_file(),
        "active": BATCH_ACTIVE_MARK in log_text,
        "fallback": BATCH_FALLBACK_MARK in log_text,
        "log": str(log_path),
    }
    return metrics, config_path


def test_fgvp_ens_mean_matches_golden(run):
    """Core: the ensemble mean must not deviate from the golden mean beyond the predicted sampling noise (self-calibrated)."""
    metrics, _ = run
    rows = metrics["rows"]
    assert rows, "没有可比对象 —— 产物和标准答案没有共同的 (成员, 步, 变量)"

    def fail_line(r):
        return _FAIL_FRACTION.get(r["var"], EXCEED_FRACTION_CLOSE)

    bad = [r for r in rows if r["fail_fraction"] > fail_line(r)]
    if bad:
        bad.sort(key=lambda r: -(r["fail_fraction"] / fail_line(r)))
        detail = "\n".join(
            f"  step={r['step']:03d} var={r['var']:<6} 违规占比={r['fail_fraction']:.4g}"
            f"（线={fail_line(r):g}）"
            f" max|diff|={r['max_abs_diff']:.4g} max|diff|/场std={r['diff_over_std']:.4g}"
            f" run散度={r['run_spread_mean']:.4g} golden散度={r['gold_spread_mean']:.4g}"
            for r in bad[:20]
        )
        pytest.fail(
            f"{len(bad)}/{len(rows)} 个 (步, 变量) 的集合均值偏离 golden 超标：\n{detail}\n"
            "散度两侧同量级而均值整片偏 ⇒ 链路变了（前处理/算子/回填）；"
            "散度本身差一个量级 ⇒ 扰动强度变了。"
        )


def test_fgvp_ens_perturbation_alive(run):
    """Ensemble sanity: members must differ from one another, and the spread of every (step, variable) must be positive."""
    metrics, _ = run

    assert not metrics["twins"], (
        "以下成员与 member_000 逐位相同 —— 扰动路径死了（51 次前向算出同一份）：\n"
        + "\n".join(f"  member_{m} step={s:03d}" for m, s in metrics["twins"][:20])
    )

    dead = [r for r in metrics["rows"] if not (r["run_spread_mean"] > 0.0)]
    assert not dead, (
        "以下 (步, 变量) 的成员间散度为 0 —— 该输出通道的扰动断了：\n"
        + "\n".join(f"  step={r['step']:03d} var={r['var']}"
                    f" golden散度={r['gold_spread_mean']:.4g}"
                    for r in dead[:20])
        + "\ngolden 侧同为 0 ⇒ 从来就没通过扰动（找模型侧）；golden 侧非 0 ⇒ 这"
          "次跑丢的（找框架侧）。"
    )


def test_fgvp_ens_coverage(run):
    """Coverage guard: no member, step, or variable may be silently skipped; also checks that tp is non-negative."""
    metrics, config_path = run
    rows, skipped = metrics["rows"], metrics["skipped"]

    assert {r["step"] for r in rows} == set(range(1, STEPS + 1)), (
        f"比到的步不是 1..{STEPS}，实得 {sorted({r['step'] for r in rows})}"
    )

    expected = expected_vars(config_path)
    compared = {r["var"] for r in rows}
    assert compared == expected, (
        f"比到的变量和配方声明的不一致：\n"
        f"  少比了 {sorted(expected - compared)}\n"
        f"  多比了 {sorted(compared - expected)}"
    )

    level_skips = [s for s in skipped if s[0] in ("成员", "步", "变量")]
    assert not level_skips, f"有成员/步/变量被静默跳过：{level_skips}"

    tp_rows = [r for r in rows if r["var"] == "tp"]
    if tp_rows:
        negative = [r for r in tp_rows if r["run_min"] < 0]
        assert not negative, (
            "tp 出现负值（to_dataset 的非负兜底丢了）：\n"
            + "\n".join(f"  step={r['step']:03d} min={r['run_min']:.4g}"
                        for r in negative[:20])
        )


def test_fgvp_ens_no_nan_mismatch(run):
    """Cells that are NaN on one side and numeric on the other do not enter the exceed statistics; missing-data positions are checked per member."""
    metrics, _ = run
    bad = [r for r in metrics["rows"] if r["nan_mismatch"] > 0]
    assert not bad, (
        "以下 (步, 变量) 的 NaN 位置与 golden 不一致，多半是缺测处理路径变了：\n"
        + "\n".join(f"  step={r['step']:03d} var={r['var']} 不一致点={r['nan_mismatch']}"
                    for r in bad[:20])
    )


def test_fgvp_ens_model_batch_takes_effect(run):
    """Batching layer: ``model_batch>1`` must actually enter the graph, not silently fall back to per-sample.

    When the graph has no dynamic batch axis, the runner emits a single WARNING and falls back
    to 1; the run still completes and the mean/spread/member-distinctness checks all pass, so
    this suite would prove nothing about batching. The rank log is therefore authoritative:
    exactly one of "active" or "fell back" must appear.
    """
    metrics, _ = run
    info = metrics["model_batch"]

    if not info["log_exists"]:
        pytest.fail(f"rank 日志不在 {info['log']} —— cli 单卡的 --log-file 落点改了？")

    if not info["active"]:
        pytest.fail(
            f"model_batch={info['requested']} 没生效"
            + ("：runner 判定图的输入/输出 batch 轴不是符号维、退回了逐条 —— 要重导"
               "动态 batch 的图（正式配置当前就是这个值），或把本档 MODEL_BATCH 调回 1"
               if info["fallback"] else
               "：日志里既没有「生效」也没有「退回」—— runner 里的判定分支根本没走到"
               "（日志标记常量被改过？）")
            + f"\n  日志：{info['log']}")
