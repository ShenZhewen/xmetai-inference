# -*- coding: utf-8 -*-
"""fgvp realdata suite: run the real ONNX model on sliced data and compare against golden.

The recipe is derived by deepcopying configs/weather_era_0p25_fgvp.py and overrides only
fixture-specific entries; PASS requires the worst verdict to be in {IDENTICAL, CLOSE}. The
classification criteria are frozen in realdata_common and bit-identical output is not required.
The provenance of the golden data is unconfirmed: it guards against regressions only and does
not prove the current results are correct.

Usage: pytest -m realdata tests/integration/test_fgvp_realdata.py
"""
import sys
from pathlib import Path
from string import Template

import pytest

pytestmark = pytest.mark.realdata
pytest.importorskip("onnxruntime")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realdata_common import (ATOL, RTOL, VERDICT_RANK, FGVP_FIXTURE_ROOT,
                             compare_dirs, data_root_hint, expected_vars)

#: Fixture root. Four components: dataset/*.zarr (3 sliced stores), the ONNX under test,
#: the operator .so, and golden. The absolute path is configured at the top of realdata_common.py.
DATA_ROOT = FGVP_FIXTURE_ROOT

#: Initialization time. The slice covers only this one, and the golden data is for it alone.
TIMES = "2025010200"

#: Step indices compared (1-based, aligned with the written ``001.nc`` files).
STEPS = 3

#: Single GPU, single sample: the test needs correctness only, not throughput.
MEMBERS = 1
GPUS = 1

# On a single GPU, the device is chosen by the outer CUDA_VISIBLE_DEVICES: ``cli._run_infer``
# reads ``cfg.cuda_devices`` only when gpus>1, so this knob is deliberately left unset.

#: Golden reference root.
GOLDEN_ROOT = DATA_ROOT / "golden"

#: Hourly initialization -> the CLI writes the directory name as ``%Y%m%d`` (8 digits).
INIT_DIR = "20250102"

_PASS_RANK = VERDICT_RANK["CLOSE"]

_hint = data_root_hint(DATA_ROOT)
if _hint:
    pytest.skip(_hint, allow_module_level=True)

_missing = []
if not (DATA_ROOT / "dataset").is_dir():
    _missing.append("dataset/")
elif len(list((DATA_ROOT / "dataset").glob("*.zarr"))) != 3:
    _missing.append("dataset/ 下的 3 个切片 store")
if not (DATA_ROOT / "iwc_fgvp_gdn2_260901.onnx").is_file():
    _missing.append("iwc_fgvp_gdn2_260901.onnx")
if not list(DATA_ROOT.glob("xmetai_onnx_plugins*.so")):
    _missing.append("xmetai_onnx_plugins*.so")
if not (GOLDEN_ROOT / INIT_DIR).is_dir():
    _missing.append(f"golden/{INIT_DIR}/")

if _missing:
    pytest.skip(
        f"fixture 不齐: {DATA_ROOT}（缺 {', '.join(_missing)}）—— "
        "把切片数据、onnx、.so 放进该目录后再跑；"
        "DATA_ROOT 在 tests/integration/realdata_common.py 顶部配置",
        allow_module_level=True,
    )


# string.Template rather than f-string/format: the template is full of braces.
_CONFIG_TEMPLATE = Template('''\
# -*- coding: utf-8 -*-
"""fgvp realdata inference recipe; generated at runtime by test_fgvp_realdata.py. Do not edit."""
import copy
from pathlib import Path

from xmetai_inference.configs.weather_era_0p25_fgvp import cfg as _production

# deepcopy: import returns the same cached object; mutating it directly would point the production recipe at the fixture.
cfg = copy.deepcopy(_production)

DATA_ROOT = Path("$data_root")

# The sliced stores keep their source file names, so glob alignment is automatic; channels are selected by name, so order does not matter.
_paths = sorted((DATA_ROOT / "dataset").glob("*.zarr"))
if len(_paths) != 3:
    raise SystemExit(
        f"{DATA_ROOT}/dataset 下应有 3 个切片 store，实得 {len(_paths)} 个："
        f"{[p.name for p in _paths]}。"
    )
cfg.dataset["paths"] = [str(p) for p in _paths]

cfg.model_path = f"{DATA_ROOT}/iwc_fgvp_gdn2_260901.onnx"
# The .so is ABI-bound to onnxruntime/CUDA; use the copy shipped with the fixture.
cfg.ops_library = "$ops_library"

# Autoregressive step i depends only on steps < i, so the first 3 steps of a 60-step run are numerically identical.
cfg.times = "$times"
cfg.steps = $steps
cfg.members = $members
cfg.gpus = $gpus
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
            data_root=str(DATA_ROOT),
            ops_library=_ops_library_path(),
            times=TIMES,
            steps=STEPS,
            members=MEMBERS,
            gpus=GPUS,
        ),
        encoding="utf-8",
    )
    return dest


@pytest.fixture(scope="session")
def run(tmp_path_factory):
    """Run the real inference once and compare, shared across the module (goes through the
    public entry point ``main``, the only path that exercises config loading, scheduling,
    and output writing). Returns ``(output directory, rows, skipped, recipe path)``.
    """
    from xmetai_inference import main

    config_path = _write_config(
        tmp_path_factory.mktemp("fgvp_config") / "fgvp_realdata_cfg.py")
    out_dir = tmp_path_factory.mktemp("fgvp_realdata")
    print(f"\n[fixture] 本次用的配方：{config_path}")

    # Catch SystemExit separately so upstream validation failures are not buried in pytest internals.
    try:
        rc = main(["--config", str(config_path), "--out", str(out_dir)])
    except SystemExit as exc:
        pytest.fail(f"推理链以 SystemExit 退出：{exc}")
    assert rc == 0, f"推理链返回 {rc}（非 0 见上面日志）"

    produced = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert INIT_DIR in produced, f"产物目录里没有 {INIT_DIR}/，实有：{produced}"

    rows, skipped = compare_dirs(out_dir, GOLDEN_ROOT, step_limit=STEPS,
                                 rtol=RTOL, atol=ATOL)
    return out_dir, rows, skipped, config_path


def test_fgvp_matches_golden(run):
    """Core: the worst verdict for every (step, variable) must be in {IDENTICAL, CLOSE}."""
    _, rows, _, _ = run
    assert rows, "没有可比对象 —— 产物和标准答案没有共同的 (起报, 步, 变量)"

    bad = [r for r in rows if VERDICT_RANK[r["verdict"]] > _PASS_RANK]
    if bad:
        bad.sort(key=lambda r: -VERDICT_RANK[r["verdict"]])
        detail = "\n".join(
            f"  step={r['step']:03d} var={r['var']:<6} {r['verdict']:<9}"
            f" max|diff|={r['max_abs_diff']:.4g} 超容差占比={r['exceed_frac']:.4g}"
            f" max|diff|/场std={r['diff_over_std']:.4g}"
            for r in bad[:20]
        )
        pytest.fail(
            f"{len(bad)}/{len(rows)} 个 (步, 变量) 超出 CLOSE 档：\n{detail}\n"
            "先看有没有某个变量**从第 1 步**就偏 —— 那是前处理或算子注册的问题；"
            "只有后面几步才偏则是 rollout 回填的锅。"
        )


def test_fgvp_coverage(run):
    """Coverage guard: writing fewer steps makes passing easier, so step- and variable-level skips are failures (initialization-level skips are tolerated)."""
    _, rows, skipped, config_path = run

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

    level_skips = [s for s in skipped if s[0] in ("步", "变量")]
    assert not level_skips, f"有步/变量被静默跳过：{level_skips}"


def test_fgvp_no_nan_mismatch(run):
    """Cells that are NaN on one side and numeric on the other do not enter the exceed statistics; missing-data positions are checked separately."""
    _, rows, _, _ = run
    bad = [r for r in rows if r["nan_mismatch"] > 0]
    assert not bad, (
        "以下 (步, 变量) 的 NaN 位置两侧不一致，多半是缺测处理路径变了：\n"
        + "\n".join(f"  step={r['step']:03d} var={r['var']} 不一致点={r['nan_mismatch']}"
                    for r in bad[:20])
    )
