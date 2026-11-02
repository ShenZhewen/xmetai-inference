# -*- coding: utf-8 -*-
"""fuxi (fuxi21 / FuXi-2.1) realdata suite: run the real .pt2 model on sliced data and compare against golden.

The recipe is derived by deepcopying configs/weather_era_0p25_fuxi21.py and overrides only
fixture-specific entries. fuxi2.1 is not bit-reproducible on GPU: step 1 is classified per
variable (surface CLOSE, upper-air DRIFT), and from step 2 on the noise spreads into every
channel through feedback, so only the catastrophe limit applies there. A deviation on all
variables at step 1 indicates a preprocessing or time-feature problem.

Usage: pytest -m realdata tests/integration/test_fuxi_realdata.py
"""
import sys
from pathlib import Path
from string import Template

import pytest

pytestmark = pytest.mark.realdata
pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realdata_common import (ATOL, RTOL, VERDICT_RANK, FUXI_FIXTURE_ROOT,
                             compare_dirs, data_root_hint, expected_vars)

#: Fixture root. The three model files (fuxi-2.1.pt2 / mean.nc / std.nc) are placed manually,
#: plus 5 sliced stores. The absolute path is configured at the top of realdata_common.py.
DATA_ROOT = FUXI_FIXTURE_ROOT

#: Initialization time. The slice covers only this one, and the golden data is for it alone (within the production recipe's times range).
TIMES = "2025010200"

#: Step indices compared (1-based, aligned with the written ``001.nc`` files).
STEPS = 3

#: Deterministic suite: members=1, comparing each (step, variable) field directly.
MEMBERS = 1
GPUS = 1

# On a single GPU, the device is chosen by the outer CUDA_VISIBLE_DEVICES (see the note at the top of the fgvp suite).

#: GPU nondeterminism affects only upper-air channels; these variables are relaxed to DRIFT at step 1, the rest stay at CLOSE.
UPPER_AIR_VARS = {"z500", "q700", "t700", "t850", "u850", "v850"}


def _pass_rank(variable: str) -> int:
    return (VERDICT_RANK["DRIFT"] if variable in UPPER_AIR_VARS
            else VERDICT_RANK["CLOSE"])


#: Classification comparison applies only to this step; from step 2 on the noise spreads into every channel through feedback, so a strict threshold would always fail.
STRICT_STEP = 1

#: Catastrophe limit (all steps): fail when max|diff| exceeds this fraction of field variability.
CATASTROPHE_STD_LIMIT = 0.5

#: Golden reference root.
GOLDEN_ROOT = DATA_ROOT / "golden"

#: Hourly initialization -> the CLI writes the directory name as ``%Y%m%d`` (8 digits).
INIT_DIR = "20250102"

#: Number of sliced stores (fuxi21 uses 5 dynamic stores, adding soil / wave over fgvp).
_N_STORES = 5

_hint = data_root_hint(DATA_ROOT)
if _hint:
    pytest.skip(_hint, allow_module_level=True)

_missing = []
if not (DATA_ROOT / "dataset").is_dir():
    _missing.append("dataset/")
elif len(list((DATA_ROOT / "dataset").glob("*.zarr"))) != _N_STORES:
    _missing.append(f"dataset/ 下的 {_N_STORES} 个切片 store")
if not (DATA_ROOT / "fuxi-2.1.pt2").is_file():
    _missing.append("fuxi-2.1.pt2")
if not (DATA_ROOT / "mean.nc").is_file():
    _missing.append("mean.nc")
if not (DATA_ROOT / "std.nc").is_file():
    _missing.append("std.nc")
if not (GOLDEN_ROOT / INIT_DIR).is_dir():
    _missing.append(f"golden/{INIT_DIR}/")

if _missing:
    pytest.skip(
        f"fixture 不齐: {DATA_ROOT}（缺 {', '.join(_missing)}）—— "
        "把切片 store、模型三件套、golden 放进该目录后再跑；"
        "DATA_ROOT 在 tests/integration/realdata_common.py 顶部配置",
        allow_module_level=True,
    )


# string.Template is used for the same reason as in the fgvp suite.
_CONFIG_TEMPLATE = Template('''\
# -*- coding: utf-8 -*-
"""fuxi21 realdata inference recipe; generated at runtime by test_fuxi_realdata.py. Do not edit."""
import copy
from pathlib import Path

from xmetai_inference.configs.weather_era_0p25_fuxi21 import cfg as _production

# deepcopy: import returns the same cached object; mutating it directly would point the production recipe at the fixture.
cfg = copy.deepcopy(_production)

DATA_ROOT = Path("$data_root")

# The sliced stores keep their source file names, so glob alignment is automatic; channels are selected by name, so order does not matter.
_paths = sorted((DATA_ROOT / "dataset").glob("*.zarr"))
if len(_paths) != 5:
    raise SystemExit(
        f"{DATA_ROOT}/dataset 下应有 5 个切片 store，实得 {len(_paths)} 个："
        f"{[p.name for p in _paths]}。"
    )
cfg.dataset["paths"] = [str(p) for p in _paths]

# Point the model at the fixture. mean.nc/std.nc resolve relative to the model_path directory and ship with the model.
cfg.model_path = f"{DATA_ROOT}/fuxi-2.1.pt2"

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


def _write_config(dest: Path) -> Path:
    """Materialize the template as a real file. The CLI accepts paths, not objects."""
    dest.write_text(
        _CONFIG_TEMPLATE.substitute(
            data_root=str(DATA_ROOT),
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
    """Run the real inference once and compare, shared across the module. Returns ``(output directory, rows, skipped, recipe path)``."""
    from xmetai_inference import main

    config_path = _write_config(
        tmp_path_factory.mktemp("fuxi_config") / "fuxi_realdata_cfg.py")
    out_dir = tmp_path_factory.mktemp("fuxi_realdata")
    print(f"\n[fixture] 本次用的配方：{config_path}")

    # Catch SystemExit separately, for the same reason as the fgvp suite.
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


def test_fuxi_matches_golden(run):
    """Core: classify step 1 per variable, and apply the catastrophe limit to all steps (see the module header)."""
    _, rows, _, _ = run
    assert rows, "没有可比对象 —— 产物和标准答案没有共同的 (起报, 步, 变量)"

    strict = [r for r in rows if r["step"] == STRICT_STEP]
    bad = [r for r in strict
           if VERDICT_RANK[r["verdict"]] > _pass_rank(r["var"])]

    # Comparisons involving NaN are false, so `not` sends them to blown -- NaN should fail anyway.
    blown = [r for r in rows
             if not (r["diff_over_std"] < CATASTROPHE_STD_LIMIT)]

    if bad or blown:
        detail = ""
        if bad:
            bad.sort(key=lambda r: -VERDICT_RANK[r["verdict"]])
            detail += "\n".join(
                f"  step={r['step']:03d} var={r['var']:<6} {r['verdict']:<9}"
                f" max|diff|={r['max_abs_diff']:.4g} 超容差占比={r['exceed_frac']:.4g}"
                f" max|diff|/场std={r['diff_over_std']:.4g}"
                f"（门槛 {'DRIFT' if r['var'] in UPPER_AIR_VARS else 'CLOSE'}）"
                for r in bad[:20]
            )
        if blown:
            blown.sort(key=lambda r: -(r["diff_over_std"] or 0.0))
            detail += ("\n  ——\n" if detail else "") + "\n".join(
                f"  step={r['step']:03d} var={r['var']:<6} 超灾难线"
                f" max|diff|={r['max_abs_diff']:.4g} max|diff|/场std="
                f"{r['diff_over_std']:.4g}（线={CATASTROPHE_STD_LIMIT:g}）"
                for r in blown[:20]
            )
        pytest.fail(
            f"第 {STRICT_STEP} 步判档超标 {len(bad)}/{len(strict)} 条，"
            f"超灾难线 {len(blown)}/{len(rows)} 条：\n{detail}\n"
            "第 1 步全变量就偏 ⇒ 前处理（单位换算 / mean/std 归一化）或时间条件"
            "的问题；只有高空通道第 1 步轻微超标且幅度 <1% 场变率的，按 GPU 噪声底"
            "排查（见文件头）；灾难线红的是回填 / zero_channels 级别的回归，场整个不对。"
        )


def test_fuxi_coverage(run):
    """Coverage guard: silently comparing fewer files passes more easily (same as the fgvp suite)."""
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


def test_fuxi_no_nan_mismatch(run):
    """Cells that are NaN on one side and numeric on the other do not enter the exceed statistics; missing-data positions are checked separately."""
    _, rows, _, _ = run
    bad = [r for r in rows if r["nan_mismatch"] > 0]
    assert not bad, (
        "以下 (步, 变量) 的 NaN 位置两侧不一致，多半是缺测处理路径变了：\n"
        + "\n".join(f"  step={r['step']:03d} var={r['var']} 不一致点={r['nan_mismatch']}"
                    for r in bad[:20])
    )
