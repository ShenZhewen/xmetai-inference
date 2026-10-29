# -*- coding: utf-8 -*-
"""fuxi（fuxi21 / FuXi-2.1）真实数据档：真 .pt2 在切片数据上跑一遍，和标准答案对拍。

配方是 configs/weather_era_0p25_fuxi21.py 的 deepcopy 派生，只覆盖 fixture 特有项。fuxi2.1 在 GPU 上
不可逐位复现：第 1 步按变量判档（地面 CLOSE、高空 DRIFT），第 2 步起噪声经回填扩散到
全部通道，只挂灾难线；第 1 步全变量就偏 ⇒ 前处理或时间条件的问题。

跑法：pytest -m realdata tests/integration/test_fuxi_realdata.py
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

from realdata_common import ATOL, RTOL, VERDICT_RANK, compare_dirs, expected_vars

#: fixture 根。模型三件套（fuxi-2.1.pt2 / mean.nc / std.nc）手工放，5 个切片 store。
DATA_ROOT = Path("/workspace/data/shenzw/inference_test_data/fuxi_test_data")

#: 起报时间。切片只覆盖这一个，标准答案也只对这一个（在正式配方 times 范围内）。
TIMES = "2025010200"

#: 参与对拍的步序号（1-based，与落盘的 ``001.nc`` 对齐）。
STEPS = 3

#: 确定性档：members=1，每个 (步, 变量) 逐场对拍。
MEMBERS = 1
GPUS = 1

# 单卡跑哪张卡由外层 CUDA_VISIBLE_DEVICES 决定（同 fgvp 档顶部说明）。

#: GPU 非确定性只打高空通道 —— 这组变量第 1 步放宽到 DRIFT，其余仍按 CLOSE。
UPPER_AIR_VARS = {"z500", "q700", "t700", "t850", "u850", "v850"}


def _pass_rank(variable: str) -> int:
    return (VERDICT_RANK["DRIFT"] if variable in UPPER_AIR_VARS
            else VERDICT_RANK["CLOSE"])


#: 判档对拍只看这一步；第 2 步起噪声经回填扩散到全部通道，紧门槛必红。
STRICT_STEP = 1

#: 灾难线（全步生效）：max|diff| 超场变率这个比例即红。
CATASTROPHE_STD_LIMIT = 0.5

#: 标准答案根。
GOLDEN_ROOT = DATA_ROOT / "golden"

#: 整点起报 → cli 的目录名写 ``%Y%m%d``（8 位）。
INIT_DIR = "20250102"

#: 切片 store 个数（fuxi21 用 5 个动态 store，比 fgvp 多 soil / wave）。
_N_STORES = 5

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
        "把切片 store、模型三件套、golden 放进该目录后再跑",
        allow_module_level=True,
    )


# string.Template 的理由同 fgvp 档。
_CONFIG_TEMPLATE = Template('''\
# -*- coding: utf-8 -*-
"""fuxi21 真实数据档的推理配方 —— 由 test_fuxi_realdata.py 运行时生成，别手改。"""
import copy
from pathlib import Path

from xmetai_inference.configs.weather_era_0p25_fuxi21 import cfg as _production

# deepcopy：import 拿到的是缓存的同一个对象，直接改会把正式配方改成指向 fixture。
cfg = copy.deepcopy(_production)

DATA_ROOT = Path("$data_root")

# 切片 store 沿用源文件名，glob 自动对齐；通道按名字选，顺序不影响结果。
_paths = sorted((DATA_ROOT / "dataset").glob("*.zarr"))
if len(_paths) != 5:
    raise SystemExit(
        f"{DATA_ROOT}/dataset 下应有 5 个切片 store，实得 {len(_paths)} 个："
        f"{[p.name for p in _paths]}。"
    )
cfg.dataset["paths"] = [str(p) for p in _paths]

# 模型指到 fixture。mean.nc/std.nc 按 model_path 所在目录解析，跟着模型过来。
cfg.model_path = f"{DATA_ROOT}/fuxi-2.1.pt2"

# 自回归第 i 步只依赖第 <i 步，跑 3 步的前 3 步与跑 60 步数值相同。
cfg.times = "$times"
cfg.steps = $steps
cfg.members = $members
cfg.gpus = $gpus
# worker 设 0 免得 DataLoader 再起子进程。
cfg.dataloader = {
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": True,
    "shuffle": False,
    "drop_last": False,
}
''')


def _write_config(dest: Path) -> Path:
    """把模板落成真文件。CLI 收的是路径，给不了对象。"""
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
    """跑一次真推理并对拍，整模块共用。返回 ``(产物目录, rows, skipped, 配方路径)``。"""
    from xmetai_inference import main

    config_path = _write_config(
        tmp_path_factory.mktemp("fuxi_config") / "fuxi_realdata_cfg.py")
    out_dir = tmp_path_factory.mktemp("fuxi_realdata")
    print(f"\n[fixture] 本次用的配方：{config_path}")

    # SystemExit 单独接住，理由同 fgvp 档。
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
    """核心：第 1 步按变量判档，全部步挂灾难线（口径见文件头）。"""
    _, rows, _, _ = run
    assert rows, "没有可比对象 —— 产物和标准答案没有共同的 (起报, 步, 变量)"

    strict = [r for r in rows if r["step"] == STRICT_STEP]
    bad = [r for r in strict
           if VERDICT_RANK[r["verdict"]] > _pass_rank(r["var"])]

    # NaN 的比较结果为假，not 之后进 blown —— NaN 本来就该红。
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
    """覆盖面兜底：缺文件静默少比 = 更容易绿（同 fgvp 档）。"""
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
    """NaN↔数值的格子不进超容差统计，缺测位置得单独盯。"""
    _, rows, _, _ = run
    bad = [r for r in rows if r["nan_mismatch"] > 0]
    assert not bad, (
        "以下 (步, 变量) 的 NaN 位置两侧不一致，多半是缺测处理路径变了：\n"
        + "\n".join(f"  step={r['step']:03d} var={r['var']} 不一致点={r['nan_mismatch']}"
                    for r in bad[:20])
    )
