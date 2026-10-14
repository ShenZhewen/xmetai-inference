# -*- coding: utf-8 -*-
"""fgvp_ens 真实数据档：真 51 成员集合 ONNX 在切片数据上跑一遍，和标准答案统计对拍。

除链路端到端外还要证明**集合语义没坏**（扰动在触发、成员互异、spread>0）：FGVP 集合
的扰动路径静默死了照样全绿，确定性档发现不了。逐成员对拍在原理上不成立（扰动是图内
随机算子、无 seed）；golden 是 8 卡正式跑的 51 成员，接近独立采样。

判 PASS 三层：
  * 确定性层：51 成员 × 步 × 配方声明变量的覆盖面、NaN 掩码、网格与 golden 一致；
  * 集合层：成员互异、逐变量 spread>0、集合均值对拍 —— 阈值从 golden 自身标定：
    两次独立 51 成员采样的均值差 ~ N(0, σ·√(2/51))，超 4 倍预测噪声记违规点，违规
    占比 ≤ 1e-3（与 ``EXCEED_FRACTION_CLOSE`` 同档）；σ≈0 的确定性格点回落 RTOL/ATOL。
    **tp 例外**：零膨胀重尾让 4σ 线的实测越线率达 0.6~0.7%（其余变量最高 2.6e-4，
    符合高斯预期），tp 的占比线放宽到 ``_FAIL_FRACTION`` 的 0.05 —— 真实回归会让
    降水区整片越线（百分之几十），与噪声仍隔一个量级。
  * 批量化层：配方显式给 ``model_batch=6``（一次 forward 堆 6 条成员轨迹）。随机算子
    若按 (1,...) 出噪声再 broadcast 到 batch 维，k 行拿到同一份扰动、成员互异用例
    立刻红。批量化是否真进图另有一条用例盯 rank 日志：图不是动态 batch 轴时 runner
    会打一条 WARNING 退回 1（这一趟照跑完，其余用例照样全绿，却跟批量化毫无关系）。

成员目录 0 基（``member_000..050``）。golden 必须挑 fgvp_ens_merged **新 .so 批次**
（20250721 起）的完整日期 —— 批次混杂，旧批均值对拍可能真红。

跑法：pytest -m realdata tests/integration/test_fgvp_ens_realdata.py
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

from realdata_common import (ATOL, EXCEED_FRACTION_CLOSE, RTOL, expected_vars,
                             load_fields, steps_in, variables_in)

# runner 侧「批量化生效 / 被退回」的日志标记。从 runner import 而不抄字符串：措辞
# 改了这里跟着变，不会退化成一个永远搜不到的旧标记、把判据变成假绿。
from xmetai_inference.runner import BATCH_ACTIVE_MARK, BATCH_FALLBACK_MARK

#: fixture 根。
DATA_ROOT = Path("/workspace/data/shenzw/inference_test_data/fgvp_ens_test_data")

#: 本档自己的切片 store（fgvp 档的切片只覆盖 20250102，golden 归档没有那天）。
DATASET_ROOT = DATA_ROOT / "dataset"

#: 被检 ONNX：iwc_gssp_im = fgvp_ens 现役图（20261008 导出，与仓库配方同源）。
ONNX_NAME = "iwc_gssp_im.onnx"

#: ONNX 外部权重（ORT 从 .onnx 同目录读，缺了加载即崩，必须一起自检）。
ONNX_DATA_NAME = "iwc_gssp_im.data"

#: 起报时间。必须是 fgvp_ens_merged 新 .so 批次（20250721 起）的完整日期。
TIMES = "2025072100"

#: 参与对拍的步序号（1-based，与落盘的 ``001.nc`` 对齐）。
STEPS = 3

#: 全员 51：集合统计的采样噪声按 √51 缩，测的就是正式配方语义。
MEMBERS = 51

#: 前向批：一次 forward 堆几条成员轨迹（51 成员 = 9 块：8×6 + 3）。显式写死而非靠配方
#: 漂过来 —— 模板是 deepcopy 的正式配方（configs/fgvp_ens.py 本身就是 6），不写的话
#: 正式配方一改值这档会跟着变，覆盖面却没人知道。
MODEL_BATCH = 6

#: 单卡（cuda_devices 单卡空转，同 fgvp 档顶部说明）。
GPUS = 1

#: 标准答案根。出处记在 golden/PROVENANCE.md。
GOLDEN_ROOT = DATA_ROOT / "golden"

#: 整点起报 → cli 的目录名写 ``%Y%m%d``（8 位）。
INIT_DIR = "20250721"

#: 均值对拍的噪声倍数：|z|>4 概率 ≈ 6.3e-5，远低于 1e-3 的判据线。
Z_MAX = 4.0

#: golden 逐点成员散度低于它就算确定性格点，回落 RTOL/ATOL。
#: 绝对值而非相对值：输出变量量级 O(1)~O(1e5)，1e-6 之下纯属 float 噪声。
SIGMA_FLOOR = 1e-6

#: 均值对拍违规占比的失败线，按变量覆盖。tp 的 0.05 是实测定的（见文件头「tp 例外」）。
_FAIL_FRACTION = {"tp": 0.05}

#: 成员目录名：member_000 起步（0 基）。
_MEMBER_RE = re.compile(r"^member_(\d{3})$")

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
        "把切片 store、模型、.so、golden 放进该目录后再跑",
        allow_module_level=True,
    )


_CONFIG_TEMPLATE = Template('''\
# -*- coding: utf-8 -*-
"""fgvp_ens 真实数据档的推理配方 —— 由 test_fgvp_ens_realdata.py 运行时生成，别手改。"""
import copy
from pathlib import Path

from xmetai_inference.configs.fgvp_ens import cfg as _production

# deepcopy：import 拿到的是缓存的同一个对象，直接改会把正式配方改成指向 fixture。
cfg = copy.deepcopy(_production)

# 三个切片 store glob 出来即用；通道按名字选，顺序不影响结果。
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
# .so 与 onnxruntime/CUDA 是 ABI 绑定的，必须用 fixture 里配套的那份。
cfg.ops_library = "$ops_library"

# 自回归第 i 步只依赖第 <i 步。集合语义下 golden 也只存 3 步。
cfg.times = "$times"
cfg.steps = $steps
cfg.members = $members
cfg.gpus = $gpus
# 前向批：本档要证明批量化路径本身没坏（切块 → 一次 forward k 条 → 逐行回填），
# 所以显式给值，不靠正式配方漂过来（见 MODEL_BATCH 的说明）。图不是动态 batch 轴
# 时 runner 会打 WARNING 退回 1，由 test_..._model_batch_takes_effect 拦。
cfg.model_batch = $model_batch
# worker 设 0 免得 DataLoader 再起子进程。
cfg.dataloader = {
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": True,
    "shuffle": False,
    "drop_last": False,
}
''')


def _ops_library_path() -> str:
    """fixture 里那份 .so 的路径（同名可能带版本后缀，glob 出来）。"""
    matches = sorted(DATA_ROOT.glob("xmetai_onnx_plugins*.so"))
    return str(matches[0]) if matches else "xmetai_onnx_plugins.so"


def _write_config(dest: Path) -> Path:
    """把模板落成真文件。CLI 收的是路径，给不了对象。"""
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


# 51 成员 × 3 步 × 5 变量的场 ≈6GB float64 持不住，故逐成员流式累计 sum/sumsq/count。
# float64 累计是必须的：z500 平方和方差比 mean² 小 ~1e-6，float32 舍入会吃掉散度。

def _member_dirs(root: Path) -> dict:
    """成员目录根 -> ``{成员号(3位): 路径}``，只认 ``member_NNN`` 命名。"""
    found = {}
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        match = _MEMBER_RE.fullmatch(path.name)
        if match:
            found[match.group(1)] = path
    return found


def _mean_std(sum_, sumsq, count):
    """流式累计 -> (逐点均值, 逐点标准差)。未累计到的点计为 NaN。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, sum_ / np.maximum(count, 1), np.nan)
        var = np.where(count > 1,
                       sumsq / np.maximum(count, 1) - mean * mean, np.nan)
    # 平方和法的微负截回 0。
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std


def _collect(run_members_root: Path, golden_members_root: Path) -> dict:
    """两侧 51 成员目录流式对拍，返回 ``{rows, skipped, twins}``。

    ``rows`` 每个 (步, 变量) 一条（均值对拍 + NaN/spread 指标）；``skipped`` 记只在
    一侧存在的 (成员/步/变量)；``twins`` 记与 member_000 逐位相同的 (成员, 步)。
    网格不一致直接抛 ValueError。
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

    # 变量清单从第一对文件探测（同一次运行里所有成员/步都一样）。
    first_run_dir = run_members[common[0]]
    first_golden_dir = golden_members[common[0]]
    names = None
    base = {}          # member_000 各 (step, var) 的场，成员互异性的比对基准
    twin_members = []
    acc = {}            # (step, var) -> 累计器；run/golden 两侧各一套

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

        member_is_twin = {}  # step -> 是否与基准成员完全相同
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
            # 网格对不上说明网格本身变了，逐格相减没有意义。
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

    noise_scale = np.sqrt(2.0 / MEMBERS)   # 两次独立 51 成员采样均值差的 σ 倍率
    rows = []
    for (step, var), slot in sorted(acc.items()):
        run_mean, run_std = _mean_std(slot["run_sum"], slot["run_sumsq"],
                                      slot["run_count"])
        gold_mean, gold_std = _mean_std(slot["gold_sum"], slot["gold_sumsq"],
                                        slot["gold_count"])
        compared = (slot["run_count"] > 0) & (slot["gold_count"] > 0)
        diff = run_mean - gold_mean
        noise = gold_std * noise_scale
        # 确定性格点（golden 散度可忽略）回落 RTOL/ATOL；集合格点 4 倍预测噪声。
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
    """跑一次真 51 成员推理并对拍，整模块共用。返回 ``(指标, 配方路径)``。"""
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

    # SystemExit 单独接住，理由同 fgvp 档。
    try:
        rc = main(["--config", str(config_path), "--out", str(out_dir)])
    except SystemExit as exc:
        pytest.fail(f"推理链以 SystemExit 退出：{exc}")
    assert rc == 0, f"推理链返回 {rc}（非 0 见上面日志）"

    run_root = out_dir / INIT_DIR
    assert run_root.is_dir(), (
        f"产物目录里没有 {INIT_DIR}/，实有：{sorted(p.name for p in out_dir.iterdir())}")
    metrics = _collect(run_root, GOLDEN_ROOT / INIT_DIR)

    # 批量化进没进图只有 rank 日志说得清：单卡走 cli._run_infer 的 gpus<=1 分支，日志
    # 固定落 <out>/rank_0.log（多卡才是 rank_N.log）。判定放在用例里，免得日志路径一变
    # 连坐四条与批量化无关的用例。
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
    """核心：集合均值与 golden 均值的偏离不得超过预测采样噪声（自标定口径）。"""
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
    """集合性：成员必须互不相同，且每个 (步, 变量) 的 spread 必须为正。"""
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
    """覆盖面兜底：51 成员 × 步 × 变量一个都不能静默少比；另加 tp 非负。"""
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
    """NaN↔数值的格子不进超容差统计，缺测位置得逐成员盯。"""
    metrics, _ = run
    bad = [r for r in metrics["rows"] if r["nan_mismatch"] > 0]
    assert not bad, (
        "以下 (步, 变量) 的 NaN 位置与 golden 不一致，多半是缺测处理路径变了：\n"
        + "\n".join(f"  step={r['step']:03d} var={r['var']} 不一致点={r['nan_mismatch']}"
                    for r in bad[:20])
    )


def test_fgvp_ens_model_batch_takes_effect(run):
    """批量化层：``model_batch>1`` 必须真进图，不能悄悄退回逐条。

    图不是动态 batch 轴时 runner 只打一条 WARNING 就退回 1，这一趟照跑完、均值/散度/
    成员互异照样全绿 —— 那时这条档与批量化无关。故以 rank 日志为凭：「生效」或
    「退回」必有其一。
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
