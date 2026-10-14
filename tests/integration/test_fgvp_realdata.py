# -*- coding: utf-8 -*-
"""fgvp 真实数据档：真 ONNX 在切片数据上跑一遍，和标准答案对拍。

配方是 ``configs/fgvp.py`` 的 deepcopy 派生，落成临时 ``.py`` 喂 CLI（链路的
契约是路径不是对象，多卡 worker 还要按 ``--config-source`` 自己再加载一次），
只覆盖 fixture 特有项。判 PASS 口径：最坏一档 ∈ {IDENTICAL, CLOSE}，分档判据
冻结在 ``realdata_common``，不要改。不要求逐位相同（GPU 归约次序 / golden 当年
的 batch / 切片 store 的分块路径都在 CLOSE 范畴内）。

已知弱点：golden 的出处（哪个 commit、几张卡）没能确证 —— 本档挡得住**今后**的
回归（改了代码就红），但不能用来论证「当前结果是对的」。集合档的 golden 已按
规矩配了 PROVENANCE.md。

跑法：pytest -m realdata tests/integration/test_fgvp_realdata.py
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

from realdata_common import (ATOL, RTOL, VERDICT_RANK, compare_dirs,
                             expected_vars)

#: fixture 根。四件套：dataset/*.zarr（3 个切片 store）、被检 onnx、算子 .so、golden。
DATA_ROOT = Path("/workspace/data/shenzw/inference_test_data/fgvp_test_data")

#: 起报时间。切片只覆盖这一个，标准答案也只对这一个。
TIMES = "2025010200"

#: 参与对拍的步序号（1-based，与落盘的 ``001.nc`` 对齐）。
STEPS = 3

#: 单卡单样本：测试只要正确性，不要吞吐。
MEMBERS = 1
GPUS = 1

# 单卡跑哪张卡由外层 CUDA_VISIBLE_DEVICES 决定：``cli._run_infer`` 只在 gpus>1
# 时才读 ``cfg.cuda_devices``，写了也没人读，所以这里故意不设这个旋钮。

#: 标准答案根。
GOLDEN_ROOT = DATA_ROOT / "golden"

#: 整点起报 → cli 的目录名写 ``%Y%m%d``（8 位）。
INIT_DIR = "20250102"

_PASS_RANK = VERDICT_RANK["CLOSE"]

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
        "把切片数据、onnx、.so 放进该目录后再跑",
        allow_module_level=True,
    )


# string.Template 而非 f-string/format：模板里全是花括号，{} 占位符要全文转义。
_CONFIG_TEMPLATE = Template('''\
# -*- coding: utf-8 -*-
"""fgvp 真实数据档的推理配方 —— 由 test_fgvp_realdata.py 运行时生成，别手改。"""
import copy
from pathlib import Path

from xmetai_inference.configs.fgvp import cfg as _production

# deepcopy：import 拿到的是缓存的同一个对象，直接改会把正式配方改成指向 fixture。
cfg = copy.deepcopy(_production)

DATA_ROOT = Path("$data_root")

# 切片 store 沿用源文件名，glob 自动对齐；通道按名字选，顺序不影响结果。
_paths = sorted((DATA_ROOT / "dataset").glob("*.zarr"))
if len(_paths) != 3:
    raise SystemExit(
        f"{DATA_ROOT}/dataset 下应有 3 个切片 store，实得 {len(_paths)} 个："
        f"{[p.name for p in _paths]}。"
    )
cfg.dataset["paths"] = [str(p) for p in _paths]

cfg.model_path = f"{DATA_ROOT}/iwc_fgvp_gdn2_260901.onnx"
# .so 与 onnxruntime/CUDA 是 ABI 绑定的，必须用 fixture 里配套的那份。
cfg.ops_library = "$ops_library"

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


def _ops_library_path() -> str:
    """fixture 里那份 .so 的路径（同名可能带版本后缀，glob 出来）。"""
    matches = sorted(DATA_ROOT.glob("xmetai_onnx_plugins*.so"))
    return str(matches[0]) if matches else "xmetai_onnx_plugins.so"


def _write_config(dest: Path) -> Path:
    """把模板落成真文件。CLI 收的是路径，给不了对象。"""
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
    """跑一次真推理并对拍，整模块共用。返回 ``(产物目录, rows, skipped, 配方路径)``。

    走公开入口 ``xmetai_inference.main(argv)``：配置加载、多卡调度、后台写盘
    只有 CLI 那一段才经过。
    """
    from xmetai_inference import main

    config_path = _write_config(
        tmp_path_factory.mktemp("fgvp_config") / "fgvp_realdata_cfg.py")
    out_dir = tmp_path_factory.mktemp("fgvp_realdata")
    print(f"\n[fixture] 本次用的配方：{config_path}")

    # SystemExit 单独接住：前置校验失败（缺 .so / 起报不在数据里）的真正原因
    # 才不会埋在 pytest 内部栈里。
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
    """核心：每个 (步, 变量) 的最坏一档都必须在 {IDENTICAL, CLOSE} 里。"""
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
    """覆盖面兜底：对拍是「谁少以谁为准、缺的记 skipped」，产物少写一步反而更容易绿，
    所以步/变量级的跳过一律当失败（起报级容忍）。"""
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
    """NaN↔数值的格子不进超容差统计，缺测位置得单独盯。"""
    _, rows, _, _ = run
    bad = [r for r in rows if r["nan_mismatch"] > 0]
    assert not bad, (
        "以下 (步, 变量) 的 NaN 位置两侧不一致，多半是缺测处理路径变了：\n"
        + "\n".join(f"  step={r['step']:03d} var={r['var']} 不一致点={r['nan_mismatch']}"
                    for r in bad[:20])
    )
