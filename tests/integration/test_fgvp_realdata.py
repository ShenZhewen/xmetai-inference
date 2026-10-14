# -*- coding: utf-8 -*-
"""fgvp 真实数据档：用真 ONNX 在切片数据上跑一遍，和已经跑好的标准答案对拍。

它证明什么
----------
整条链端到端没变：配置解析 → Zarr 读取 → 前处理链（单位换算 / 几何 / 缺测填充）
→ ONNX 推理（含自定义算子注册）→ 自回归 rollout + ``state=result`` 回填 →
落盘解码。单元测试碰不到这条链的任何一段 —— 它要真实模型、真实网格、真实数据。

四件套都在 ``DATA_ROOT`` 下（见下），由切片脚本产出：

    dataset/*.zarr            三个切片 store（起报 ±6h 共 3 帧，78 通道）
    iwc_fgvp_gdn2_260901.onnx 被检模型（手工放）
    xmetai_onnx_plugins.*.so  自定义算子库（手工放，ABI 与 onnxruntime 绑定）
    golden/20250102/*.nc      标准答案（从全年输出里拷的前 3 步）

推理配方是本文件运行时生成的
----------------------------
``_CONFIG_TEMPLATE`` 会写出一份临时 ``.py`` 配方喂给 CLI，而不是在测试里直接构造
一个 config 对象 —— 因为这条链的契约是**路径**不是对象：``configs/base.py::load_config``
用 ``importlib`` 按路径 exec 那个文件，多卡时主进程给每个 worker 子进程传的也是
``--config-source <绝对路径>``，worker 自己再加载一次。临时文件落在 pytest 的 tmp
目录里，不进仓库。

配方本体是 ``configs/fgvp.py``（线上那份）的 **deepcopy 派生**，只覆盖 fixture
特有的项（数据路径 / ONNX / 起报 / 步数 / 卡数）。派生而不是另抄一份，是为了除这几项
之外跑的就是线上那一套 —— 正式配方改了处理器链或单位换算，这里跟着变。

判 PASS 的口径
--------------
**最坏一档 ∈ {IDENTICAL, CLOSE}**，不是逐位相等。判定用本文件下半部分的
``_compare_dirs``：它按「超容差点占比」分档（逐位 / 随机舍入 / 系统漂移 / 真不一样），
而不是看 max|diff| —— float32 的舍入误差与场的量级成正比，单看最大差分不开「累积
舍入」和「真的算出了不同的场」。

不要求逐位相同的三个理由：GPU 卷积的归约次序不保证逐位稳定；标准答案当年的
batch 未必和这次一样（batch 维度会改 kernel 选择）；切片 store 的通道数与源
不同（值一样，但 zarr 的读取分块路径变了）。这些都属于 CLOSE 该罩住的范畴。

卡数**不在**此列：``members=1`` 时 ``cli`` 按起报切任务（``split_by_member`` 在
``members <= 1`` 时必为假），每个起报由一个进程独立算完，没有跨卡归约 —— 过去
那份用 8 张卡跑的和今天用 1 张卡跑的，对应起报是同一份。

一处已知的弱点
--------------
标准答案的**出处没能确证** —— ``/workspace/data/shenzw/fgvp_output`` 是全年的，
但目前不知道它是哪个 commit、几张卡跑出来的（那目录里还留着 rank_0..7 共 8 个
日志，而当时的 config 写的是 gpus=2，对不上）。如果它恰好就是当前代码跑出来的，
那这个测试对「当年」是循环论证。它仍然挡得住**今后**的回归（改了代码，数变了，
就红），这正是这个测试的主要价值；但要拿它论证「当前结果是对的」则不成立。
要根治得把标准答案换成一份出处可查的产物，并把 commit 记进 fixture。

跑法：

    pytest -m realdata tests/integration/test_fgvp_realdata.py
"""
import re
import sys
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = pytest.mark.realdata
pytest.importorskip("onnxruntime")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# 可调项：改这里就行，下面的临时配方是从这几个常量生成的
# ---------------------------------------------------------------------------

#: fixture 根。
DATA_ROOT = Path("/workspace/data/shenzw/inference_test_data/fgvp_test_data")

#: 起报时间。切片只覆盖这一个，标准答案也只对这一个。
TIMES = "2025010200"

#: 参与对拍的步序号（1-based，与落盘的 ``001.nc`` 对齐）。
STEPS = 3

#: 单卡单样本：测试只要正确性，不要吞吐。
MEMBERS = 1
GPUS = 1

# 这里**故意没有** cuda_devices 这种旋钮 —— 它在单卡时是空转的。
# ``cli._run_infer`` 只在 ``gpus > 1`` 的循环里注入 CUDA_VISIBLE_DEVICES；
# ``gpus <= 1`` 走早返回（``subprocess.call(command, env=dict(os.environ))``），
# ``cfg.cuda_devices`` 从头到尾没被读过，``_build_infer_argv`` 也不传 ``--device``。
# 所以单卡跑哪张卡由外层环境决定，配置里写什么都没用：
#
#     CUDA_VISIBLE_DEVICES=3 python -m pytest tests/integration/test_fgvp_realdata.py -m realdata
#
# 不写 ``CUDA_VISIBLE_DEVICES`` 就是进程看到的第 0 张卡。

# ---------------------------------------------------------------------------

#: 标准答案根。用 fixture 目录里拷好的那一份，测试才自足。
GOLDEN_ROOT = DATA_ROOT / "golden"

#: 标准答案目录名：起报 20250102 00Z → ``cli.py`` 的 ``_init_dirname`` 在整点时
#: 写 ``%Y%m%d``，所以是 8 位，和全年输出目录的命名一致。
INIT_DIR = "20250102"

#: 判定档位排序，越大越坏。
_RANK = {"IDENTICAL": 0, "CLOSE": 1, "DRIFT": 2, "DIFFER": 3}
_PASS_RANK = _RANK["CLOSE"]

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


# --------------------------------------------------------------------------
# 生成临时配方
# --------------------------------------------------------------------------
# 用 string.Template 而不是 f-string / str.format：下面这段代码里全是花括号
# （dict 字面量、f-string），用 {} 做占位符就得把每一个都写成 {{}}，改错一个
# 就是语法错或静默错位。$ 占位符在这里不冲突。
_CONFIG_TEMPLATE = Template('''\
# -*- coding: utf-8 -*-
"""fgvp 真实数据档的推理配方 —— 由 tests/integration/test_fgvp_realdata.py
运行时生成，别手改（改了下次跑就被覆盖）。要调参改那个测试文件顶部的常量。"""
import copy
from pathlib import Path

from xmetai_inference.configs.fgvp import cfg as _production

# 必须 deepcopy：``from ...configs.fgvp import cfg`` 拿到的是 sys.modules 里缓存的
# **同一个对象**，直接改会把正式配方在整个进程里改成指向 fixture，后续任何用到
# fgvp 的代码都会踩到。
cfg = copy.deepcopy(_production)

DATA_ROOT = Path("$data_root")

# 三个切片 store 沿用源 store 的文件名，glob 一下就和切片脚本自动对齐了，不用
# 手工同步两份名单。通道按名字选，store 的先后顺序不影响结果。
_paths = sorted((DATA_ROOT / "dataset").glob("*.zarr"))
if len(_paths) != 3:
    raise SystemExit(
        f"{DATA_ROOT}/dataset 下应有 3 个切片 store，实得 {len(_paths)} 个："
        f"{[p.name for p in _paths]}。"
    )
cfg.dataset["paths"] = [str(p) for p in _paths]

cfg.model_path = f"{DATA_ROOT}/iwc_fgvp_gdn2_260901.onnx"
# 自定义算子库是 ABI 绑定的（xmetai_plugins:GatedDeltaNet2Fn），必须和跑它的
# onnxruntime / CUDA 版本配套，所以 fixture 目录里放的是专门那份 .so。
cfg.ops_library = f"{DATA_ROOT}/xmetai_onnx_plugins.ort1_24.cuda12.so"

# 步数少不影响前几步：自回归第 i 步只依赖第 <i 步，跑 3 步和跑 60 步的前 3 步数值相同。
cfg.times = "$times"
cfg.steps = $steps
cfg.members = $members
# 单卡。故意不写 cfg.cuda_devices：那个字段只在 gpus>1 时被读（见测试文件顶部），
# 单卡时写了只是让人误以为卡号被钉住了。
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
    """把上面的模板落成真文件。CLI 收的是路径，给不了对象。"""
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


def _expected_vars(config_path: Path) -> set:
    """本配方声明要落盘的变量集合（小写）。

    从生成出来的那份配方里读、而不是在这里再抄一份 —— 抄一份就会在正式
    ``configs/fgvp.py`` 改 ``vars`` 时静默失配，而覆盖断言照绿。
    """
    from xmetai_inference.configs.base import load_config

    cfg = load_config(str(config_path))
    return {v.strip().lower() for v in str(cfg.vars).split(",") if v.strip()}


# --------------------------------------------------------------------------
# 对比判据
# --------------------------------------------------------------------------
# 原先用仓库里的 ``util/compare_outputs.py``。那一整套是给「整年目录 vs 整年目录」
# 设计的：起报发现、集合成员、CSV 汇总、进度条、实况读取，对一个只比单起报 3 步的
# 回归测试全是累赘。这里留一份精简版。
#
# **阈值和 ``_classify`` 的判据是从原实现原样搬过来的，不要改。** 改了就是偷偷
# 放宽或收紧标准，测试和以前不再可比。
#
# 代价说清楚：这是第二份实现。上游若调 ``util/`` 里的阈值，这边不会跟。对回归测试
# 而言冻结阈值反而更稳（判据不会因为别人调评测工具而漂），但要知道有这个分叉。

#: 默认容差：rtol=1e-5 能容忍 float32 往返与代数化简的舍入（约 100 eps 的余量），
#: 又足以抓住任何真实的物理量变化。atol 给 tp 这类大片为 0 的场兜底。
RTOL = 1e-5
ATOL = 1e-6
EXCEED_FRACTION_CLOSE = 1e-3      # 万分之一以下：随机舍入
EXCEED_FRACTION_DRIFT = 0.5       # 半数以上：多数点都变了
DRIFT_STD_FRACTION = 1e-2         # max|diff| 超过场自身变率这个比例 ⇒ 物理上可分辨

_INIT_RE = re.compile(r"^\d{8}(?:\d{2})?$")
_STEP_RE = re.compile(r"^(\d{3})\.nc$")


def _init_dirs(root: Path) -> dict:
    """子目录名 -> 起报时刻。兼容 YYYYMMDD 与 YYYYMMDDHH 两种命名。"""
    found = {}
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        if _INIT_RE.fullmatch(entry.name):
            fmt = "%Y%m%d" if len(entry.name) == 8 else "%Y%m%d%H"
            found[entry.name] = pd.to_datetime(entry.name, format=fmt)
    return found


def _steps_in(directory: Path) -> list:
    """目录里的 ``NNN.nc`` 步序号。"""
    return sorted(
        int(match.group(1))
        for path in directory.iterdir()
        if path.is_file() and (match := _STEP_RE.fullmatch(path.name))
    )


def _variables_in(path: Path) -> set:
    """NetCDF 里的二维变量名（小写）。"""
    with xr.open_dataset(path) as ds:
        return {str(name).lower() for name in ds.data_vars if ds[name].ndim == 2}


def _align_grid(values, lat, lon):
    """把场摆成「北→南、经度 0..360 升序」，和原实现一致。

    纬度反序要连数据一起翻；经度只重排（不翻转），这样两份不同存储顺序的输出能
    逐格对上。shape 与坐标数不符直接报错，免得后面按错位的格子比。
    """
    values = np.asarray(values, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)
    if values.shape[-2:] != (lat.size, lon.size):
        raise ValueError(
            f"数据 shape {values.shape[-2:]} 与坐标 {(lat.size, lon.size)} 不一致")
    if lat.size > 1 and lat[0] < lat[-1]:
        lat = lat[::-1]
        values = np.flip(values, axis=-2)
    lon360 = np.mod(lon, 360.0)
    order = np.argsort(lon360)
    return np.take(values, order, axis=-1), lat, lon360[order]


def _load_fields(path: Path, variables):
    """读一个预测文件里的二维场 -> (``{变量: (H,W)}``, lat, lon)。

    变量名大小写不敏感：落盘走的是 ``upper()``，配置里写的是小写。
    """
    with xr.open_dataset(path) as ds:
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError(f"{path} 缺少 lat/lon 坐标")
        lower = {str(name).lower(): name for name in ds.data_vars}
        fields, lat, lon = {}, None, None
        for variable in variables:
            if variable not in lower:
                raise KeyError(f"{path} 缺少变量 {variable!r}")
            values = np.asarray(ds[lower[variable]].values, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"{path} 的 {variable} 不是二维规则网格")
            values, lat, lon = _align_grid(values, ds["lat"].values, ds["lon"].values)
            fields[variable] = values
        return fields, lat, lon


def _classify(max_diff, ref_std, exceed_fraction):
    """给一档判定。

    **主判据是超容差点的占比，不是 max|diff|。** 原因：float32 的舍入误差与场的
    量级成正比 —— z500 在 54000 量级上单次运算的 eps 就有 6.5e-3，rollout 累积
    到 0.1 完全正常。所以单看 max|diff| 分不开「累积舍入」和「真的算出了不同的
    场」，两者可以是同一个数量级。能分开的是**分布**：舍入是散布的少数点，真实的
    数值改动会让绝大多数点都偏离。

    另外若 max|diff| 超过场自身空间变率的 1%，无论占比多少都判 DIFFER —— 那已经
    是物理上可分辨的差异。
    """
    if max_diff == 0.0:
        return "IDENTICAL"
    if ref_std > 0 and max_diff > DRIFT_STD_FRACTION * ref_std:
        return "DIFFER"
    if exceed_fraction < EXCEED_FRACTION_CLOSE:
        return "CLOSE"
    if exceed_fraction < EXCEED_FRACTION_DRIFT:
        return "DRIFT"
    return "DIFFER"


def _field_metrics(a, b, rtol=RTOL, atol=ATOL):
    """两个 (H,W) 场 -> 指标 dict。NaN 位置不一致会被单独标出。

    容差同时看相对和绝对两把尺子（满足其一即通过）：``|a-b| <= atol + rtol*|b|``。
    分母取两个都有限的格子 —— 一侧缺测的格子不进这个统计，改了缺测路径不该在这里
    表现为「差异变小」。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    nan_mismatch = int(np.count_nonzero(finite_a != finite_b))
    both = finite_a & finite_b
    if not both.any():
        return {
            "verdict": "DIFFER",
            "max_abs_diff": float("nan"),
            "exceed_frac": float("nan"),
            "diff_over_std": float("nan"),
            "nan_mismatch": nan_mismatch,
        }

    diff = np.abs(a - b)[both]
    max_diff = float(diff.max())
    # 参考侧（标准答案）自身的空间变率，用来判断差异是否已物理可分辨
    ref_std = float(b[both].std())
    exceed_frac = float(np.count_nonzero(diff > atol + rtol * np.abs(b[both]))
                        / diff.size)
    return {
        "verdict": _classify(max_diff, ref_std, exceed_frac),
        "max_abs_diff": max_diff,
        "exceed_frac": exceed_frac,
        "diff_over_std": max_diff / ref_std if ref_std > 0 else float("nan"),
        "nan_mismatch": nan_mismatch,
    }


def _compare_dirs(root_a: Path, root_b: Path, step_limit=None,
                  rtol=RTOL, atol=ATOL):
    """两份输出根目录 -> ``(rows, skipped)``。

    ``rows`` 每个 (起报, step, 变量) 一条；``skipped`` 是只在一侧存在的东西，形状
    ``(级别, 什么, 为什么)``，级别 ∈ {"起报", "步", "变量"} —— 覆盖面断言靠它抓
    「产物少写了一步，测试反而更容易绿」这种情况。

    变量清单与网格只在第一对上探测一次：同一次运行里所有步都一样。
    """
    inits_a, inits_b = _init_dirs(root_a), _init_dirs(root_b)
    if not inits_a:
        raise ValueError(f"{root_a} 下没有起报目录")
    if not inits_b:
        raise ValueError(f"{root_b} 下没有起报目录")

    skipped = [("起报", name, "只在 A 侧") for name in sorted(set(inits_a) - set(inits_b))]
    skipped += [("起报", name, "只在 B 侧") for name in sorted(set(inits_b) - set(inits_a))]

    common_inits = sorted(set(inits_a) & set(inits_b))
    if not common_inits:
        raise ValueError(
            f"两侧没有共同的起报目录。\n  A: {sorted(inits_a)}\n  B: {sorted(inits_b)}")

    rows = []
    for init_dir in common_inits:
        dir_a, dir_b = root_a / init_dir, root_b / init_dir
        steps_a, steps_b = set(_steps_in(dir_a)), set(_steps_in(dir_b))
        for step in sorted(steps_a - steps_b):
            skipped.append(("步", f"{init_dir}/{step:03d}", "只在 A 侧"))
        for step in sorted(steps_b - steps_a):
            skipped.append(("步", f"{init_dir}/{step:03d}", "只在 B 侧"))

        steps = sorted(steps_a & steps_b)
        if step_limit is not None:
            steps = [s for s in steps if s <= step_limit]

        names = None
        for step in steps:
            path_a = dir_a / f"{step:03d}.nc"
            path_b = dir_b / f"{step:03d}.nc"
            if names is None:
                vars_a, vars_b = _variables_in(path_a), _variables_in(path_b)
                for name in sorted(vars_a ^ vars_b):
                    skipped.append(("变量", name, "只在一侧"))
                names = sorted(vars_a & vars_b)
                if not names:
                    continue

            fields_a, lat_a, lon_a = _load_fields(path_a, names)
            fields_b, lat_b, lon_b = _load_fields(path_b, names)
            # 坐标对不上就别比了：两边都摆成「北→南、经度升序」之后若还不同，说明
            # 网格本身不一样，逐格相减毫无意义（会得到一片看似"漂移"的噪声）。
            # 原实现只在第一对上校验（整年目录下逐对 allclose 是浪费），这里每步都查
            # —— 多出来的成本是每步两千多个浮点数的比较，相比读上 G 的数据可以忽略；
            # 严格说是比原实现更紧的一道闸，只会在真有网格不一致时才响。
            if lat_a.shape != lat_b.shape or lon_a.shape != lon_b.shape:
                raise ValueError(f"网格 shape 不一致：{path_a} 与 {path_b}")
            if not (np.allclose(lat_a, lat_b, atol=1e-4)
                    and np.allclose(lon_a, lon_b, atol=1e-4)):
                raise ValueError(f"网格坐标不一致：{path_a} 与 {path_b}")

            for name in names:
                rows.append({
                    "init": init_dir,
                    "step": step,
                    "var": name,
                    **_field_metrics(fields_a[name], fields_b[name], rtol, atol),
                })
    return rows, skipped


# --------------------------------------------------------------------------
# 跑链
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def run(tmp_path_factory):
    """跑一次真推理并对拍，整模块共用。返回 ``(产物目录, rows, skipped, 配方路径)``。

    走的是公开入口 ``xmetai_inference.main(argv)``（就是 CLI 的 ``main``），不是在
    测试里手搓 Rollout —— 配置加载、多卡调度、后台写盘这些只有 CLI 那一段才经过。
    """
    from xmetai_inference import main

    config_path = _write_config(
        tmp_path_factory.mktemp("fgvp_config") / "fgvp_realdata_cfg.py")
    out_dir = tmp_path_factory.mktemp("fgvp_realdata")
    print(f"\n[fixture] 本次用的配方：{config_path}")

    # 链路的失败有两条出口：写完盘发现错误返回 1，前置校验失败直接 SystemExit。
    # 后者不接住的话会以「fixture 里抛了 SystemExit」的形式报出来，把真正的原因
    # （缺 .so / 起报不在数据里 / ONNX 路径不对）埋在一堆 pytest 内部栈里。
    try:
        rc = main(["--config", str(config_path), "--out", str(out_dir)])
    except SystemExit as exc:
        pytest.fail(f"推理链以 SystemExit 退出：{exc}")
    assert rc == 0, f"推理链返回 {rc}（非 0 见上面日志）"

    produced = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert INIT_DIR in produced, f"产物目录里没有 {INIT_DIR}/，实有：{produced}"

    rows, skipped = _compare_dirs(out_dir, GOLDEN_ROOT, step_limit=STEPS)
    return out_dir, rows, skipped, config_path


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------

def test_fgvp_matches_golden(run):
    """核心：每个 (步, 变量) 的最坏一档都必须在 {IDENTICAL, CLOSE} 里。"""
    _, rows, _, _ = run
    assert rows, "没有可比对象 —— 产物和标准答案没有共同的 (起报, 步, 变量)"

    bad = [r for r in rows if _RANK[r["verdict"]] > _PASS_RANK]
    if bad:
        bad.sort(key=lambda r: -_RANK[r["verdict"]])
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
    """覆盖面兜底：比到的步和变量得是**全部**，不能因为一侧缺文件静默少比。

    对比函数的设计是「谁少以谁为准，缺的记进 skipped」。这个性质对整年目录很方便，
    但对回归测试是危险的 —— 产物少写了一步，测试反而更容易绿。所以这里把「步 /
    变量」两级的跳过一律当失败；起报级的跳过容忍（标准答案那侧可能有别的起报）。
    """
    _, rows, skipped, config_path = run

    assert {r["step"] for r in rows} == set(range(1, STEPS + 1)), (
        f"比到的步不是 1..{STEPS}，实得 {sorted({r['step'] for r in rows})}"
    )

    expected = _expected_vars(config_path)
    compared = {r["var"] for r in rows}
    assert compared == expected, (
        f"比到的变量和配方声明的不一致：\n"
        f"  少比了 {sorted(expected - compared)}\n"
        f"  多比了 {sorted(compared - expected)}"
    )

    level_skips = [s for s in skipped if s[0] in ("步", "变量")]
    assert not level_skips, f"有步/变量被静默跳过：{level_skips}"


def test_fgvp_no_nan_mismatch(run):
    """缺测位置必须两侧一致。

    单独一条是因为分档是按超容差点占比算的，而 NaN↔数值的格子根本不进那个统计
    —— 缺测路径改坏了照样能判 CLOSE。
    """
    _, rows, _, _ = run
    bad = [r for r in rows if r["nan_mismatch"] > 0]
    assert not bad, (
        "以下 (步, 变量) 的 NaN 位置两侧不一致，多半是缺测处理路径变了：\n"
        + "\n".join(f"  step={r['step']:03d} var={r['var']} 不一致点={r['nan_mismatch']}"
                    for r in bad[:20])
    )
