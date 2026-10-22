# -*- coding: utf-8 -*-
"""P4：**逐尺度的 SER** —— 剩下的那点离散度不足，缺在哪个尺度上。

为什么必须另算
--------------
评测流程里的 ``spherical_bands`` 吃的是 ``ensemble_mean`` 变换，成员维在 matcher
里就被平均掉了，算出来的是**平均场**的谱。``configs/weather_rmse_ens_fgvp_ar1v14.py``
的 docstring 自己写明了这一点：要读**成员离散**的谱必须从原始场离线另算。
本脚本就是那笔账。

它回答什么
----------
标量 SER ≡ ``spread / rmse@ensemble`` 是一个**全尺度混合**的量：它说"离散度整体
不够"，但不说"缺在行星尺度还是缺在天气尺度"。这两者的处置完全不同 —— 大尺度缺
可以靠加大 σ / 拉长 τ 补，小尺度缺则说明扰动**进不去**那个尺度，加 σ 只会先把
大尺度顶过头。所以要把 SER 按球谐带拆开：

    逐成员偏差场   d_m(x) = f_m(x) − mean_member(f_m(x))
    逐尺度离散度   P_s(k) = mean_m  B( d_m )(k)          B = 球谐带功率
    集合平均误差   e(x)   = mean_member(f_m(x)) − o(x)
    逐尺度误差     P_e(k) = B( e )(k)
    逐尺度 SER     R(k)   = sqrt( P_s(k) / P_e(k) )
    标量口径       spread² = 面积加权 mean_x mean_m d_m²

``R(k)`` 是标量 SER 的逐尺度推广：``< 1`` = 该尺度离散度不足，``> 1`` = 过头。

⚠ 这里是**第二版**。第一版算的是 ``B(std_member(f))`` —— 对**离散度场本身**
做球谐带。``spherical_band_power`` 会先扣掉全球面积加权平均，"离散度场"又是
正定的、空间上极光滑的，于是那个谱只反映**离散度场的空间花纹**，不是离散度的
尺度分布，``R(k)`` 会低到 0.03 而同时标量 SER ≈ 0.9。那批读数**不能用**。
判据就是下面打印的**闭合检查**：只有 ``Σ_k P_s`` 能对回 ``4π·spread²``，
这条谱才真的是标量 spread 的分解。

口径与坑
--------
* **归一化常数 ≈ 4π，且档间不变**。``_SphericalBandPower`` 的基函数是正交归一的
  （``basis[0,0]=1/√(4π)``），但 ``power()`` 末尾乘了 ``(2π)²``、且用 dθ 求积
  而非 cos-lat 求积，净效果是 ``Σ_k B(x) ≈ 4π × 面积加权 mean(x²)``。
  实测合成 z500 场落在 **0.95 × 4π**，不是 1.00 × 4π —— 这个 ~5% 偏移随场型走，
  **别拿 1.000 当判据**。真正的不变式是：正确构造下这个常数**不随时效变**
  （合成场上离散度幅度差 40 倍、常数只飘 0.5%），而旧构造低 150 倍且飘 35%。
  ``R(k)`` 是比值，常数自己会约掉；它只用于闭合检查。
* **球谐带从 l=1 起**（``DEFAULT_SPHERICAL_BANDS`` 最小带是 (1,4)），不含 l=0，
  所以误差侧的**全球均值偏差**不在谱里。打印的 ``closure_err`` 就是把这部分
  显式量出来（<100% 的缺口 ≈ 偏差占 rmse² 的比例），别当成 bug。
* 球谐带分段与归档里的 ``spherical_bands_*`` 一一对得上，``band`` 列可直接拼接。
* 预报侧**直接读 .nc**（省内存：一步 51 成员只 53MB）。文件名 ``NNN.nc`` =
  提前 ``NNN × 6h``（``cli.py:658`` 落盘用 ``step_idx = step + 1``）。这个映射
  会**打印出来**，并且用「标量 spread 对得上归档吗」再兜一道底。
* 观测侧走评测仓自己的 ``era5_zarr`` 读取器 —— 单位、时刻语义（``init_times``
  传的是**有效时刻** ``init + lead``）都由它保证。
* ⚠ **经度原点不一样，必须显式对齐**。预报 netCDF 的 lon 是 ``0..359.75``，
  era5 zarr store 的 lon 是 ``-180..179.75``。这个错位对本脚本**其余的每一道闸
  都是隐形的**：经度滚移不改任何一条纬圈上的功率谱，也不改 cos-lat 加权均值 ——
  所以闸1（对归档 spread）、闸2（闭合）、均值绝对值表全都看不出来。实测后果：
  spread 与归档**逐位相同**，而 6h 的 rmse 虚高 43 倍（915 vs 归档 21.09）。
  ``_align_lon_axis`` 按坐标算滚移格数，闸0 再**回读验证**（对齐后残余滚移必须
  是 0、corr 必须 >0.99）。
* 同理，球谐带功率对**纬度行序翻转**也是恒等的（lat 至少还有一道 raise 兜着）。

代价
----
每个 (臂, 起报) ≈ 3060 次小文件读（约 190MB/次×3060 ≈ 11GB），加约 50s 球谐。
5 臂 × 3 起报大约 **30~45 分钟**。冒烟先把 ``LEADS`` 收窄（见下）。

用法
----
在服务器上（要 netcdf4 + 评测仓；脚本自己会把 EVAL_ROOT 塞进 sys.path，
所以**不用**设 PYTHONPATH）::

    cd /workspace/szwCode/xmetai-evalation-v2
    python /workspace/szwCode/xmetai-inference2/xmetai_inference/analysis/ar1_spread_spectrum.py

要改臂表 / 起报 / 时效 / 输出路径，**直接改下面的字面量**（不用环境变量、不用命令行）。
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ── 可调项：直接改这里 ────────────────────────────────────────────────────
#: 预报根：<FORECAST_ROOT>/<臂>/<YYYYMMDD>/member_XXX/NNN.nc
FORECAST_ROOT = "/workspace/data/shenzw/ar1_v14"

#: 评测仓根（提供 era5 zarr 读取器 + 球谐带功率，口径与归档逐位一致）
EVAL_ROOT = "/workspace/szwCode/xmetai-evalation-v2"
ERA5_PL_STORE = ("/workspace/data/liujunjie/era5_foundation_store2/"
                 "era5_pl_2025.01-2026.07.c84.p25.h6.zarr")

#: 只跑 run-b 那五条（**同批**，可横比 —— 跨 Job 的 run-a 臂请另开一次输出文件）。
#: aB 与 s15 是 config 里定死的「形状 vs 大小」配对（时均等效 σ 0.1507 / 0.1500），
#: 这条对照是本脚本的主读：同 σ 平均强度下，包络把离散度挪到哪个尺度去了。
ARMS = ("fgvp_ens_ar1state_aB", "fgvp_ens_ar1state_s15",
        "fgvp_ens_ar1state_zm15", "fgvp_ens_ar1state_zmaB",
        "fgvp_ens_ar1state_t005s15")

#: 起报时刻（与推理侧 cfg.times 逐字一致）
INITS = ("2025070200", "2025071000", "2025071800")

#: 时效筛选（小时）。None = 全部 60 档。**冒烟先写 (24.0, 240.0)**，I/O 省 30 倍。
LEADS = None

#: 变量（本批单要素）
VAR = "z500"

#: 输出。空字符串 = 只打印不落盘。
OUT_CSV = "/workspace/data/shenzw/ar1_v14/_spread_spectrum.csv"

#: 归档 scores.csv 根（用来核对"标量 spread 算对了吗 / 时效映射错没错"）。
#: 找不到会**明确打印**是哪个路径没找到，然后跳过（上一版是静默跳过，白跑一轮）。
SCORES_ROOT = "/workspace/data/shenzw/evaluation_results/v14"

HOUR_INTERVAL = 6.0

#: ``_SphericalBandPower`` 的净归一化常数（见 module docstring）。仅用于闭合检查。
CLOSURE_REF = 4.0 * np.pi

#: 闭合检查的两道判据（单位都是「倍数 4π」）：
#: * **档间极差** < ``CLOSURE_DRIFT`` —— 这才是真判据。正确构造下这个常数在
#:   各时效之间**不变**（合成 z500 场实测：离散度幅度差 40 倍，常数只飘 0.5%）；
#:   旧构造（对离散度场本身做带功率）飘 35%，且低 150 倍。
#: * 绝对量落在 ``CLOSURE_BAND`` 内 —— 粗闸，一枪打死数量级级别的构造错。
#:   注意 4π 只是理论值，常数随场型有 ~5% 系统偏移（合成场实测 0.95），
#:   **不是** 1.000 才算对。
CLOSURE_DRIFT = 0.10
CLOSURE_BAND = (0.5, 1.5)

#: 6h 一定要在里面 —— 归档在 6h 是 spread 21.27 / rmse 21.09（几乎完美校准），
#: 那是最锐利的一格：短时效对不上就说明观测侧读错了，长时效才能对上就说明是别的。
_SHOW_LEADS = (6.0, 24.0, 72.0, 120.0, 168.0, 240.0, 360.0)
_STEP_RE = re.compile(r"^(\d{3})\.nc$")


def _load_eval_api():
    if EVAL_ROOT not in sys.path:
        sys.path.insert(0, EVAL_ROOT)
    from xmetai_evaluation.core.contracts import DataRequest          # noqa: PLC0415
    from xmetai_evaluation.io.era5_zarr_reader import (               # noqa: PLC0415
        Era5ZarrCatalog, Era5ZarrReader)
    from xmetai_evaluation.metrics.specialized import (               # noqa: PLC0415
        DEFAULT_SPHERICAL_BANDS, spherical_band_power)
    return DEFAULT_SPHERICAL_BANDS, spherical_band_power, DataRequest, \
        Era5ZarrCatalog, Era5ZarrReader


def _find_var(ds, want):
    """大小写不敏感地找一个 2D 要素变量（推理侧落盘是 Z500 大写）。"""
    for name in ds.variables:
        if name.lower() == want.lower() and ds.variables[name].ndim == 2:
            return ds.variables[name]
    raise KeyError("文件里没有 2D 变量 %r（有：%s）" % (want, list(ds.variables)))


def _resolve_member_root(arm_dir: Path):
    """找到装 ``member_XXX`` 的那一层。

    正常是 ``<臂>/<YYYYMMDD>/member_XXX``；但老仓的排布也出现过
    ``<臂>/<YYYYMMDD>/<YYYYMMDD>/member_XXX`` 这种多套一层的。两种都认，
    并且**把选中的路径打出来**——选错了就是另一条臂的数，必须让人看得见。
    """
    cands = [arm_dir]
    cands += sorted(d for d in arm_dir.iterdir() if d.is_dir())
    cands += sorted(d for d in arm_dir.iterdir() if d.is_dir()
                    for d in d.iterdir() if d.is_dir())
    for c in cands:
        try:
            if any(p.is_dir() and p.name.startswith("member_")
                   for p in c.iterdir()):
                return c
        except OSError:
            continue
    raise FileNotFoundError("%s 底下（含两层）没有 member_* 目录" % arm_dir)


def _member_dirs(root: Path):
    """member_XXX 目录，按名字里的数字升序 —— 与评测 reader 的排序一致。"""
    out = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("member_"):
            m = re.search(r"(\d+)", p.name)
            out.append((int(m.group(1)) if m else 0, p))
    if not out:
        raise FileNotFoundError("%s 下没有 member_* 目录" % root)
    return [p for _, p in sorted(out)]


def _step_files(member_dir: Path):
    """(编号, 路径) 按编号升序。跳过隐藏的 ``.NNN.nc.tmp-*`` 半成品。"""
    out = []
    for p in member_dir.iterdir():
        m = _STEP_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    if not out:
        raise FileNotFoundError("%s 下没有 NNN.nc 步文件" % member_dir)
    return sorted(out)


def _scan_ensemble(root: Path, varname: str, want_leads, bands, sph_power):
    """扫一条臂的一个起报 → 逐时效的 (平均场, 离散度平方, 离散度带功率)。

    循环是**时效在外、成员在内**的：算 ``mean_m B(f_m − mean_m f_m)`` 必须同时
    握住一个时效的全部成员，所以不能像标量版那样"一个成员读完整条时间轴"。
    峰值内存 = 51 成员 × 一格 = 53MB（float32），仍然很小。

    **时效按文件编号推**（``lead = 编号 × 6h``），不按位置 —— 中间缺一个文件时，
    按位置推会静默错位，按编号推不会。成员步文件编号不一致直接报错（成员不齐，
    不能算集合量）。

    Returns: dict(mean, var, p_spec, lat, lon, leads, nmem)
    """
    import netCDF4                                                # noqa: PLC0415

    members = _member_dirs(root)
    steps0 = _step_files(members[0])
    ref_nums = [num for num, _ in steps0]
    all_leads = np.array([num * HOUR_INTERVAL for num in ref_nums], dtype="f8")
    if want_leads is None:
        keep = list(range(len(ref_nums)))
    else:
        wanted = {float(x) for x in want_leads}
        keep = [i for i, L in enumerate(all_leads) if L in wanted]
        if not keep:
            raise ValueError("LEADS=%s 一个都对不上（这条臂有 %s..%s）"
                             % (want_leads, all_leads[:3], all_leads[-3:]))
    leads = all_leads[keep]

    # 路径先列一遍存起来：放循环里会把目录列 3060 次。
    files = []
    for mdir in members:
        st = _step_files(mdir)
        if [num for num, _ in st] != ref_nums:
            raise ValueError("%s 的步文件编号与 %s 不一致（成员不齐，不能算集合量）"
                             % (mdir.name, members[0].name))
        files.append(st)

    nmem = len(members)
    nband = len(bands)
    p_spec = np.full((len(keep), nband), np.nan, dtype="f8")

    # 网格先探一次（多开 1 个文件，换掉"每档都要判断要不要重分配"那套）。
    # ⚠ 三个缓冲必须在**循环外**一次分配好：曾经写成"首档顺手分配"，结果第 2 档
    #   起 ``lat`` 已非 None，缓冲再也不会重建，一路拿 (nmem,0,0) 占位符算到底。
    with netCDF4.Dataset(files[0][keep[0]][1]) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype="f8")
        lon = np.asarray(ds.variables["lon"][:], dtype="f8")
    stack = np.empty((nmem, lat.size, lon.size), dtype="f4")
    mean = np.empty((len(keep), lat.size, lon.size), dtype="f4")
    var = np.empty((len(keep), lat.size, lon.size), dtype="f8")

    for k, i in enumerate(keep):
        for m in range(nmem):
            with netCDF4.Dataset(files[m][i][1]) as ds:
                stack[m] = np.asarray(_find_var(ds, varname)[:], dtype="f4")

        fbar = stack.mean(axis=0, dtype="f8")
        dev = stack - fbar                                         # (M, lat, lon)
        # ddof=0 的成员间方差，恒等于评测侧 spread 的平方；einsum 免得再开 106MB 临时量
        var[k] = np.einsum("mij,mij->ij", dev, dev, dtype="f8") / float(nmem)
        mean[k] = fbar.astype("f4")
        # ★ 关键：对**偏差场**做带功率，再对成员平均 —— 这才是逐尺度离散度
        p_spec[k] = sph_power(dev, lat=lat, bands=bands).mean(axis=0)
        if k % 10 == 9 or k == len(keep) - 1:
            print("      已扫 %d/%d 档" % (k + 1, len(keep)), flush=True)

    return {"mean": mean, "var": var, "p_spec": p_spec,
            "lat": lat, "lon": lon, "leads": leads, "nmem": nmem}


def _read_obs(DataRequest, Era5ZarrCatalog, Era5ZarrReader, init, leads, varname):
    """按 (起报, 时效) 读观测 —— 传进去的是**有效时刻** ``init + lead``。

    返回 ``(values, lat, lon)``。``lon`` 是**必须带出来**的：lat 有 guard（见
    ``main`` 里那道 raise），lon 一直没有，而经度错位对联里的每一道闸都是隐形的
    —— 见 ``_lon_probe``。取不到时给 None，不抛（有的 store 只存 lat）。
    """
    stores = {"pl": ERA5_PL_STORE}
    valid = [init + timedelta(hours=float(L)) for L in leads]
    req = DataRequest(source_id="era5_zarr", variables=[varname], init_times=valid)
    cat = Era5ZarrCatalog(stores)
    rd = Era5ZarrReader(source_id="era5_zarr", stores=stores)
    da = rd.read(req, cat.discover(req)).payload[varname]

    # ⚠ datetime64 的单位不一致时，字典键会**静默不相等**（'us' vs 'ns'）。
    #   统一抬到 ns 再对齐，别拿裸 np.datetime64 当键。
    have = np.asarray(da.coords["valid_time"].values).astype("datetime64[ns]")
    want = np.array([np.datetime64(t, "ns") for t in valid])
    if np.array_equal(have, want):
        order = np.arange(want.size)
    else:
        idx = {t: i for i, t in enumerate(have)}
        missing = [str(t) for t in want if t not in idx]
        if missing:
            raise KeyError("观测里没有这些时刻：%s" % missing[:5])
        order = np.array([idx[t] for t in want])
    lon = None
    for key in ("lon", "longitude"):
        if key in da.coords:
            lon = np.asarray(da.coords[key].values, dtype="f8")
            break
    return np.asarray(da.values, dtype="f8")[order], \
        np.asarray(da.coords["lat"].values, dtype="f8"), lon


def _align_lon_axis(vals, obs_lon, fcast_lon):
    """把观测的经度轴滚到预报经度轴上 —— **按坐标算，别硬编码 720**。

    实测：预报 lon ``0..359.75``、era5 zarr store lon ``-180..179.75``，原点差半圈。
    这里恰好是半圈，所以 ±720 等价（1440 的一半）；换个 store 就不一定了，按坐标
    算才通用。理由与后果见 module docstring，以及 ``_lon_probe``。

    返回 ``(滚好的值, 实际滚了几格)``；坐标缺失/不可比时原样返回、滚 0 格，
    由闸0 去喊 —— 这里**不抛**，因为"没对齐"是要打印出来给人看的结论。
    """
    n = fcast_lon.size
    if obs_lon is None or obs_lon.size != n or n < 2:
        return vals, 0
    sf = float(fcast_lon[1] - fcast_lon[0])
    so = float(obs_lon[1] - obs_lon[0])
    if sf == 0.0 or np.sign(sf) != np.sign(so):
        return vals, 0                    # 轴反向或退化，交给闸0 报
    # 要 aligned[i] 落在 fcast_lon[i] 上：obs_lon[j] = fcast_lon[i] ⇒ j = i + k
    k = int(round((fcast_lon[0] - obs_lon[0]) / sf))
    if k % n == 0:
        return vals, 0
    return np.roll(vals, -k, axis=-1), (-k) % n


def _lon_probe(f_field, o_field, lat, nlon):
    """一次 FFT 把**经度对齐**摊开：最优滚移 / 零滚移处的相关 / 滚移前后的 rmse。

    为什么必须单独查
    ----------------
    ``_wavg2`` 的面积加权均值和 ``spherical_band_power`` 对**经度滚移都是恒等**的：
    滚移不改任何一条纬圈上的功率谱，也不改 cos-lat 加权均值。所以本脚本其余的每一
    道闸——闸1 对归档 spread、闸2 闭合、绝对值均值表——**全都看不见经度错位**。
    lat 至少还有一道 raise，lon 从头到尾没被看过一眼。这就是那个盲区。

    做法
    ----
    把「ΦΔ 扫一遍」写成循环相关，一次 FFT 出全部候选::

        irfft(rfft(f)·conj(rfft(o)))[d] = Σ_x f[x]·o[x−d]  =  ⟨f, roll(o, d)⟩

    所以取 argmax 就是让 ``‖f − roll(o, Δ)‖`` 最小的 Δ。DC 不影响 argmax，
    直接拿原始场（≈5.6e4 m²/s²）算也在 float64 的安全范围内。

    返回 ``(d_best, corr0, rmse0, rmse_best)``；``corr0`` 是面加权相关（已去均值）。
    """
    w = np.cos(np.deg2rad(lat))[:, None]
    wn = w.sum() * nlon
    cc = np.fft.irfft(np.fft.rfft(f_field, axis=1)
                      * np.conj(np.fft.rfft(o_field, axis=1)),
                      n=nlon, axis=1)
    d = int(np.argmax((cc * w).sum(axis=0)))

    fa = f_field - (f_field * w).sum() / wn
    oa = o_field - (o_field * w).sum() / wn
    corr0 = float((fa * oa * w).sum()
                  / np.sqrt((fa * fa * w).sum() * (oa * oa * w).sum()))
    rmse0 = float(np.sqrt(((f_field - o_field) ** 2 * w).sum() / wn))
    rmse_d = float(np.sqrt(((f_field - np.roll(o_field, d, axis=1)) ** 2 * w).sum()
                           / wn))
    return d, corr0, rmse0, rmse_d


def _archived_scalar_spread(arm, init_s, leads):
    """归档里的标量 spread，用来核对读法没读错时效。

    返回 ``(got, path)``：``got`` 是 ``{(YYYYMMDD, lead_h): value}``，找不到时
    为 None，同时把**试过的路径**带回去打印 —— 上一版这里静默跳过，等于这道闸
    根本没跑，白丢一轮。
    """
    path = Path(SCORES_ROOT) / arm / "scores.csv"
    if not path.is_file():
        return None, path
    want = {(init_s[:8], float(L)) for L in leads}
    got = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("variable") != VAR or row.get("metric") != "spread"
                    or row.get("product_kind") != "ensemble"
                    or row.get("status") != "success"
                    or row.get("region", "") != ""):       # ← region 必须过滤
                continue
            try:
                key = (row["init_time"][:8], float(row["lead_h"]))
            except (TypeError, ValueError):
                continue
            if key in want:
                got[key] = float(row["value"])
    return (got or None), path


def _wavg2(field, lat, lon_size):
    """面积加权（cos lat）的均方 → 与评测 ``_spread_by_lead`` 同口径。"""
    w = np.cos(np.deg2rad(lat))[:, None]
    return (field ** 2 * w).sum(axis=(1, 2)) / (w.sum() * lon_size)


def main() -> int:
    (bands, sph_power, DataRequest, Era5ZarrCatalog,
     Era5ZarrReader) = _load_eval_api()
    band_tags = ["%d_%d" % (lo, hi) for lo, hi in bands]
    nband = len(band_tags)

    print("=" * 104)
    print("P4 逐尺度 SER    球谐带 = %s" % (band_tags,))
    print("P_s(k) = mean_m B(f_m − f̄)(k)    P_e(k) = B(f̄ − o)(k)    R(k) = sqrt(P_s/P_e)")
    print("闭合检查：Σ_k P_s / spread² 应是**档间不变**的常数（理论 4π=%.4f，"
          "实测偏 ~5%%）——判据是档间极差，不是绝对值" % CLOSURE_REF)
    print("=" * 104)

    rows = []
    for arm in ARMS:
        arm_dir = Path(FORECAST_ROOT) / arm
        if not arm_dir.is_dir():
            print("\n[%s] !!!! 目录不存在：%s（跳过）" % (arm, arm_dir), flush=True)
            continue
        for init_s in INITS:
            try:
                root = _resolve_member_root(arm_dir / init_s[:8])
            except (FileNotFoundError, OSError) as exc:
                print("\n[%s %s] !!!! %s（跳过）" % (arm, init_s, exc), flush=True)
                continue
            init = datetime.strptime(init_s, "%Y%m%d%H")
            print("\n[%s %s] 成员根 = %s，开扫 …" % (arm, init_s, root), flush=True)
            got = _scan_ensemble(root, VAR, LEADS, bands, sph_power)
            lat, lon, leads = got["lat"], got["lon"], got["leads"]
            print("   成员 %d，时效 %d 档 (%g..%gh)，网格 %d×%d"
                  % (got["nmem"], leads.size, leads[0], leads[-1],
                     lat.size, lon.size), flush=True)

            spread = np.sqrt(got["var"])                            # (T, lat, lon)
            obs, obs_lat, obs_lon = _read_obs(DataRequest, Era5ZarrCatalog,
                                              Era5ZarrReader, init, leads, VAR)
            if obs.shape[1:] != spread.shape[1:]:
                raise ValueError("观测网格 %s != 预报网格 %s"
                                 % (obs.shape[1:], spread.shape[1:]))
            if obs_lat.size == lat.size and np.abs(obs_lat - lat).max() > 1e-6:
                raise ValueError("观测 lat 与预报 lat 不一致（差 %.4g）—— 先对齐再算"
                                 % np.abs(obs_lat - lat).max())
            # ── 经度对齐：**必须**，理由见 module docstring ──────────────
            # 预报 lon 0..359.75 / era5 store lon −180..179.75，原点差半圈。
            # 不做这一步，下面每一条 rmse 都是拿 0°E 去对 180°W，虚高一个量级。
            obs, lon_shift = _align_lon_axis(obs, obs_lon, lon)

            mean = np.asarray(got["mean"], dtype="f8")
            err = mean - obs                                        # (T, lat, lon)

            # ── 闸0：对齐**回读验证**（不是探测）────────────────────────
            # 滚完之后，残余滚移必须是 0、corr 必须接近 1。为什么非有不可：滚移不改
            # 面积加权均值、也不改球谐带功率，所以上面下面每一道闸对它都是瞎子。
            o_lon_s = ("（store 里没有 lon 坐标）" if obs_lon is None else
                       "%.4g..%.4g (n=%d)" % (obs_lon[0], obs_lon[-1],
                                              obs_lon.size))
            print("   闸0 经度  预报 %.4g..%.4g (n=%d)   观测 %s   已滚 %+d 格"
                  % (lon[0], lon[-1], lon.size, o_lon_s, lon_shift))
            if obs_lon is None or obs_lon.size != lon.size:
                print("        !!!! 经度轴不可比，没法对齐 —— 下面所有 rmse 都要打问号")
            else:
                for i in sorted({0, 1, min(3, leads.size - 1)}):
                    d, corr0, r0, _ = _lon_probe(mean[i], obs[i], lat, lon.size)
                    ok = (d == 0 and corr0 > 0.99)
                    print("        lead %-5g 残余滚移 %+5d 格   rmse %.3f   "
                          "corr(集合均值, 观测) %+.4f   %s"
                          % (leads[i], d, r0, corr0,
                             "OK" if ok else "!!!! 还是没对齐"))

            # ⚠ 这里传的是 **std**，不是 var：``_wavg2`` 自己会平方（面积加权均方）。
            #   传 var 进去等于算四阶矩，标量 spread 会大出一个量级还看不出来。
            scalar_spread2 = _wavg2(spread, lat, lon.size)
            scalar_spread = np.sqrt(scalar_spread2)
            scalar_rmse2 = _wavg2(err, lat, lon.size)
            scalar_rmse = np.sqrt(scalar_rmse2)
            ser = scalar_spread / scalar_rmse                       # 标量 SER，逐时效

            # ── 闸1：标量 spread 与归档对得上，才说明时效映射/口径没错位 ──
            arch, arch_path = _archived_scalar_spread(arm, init_s, leads)
            if arch is None:
                print("   ⚠ 归档核对跳过：没有 %s（要启用就把 SCORES_ROOT 指对）"
                      % arch_path)
            else:
                diffs = [100.0 * (scalar_spread[i] - arch[(init_s[:8], float(L))])
                         / arch[(init_s[:8], float(L))]
                         for i, L in enumerate(leads)
                         if (init_s[:8], float(L)) in arch]
                if not diffs:
                    print("   ⚠ 归档核对：%s 在，但这些起报/时效一档都没匹配上"
                          % arch_path)
                else:
                    worst = max(abs(d) for d in diffs)
                    print("   闸1 标量 spread vs 归档：%d 档匹配，最大偏离 %.2f%%%s"
                          % (len(diffs), worst,
                             "" if worst < 1.0 else "   !!!! >1%，时效映射或口径有问题"))

            # ── 逐尺度 ──
            p_e = sph_power(err, lat=lat, bands=bands)              # (T, nband)
            p_s = got["p_spec"]
            with np.errstate(divide="ignore", invalid="ignore"):
                R = np.sqrt(p_s / p_e)

            # ── 闸2：闭合检查（本版构造对不对，全看这个）──
            c_s = p_s.sum(axis=1) / (CLOSURE_REF * scalar_spread2)   # (T,) 逐档
            c_e = p_e.sum(axis=1) / (CLOSURE_REF * scalar_rmse2)
            cs_mid = float(np.nanmean(c_s))
            cs_drift = float(np.nanmax(c_s) - np.nanmin(c_s))
            ce_mid = float(np.nanmean(c_e))
            ce_drift = float(np.nanmax(c_e) - np.nanmin(c_e))
            why = []
            if not (CLOSURE_BAND[0] <= cs_mid <= CLOSURE_BAND[1]):
                why.append("绝对量 %.4f 掉出 %s —— 构造很可能错了"
                           % (cs_mid, CLOSURE_BAND))
            if cs_drift > CLOSURE_DRIFT:
                why.append("档间极差 %.1f%% > %.0f%% —— 不是同一个常数，构造有问题"
                           % (100 * cs_drift, 100 * CLOSURE_DRIFT))
            print("   闸2 闭合 Σ_k P_s/(4π·spread²)：均值 %.4f，档间极差 %.2f%%   %s"
                  % (cs_mid, 100 * cs_drift,
                     "✓ 常数档间不变（4π 是理论值，实测 ~0.95 属正常）"
                     if not why else "!!!! " + "；".join(why)))
            print("       误差侧 Σ_k P_e/(4π·rmse²)：均值 %.4f，档间极差 %.2f%%"
                  "   ← 相对离散度侧的缺口 ≈ 全球均值偏差占 rmse² 的比例"
                  "（球谐带不含 l=0）" % (ce_mid, 100 * ce_drift))

            # ── 读数 ──
            # 绝对值必须打出来：SER 是个比值，分子分母任一读错都看不出来。
            # 归档侧 aB 24h 的口径是 spread 51.25 / rmse 62.15（见 scores.csv），
            # 对不上就在这张表上一眼定位——是 spread 小了还是 rmse 大了。
            wa = np.cos(np.deg2rad(lat))
            gmean_f = (mean.mean(axis=2) * wa).sum(axis=1) / wa.sum()
            gmean_o = (obs.mean(axis=2) * wa).sum(axis=1) / wa.sum()
            print("   %-8s %10s %10s %8s %13s %13s   %s"
                  % ("lead_h", "spread", "rmse", "标量SER", "mean(集合均值)",
                     "mean(观测)", "偏差"))
            for i, L in enumerate(leads):
                if float(L) not in _SHOW_LEADS:
                    continue
                print("   %-8g %10.3f %10.3f %8.3f %13.1f %13.1f   %+.1f"
                      % (L, scalar_spread[i], scalar_rmse[i], ser[i],
                         gmean_f[i], gmean_o[i], gmean_f[i] - gmean_o[i]))
            print("   %-8s %10s   %s" % ("lead_h", "标量SER",
                                         "".join("%14s" % t for t in band_tags)))
            for i, L in enumerate(leads):
                if float(L) not in _SHOW_LEADS:
                    continue
                print("   %-8g %10.3f   %s" % (L, ser[i],
                                               "".join("%14.3f" % R[i, j]
                                                       for j in range(nband))))
            for lab, sel in (("短 6-120h", leads <= 120.0),
                             ("长 126-360h", leads > 120.0)):
                if not sel.any():
                    continue
                fs = p_s[sel].sum(axis=0) / p_s[sel].sum()
                fe = p_e[sel].sum(axis=0) / p_e[sel].sum()
                print("   %s  带占比：离散度 %s" % (lab, "  ".join(
                    "%s=%.1f%%" % (t, 100 * f) for t, f in zip(band_tags, fs))))
                print("   %s  带占比：误差   %s" % (" " * len(lab), "  ".join(
                    "%s=%.1f%%" % (t, 100 * f) for t, f in zip(band_tags, fe))))
                print("   %s  带平均 R：%s" % (lab, "  ".join(
                    "%s=%.3f" % (t, float(np.nanmean(R[sel, j])))
                    for j, t in enumerate(band_tags))))

            for i in range(leads.size):
                for j, tag in enumerate(band_tags):
                    rows.append({
                        "arm": arm, "init": init_s, "lead_h": float(leads[i]),
                        "band": tag,
                        "spread_power": float(p_s[i, j]),
                        "err_power": float(p_e[i, j]),
                        "per_scale_ser": float(R[i, j]),
                        "spread_frac": float(p_s[i, j] / p_s[i].sum()),
                        "err_frac": float(p_e[i, j] / p_e[i].sum()),
                        "scalar_ser": float(ser[i]),
                        "scalar_spread": float(scalar_spread[i]),
                        "scalar_rmse": float(scalar_rmse[i]),
                        "closure_spread": float(c_s[i]),
                        "closure_err": float(c_e[i]),
                    })
            print("   ← %s %s 完成" % (arm, init_s), flush=True)

    if not rows:
        print("\n没有产出任何行 —— 先检查 FORECAST_ROOT 下的臂名与起报目录。")
        return 1
    if OUT_CSV:
        out = Path(OUT_CSV)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print("\n写入 %s（%d 行）" % (out, len(rows)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
