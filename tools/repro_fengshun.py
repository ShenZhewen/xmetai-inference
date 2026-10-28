# -*- coding: utf-8 -*-
"""复现第三方 fengshun2.0 推理，并标定 step/hour/doy 的语义。

背景
----
第三方在 `infer1017/20110105/member/10` 下产出了 `10.nc`（t2m/t2m_min/t2m_max/tp，
单成员）和 `10mean.nc`（30 通道，11 成员均值）。两者的 `time` 都等于**起报日**、
`lead_time` 都等于 10，说明他们喂给图的 `step` 原值就是 10。

本脚本要回答三件事：
  1. 用同一个起报日 2011-01-05、同一个 step=10 复跑，能不能对上 `10.nc`；
  2. `step` 是「天」还是「小时」，以及有效时刻是 init+step 还是 init+(step-1)；
  3. 输入窗两帧的顺序是 [t-1, t] 还是 [t, t-1]。

只读数据盘，只往 --out 目录写一个小 JSON。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# 只读盘约定：除 tp 外 zscore（phys = d*std + mean），tp 存 log1p(mm)
TP = "tp"


# ---------------------------------------------------------------- 环境 / session
def preload_ort_lib():
    """先 RTLD_GLOBAL 预加载 libonnxruntime，插件库注册才找得到符号。"""
    import ctypes
    import glob

    try:
        import onnxruntime as _ort

        capi = os.path.join(os.path.dirname(_ort.__file__), "capi")
        for pattern in ("libonnxruntime.so.1", "libonnxruntime.so.1.*"):
            matches = sorted(glob.glob(os.path.join(capi, pattern)))
            if matches:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                return matches[0]
    except Exception:  # noqa: BLE001
        pass
    return None


def make_session(model: str, plugin: str):
    import torch  # noqa: F401  必须早于 onnxruntime：torch 才会挂上 nvidia 的 .so 路径
    import onnxruntime as ort

    print("onnxruntime      :", ort.__version__)
    print("providers        :", ort.get_available_providers())
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_mem_pattern = False
    print("预加载 libonnxruntime:", preload_ort_lib() or "<跳过>")
    opts.register_custom_ops_library(plugin)
    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        model, sess_options=opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("session          : ok  %.2fs  实际 provider=%s" % (
        time.perf_counter() - t0, sess.get_providers()))
    print("输入: " + ", ".join(f"{i.name}{i.shape}" for i in sess.get_inputs()))
    print("输出: " + ", ".join(f"{o.name}{o.shape}" for o in sess.get_outputs()))
    return sess


# ---------------------------------------------------------------- 数据盘
class Store:
    def __init__(self, path: str):
        import zarr

        self.g = zarr.open_group(path, mode="r")
        self.ch = [str(x) for x in np.asarray(self.g["channel"][:]).ravel()]
        t = np.asarray(self.g["time"][:])
        if t.dtype.kind != "M":
            raise TypeError(f"盘里 time 不是 datetime64（{t.dtype}），先确认单位再跑")
        self.time = t.astype("datetime64[s]")
        self.mean = np.asarray(self.g["mean"][:]).ravel().astype(np.float32)
        self.std = np.asarray(self.g["std"][:]).ravel().astype(np.float32)
        self.tp_i = self.ch.index(TP)

    def idx(self, names):
        return [self.ch.index(n) for n in names]

    def frame(self, day: str) -> np.ndarray:
        """读某天 00Z 的 87 通道物理场（tp 已 expm1 回 mm）。"""
        t = np.datetime64(day + "T00:00:00")
        i = int(np.searchsorted(self.time, t))
        if i >= self.time.size or str(self.time[i])[:10] != day:
            raise KeyError(f"盘里没有 {day}（最近的是 {self.time[min(i, self.time.size - 1)]}）")
        d = np.asarray(self.g["data"][i, :, :, :]).astype(np.float32)
        phys = d * self.std[:, None, None] + self.mean[:, None, None]
        phys[self.tp_i] = np.expm1(d[self.tp_i])
        return phys


def shift(day: str, n: int) -> str:
    y, m, d = map(int, day.split("-"))
    return (date(y, m, d) + timedelta(days=n)).isoformat()


def day_of_year(day: str) -> int:
    y, m, d = map(int, day.split("-"))
    return date(y, m, d).timetuple().tm_yday


# ---------------------------------------------------------------- 指标
def area_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat.astype(np.float64))).clip(0.0, None)
    return np.broadcast_to(w[:, None], (lat.size, 1440))


def wcorr(a, b, w):
    am = (a * w).sum() / w.sum()
    bm = (b * w).sum() / w.sum()
    num = (w * (a - am) * (b - bm)).sum()
    den = np.sqrt((w * (a - am) ** 2).sum() * (w * (b - bm) ** 2).sum())
    return float(num / den) if den > 0 else float("nan")


def wrmse(a, b, w):
    return float(np.sqrt((w * (a - b) ** 2).sum() / w.sum()))


def load_ref(path: str):
    """参考场存成 .npz（channel, data[ch,lat,lon]），省掉 Job 里的 xarray 依赖。"""
    d = np.load(path, allow_pickle=False)
    return [str(x) for x in d["channel"]], np.asarray(d["data"], dtype=np.float32)


# ---------------------------------------------------------------- 推理
def infer(sess, x2: np.ndarray, step: int, hour: float, doy: float,
          n: int, keep):
    """跑 n 个成员（图里 RandomNormalLike ⇒ 一次前向 = 一个成员），返回 (均值, 首成员)。"""
    names = [i.name for i in sess.get_inputs()]
    x = np.ascontiguousarray(x2[None], dtype=np.float32)
    feeds_common = {
        "step": np.array([step], np.float32),
        "hour": np.array([hour], np.float32),
        "doy": np.array([doy], np.float32),
    }
    acc = None
    m0 = None
    for k in range(n):
        feeds = {"input": x}
        for key, val in feeds_common.items():
            if key in names:
                feeds[key] = val
        y = np.asarray(sess.run(None, feeds)[0], dtype=np.float32)[0, 0]  # (87,721,1440)
        ys = y[keep]
        if k == 0:
            acc = np.zeros_like(ys, dtype=np.float64)
            m0 = ys.copy()
        acc += ys
    return (acc / n).astype(np.float32), m0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/tmp/douzsh/models/fengshun2.0_260901.onnx")
    ap.add_argument("--plugin", default="/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so")
    ap.add_argument("--store", default="/workspace/data/zarr_refresh_v3/era5.1959_2025.c87.p25.h24.v1")
    ap.add_argument("--ref-dir", default="/workspace/szw_repro")
    ap.add_argument("--json", default="/workspace/szw_repro/repro_fengshun.json")
    args = ap.parse_args()

    t_start = time.perf_counter()
    st = Store(args.store)
    print("盘通道 %d 个；时间 %s → %s" % (
        len(st.ch), str(st.time[0])[:10], str(st.time[-1])[:10]))
    lat = np.asarray(st.g["lat"][:]).ravel()
    W = area_weights(lat)
    print("lat %.2f→%.2f" % (lat[0], lat[-1]))

    sess = make_session(args.model, args.plugin)
    report: dict = {}

    # ================================================================ A: 复现 10.nc
    print("\n" + "=" * 74)
    print("A. 复现 10.nc：起报 2011-01-05，step=10")
    print("=" * 74)
    init = "2011-01-05"
    f_prev, f_init = st.frame(shift(init, -1)), st.frame(init)
    keepA = st.idx(["z500", "t2m", "t2m_min", "t2m_max", "tp"])
    refA = Path(args.ref_dir) / "10.npz"
    refA_ch, refA_v = load_ref(str(refA)) if refA.is_file() else ([], None)
    print("参考 %s 通道 %s" % (refA.name, refA_ch))

    runsA = {}
    t0 = time.perf_counter()
    meanA, mem0A = infer(sess, np.stack([f_prev, f_init]), 10, 0.0,
                         float(day_of_year(init)), 11, keepA)
    print("  11 成员用时 %.1fs" % (time.perf_counter() - t0))
    # 0 = RandomNormalLike 没换种子，11 个成员其实是同一个场
    print("  成员间差异 max|m0-ens_mean| = %.4g" % float(np.abs(mem0A[0] - meanA[0]).max()))
    runsA["chrono"] = (meanA, mem0A)

    t0 = time.perf_counter()
    meanS, mem0S = infer(sess, np.stack([f_init, f_prev]), 10, 0.0,
                         float(day_of_year(init)), 3, keepA)
    print("  帧序反置 3 成员用时 %.1fs" % (time.perf_counter() - t0))
    runsA["swapped"] = (meanS, mem0S)

    # step=0：看输出帧相对输入窗末是 t 还是 t+1
    meanZ, _ = infer(sess, np.stack([f_prev, f_init]), 0, 0.0,
                     float(day_of_year(init)), 2, keepA)

    def block(tag, mean, mem0):
        out = {}
        print("  -- %s --  全球均值: %s" % (tag, ", ".join(
            "%s=%.4g" % (n, float(mean[i].mean())) for i, n in enumerate(
                ["z500", "t2m", "t2m_min", "t2m_max", "tp"]))))
        if refA_v is None:
            return out
        for name in refA_ch:
            if name not in ["z500", "t2m", "t2m_min", "t2m_max", "tp"]:
                continue
            i = ["z500", "t2m", "t2m_min", "t2m_max", "tp"].index(name)
            j = refA_ch.index(name)
            r = refA_v[j]
            for label, fld in (("ens_mean", mean[i]), ("member0", mem0[i])):
                k = "%s/%s" % (name, label)
                out[k] = {
                    "corr": wcorr(fld, r, W),
                    "rmse": wrmse(fld, r, W),
                    "bias": float(((fld - r) * W).sum() / W.sum()),
                    "mean": float(fld.mean()),
                    "ref_mean": float(r.mean()),
                }
                print("     %-9s %-9s corr=%+.4f rmse=%9.4f bias=%+9.4f (mean %.4g vs %.4g)"
                      % (name, label, out[k]["corr"], out[k]["rmse"],
                         out[k]["bias"], out[k]["mean"], out[k]["ref_mean"]))
        return out

    report["A"] = {k: block(k, *v) for k, v in runsA.items()}

    print("  -- step=0 输出 vs 盘里真值（判输出帧是 t 还是 t+1）--")
    zi = keepA.index(st.ch.index("z500"))
    for day in (init, shift(init, 1)):
        truth = st.frame(day)[st.ch.index("z500")]
        print("     step=0 vs %s : corr=%+.4f rmse=%9.1f" % (
            day, wcorr(meanZ[zi], truth, W), wrmse(meanZ[zi], truth, W)))
    report["A_step0"] = {
        d: {"corr": wcorr(meanZ[zi], st.frame(d)[st.ch.index("z500")], W),
            "rmse": wrmse(meanZ[zi], st.frame(d)[st.ch.index("z500")], W)}
        for d in (init, shift(init, 1))}

    # ================================================================ B: step 语义
    print("\n" + "=" * 74)
    print("B. step 语义：起报 2024-12-24，step∈{0,1,2,3,10,24} 对真值")
    print("=" * 74)
    initB = "2024-12-24"
    g_prev, g_init = st.frame(shift(initB, -1)), st.frame(initB)
    keepB = st.idx(["z500", "t2m"])
    zi_b = keepB.index(st.ch.index("z500"))
    cand = [initB] + [shift(initB, k) for k in (1, 2, 3, 9, 10)]
    truth = {d: st.frame(d)[st.ch.index("z500")] for d in cand}
    print("  候选真值: " + ", ".join(cand))
    table = {}
    for step in (0, 1, 2, 3, 10, 24):
        t0 = time.perf_counter()
        m, _ = infer(sess, np.stack([g_prev, g_init]), step, 0.0,
                     float(day_of_year(initB)), 2, keepB)
        row = {d: [round(wcorr(m[zi_b], truth[d], W), 4),
                   round(wrmse(m[zi_b], truth[d], W), 1)] for d in cand}
        table[step] = row
        best = max(row, key=lambda d: row[d][0])
        print("  step=%-3d 最佳 ACC: %s (%.4f)   [%.1fs]" % (
            step, best, row[best][0], time.perf_counter() - t0))
        print("           " + "  ".join("%s:%.3f" % (d[5:], row[d][0]) for d in cand))
    report["B"] = {str(k): v for k, v in table.items()}

    # ================================================================ C: 复现 10mean.nc
    print("\n" + "=" * 74)
    print("C. 复现 10mean.nc：起报 2024-12-24，step=10，11 成员")
    print("=" * 74)
    refC = Path(args.ref_dir) / "10mean.npz"
    refC_ch, refC_v = load_ref(str(refC)) if refC.is_file() else ([], None)
    print("参考 %s 通道 %d 个" % (refC.name, len(refC_ch)))
    if refC_v is not None:
        keepC = st.idx(refC_ch)
        t0 = time.perf_counter()
        meanC, mem0C = infer(sess, np.stack([g_prev, g_init]), 10, 0.0,
                             float(day_of_year(initB)), 11, keepC)
        print("  11 成员用时 %.1fs" % (time.perf_counter() - t0))
        report["C"] = {}
        for i, name in enumerate(refC_ch):
            r = refC_v[i]
            e = {
                "corr_mean": wcorr(meanC[i], r, W),
                "rmse_mean": wrmse(meanC[i], r, W),
                "corr_mem0": wcorr(mem0C[i], r, W),
                "rmse_mem0": wrmse(mem0C[i], r, W),
                "mean": float(meanC[i].mean()),
                "ref_mean": float(r.mean()),
                "ref_min": float(r.min()),
                "ref_max": float(r.max()),
            }
            report["C"][name] = e
            print("  %-8s corr(ens)=%+.4f corr(m0)=%+.4f rmse(ens)=%9.4f  mean %.5g vs %.5g"
                  % (name, e["corr_mean"], e["corr_mem0"], e["rmse_mean"],
                     e["mean"], e["ref_mean"]))

    # ================================================================ D: doy 口径
    print("\n" + "=" * 74)
    print("D. doy 取起报日还是有效日：起报 2024-12-24 step=10，各 5 成员")
    print("=" * 74)
    keepD = st.idx(["z500", "t2m"])
    zD = keepD.index(st.ch.index("z500"))
    doy_init, doy_valid = float(day_of_year(initB)), float(day_of_year(shift(initB, 10)))
    mDi, _ = infer(sess, np.stack([g_prev, g_init]), 10, 0.0, doy_init, 5, keepD)
    mDv, _ = infer(sess, np.stack([g_prev, g_init]), 10, 0.0, doy_valid, 5, keepD)
    mDc, _ = infer(sess, np.stack([g_prev, g_init]), 10, 0.0, doy_init, 5, keepD)
    d_iv = float(np.abs(mDi[zD] - mDv[zD]).mean())
    d_cc = float(np.abs(mDi[zD] - mDc[zD]).mean())  # 同配置重复 = 抽样噪声地板
    print("  doy=%g vs doy=%g : |Δz500|=%.2f m²/s²" % (doy_init, doy_valid, d_iv))
    print("  噪声地板（同配置重跑）      : |Δz500|=%.2f m²/s²" % d_cc)
    report["D"] = {"doy_init": doy_init, "doy_valid": doy_valid,
                   "delta_doy": d_iv, "noise_floor": d_cc}

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\nJSON 已写入 %s   总用时 %.1fs" % (args.json, time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
