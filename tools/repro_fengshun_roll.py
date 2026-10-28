# -*- coding: utf-8 -*-
"""fengshun2.0 复现 v2：10 步自回归 rollout + 全 87 通道单步诊断。

v1（repro_fengshun.py）教了两件事
---------------------------------
1. **一次前向只前进 1 天**，`step` 几乎不起作用（step=0/1/2/3/10/24 的最佳 ACC
   都落在同一天）。所以「step=10」不等于「一步跑到第 10 天」。
2. 第三方的 `10.nc` / `10mean.nc` 是**第 10 步**的场（它们的场贴 +9/+10 天真值，
   不贴 +1 天）。v1 只跑了 1 步就去比，比错了，0.96 的天花板就是这么来的。

另外：盘的 `sst` 有 33.89% 的 NaN（陆地），喂进图之前必须替掉。这里填**气候均值**
（即归一化空间的 0，物理上 ≈287.2 K），填完把格点数打出来。

本脚本做三件事
--------------
E.  单步诊断：两个起报日各跑 1 步，与盘里 t+1 真值逐个通道比 ACC/RMSE。
    —— 用来判定 `v` 到底是「我们喂错了」还是「模型不擅长」（sst 只算海洋格点）。
A2. 2011-01-05 起报，**11 成员 × 10 步自回归**，逐步对真值给 ACC：
    既看 rollout 收敛到哪，也在第 10 步与第三方 `10.npz` 对表。
C2. 2024-12-24 起报，同上，第 10 步与第三方 `10mean.npz`（30 通道）对表。

只读数据盘，只往 --json 写一个小 JSON。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

TP = "tp"
SST = "sst"


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
        self.sst_i = self.ch.index(SST)

    def idx(self, names):
        return [self.ch.index(n) for n in names]

    def _raw(self, day: str) -> np.ndarray:
        t = np.datetime64(day + "T00:00:00")
        i = int(np.searchsorted(self.time, t))
        if i >= self.time.size or str(self.time[i])[:10] != day:
            raise KeyError(f"盘里没有 {day}（最近的是 {self.time[min(i, self.time.size - 1)]}）")
        return np.asarray(self.g["data"][i, :, :, :]).astype(np.float32)

    def truth(self, day: str) -> np.ndarray:
        """真值：物理场，**保留 NaN**（sst 陆地），比较时当掩膜用。"""
        d = self._raw(day)
        phys = d * self.std[:, None, None] + self.mean[:, None, None]
        phys[self.tp_i] = np.expm1(d[self.tp_i])
        return phys

    def frame(self, day: str) -> np.ndarray:
        """喂给图的输入：物理场，NaN 一律替成该通道气候均值（归一化空间 = 0）。"""
        phys = self.truth(day)
        bad = ~np.isfinite(phys)
        if bad.any():
            rows = np.nonzero(bad.any(axis=(1, 2)))[0]
            for c in rows:
                phys[c][bad[c]] = self.mean[c]
            self._filled = int(bad.sum())
        return phys


def sanitize(phys: np.ndarray, st: Store) -> tuple[np.ndarray, int]:
    """模型输出里的非有限值（sst 陆地）同样替成气候均值，保证能喂回下一步。"""
    bad = ~np.isfinite(phys)
    n = int(bad.sum())
    if n:
        for c in np.nonzero(bad.any(axis=(1, 2)))[0]:
            phys[c][bad[c]] = st.mean[c]
    return phys, n


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


def wstats(a, b, W):
    """面积加权 corr / rmse / bias，只在两边都有限的格点上算。"""
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 100:
        return float("nan"), float("nan"), float("nan"), int(ok.sum())
    w = W[ok]
    aa = a[ok].astype(np.float64)
    bb = b[ok].astype(np.float64)
    ws = w.sum()
    am = (w * aa).sum() / ws
    bm = (w * bb).sum() / ws
    num = (w * (aa - am) * (bb - bm)).sum()
    den = np.sqrt((w * (aa - am) ** 2).sum() * (w * (bb - bm) ** 2).sum())
    corr = float(num / den) if den > 0 else float("nan")
    rmse = float(np.sqrt((w * (aa - bb) ** 2).sum() / ws))
    bias = float((w * (aa - bb)).sum() / ws)
    return corr, rmse, bias, int(ok.sum())


def load_ref(path: str):
    """参考场存成 .npz（channel, data[ch,lat,lon]）。"""
    d = np.load(path, allow_pickle=False)
    return [str(x) for x in d["channel"]], np.asarray(d["data"], dtype=np.float32)


# ---------------------------------------------------------------- 推理
def forward(sess, names, x2: np.ndarray, step: int, hour: float, doy: float) -> np.ndarray:
    feeds = {"input": np.ascontiguousarray(x2[None], dtype=np.float32)}
    for key, val in (("step", step), ("hour", hour), ("doy", doy)):
        if key in names:
            feeds[key] = np.array([val], dtype=np.float32)
    return np.asarray(sess.run(None, feeds)[0], dtype=np.float32)[0, 0]


def ens_step(sess, names, x2, step, hour, doy, n):
    """n 次前向 → (集合均值, 首成员)。图里 RandomNormalLike ⇒ 一次前向一个成员。"""
    acc = None
    m0 = None
    for k in range(n):
        y, _ = sanitize(forward(sess, names, x2, step, hour, doy), STORE)
        if k == 0:
            acc = y.astype(np.float64)
            m0 = y.copy()
        else:
            acc += y
    return (acc / n).astype(np.float32), m0


def rollout(sess, names, init: str, n_members: int, n_steps: int):
    """从 init 起报，每个成员独立自回归跑 n_steps 步，返回每步的 (集合均值, 首成员)。"""
    prev = STORE.frame(shift(init, -1))
    cur = STORE.frame(init)
    C, H, Wd = cur.shape
    accs = [np.zeros((C, H, Wd), np.float64) for _ in range(n_steps + 1)]
    m0s = [None] * (n_steps + 1)
    doy = float(day_of_year(init))
    t0 = time.perf_counter()
    for m in range(n_members):
        p, c = prev, cur
        for k in range(1, n_steps + 1):
            y, _ = sanitize(forward(sess, names, np.stack([p, c]), k, 0.0, doy), STORE)
            accs[k] += y
            if m == 0:
                m0s[k] = y.copy()
            p, c = c, y
        print("    成员 %2d/%2d 完成  [%.0fs]" % (m + 1, n_members, time.perf_counter() - t0))
    means = [(a / n_members).astype(np.float32) for a in accs[1:]]
    return means, m0s[1:]


# ---------------------------------------------------------------- 主流程
STORE: Store = None  # noqa: F821  ens_step/rollout 里用


def main() -> int:
    global STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/tmp/douzsh/models/fengshun2.0_260901.onnx")
    ap.add_argument("--plugin", default="/workspace/tmp/douzsh/models/xmetai_onnx_plugins.ort1_24.cuda12.so")
    ap.add_argument("--store", default="/workspace/data/zarr_refresh_v3/era5.1959_2025.c87.p25.h24.v1")
    ap.add_argument("--ref-dir", default="/workspace/szw_repro")
    ap.add_argument("--json", default="/workspace/szw_repro/repro_fengshun_roll.json")
    ap.add_argument("--members", type=int, default=11, help="成员数（第三方 10mean 是 11 成员）")
    ap.add_argument("--steps", type=int, default=10, help="自回归步数")
    args = ap.parse_args()

    t_start = time.perf_counter()
    STORE = Store(args.store)
    st = STORE
    print("盘通道 %d 个；时间 %s → %s" % (
        len(st.ch), str(st.time[0])[:10], str(st.time[-1])[:10]))
    lat = np.asarray(st.g["lat"][:]).ravel()
    W = area_weights(lat)
    print("lat %.2f→%.2f  lon 0→%.2f" % (lat[0], lat[-1],
          float(np.asarray(st.g["lon"][:]).ravel()[-1])))

    sess = make_session(args.model, args.plugin)
    names = [i.name for i in sess.get_inputs()]
    report: dict = {"members": args.members, "steps": args.steps}

    # ================================================================ E: 单步全通道诊断
    print("\n" + "=" * 78)
    print("E. 单步诊断：1 次前向 vs 盘里 t+1 真值，87 通道")
    print("=" * 78)
    speed = {}
    for init in ("2011-01-05", "2024-12-24"):
        x2 = np.stack([st.frame(shift(init, -1)), st.frame(init)])
        truth = st.truth(shift(init, 1))
        base = st.truth(init)
        print("  输入 NaN 填充：%d 格（%s 陆地）" % (
            getattr(st, "_filled", 0), ", ".join(
                "%s=%.2f%%" % (n, 100.0 * (~np.isfinite(truth[c])).mean())
                for c, n in enumerate(st.ch) if (~np.isfinite(truth[c])).any())))
        t0 = time.perf_counter()
        em, _ = ens_step(sess, names, x2, 1, 0.0, float(day_of_year(init)), args.members)
        dt = (time.perf_counter() - t0) / args.members
        speed[init] = dt
        rows = {}
        for c, name in enumerate(st.ch):
            corr, rmse, bias, n = wstats(em[c], truth[c], W)
            pcorr, prmse, _, _ = wstats(base[c], truth[c], W)  # 持续性基线
            rows[name] = {"corr": corr, "rmse": rmse, "bias": bias,
                          "persist_corr": pcorr, "persist_rmse": prmse, "n": n}
        report.setdefault("E", {})[init] = rows
        print("\n  -- 起报 %s（%.2fs/前向，%d 成员均值）--" % (init, dt, args.members))
        print("  %-9s %8s %8s %10s %10s   %s" % (
            "channel", "ACC", "持续性", "RMSE", "持续性R", "判读"))
        order = sorted(st.ch, key=lambda n: -(rows[n]["corr"] if np.isfinite(rows[n]["corr"]) else -9))
        for name in order:
            e = rows[name]
            tag = ""
            if not np.isfinite(e["corr"]):
                tag = "!! 无法计算"
            elif e["corr"] < 0.5:
                tag = "!! 崩"
            elif e["corr"] < e["persist_corr"] - 0.05:
                tag = "< 不如持续性"
            print("  %-9s %+.4f  %+.4f  %10.4f  %10.4f   %s" % (
                name, e["corr"], e["persist_corr"], e["rmse"], e["persist_rmse"], tag))

    # ================================================================ A2 / C2: rollout
    cases = [
        ("A2", "2011-01-05", "10.npz"),
        ("C2", "2024-12-24", "10mean.npz"),
    ]
    key_ch = ["z500", "t2m", "u200", "v200", "sst", "tp"]
    key_i = st.idx(key_ch)
    for tag, init, refname in cases:
        print("\n" + "=" * 78)
        print("%s. 自回归 rollout：起报 %s，%d 成员 × %d 步" % (tag, init, args.members, args.steps))
        print("=" * 78)
        means, m0s = rollout(sess, names, init, args.members, args.steps)
        truths = {k: st.truth(shift(init, k)) for k in range(1, args.steps + 1)}
        base = st.truth(init)

        print("  逐步 ACC（集合均值 / 持续性）：")
        print("  %-5s " % "step" + " ".join("%18s" % c for c in key_ch))
        curve = {}
        for k in range(1, args.steps + 1):
            row = {}
            cells = []
            for name, c in zip(key_ch, key_i):
                corr, rmse, _, _ = wstats(means[k - 1][c], truths[k][c], W)
                pcorr, prmse, _, _ = wstats(base[c], truths[k][c], W)
                row[name] = {"corr": corr, "rmse": rmse,
                             "persist_corr": pcorr, "persist_rmse": prmse}
                cells.append("%+.3f/%+.3f" % (corr, pcorr))
            curve[k] = row
            print("  %-5d " % k + " ".join("%18s" % x for x in cells))
        report.setdefault(tag, {"init": init, "curve": curve})

        # 第 N 步 vs 第三方参考
        refp = Path(args.ref_dir) / refname
        if not refp.is_file():
            print("  （缺参考 %s，跳过对表）" % refname)
            continue
        ref_ch, ref_v = load_ref(str(refp))
        fin = means[-1]
        ocean = np.isfinite(truths[args.steps][st.sst_i])  # sst 只比海洋
        print("\n  第 %d 步 vs 第三方 %s（%d 通道）：" % (args.steps, refname, len(ref_ch)))
        print("  %-9s %10s %10s %10s %12s %12s" % (
            "channel", "ACC", "RMSE", "bias", "我们均值", "他们均值"))
        refrows = {}
        for j, name in enumerate(ref_ch):
            c = st.ch.index(name)
            pred = fin[c].copy()
            ref = ref_v[j]
            if name == SST:
                pred = np.where(ocean, pred, np.nan)
            corr, rmse, bias, n = wstats(pred, ref, W)
            refrows[name] = {"corr": corr, "rmse": rmse, "bias": bias,
                             "mean": float(np.nanmean(pred)), "ref_mean": float(np.nanmean(ref)), "n": n}
            print("  %-9s %+.4f %10.4f %+10.4f %12.5g %12.5g" % (
                name, corr, rmse, bias, refrows[name]["mean"], refrows[name]["ref_mean"]))
        report[tag]["ref"] = {"file": refname, "channels": refrows}

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=float)
    print("\nJSON 已写入 %s   总用时 %.1fs" % (args.json, time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
