#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测：预测 (runner.py 输出 NetCDF) vs 实况 (era5_store)，算 RMSE/MAE/CRPS/Spread/BSS/AROC。

指标算法移植自 D:\\weather_test_bash\\ensemble_verifier.py（CRPS 排序系数法、
Spread-Error Ratio、BSS、AROC），数据读取换成框架自己的：
  预测 = runner.py 落盘的 {out}/{起报}/member_{成员}/step.nc
  实况 = era5_store（与输入同一数据源，用同一 loader 读「有效时间」真值）

两种模式：
  * --init 2025010600     只评估这一个起报（指定起报时间）
  * 不写 --init           扫描 --fcst 下所有起报目录（YYYYMMDD 或 YYYYMMDDHH），逐个评估

成员处理口径（集合预报，forecast_type=ensemble）：
  * RMSE / MAE   —— 先取 50 成员集合平均，再用「集合平均 vs 实况」算（先平均再算指标）
  * CRPS / Spread / SSR —— 直接用整个集合分布（不先平均）
  * BSS / AROC（降水）   —— 把「超过阈值的成员比例」当作概率，vs 二值实况
确定性模型（forecast_type=deterministic）只算 RMSE / MAE，集合指标留 NaN。

用法：
  python evaluate.py --fcst /path/output --init 2025010600 \
      --steps 61 --vars z500,u200,v200,msl,tp \
      --loader era5_store --spec specs/fuxi_ens.json --out ./eval_results
  python evaluate.py --fcst /path/output --steps 61 \
      --vars z500,u200,v200,msl,tp --spec specs/fuxi_ens.json --out ./eval_results
"""
import argparse
import logging
import os
import re
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr

from loaders import create_loader
from adapters.build_input import load_spec

PRECIP_THRESHOLDS = [0.1, 4.0, 13.0, 25.0]
CHINA_LAT_RANGE = (15.0, 55.0)
CHINA_LON_RANGE = (70.0, 140.0)

log = logging.getLogger("eval")


# ---------------------------------------------------------------------------
# 指标算法（自 ensemble_verifier.py 移植，接口统一为 (M, lat, lon)）
# ---------------------------------------------------------------------------
def _lat_weights(lats):
    return np.cos(np.deg2rad(np.abs(np.asarray(lats, dtype=np.float64))))


def crps_field(ensemble, obs):
    """逐点 CRPS 场 (lat, lon)。ensemble (M, lat, lon)，obs (lat, lon)。
    闭式解：E|X−y| − ½E|X−X'|，第二项用排序线性系数法 O(M log M)。"""
    M = ensemble.shape[0]
    term1 = np.mean(np.abs(ensemble - obs[np.newaxis, ...]), axis=0)
    srt = np.sort(ensemble, axis=0)
    k = np.arange(M, dtype=np.float64)
    coeff = (2.0 * k + 1.0 - M) / (M ** 2)
    coeff = coeff.reshape((-1,) + (1,) * (ensemble.ndim - 1))
    term2 = np.sum(srt * coeff, axis=0)
    return np.clip(term1 - term2, 0.0, None)


def weighted_mean(field, lats):
    """cos(lat) 加权平均。field (lat, lon)。"""
    w = _lat_weights(lats)
    w2d = np.broadcast_to(w[:, np.newaxis], field.shape)
    wsum = float(np.nansum(w2d))
    return float(np.nansum(w2d * field) / wsum) if wsum > 0 else float("nan")


def spread_rmse_ratio(ensemble, obs, lats):
    """返回 (spread, rmse, ratio)。spread 用 ddof=1 无偏；rmse 为集合平均的。"""
    M = ensemble.shape[0]
    emean = np.mean(ensemble, axis=0)
    w = _lat_weights(lats)
    w2d = np.broadcast_to(w[:, np.newaxis], emean.shape)
    wsum = float(np.nansum(w2d))
    if M > 1:
        sv = np.nansum(w2d * np.sum((ensemble - emean[np.newaxis, ...]) ** 2, axis=0))
        spread = float(np.sqrt(sv / (wsum * (M - 1)))) if wsum > 0 else float("nan")
    else:
        spread = float("nan")
    rv = np.nansum(w2d * (emean - obs) ** 2)
    rmse = float(np.sqrt(rv / wsum)) if wsum > 0 else float("nan")
    ratio = spread / rmse if (M > 1 and rmse and rmse > 0 and not np.isnan(rmse)) else float("nan")
    return spread, rmse, ratio


def bss(prob, obs_bin, weights=None):
    """Brier 技巧评分。prob/obs_bin (lat, lon)；weights 可选。"""
    if weights is None:
        weights = np.ones_like(obs_bin, dtype=np.float64)
    wsum = float(np.nansum(weights))
    if wsum <= 0:
        return float("nan")
    bs = float(np.nansum(weights * (prob - obs_bin) ** 2) / wsum)
    p_clim = float(np.nansum(weights * obs_bin) / wsum)
    bs_ref = p_clim * (1.0 - p_clim)
    if bs_ref <= 0:
        return float("nan")
    return float(1.0 - bs / bs_ref)


def aroc(prob, obs_bin, weights=None):
    """ROC 曲线下面积。以概率唯一非零取值为阈值，梯形积分。"""
    if weights is None:
        weights = np.ones_like(obs_bin, dtype=np.float64)
    total_obs = float(np.nansum(weights * obs_bin))
    total_non = float(np.nansum(weights * (1 - obs_bin)))
    if total_obs <= 0 or total_non <= 0:
        return float("nan")
    thresholds = np.unique(prob[prob > 0])
    pts = [(0.0, 0.0)]
    for p in thresholds:
        yes = prob >= p
        pod = float(np.nansum(weights * (yes & (obs_bin == 1))) / total_obs)
        pofd = float(np.nansum(weights * (yes & (obs_bin == 0))) / total_non)
        pts.append((pofd, pod))
    pts.append((1.0, 1.0))
    pts = sorted(pts, key=lambda x: x[0])
    area = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        area += (x1 - x0) * (y0 + y1) / 2.0
    return float(max(0.0, min(1.0, area)))


# ---------------------------------------------------------------------------
# 数据读取 / 对齐
# ---------------------------------------------------------------------------
def _align_grid(vals, lat, lon):
    """统一到「lat 北→南、lon 0→360」。vals (lat, lon)。"""
    vals = np.asarray(vals, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    if lat.size > 1 and lat[0] < lat[-1]:        # 南→北，翻转为北→南
        vals = vals[::-1, ...]
    if lon.size > 1 and lon[0] < 0:              # -180..180 → 0..360
        vals = np.roll(vals, int(lon.size // 2), axis=-1)
    return vals


def _apply_unit(var, arr, is_obs):
    """单位对齐到「比较口径」。z 保持 m²/s²（论文口径，GraphCast/FuXi 同款，不 ÷g）；tp：实况 m→mm（预测已是 mm）。"""
    if var == "tp" and is_obs:
        return arr * 1000.0
    return arr


def _read_pred(fcst_root, init_dir, step_idx, var, members, multi):
    """读某 step 全体成员的某变量，返回 (M, lat, lon)。"""
    name = var.upper()  # z500 -> Z500, tp -> TP, msl -> MSL
    preds = []
    for m in range(members):
        if multi:
            nc = os.path.join(fcst_root, init_dir, f"member_{m:03d}", f"{step_idx:03d}.nc")
        else:
            nc = os.path.join(fcst_root, init_dir, f"{step_idx:03d}.nc")
        with xr.open_dataset(nc) as ds:
            preds.append(np.asarray(ds[name].values, dtype=np.float64))
    return np.stack(preds, axis=0)


# ---------------------------------------------------------------------------
# 参数解析 / 类型与成员数
# ---------------------------------------------------------------------------
def _resolve_members(args, spec):
    """从 spec 的 forecast_type 定 members 默认值；类型↔成员数不一致时告警。"""
    forecast_type = spec["model"].get("forecast_type", "deterministic")
    members = args.members if args.members is not None else spec["model"].get("members", 1)
    if members < 1:
        raise SystemExit("成员数必须 >= 1")
    if forecast_type == "deterministic" and members > 1:
        log.warning("spec 声明确定性模型（forecast_type=deterministic），但 members=%d>1；"
                    "确定性输出没有成员维度，按 1 处理", members)
        members = 1
    if forecast_type == "ensemble" and members <= 1:
        log.warning("spec 声明集合模型（forecast_type=ensemble），但 members=%d；"
                    "集合指标（CRPS/Spread/BSS/AROC）将无法计算", members)
    return forecast_type, members


def _scan_inits(fcst_root, init_hour):
    """扫描 --fcst 下的起报目录，返回 [(init_ts, dirname), ...]（按时间排序）。

    runner.py 落盘是「日期目录」YYYYMMDD（无小时），缺的起报小时用 --init-hour 补；
    若目录是 YYYYMMDDHH（带小时）则直接用自己的小时。
    """
    if not os.path.isdir(fcst_root):
        raise SystemExit(f"预测目录不存在：{fcst_root}")
    found = []
    for name in os.listdir(fcst_root):
        if not os.path.isdir(os.path.join(fcst_root, name)):
            continue
        m8 = re.fullmatch(r"\d{8}", name)
        m10 = re.fullmatch(r"\d{10}", name)
        if m10:
            init = pd.to_datetime(name, format="%Y%m%d%H")
        elif m8:
            init = pd.to_datetime(name + f"{init_hour:02d}", format="%Y%m%d%H")
        else:
            continue
        found.append((init, name))
    if not found:
        raise SystemExit(f"{fcst_root} 下没有 YYYYMMDD 或 YYYYMMDDHH 起报目录")
    found.sort(key=lambda t: t[0])
    return found


def _parse_init_token(token):
    """把 --inits 里的一个 token 解析成 (init_ts, dirname)。

    支持两种写法：YYYYMMDD（起报小时取 0，目录就是这 8 位）和
    YYYYMMDDHH（带小时，目录保持 10 位，与 runner.py 落盘命名一致）。
    """
    token = token.strip()
    if re.fullmatch(r"\d{8}", token):
        return pd.to_datetime(token, format="%Y%m%d"), token
    if re.fullmatch(r"\d{10}", token):
        return pd.to_datetime(token, format="%Y%m%d%H"), token
    raise SystemExit(f"无法解析起报时间 {token!r}（应为 YYYYMMDD 或 YYYYMMDDHH）")


def _setup_logging(out_dir):
    """日志：控制台(stderr，INFO) + 文件({out}/evaluate.log，全量 DEBUG)。"""
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(os.path.join(out_dir, "evaluate.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)


_PROGRESS_INTERVAL = 5  # 评测进度：每隔多少步打一条完整行（逐步单行覆盖在多起报/日志下会堆叠）
_SUMMARY_INTERVAL = 10  # 每评多少个起报打印一次累计平均（避免逐起报刷屏）

def _fmt_dur(sec):
    """秒 -> 人类可读时长（'45s' / '12m34s' / '1h02m'）。"""
    sec = int(round(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _progress_bar(frac, label, eta_s=None):
    """单行进度条（stdout，\\r 覆盖）。"""
    n = 24
    filled = int(round(n * frac))
    bar = "#" * filled + "-" * (n - filled)
    eta = f"  ETA {eta_s:.0f}s" if eta_s is not None else ""
    print(f"\r  {label}  [{bar}] {frac * 100:5.1f}%{eta}", end="", flush=True)


def _print_summary(rows, vars_, forecast_type):
    """打印当前累计的每个变量平均 RMSE/MAE（跨所有起报与 lead）。

    这是给「想看整体平均」的汇总视图；逐起报×逐 lead 的明细仍在 CSV 里。
    集合模型多打一列 CRPS（tp 降水走 BSS/AROC，无 CRPS，显示 '-'）。
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    is_ensemble = forecast_type == "ensemble"
    hdr = f"  {'var':<6} {'RMSE':>10} {'MAE':>10}"
    if is_ensemble:
        hdr += f" {'CRPS':>10}"
    print(hdr)
    for var in vars_:
        sub = df[df["var"] == var]
        if sub.empty:
            continue
        line = f"  {var:<6} {sub['rmse'].mean():>10.4f} {sub['mae'].mean():>10.4f}"
        if is_ensemble:
            crps = sub["crps"].dropna()
            line += f" {crps.mean():>10.4f}" if len(crps) else f" {'-':>10}"
        print(line)


# ---------------------------------------------------------------------------
# 单个起报的评测
# ---------------------------------------------------------------------------
def _eval_init(fcst_root, rows, init, init_dir, steps, interval, members, multi,
               vars_, loader, lat, china_mask, forecast_type):
    """评估一个起报的所有 step × 变量，逐行 append 到 rows。"""
    is_ensemble = forecast_type == "ensemble" and members > 1
    step_times = []
    for step_idx in range(1, steps + 1):
        t0 = perf_counter()
        lead = step_idx * interval
        valid = init + pd.Timedelta(hours=lead)
        obs_ds = loader.load(valid)                      # 每个有效时间读一次实况
        chans = [str(c) for c in np.atleast_1d(obs_ds["channel"].values)]

        for var in vars_:
            chan = var.lower()
            if chan not in chans:
                log.debug("  [跳过] 实况无通道 %s（lead %dh）", chan, lead)
                continue
            obs = _align_grid(obs_ds["data"].sel(channel=chan).values,
                              obs_ds["lat"].values, obs_ds["lon"].values)
            obs = _apply_unit(var, obs, is_obs=True)
            pred = _read_pred(fcst_root, init_dir, step_idx, var, members, multi)
            pred = _apply_unit(var, pred, is_obs=False)

            emean = pred.mean(axis=0)
            rmse = float(np.sqrt(weighted_mean((emean - obs) ** 2, lat)))
            mae = weighted_mean(np.abs(emean - obs), lat)

            row = {"init": init.strftime("%Y%m%d%H"), "forecast_type": forecast_type,
                   "n_members": members, "var": var, "lead_hour": lead,
                   "valid_time": valid.strftime("%Y%m%d%H"),
                   "rmse": rmse, "mae": mae,
                   "crps": float("nan"), "spread": float("nan"), "ssr": float("nan"),
                   "bss": float("nan"), "aroc": float("nan")}

            if is_ensemble and var != "tp":             # 连续场：CRPS / spread / SSR
                spread, _, ratio = spread_rmse_ratio(pred, obs, lat)
                row["crps"] = weighted_mean(crps_field(pred, obs), lat)
                row["spread"] = spread
                row["ssr"] = ratio

            if is_ensemble and var == "tp":             # 降水：中国区域 BSS / AROC
                bss_vals, aroc_vals = [], []
                w = _lat_weights(lat)[:, np.newaxis] * china_mask
                for thr in PRECIP_THRESHOLDS:
                    pr = (pred >= thr).mean(axis=0)
                    ob = (obs >= thr).astype(float)
                    bss_vals.append(bss(pr, ob, weights=w))
                    aroc_vals.append(aroc(pr, ob, weights=w))
                row["bss"] = float(np.nanmean(bss_vals))
                row["aroc"] = float(np.nanmean(aroc_vals))

            rows.append(row)
            crps_s = f"{row['crps']:.4f}" if np.isfinite(row["crps"]) else "-"
            ssr_s = f"{row['ssr']:.4f}" if np.isfinite(row["ssr"]) else "-"
            log.debug("  %-5s lead %3dh valid %s  RMSE=%.4f MAE=%.4f CRPS=%s SSR=%s",
                      var, lead, valid.strftime("%Y%m%d%H"), rmse, mae, crps_s, ssr_s)

        dt = perf_counter() - t0
        step_times.append(dt)
        if step_idx == steps or step_idx % _PROGRESS_INTERVAL == 0:
            # 逐步进度：只进 evaluate.log（避免 220 起报 × 每 5 步把控制台刷屏），用平均耗时外推 ETA。
            avg = sum(step_times) / step_idx
            eta = avg * (steps - step_idx)
            log.debug("  %s 已评 %d/%d 步 · 还要 %s", init.strftime("%Y%m%d%H"),
                      step_idx, steps, _fmt_dur(eta))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="预测 vs era5_store 实况的集合检验")
    p.add_argument("--fcst", default="/workspace/szwCode/xmetai-inference/output",
                   help="预测输出根目录（runner.py 的 --out）")
    p.add_argument("--init", default=None,
                   help="起报时间 YYYYMMDDHH；只评这一个起报")
    p.add_argument("--inits", default=None,
                   help="起报时间列表，逗号/空格分隔（YYYYMMDD 或 YYYYMMDDHH）；"
                        "只评列出的这几个起报，不扫描全部")
    p.add_argument("--init-hour", type=int, default=0,
                   help="扫描模式下，日期目录(YYYYMMDD)缺的起报小时（默认 0=00UTC）")
    p.add_argument("--steps", type=int, default=61, help="预报步数")
    p.add_argument("--members", type=int, default=None,
                   help="集合成员数（缺省读 spec 的 model.members；确定性=1）")
    p.add_argument("--vars", default="z500,u200,v200,msl,tp", help="要检验的变量，逗号分隔")
    p.add_argument("--loader", default="era5_store", help="实况数据源（地址在 loader 里内置）")
    p.add_argument("--spec", default="specs/fuxi_ens.json", help="模型 spec JSON")
    p.add_argument("--out", default="./eval_results", help="CSV 输出目录")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    _setup_logging(args.out)

    spec = load_spec(args.spec)
    forecast_type, members = _resolve_members(args, spec)
    loader = create_loader(args.loader, spec=spec)
    interval = spec["model"].get("hour_interval", 6)
    multi = members > 1
    vars_ = [v.strip() for v in args.vars.split(",") if v.strip()]

    if args.init is not None:
        init = pd.to_datetime(args.init, format="%Y%m%d%H")
        inits = [(init, init.strftime("%Y%m%d"))]
        mode = "single"
    elif args.inits:
        inits = [_parse_init_token(t) for t in re.split(r"[,\s]+", args.inits) if t]
        mode = "list"
    else:
        inits = _scan_inits(args.fcst, args.init_hour)
        mode = "folder"

    # 目标网格坐标（模型网格，北→南 lat，0→360 lon）
    glat, glon = spec["grid"]["lat"], spec["grid"]["lon"]
    lat = np.arange(glat["size"], dtype=np.float64) * glat["step"] + glat["start"]
    lon = np.arange(glon["size"], dtype=np.float64) * glon["step"] + glon["start"]
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    china_mask = ((lat2d >= CHINA_LAT_RANGE[0]) & (lat2d <= CHINA_LAT_RANGE[1]) &
                  (lon2d >= CHINA_LON_RANGE[0]) & (lon2d <= CHINA_LON_RANGE[1]))

    log.info("评测启动：mode=%s | inits=%d | steps=%d | members=%d | forecast_type=%s | vars=%s",
             mode, len(inits), args.steps, members, forecast_type, ",".join(vars_))
    log.info("预测目录 %s | spec=%s | loader=%s | 输出 %s", args.fcst, args.spec, args.loader, args.out)

    rows = []
    t_total0 = perf_counter()
    for ii, (init, init_dir) in enumerate(inits):
        log.debug("=== 起报 %s（%d/%d）===", init.strftime("%Y%m%d%H"), ii + 1, len(inits))
        _eval_init(args.fcst, rows, init, init_dir, args.steps, interval, members, multi,
                   vars_, loader, lat, china_mask, forecast_type)
        done = ii + 1
        # 每 _SUMMARY_INTERVAL 个起报打一次累计平均（最后那个留到末尾统一出最终汇总）
        if done % _SUMMARY_INTERVAL == 0 and done != len(inits):
            print(f"已评 {done}/{len(inits)} 个起报 · 还剩 {len(inits) - done} 个起报", flush=True)
            _print_summary(rows, vars_, forecast_type)

    df = pd.DataFrame(rows)
    csv_name = {"single": f"eval_{args.init}.csv", "list": "eval_selected.csv",
                "folder": "eval_all.csv"}[mode]
    csv_path = os.path.join(args.out, csv_name)
    df.to_csv(csv_path, index=False, float_format="%.6g")
    log.info("完成：%d 行写入 %s，总耗时 %.1fs", len(df), csv_path, perf_counter() - t_total0)

    # 最终汇总：所有起报的平均 RMSE/MAE（明细在 CSV，此处只给每个变量的整体平均）
    print(f"\n== 全部 {len(inits)} 个起报平均（{len(df)} 行）==", flush=True)
    _print_summary(rows, vars_, forecast_type)


if __name__ == "__main__":
    main()
