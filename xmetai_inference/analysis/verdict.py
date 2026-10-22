# -*- coding: utf-8 -*-
"""判决页：把 README 2M.3 的「注入点判据」做成一条命令（零推理、零 GPU）。

只读 v2 评测归档（每个臂一个目录，里面有 scores.csv 与 manifest.json）。它回答一个问题：

    这个变体到底改了「形状」，还是只改了「幅度」？

判据（可证伪，见 perturbation/README 的 2M.3）：

    r(lead) = spread_X(lead) / spread_REF(lead)
    k(lead) = r(lead) ** (1 / p(lead))        # 折算成「等效 sigma 倍数」

* **r 平** => 只是幅度（等价于把 sigma 乘一个常数）；
* **r 随时效上升** => 长时效响应更强 = 形状真被治好。

k 是在「sigma 用多少」这把尺子上读 r，所以跨臂可比；把它和「补到标定比 1.0 所需的 k」比一下，
就能判「这一族够不够」，不必等 CRPSS、也不必凑起报数。

p(lead) 是 sigma 的响应指数，默认取 2L.4 的实测表（B: sigma_full=0.01 vs F: 0.04，成员对齐 4v4）。
它是「注入点在骨干上的位置」的属性 —— **换骨干必须重测**，用 --p-from 传两个推理输出目录
（sigma 不同、其余相同），本命令会现场量。

注意：判据的**定性那一半（r 平不平）不需要 p**；p 只影响「还要加多少 sigma」这个量级读数。

用法：

    python -m xmetai_inference.perturbation verdict F S T V E3
    python -m xmetai_inference.perturbation verdict F S2 --base /data/eval
    python -m xmetai_inference.perturbation verdict F S --dir S=/tmp/other/S
    python -m xmetai_inference.perturbation verdict F S --p-from <小sigma目录> <大sigma目录>
    python -m xmetai_inference.perturbation verdict F S --csv verdict.csv
"""
from __future__ import annotations

import json
import math
import os
import os.path as osp

DEFAULT_BASE = os.environ.get("SPW_EVAL_BASE", "D:\\xmetai-evaluation-result")
DEFAULT_PREFIX = "weather_rmse_ens_spw_"
DEFAULT_VAR = "z500"
GLOBAL = "global"

# 判决口径：与 README 2L.11 / 2L.12、评测侧 compare.py 逐字一致
SCALARS = {
    "spread": ("ensemble", "area_weighted", "spread"),
    "ratio": ("ensemble", "area_weighted", "spread_error_ratio"),
    "crps": ("ensemble", "area_weighted", "crps"),
    "rmse": ("ensemble", "area_weighted", "rmse"),
    "acc": ("deterministic", "spatial_correlation", "acc"),
    "act": ("specialized", "area_weighted", "activity_ratio"),
}
BANDS = ("1_4", "5_20", "21_40", "41_64", "65_128")
# 判据 3 只认这两个带（2L.8：唯一没被系统抑制的通道）
GAIN_BANDS = ("1_4", "5_20")
BINS = ((6, 48), (72, 120), (168, 240), (240, 360), (6, 360))
SHORT = (6, 48)
LONG = (240, 360)

# 2L.4 实测：sigma 的响应指数 p(lead)。跨骨干要重测（见 --p-from）。
P_TABLE = {6: 1.007, 54: 0.990, 102: 0.946, 150: 0.856,
           198: 0.755, 246: 0.672, 294: 0.541, 342: 0.466}
P_SIGMA_PAIR = (0.01, 0.04)
P_SOURCE = "2L.4（B sigma_full=0.01 vs F 0.04，成员对齐 4v4）"


# --------------------------------------------------------------------------- 读数

def arm_dir(base, prefix, arm):
    return osp.join(base, prefix + arm)


def read_manifest(path):
    """manifest.json 里的硬事实：起报日/成员数/步数/变量/协议。"""
    with open(osp.join(path, "manifest.json"), encoding="utf-8") as handle:
        raw = json.load(handle)
    config = raw.get("resolved_config", {})
    forecast = config.get("forecast", {}).get("params", {})
    return {
        "created": str(raw.get("created_at", ""))[:19],
        "root_dir": forecast.get("root_dir"),
        "start": config.get("start_date"),
        "end": config.get("end_date"),
        "var": config.get("variables", {}).get("forecast"),
        "protocol": raw.get("protocol_id"),
        "step_hours": forecast.get("step_hours"),
        "n_files": raw.get("n_forecast_input_files"),
        "n_results": raw.get("n_results"),
    }


def load_arm(path):
    """读 scores.csv 成长表。region 的 NaN 就是「全球」。"""
    import pandas as pd

    frame = pd.read_csv(osp.join(path, "scores.csv"), low_memory=False)
    frame["init_day"] = frame.init_time.str.slice(0, 10).str.replace("-", "", regex=False)
    frame["region"] = frame.region.fillna(GLOBAL)
    return frame


def curve(frame, key, var=DEFAULT_VAR, region=GLOBAL, verbose=True):
    """一个标量指标的逐时效曲线（跨起报日取均值）。"""
    product_kind, aggregation, metric = SCALARS[key]
    query = frame[(frame.variable == var) & (frame.metric == metric)
                  & (frame.product_kind == product_kind)
                  & (frame.aggregation == aggregation) & (frame.region == region)]
    if verbose and not len(query):
        raise SystemExit(
            "读不到指标 %s（variable=%s region=%s）—— 检查 --var / --region 与归档是否完整"
            % (metric, var, region))
    return query.groupby("lead_h").value.mean()


def band_curve(frame, kind, var=DEFAULT_VAR, region=GLOBAL):
    """球谐带功率曲线：返回 {带: Series(lead_h)}。kind = forecast / observation / ratio。"""
    metric = "spherical_band_power_%s" % kind
    query = frame[(frame.variable == var) & (frame.metric == metric) & (frame.region == region)]
    out = {}
    for band in BANDS:
        rows = query[query.group == band]
        if len(rows):
            out[band] = rows.groupby("lead_h").value.mean()
    return out


def daily_curve(frame, key, var=DEFAULT_VAR, region=GLOBAL):
    """逐起报日的曲线（判「符号一致性」用）。返回 {起报日: Series(lead_h)}。"""
    product_kind, aggregation, metric = SCALARS[key]
    query = frame[(frame.variable == var) & (frame.metric == metric)
                  & (frame.product_kind == product_kind)
                  & (frame.aggregation == aggregation) & (frame.region == region)]
    return {day: rows.groupby("lead_h").value.mean()
            for day, rows in query.groupby("init_day")}


# --------------------------------------------------------------------- p(lead)

def p_from_dirs(low_dir, high_dir, sigma_pair, var=DEFAULT_VAR):
    """现场量 p(lead)：两个只有 sigma 不同的推理输出目录（零 GPU，读产物）。

    p = ln(sigma_hat_high / sigma_hat_low) / ln(sigma_high / sigma_low)
    """
    from .spread import analyse_run

    low_sigma, high_sigma = sigma_pair
    if not (0 < low_sigma < high_sigma):
        raise SystemExit("--p-sigma 要给「小,大」两个正数，且前小后大")
    low = {row["lead_h"]: row["sigma"] for row in analyse_run(
        low_dir, var=var, verbose=False)["rows"]}
    high = {row["lead_h"]: row["sigma"] for row in analyse_run(
        high_dir, var=var, verbose=False)["rows"]}
    shared = sorted(set(low) & set(high))
    if len(shared) < 2:
        raise SystemExit("--p-from 的两个目录没有共同时效，量不出 p")
    scale = math.log(high_sigma / low_sigma)
    return {lead: math.log(high[lead] / low[lead]) / scale for lead in shared}


def p_at(table, lead):
    """把离散的 p 表插到任意时效（对数时效上线性插值）。"""
    leads = sorted(table)
    if lead <= leads[0]:
        return table[leads[0]]
    if lead >= leads[-1]:
        return table[leads[-1]]
    for index in range(len(leads) - 1):
        first, second = leads[index], leads[index + 1]
        if first <= lead <= second:
            weight = (math.log(lead) - math.log(first)) / (math.log(second) - math.log(first))
            return table[first] * (1 - weight) + table[second] * weight
    return table[leads[-1]]


# ------------------------------------------------------------------------ 判决

def bin_mean(series, low, high):
    part = series.loc[(series.index >= low) & (series.index <= high)]
    return float(part.mean()) if len(part) else float("nan")


def k_profile(spread_arm, spread_ref, p_table):
    """每个时效的 r 与 k。"""
    rows = []
    for lead in sorted(spread_ref.index):
        if lead not in spread_arm.index:
            continue
        ratio = float(spread_arm[lead]) / float(spread_ref[lead])
        p_value = p_at(p_table, lead)
        rows.append({"lead": lead, "r": ratio, "p": p_value,
                     "k": ratio ** (1.0 / p_value) if ratio > 0 else float("nan")})
    return rows


def k_needed(rmse_ref_ratio, p_table):
    """参考臂自己的长时效标定缺口：要把标定比补到 1.0，需要把 sigma 乘多少。"""
    rows = []
    for lead in sorted(rmse_ref_ratio.index):
        gap = float(rmse_ref_ratio[lead])
        if not (gap > 0):
            continue
        p_value = p_at(p_table, lead)
        rows.append({"lead": lead, "ratio_ref": gap,
                     "need_r": 1.0 / gap,
                     "k_needed": (1.0 / gap) ** (1.0 / p_value)})
    return rows


def flatness(rows):
    """k 的「平不平」：极差 + 趋势。趋势 = 后 3 个时效 / 前 3 个时效。"""
    values = [row["k"] for row in rows if math.isfinite(row["k"])]
    if len(values) < 4:
        return {"span": float("nan"), "trend": float("nan")}
    ordered = sorted(values)
    head = sum(values[:3]) / 3.0
    tail = sum(values[-3:]) / 3.0
    return {"span": ordered[-1] / ordered[0] if ordered[0] else float("nan"),
            "trend": tail / head if head else float("nan"),
            "first3": head, "last3": tail}


def criteria(arm_curves, ref_curves, arm_bands, ref_bands):
    """2L.7 的四条判据（判据 4 本命令不做，需 $P spread 的 domain_spread）。"""
    spread = arm_curves["spread"]
    ref_spread = ref_curves["spread"]
    long_ratio = bin_mean(spread, *LONG) / bin_mean(ref_spread, *LONG)
    short_ratio = bin_mean(spread, *SHORT) / bin_mean(ref_spread, *SHORT)
    verdict = {
        "long_ratio": long_ratio,
        "short_ratio": short_ratio,
        "pass_long": long_ratio > 1.0,
        "pass_short": short_ratio <= 1.0,
    }
    # 判据 3：+240..360 h 的带功率（相对参考臂）
    relative = {}
    for band in BANDS:
        mine, theirs = arm_bands.get(band), ref_bands.get(band)
        if mine is None or theirs is None:
            continue
        value = bin_mean(mine, *LONG) / bin_mean(theirs, *LONG)
        relative[band] = value
    verdict["bands"] = relative
    # 判据 3 的分类：先问「有没有可辨识的净增益」，再问增益落在哪个尺度。
    # 阈值是拿四个臂逐一核对 README 2L.11 / 2L.12 的判决拍出来的：
    #   S   最大净增益 +6.4%（21-40 反而 -15.5%） -> 没有净增益
    #   T   +61.0% 落在 21-40                     -> 整体锐化（5-20 与 21-40 一起亮）
    #   E3  +47.1% 落在 21-40                     -> 同上，更弱
    #   V   +748% 落在 65-128                     -> 增益落在更小尺度（谱尾部变脏）
    # 靶心（2L.8）：增益要落在 5-20（天气尺度）；1-4（行星）只要求不被弄坏 —— 行星带功率本来就小，
    # 拿 min(1-4, 5-20) 当门槛会把「只抬 5-20」这种正解误判成判负，所以只看 5-20 过没过可辨识门槛。
    if not all(band in relative for band in BANDS):
        verdict["band_kind"] = "缺数据"
    else:
        gain = {band: relative[band] - 1.0 for band in BANDS}
        fine = max(gain["41_64"], gain["65_128"])
        biggest = max(gain.values())
        if biggest < 0.08:
            verdict["band_kind"] = ("没有可辨识的净增益（最大 %+.1f%%，21-40 %+.1f%%）=> 判负"
                                    % (biggest * 100, gain["21_40"] * 100))
        elif (gain["5_20"] >= 0.08 and gain["1_4"] >= -0.02
              and max(gain["21_40"], gain["41_64"], gain["65_128"]) <= 0.03):
            verdict["band_kind"] = "选择性（增益落在 1-4 / 5-20）"
        elif (gain[GAIN_BANDS[0]] >= -0.02 and gain[GAIN_BANDS[1]] >= 0.10
              and gain["21_40"] >= 0.10):
            verdict["band_kind"] = "整体锐化（5-20 与 21-40 一起变亮，非尺度选择性）"
        elif fine >= 0.10 and fine >= biggest - 1e-9:
            verdict["band_kind"] = "增益落在更小尺度（谱尾部变脏）=> 判负"
        else:
            verdict["band_kind"] = "增益落在被抑制的通道（21-40）=> 判负"
    verdict["pass_band"] = verdict["band_kind"].startswith("选择性")
    return verdict


def _band_symbol(crit):
    """判据 3 的记号：✅ 选择性；➖ 中性（整体锐化，既没补对也没补错）；❌ 判负。"""
    if crit.get("pass_band"):
        return "✅"
    return "➖" if "整体锐化" in str(crit.get("band_kind", "")) else "❌"


def verdict_line(name, flat, crit, rows, needs):
    """一句话判决。"""
    large = [row for row in rows if row["lead"] >= LONG[0]]
    worst_need = max((row["k_needed"] for row in needs if row["lead"] >= LONG[0]),
                     default=float("nan"))
    best_k = max((row["k"] for row in large), default=float("nan"))
    if not (math.isfinite(flat.get("trend", float("nan")))):
        return "%s：样本太少，判不了" % name
    if flat["trend"] >= 1.10:
        head = "有形状红利（k 随时效上升 %+.0f%%，长时效 k=%.2f）" % (
            (flat["trend"] - 1.0) * 100.0, flat["last3"])
    elif flat["trend"] <= 0.90:
        head = "纯幅度（k 反而随时效下降 %+.0f%%）" % ((flat["trend"] - 1.0) * 100.0)
    else:
        head = "纯幅度（k 平，极差 %.2fx）" % flat["span"]
    if math.isfinite(worst_need) and best_k < worst_need:
        tail = "且幅度不够：长时效 k=%.2f < 需要的 %.2f => 判负" % (best_k, worst_need)
    elif math.isfinite(worst_need):
        tail = "长时效 k=%.2f >= 需要的 %.2f => 过幅度门槛" % (best_k, worst_need)
    else:
        tail = ""
    passed = sum(1 for key in ("pass_long", "pass_short", "pass_band") if crit.get(key))
    return "%s：%s；%s；判据 1/2/3 过了 %d 条" % (name, head, tail, passed)


# ------------------------------------------------------------------------ 打印

def format_page(names, facts, curves, bands, p_table, p_source, var,
                daily=None, days=(), check=True):
    ref = names[0]
    lines = []
    lines.append("判决页（零推理）：臂 = %s   基准 = %s   变量 = %s" % (", ".join(names), ref, var))
    lines.append("p(lead) 来源：%s" % p_source)
    lines.append("")

    # [0] 可比性
    lines.append("=== [0] 可比性核对（manifest 的硬事实；不一致的行前面加 ⚠）==="
                 if check else "=== [0] 可比性核对（--no-check：不核，只列事实）===")
    keys = [("start", "起报起"), ("end", "起报止"), ("var", "变量"), ("protocol", "协议"),
            ("step_hours", "步长 h"), ("n_files", "输入文件数")]
    base_facts = facts[ref]
    lines.append("  臂        " + "  ".join("%-11s" % label for _k, label in keys)
                 + "  目录")
    for name in names:
        item = facts[name]
        cells = []
        for key, _label in keys:
            value = item.get(key)
            same = value == base_facts.get(key)
            text = "-" if value is None else str(value)
            cells.append("%-11s" % (text if same else "⚠" + text))
        lines.append("  %-9s " % name + "  ".join(cells) + "  " + str(item.get("root_dir")))
    lines.append("")

    # [1] r / k
    lines.append("=== [1] 判决量：r(lead) = spread_臂/spread_基准，k(lead) = r^(1/p(lead)) ===")
    lead_list = sorted(curves[ref]["spread"].index)
    head = "  lead | p(lead) | spread基准 |"
    for name in names[1:]:
        head += " %-22s |" % ("spread%s  r      k" % name)
    lines.append(head)
    for lead in lead_list:
        row = "  %5d | %7.3f | %10.2f |" % (lead, p_at(p_table, lead),
                                            float(curves[ref]["spread"][lead]))
        for name in names[1:]:
            spread = curves[name]["spread"]
            if lead in spread.index:
                ratio = float(spread[lead]) / float(curves[ref]["spread"][lead])
                row += " %9.2f %6.3f %6.3f |" % (
                    float(spread[lead]), ratio, ratio ** (1.0 / p_at(p_table, lead)))
            else:
                row += " %-22s |" % "-"
        lines.append(row)
    lines.append("")

    # [1b] 逐日起报的符号一致性（3 天样本里唯一能看显著性的地方）
    if daily and len(days) > 1:
        lines.append("=== [1b] 逐日起报（%s）：判据 1 / 2 的符号一致性 ===" % ",".join(days))
        for name in names[1:]:
            pieces = []
            for low, high, want_more in ((SHORT[0], SHORT[1], False),
                                         (LONG[0], LONG[1], True)):
                values = []
                for day in days:
                    mine, theirs = daily[name].get(day), daily[ref].get(day)
                    if mine is None or theirs is None:
                        continue
                    values.append(bin_mean(mine, low, high) / bin_mean(theirs, low, high))
                if values:
                    hits = sum(1 for value in values if (value > 1) == want_more)
                    pieces.append("+%d..%d %s（合判据 %d/%d）"
                                  % (low, high, " ".join("%.3f" % value for value in values),
                                     hits, len(values)))
            lines.append("  %-7s %s" % (name, "  |  ".join(pieces)))
        lines.append("")

    # [2] 参考臂的缺口 <-> 需要的 k
    needs = k_needed(curves[ref]["ratio"], p_table)
    lines.append("=== [2] 基准臂自己的标定缺口，与「补到 1.0 需要的 k」 ===")
    lines.append("  lead | 基准标定比 | 需要 spread x | 需要 k（等效 sigma 倍数）")
    for row in needs:
        mark = "  <==" if row["lead"] >= LONG[0] else ""
        lines.append("  %5d | %9.3f | %11.3f | %9.2f%s"
                     % (row["lead"], row["ratio_ref"], row["need_r"], row["k_needed"], mark))
    long_need = max((row["k_needed"] for row in needs if row["lead"] >= LONG[0]),
                    default=float("nan"))
    lines.append("  ⇒ 长时效（+%d..%d h）的门槛：k >= %.2f（判据 1+2 要同时成立才有意义）"
                 % (LONG[0], LONG[1], long_need))
    lines.append("")

    # [3] 判据
    lines.append("=== [3] 四条判据（2L.7；判据 4 本命令不做，见注释）===")
    lines.append("  臂      | 1 长时效更散 | 2 短时效不更散 | 3 增益落在 1-4/5-20 | 4 域均值离散")
    summary = {}
    for name in names[1:]:
        crit = criteria(curves[name], curves[ref], bands[name], bands[ref])
        flat = flatness(k_profile(curves[name]["spread"], curves[ref]["spread"], p_table))
        summary[name] = (crit, flat)
        lines.append("  %-7s | %-12s | %-14s | %-19s | %s"
                     % (name,
                        "%s %.3f" % ("✅" if crit["pass_long"] else "❌", crit["long_ratio"]),
                        "%s %.3f" % ("✅" if crit["pass_short"] else "❌", crit["short_ratio"]),
                        "%s %s" % (_band_symbol(crit), crit["band_kind"]),
                        "未测（用 $P spread 的 domain_spread）"))
    lines.append("  带功率（相对基准，+%d..%d h）：" % LONG)
    for name in names[1:]:
        crit = summary[name][0]
        text = "  ".join("%s=%.3f" % (band, crit["bands"][band])
                         for band in BANDS if band in crit["bands"])
        lines.append("    %-7s %s" % (name, text))
    lines.append("")

    # [4] 判决
    lines.append("=== [4] 判决 ===")
    for name in names[1:]:
        crit, flat = summary[name]
        lines.append("  " + verdict_line(name, flat, crit,
                                         k_profile(curves[name]["spread"],
                                                   curves[ref]["spread"], p_table), needs))
    lines.append("  怎么读：k 平 = 只是把 sigma 乘了个常数（可以换个 sigma 重跑同一个臂，"
                 "不必再扫别的自由度）；k 上升 = 真的改了形状。")
    lines.append("")

    # [5] 均值质量
    lines.append("=== [5] 集合均值质量（全时效 +6..360 h，相对基准）===")
    lines.append("  臂      | rmse            | acc             | act")
    for name in names:
        cells = []
        for key in ("rmse", "acc", "act"):
            mine = bin_mean(curves[name][key], *BINS[-1])
            theirs = bin_mean(curves[ref][key], *BINS[-1])
            cells.append("%9.4f (%+.1f%%)" % (mine, (mine / theirs - 1.0) * 100.0)
                         if name != ref else "%9.4f   ---  " % mine)
        lines.append("  %-7s | %s | %s | %s" % (name, cells[0], cells[1], cells[2]))
    lines.append("  提示：rmse/acc/act 变好而 spread 没变 = 「偏差杠杆」，不是集合收益（见 2M.4-5）")
    return "\n".join(lines), summary, needs


def write_profile_csv(path, names, curves, p_table):
    """把 r/k 剖面落成 CSV：一行 = (臂, 时效, p, spread, r, k)。"""
    from .spread import write_table

    rows = []
    ref = names[0]
    for name in names[1:]:
        for row in k_profile(curves[name]["spread"], curves[ref]["spread"], p_table):
            rows.append([name, row["lead"], "%.6f" % row["p"],
                         "%.6f" % float(curves[name]["spread"].get(row["lead"], float("nan"))),
                         "%.6f" % row["r"], "%.6f" % row["k"]])
    return write_table(path, ["arm", "lead_h", "p", "spread", "r", "k"], rows)


def run_verdict(arms, *, base=DEFAULT_BASE, prefix=DEFAULT_PREFIX, dirs=None, ref=None,
                var=DEFAULT_VAR, region=GLOBAL, p_from=None, p_sigma=P_SIGMA_PAIR,
                out_csv=None, check=True):
    """把一个臂集判一遍。返回 0（报告命令，不改任何东西）。"""
    dirs = dict(dirs or {})
    names = list(arms)
    if ref:
        if ref not in names:
            names.insert(0, ref)
        else:
            names.remove(ref)
            names.insert(0, ref)
    ref_name = names[0]

    paths = {}
    for name in names:
        path = dirs.get(name) or arm_dir(base, prefix, name)
        if not osp.isfile(osp.join(path, "scores.csv")):
            available = sorted(
                item[len(prefix):] for item in os.listdir(base)
                if item.startswith(prefix) and osp.isdir(osp.join(base, item)))
            raise SystemExit(
                "%s：找不到 %s\\scores.csv\n  可用（--base %s）：%s"
                % (name, path, base, ", ".join(available) or "（空）"))
        paths[name] = path

    facts = {name: read_manifest(path) for name, path in paths.items()}
    frames = {name: load_arm(path) for name, path in paths.items()}
    curves = {name: {key: curve(frames[name], key, var, region) for key in SCALARS}
              for name in names}
    bands = {name: band_curve(frames[name], "forecast", var, region) for name in names}

    if p_from:
        p_table = p_from_dirs(p_from[0], p_from[1], p_sigma, var=var)
        p_source = "实测（%s vs %s，sigma 比 %.4g，%d 个时效）" % (
            osp.basename(p_from[0].rstrip("\\/")), osp.basename(p_from[1].rstrip("\\/")),
            p_sigma[1] / p_sigma[0], len(p_table))
    else:
        p_table = dict(P_TABLE)
        p_source = "内置表 %s（用 --p-from 可现场重测）" % P_SOURCE

    if check:
        reference = facts[ref_name]
        for name in names[1:]:
            mine = facts[name]
            for key in ("start", "end", "var", "protocol", "step_hours"):
                if mine.get(key) != reference.get(key):
                    print("⚠ %s 的 %s 与基准 %s 不同（%r vs %r）—— 不是逐位可比，"
                          "结论只能当指示" % (name, key, ref_name,
                                              mine.get(key), reference.get(key)))
    daily = {name: daily_curve(frames[name], "spread", var, region) for name in names}
    days = sorted(daily[ref_name])

    page, _summary, _needs = format_page(names, facts, curves, bands, p_table, p_source, var,
                                         daily=daily, days=days, check=check)
    print(page)
    if out_csv:
        print("\nCSV -> " + write_profile_csv(out_csv, names, curves, p_table))
    return 0
