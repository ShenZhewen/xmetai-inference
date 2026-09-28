# -*- coding: utf-8 -*-
"""集合离散度的**标定口径**曲线：σ̂ 随时效 + 差值功率落在哪些尺度上。

与 enshcheck（`$P check`）的分工 —— 两个都要看，但**数字不要混用**：

* `check` 是**自洽性**体检：成员是否互不相同、离散度是否随时效增长。它报的 spread 是
  成员间 `std(ddof=0)` 再空间平均，2 成员时就是 `|d|/2`，比标定口径小约 1.77 倍（高斯场）。
  它只用来看趋势。
* 本模块算**标定口径**：`σ̂² = Σ_m (x_m - x̄)² / (M-1)` 的 cos(lat) 面积加权平均，
  也就是 `spread/rmse` 里那个 spread。σ̂ 是全集合口径，M=2 时退化成 `rms(x1-x2)/√2`。

为什么必须同时看尺度：2026-09-23 的实测里，+12 h 的成员差有 61% 的功率落在 k>180
（< 220 km）的网格尺度上，而场自身的涡动功率 99.99% 在 k<=20 —— 那些尺度上的离散度换不来
CRPS，只会把谱尾部搞脏。所以曲线里同时给「大尺度 σ̂」（只留 k<=20 口径）与 k50。

参考列（带 `*` 的都是**外部数据**，只用于对照，不参与计算，也不随本模块升级而更新）：

* `σ̂/rmse*` 与 `HOU*` 来自 `xmetai-evaluation-v2/report/fgvp_spw_vs_ens_ref`（z500、
  全球、20250702–10 九日窗口）。`HOU*` 是参考集合 `fgvp_hou` 的 spread/rmse 比值带 ——
  9 天窗口下 0.38–1.23；我们的起报窗口是 20250701–03，属**近似同口径**，读的时候当量级用。
* `σ̂/M3*` 是与 M3 对照（全权重、σ_full=0.01）在同一时效的离散度之比。

用法：

    python -m xmetai_inference.perturbation spread <输出目录> [<输出目录或 csv> ...]
    python -m xmetai_inference.perturbation spread out/A out/B --csv out/curves

    # 成员子集：在同一份 51 成员产物上抽 6 成员，量「成员数把 σ̂ 带偏了多少」
    python -m xmetai_inference.perturbation spread <51成员目录> --members 0-5 --steps 12
"""
from __future__ import annotations

import csv
import os
import os.path as osp
import re

import numpy as np

from ._tab import write_table
from .enscheck import (_dataset_dirs, _member_dirs, _pick_steps, _step_files,
                       _read_probe)

KMAX = 20                      # 「大尺度」的纬向波数上限（约 2000 km）
DEFAULT_HOUR = 6.0

# 参考：M3（fgvp_spw，全权重 σ_full=0.01）在 9 日窗口的 spread 与 rmse，z500 全球。
M3_SPREAD = {6: 4.533, 12: 6.449, 18: 8.200, 24: 9.584, 36: 12.255, 48: 15.092,
             72: 21.079, 120: 32.798, 240: 57.546, 360: 110.946}
REF_RMSE = {6: 19.49, 12: 29.98, 18: 33.12, 24: 41.96, 36: 57.57, 48: 77.01,
            72: 125.20, 120: 270.87, 240: 625.44, 360: 724.76}
# 参考带：fgvp_hou（51 成员，348 天归档）的 spread/rmse。
HOU_RATIO = {6: 0.536, 12: 0.378, 18: 0.413, 24: 0.382, 48: 0.399,
             120: 0.507, 240: 0.881, 360: 1.233}

FIELDS = ("lead_h", "sigma", "sigma_large", "rmse_ref", "ratio", "sigma_over_m3",
          "hou_ratio", "syn_frac", "k50", "zonal_frac", "eddy_corr", "domain_spread",
          # 本步**实际**用到了几个成员（各起报取最小）。它可能小于成员目录的个数 ——
          # 2026-09-23 的 A 目录就混进了早期中断 run 的残留成员（只有 5-6 帧），
          # 只报目录个数会看不见这件事。
          "members")


def _area(lat):
    """cos(lat) 权重，和为 1（配合 `_wrms` 里的 /nlon 得到面积加权平均）。"""
    weight = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return weight / weight.sum()


def _wrms(field, weight):
    """面积加权 rms：sqrt(Σ_lat w·Σ_lon x² / nlon)。"""
    return float(np.sqrt((weight[:, None] * field * field).sum() / field.shape[1]))


def _metrics(stack, weight):
    """一步的集合指标。stack: (M, lat, lon)。M>=2。"""
    count = stack.shape[0]
    deviation = stack - stack.mean(axis=0)
    variance = (deviation ** 2).sum(axis=0) / (count - 1)
    out = {"sigma": _wrms(np.sqrt(variance), weight)}

    # 大尺度 σ̂：对偏差场做纬向低通（线性算子，先低通再求方差与先求方差再低通在
    # 「集合标准差」上并不等价，所以这里必须低通**偏差场**而不是低通 σ̂ 本身）。
    spectrum = np.fft.rfft(deviation, axis=2)
    spectrum[:, :, KMAX + 1:] = 0.0
    smooth = np.fft.irfft(spectrum, n=deviation.shape[2], axis=2)
    out["sigma_large"] = _wrms(
        np.sqrt((smooth ** 2).sum(axis=0) / (count - 1)), weight)

    # 功率按纬向波数分布（只用比值，常数因子无关）
    power = (np.abs(np.fft.rfft(deviation, axis=2)) ** 2).sum(axis=0) / (count - 1)
    power = (power * weight[:, None]).sum(axis=0)
    total = power[1:].sum()
    out["syn_frac"] = float(power[1:KMAX + 1].sum() / total) if total else float("nan")
    cumulative = np.cumsum(power)
    out["k50"] = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    out["zonal_frac"] = float(power[0] / power.sum()) if power.sum() else float("nan")

    # 成员间的大尺度形态是否真的分开：去掉各自纬向平均后的两两相关（越低越分开）
    eddy = stack - stack.mean(axis=2, keepdims=True)
    correlations = []
    for first in range(count):
        for second in range(first + 1, count):
            num = float((weight[:, None] * eddy[first] * eddy[second]).sum())
            den = float(np.sqrt((weight[:, None] * eddy[first] ** 2).sum()
                                * (weight[:, None] * eddy[second] ** 2).sum()))
            if den:
                correlations.append(num / den)
    out["eddy_corr"] = float(np.mean(correlations)) if correlations else float("nan")

    # 域平均（面积加权）在成员间的离散度 —— 论文点名的「整场一致偏移」看这个
    means = np.array([float((weight[:, None] * field).sum() / field.shape[1])
                      for field in stack])
    out["domain_spread"] = float(np.sqrt(((means - means.mean()) ** 2).sum() / (count - 1)))
    return out


def _add_reference(row):
    lead = int(round(row["lead_h"]))
    rmse = REF_RMSE.get(lead)
    row["rmse_ref"] = rmse if rmse else ""
    row["ratio"] = (row["sigma"] / rmse) if rmse else ""
    row["hou_ratio"] = HOU_RATIO.get(lead, "")
    m3 = M3_SPREAD.get(lead)
    row["sigma_over_m3"] = (row["sigma"] / m3) if m3 else ""
    return row


_MEMBER_NAME_RE = re.compile(r"^member_(\d+)$")


def _member_tokens(spec) -> list | None:
    """--members 的口径：None / "all" = 全用，其余按逗号切 token。"""
    if spec is None:
        return None
    raw = str(spec).strip().lower()
    if not raw or raw == "all":
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


def select_members(members, spec):
    """从一个起报的成员列表里按 spec 选子集；None/"all" 原样返回。

    `members` 是 `_member_dirs` 的 [(目录名, 路径)]（已按成员号排序）。token 四选一：

      `7`        成员号（= 目录名 member_007 里的 7）
      `member_007` 目录名
      `0-5`      闭区间（按成员号）
      `first:6` / `last:6`  头/尾这么多个（按目录顺序）

    **存在的意义是量「成员数噪声」**：4-6 成员的 σ̂ 由单个成员主导（perturbation/README
    的 2E.5），要知道 51 成员的总体 σ̂ 与 6 成员子集的差多少，只能在**同一份产物**上抽子集
    —— 换个目录跑就不是同一个总体了。
    """
    tokens = _member_tokens(spec)
    if tokens is None:
        return list(members)
    lookup = dict(members)
    names = [name for name, _path in members]
    by_index = {}
    for name in names:
        match = _MEMBER_NAME_RE.match(name)
        if match is None:
            raise SystemExit(f"成员目录名 {name!r} 不合 member_NNN 的口径，用不了 --members")
        by_index[int(match.group(1))] = name

    usage = "（用法：0-5 / 7 / member_007 / first:6 / last:6，逗号分隔）"
    picked = []
    for token in tokens:
        if token.startswith("first:"):
            chosen = names[:int(token[6:])]
        elif token.startswith("last:"):
            count = int(token[5:])
            chosen = names[-count:] if count > 0 else []
        elif re.fullmatch(r"\d+-\d+", token):
            start, stop = (int(part) for part in token.split("-"))
            if stop < start:
                raise SystemExit(f"--members 的区间 {token!r} 是反的")
            chosen = [by_index[key] for key in sorted(by_index) if start <= key <= stop]
        elif re.fullmatch(r"\d+", token):
            key = int(token)
            if key not in by_index:
                raise SystemExit(f"--members 里的成员 {key} 不在产物里{usage}")
            chosen = [by_index[key]]
        elif token in lookup:
            chosen = [token]
        else:
            raise SystemExit(f"--members 读不懂 {token!r}{usage}")
        if not chosen:
            raise SystemExit(f"--members 的 {token!r} 一个成员都没选中{usage}")
        for name in chosen:
            if name not in picked:
                picked.append(name)
    if len(picked) < 2:
        raise SystemExit(
            f"--members 只选中 {len(picked)} 个成员，算不了离散度（至少 2 个）")
    picked.sort(key=names.index)
    return [(name, lookup[name]) for name in picked]


def analyse_run(root: str, *, var=None, init=None, steps_spec="all", stride: int = 1,
                hour_interval: float = DEFAULT_HOUR, verbose: bool = True,
                members=None) -> dict:
    """扫一个推理输出目录，返回逐时效的离散度/尺度曲线。

    `members` 给 None/"all" 以外的值时只算选中的成员子集（见 `select_members`）——
    用于在同一份产物上量「成员数噪声」，**不是**为了省时间（省时间用 --steps/--stride）。
    """
    import xarray as xr

    everything = _dataset_dirs(root)
    if init:
        wanted = {token.strip() for token in str(init).split(",") if token.strip()}
        everything = [item for item in everything if item[0] in wanted]
    if not everything:
        raise SystemExit(f"{root} 下没有找到 {{起报目录}}/member_XXX/ 结构")
    if _member_tokens(members) is not None:
        everything = [(name, path, select_members(items, members))
                      for name, path, items in everything]
        everything = [item for item in everything if len(item[2]) >= 2]
        if not everything:
            raise SystemExit(f"{root}：--members 选完后没有起报目录还剩 >=2 个成员")

    steps = None
    for _name, _path, members in everything:
        files = _step_files(members[0][1])
        steps = set(files) if steps is None else (steps & set(files))
        if not steps:
            raise SystemExit(f"{root}：各起报目录没有共同的可比步号")
    picked = _pick_steps(steps, steps_spec)

    with xr.open_dataset(_step_files(everything[0][2][0][1])[picked[0]]) as ds:
        latitudes = np.asarray(ds["lat"].values, dtype=np.float64)
    if stride and stride > 1:
        latitudes = latitudes[::stride]
    weight = _area(latitudes)

    rows = []
    for step in picked:
        per_init = []
        counts = []
        for _name, _path, members in everything:
            fields = []
            for _member, member_dir in members:
                files = _step_files(member_dir)
                if step not in files:
                    continue
                _picked_var, chunk = _read_probe(files[step], var, stride)
                fields.append(np.asarray(chunk, dtype=np.float64))
            if len(fields) < 2:
                continue
            counts.append(len(fields))
            stack = np.stack(fields)
            if stack.shape[1:] != weight.shape[:1] + (stack.shape[2],):
                raise SystemExit(
                    f"纬度维度对不上：数组 {stack.shape[1:]} vs 权重 {weight.shape}")
            per_init.append(_metrics(stack, weight))
        if not per_init:
            continue
        row = {key: float(np.mean([item[key] for item in per_init]))
               for key in per_init[0]}
        row["step"] = step
        row["lead_h"] = step * hour_interval
        row["inits"] = len(per_init)
        row["members"] = int(min(counts))
        rows.append(_add_reference(row))

    if not rows:
        raise SystemExit(f"{root}：没有任何一步能算出离散度（成员数不足 2？）")
    if verbose:
        directories = len(everything[0][2])
        used = min(row["members"] for row in rows)
        total = len(_member_dirs(everything[0][1]))
        chosen = ("" if total == directories
                  else f"（--members 选中其中 {directories}/{total} 个："
                       + ",".join(name for name, _p in everything[0][2]) + "）")
        print(f"  读 {len(everything)} 个起报；成员目录 {directories} 个、每步实际用到 {used} 个"
              f"{chosen}；{len(picked)} 步（变量 {_picked_var}）")
        if used < directories:
            print(f"  ⚠ 有成员缺帧：{directories} 个成员目录里，只有 {used} 个能覆盖全部步号 —— "
                  f"多出来的多半是别次运行留下的残留（或还没跑完），先清干净再比。")
    return {"root": osp.abspath(root), "rows": rows,
            "var": _picked_var, "inits": len(everything)}


def write_curve_csv(path: str, info: dict) -> str:
    """把一条标定曲线落成 CSV（列 = FIELDS；缺的键写空，多余的键丢掉）。"""
    fields = ["step"] + list(FIELDS)
    return write_table(path, fields,
                       ([row.get(key, "") for key in fields] for row in info["rows"]))


def load_csv(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cooked = []
    for row in rows:
        item = {"step": int(row["step"])}
        for key in FIELDS:
            raw = row.get(key, "")
            if raw in ("", None):
                item[key] = ""
            elif key == "members":
                item[key] = int(float(raw))
            else:
                item[key] = float(raw)
        cooked.append(item)
    return {"root": osp.abspath(path), "rows": cooked,
            "var": "?", "inits": 0}


def _number(value, fmt="{:.2f}", width=8):
    if value in ("", None):
        return " " * (width - 1) + "-"
    return fmt.format(value).rjust(width)


def format_curve(info: dict, *, every=None) -> str:
    rows = info["rows"]
    lines = []
    step_span = rows[-1]["lead_h"] - rows[0]["lead_h"]
    head = (f"{'lead':>6} {'成员':>4} {'σ̂':>8} {'大尺度σ̂':>9} {'σ̂/rmse*':>8} "
            f"{'HOU*':>6} {'σ̂/M3*':>7} {'k≤20':>7} {'k50':>5} {'纬向':>7} "
            f"{'涡动相关':>9} {'域均值σ':>8}")
    lines.append(f"集合离散度曲线：{info['root']}")
    lines.append(f"  变量 {info['var']}；步数 {len(rows)}（+{rows[0]['lead_h']:.0f} .. "
                 f"+{rows[-1]['lead_h']:.0f} h）；口径 σ̂ = 成员间 std(ddof=1) 的面积加权 rms")
    lines.append("")
    lines.append("  " + head)
    chosen = rows
    if every:
        chosen = rows[::max(1, int(every))]
        if rows[-1] not in chosen:
            chosen = chosen + [rows[-1]]
    elif len(rows) > 24:
        index = np.linspace(0, len(rows) - 1, 12).round().astype(int)
        chosen = [rows[i] for i in sorted(set(index))]
    for row in chosen:
        lines.append(
            "  " + f"{row['lead_h']:>6.0f} {row['members']:>4d} {row['sigma']:>8.2f} "
            f"{row['sigma_large']:>9.2f} "
            + _number(row["ratio"]) + " " + _number(row["hou_ratio"], "{:.3f}", 6)
            + " " + _number(row["sigma_over_m3"], "{:.2f}", 7)
            + " " + _number(row["syn_frac"], "{:.1%}", 7)
            + " " + _number(row["k50"], "{:.0f}", 5)
            + " " + _number(row["zonal_frac"], "{:.1%}", 7)
            + " " + _number(row["eddy_corr"], "{:.4f}", 9)
            + " " + _number(row["domain_spread"], "{:.2f}", 8))
    third = max(1, len(rows) // 3)
    early = float(np.mean([row["sigma"] for row in rows[:third]]))
    late = float(np.mean([row["sigma"] for row in rows[-third:]]))
    early_large = float(np.mean([row["sigma_large"] for row in rows[:third]]))
    late_large = float(np.mean([row["sigma_large"] for row in rows[-third:]]))
    lines.append("")
    lines.append(f"  离散度增长：前 1/3 σ̂ {early:.2f} -> 后 1/3 {late:.2f} = {late/early:.2f}x"
                 f"（大尺度 {early_large:.2f} -> {late_large:.2f} = "
                 f"{late_large/early_large:.2f}x）")
    lines.append(f"  尺度迁移：k≤20 占比 {rows[0]['syn_frac']:.1%} -> "
                 f"{rows[-1]['syn_frac']:.1%}；k50 {rows[0]['k50']:.0f} -> "
                 f"{rows[-1]['k50']:.0f}（越小越是大尺度）")
    lines.append(f"  形态分离：涡动相关 {rows[0]['eddy_corr']:.4f} -> "
                 f"{rows[-1]['eddy_corr']:.4f}（越低说明成员的大尺度形态越分开）")
    if late <= early * 1.05:
        lines.append("  ⚠ 离散度基本不增长：噪声没有进自回归回路（对照 check 的判据）")
    if min(row["members"] for row in rows) <= 4:
        lines.append("  ⚠ 有效成员数 <= 4：σ̂ 的采样噪声很大（留一法实测可差 15-35%，见 "
                     "perturbation/README 的 2D 节），百分之几的差别不要当结论")
    lines.append("  注：带 * 的参考列来自外部评测报告（见模块 docstring），只用于对照")
    return "\n".join(lines)


def compare(infos) -> str:
    names = [osp.basename(info["root"].rstrip("\\/")) or info["root"] for info in infos]
    leads = sorted({row["lead_h"] for info in infos for row in info["rows"]})
    lookup = [{row["lead_h"]: row for row in info["rows"]} for info in infos]
    width = max(9, max(len(name) for name in names) + 1)
    lines = ["变体对比（σ̂ / 大尺度σ̂，m^2/s^2）", ""]
    lines.append("  " + f"{'lead':>6} " + " ".join(f"{name:>{width}}" for name in names))
    for lead in leads:
        cells = []
        for table in lookup:
            row = table.get(lead)
            cells.append("-".rjust(width) if row is None
                         else f"{row['sigma']:.2f}/{row['sigma_large']:.2f}".rjust(width))
        lines.append("  " + f"{lead:>6} " + " ".join(cells))
    lines.append("")
    lines.append(f"  {'变体':<{width}} {'后1/3 σ̂':>10} {'前1/3 σ̂':>10} {'增速':>6} "
                 f"{'后1/3 大尺度':>12} {'涡动相关(末)':>12}")
    for name, info in zip(names, infos):
        rows = info["rows"]
        third = max(1, len(rows) // 3)
        early = float(np.mean([row["sigma"] for row in rows[:third]]))
        late = float(np.mean([row["sigma"] for row in rows[-third:]]))
        late_large = float(np.mean([row["sigma_large"] for row in rows[-third:]]))
        lines.append(f"  {name:<{width}} {late:>10.2f} {early:>10.2f} "
                     f"{late/early:>5.2f}x {late_large:>12.2f} "
                     f"{rows[-1]['eddy_corr']:>12.4f}")
    lines.append("")
    lines.append("  怎么读：先看「后1/3 σ̂」与「后1/3 大尺度」是否比基线明显大，"
                 "再看增速（约 1.0 = 离散度基本不增长）")
    lines.append("  与末步涡动相关（越低 = 成员的大尺度形态越分开）。只看 σ̂ 变大不够 —— "
                 "尺度仍在小尺度等于把谱尾部搞得更脏。")
    return "\n".join(lines)


def run_spread(paths, *, var=None, init=None, steps_spec="all", stride: int = 1,
               hour_interval: float = DEFAULT_HOUR, every=None, csv_out=None,
               members=None) -> int:
    infos = []
    for path in paths:
        is_csv = path.lower().endswith(".csv")
        print(("载入 " if is_csv else "分析 ") + osp.abspath(path))
        if is_csv and _member_tokens(members) is not None:
            raise SystemExit("--members 只对推理产物目录有效，不能和曲线 CSV 混用"
                             "（CSV 里的 σ̂ 已经是算好的整集合口径）")
        info = load_csv(path) if is_csv else analyse_run(
            path, var=var, init=init, steps_spec=steps_spec, stride=stride,
            hour_interval=hour_interval, members=members)
        print(format_curve(info, every=every))
        print()
        infos.append(info)

    if len(infos) > 1:
        print(compare(infos))

    if csv_out:
        if csv_out.lower().endswith(".csv"):
            if len(infos) != 1:
                raise SystemExit("--csv 指到具体文件时只能有一个输入；多个输入请给目录")
            targets = [csv_out]
        else:
            os.makedirs(csv_out, exist_ok=True)
            targets = [osp.join(csv_out, osp.basename(info["root"].rstrip("\\/")) + ".csv")
                       for info in infos]
        for target, info in zip(targets, infos):
            write_curve_csv(target, info)
            print(f"CSV -> {target}")
    return 0
