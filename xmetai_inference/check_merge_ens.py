#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体检 + 聚合 fgvp_ens 的多份输出目录。

目录结构（每份 output 目录下）：
    {output}/{YYYYMMDD}/member_{NNN}/{步号:03d}.nc

几份 output 装的是**不同起报日**。这个脚本做三件事：
  1) 逐日体检：成员目录齐不齐、每个成员的步文件齐不齐、有没有 0 字节残件；
  2) 报告跨目录的**重复起报日**（同一个日期出现在两份里 —— 聚合前必须先知道）；
  3) 只把**完整**的日期聚合进 DEST。残缺的原样留在原地，一个字节都不碰。

缺省**只体检，不动任何文件**。真聚合要显式给 --copy 或 --move。

用法：
    python check_merge_ens.py                 # 只体检
    python check_merge_ens.py --copy          # 完整日期**复制**到 DEST（源目录保留）
    python check_merge_ens.py --move          # 完整日期**移动**到 DEST（源目录里会消失）
    python check_merge_ens.py -v              # 残缺日期也逐条列出缺口明细

这个脚本不依赖 xmetai_inference 包，扔到服务器上任意位置都能跑。
"""
import argparse
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── 改这里 ────────────────────────────────────────────────────────────────
SRC_DIRS = [
    "/workspace/data/shenzw/fgvp_ens_output",
    "/workspace/data/shenzw/fgvp_ens_output_2",
    "/workspace/data/shenzw/fgvp_ens_output_3",
    "/workspace/data/shenzw/fgvp_ens_output_4",
]
DEST = "/workspace/data/shenzw/fgvp_ens_merged"
MEMBERS = 51      # member_000 .. member_050
STEPS = 60        # 001.nc .. 060.nc
# ──────────────────────────────────────────────────────────────────────────

DATE_RE = re.compile(r"^\d{8}$")
MEMBER_RE = re.compile(r"^member_\d{3}$")

EXPECTED_MEMBERS = [f"member_{i:03d}" for i in range(MEMBERS)]
EXPECTED_STEPS = [f"{i:03d}.nc" for i in range(1, STEPS + 1)]


def scan_date(date_dir):
    """体检一个起报日目录，返回一个结果 dict。"""
    present = {p.name for p in date_dir.iterdir()
               if p.is_dir() and MEMBER_RE.match(p.name)}
    miss_members = [m for m in EXPECTED_MEMBERS if m not in present]

    miss_steps = {}     # member -> [缺的步文件名]
    empty = []          # "member/step" —— 0 字节残件
    sizes = []
    for m in EXPECTED_MEMBERS:
        if m not in present:
            continue
        md = date_dir / m
        have = {p.name for p in md.iterdir() if p.is_file()}
        gone = [s for s in EXPECTED_STEPS if s not in have]
        if gone:
            miss_steps[m] = gone
        for s in EXPECTED_STEPS:
            if s in have:
                n = (md / s).stat().st_size
                sizes.append(n)
                if n == 0:
                    empty.append(f"{m}/{s}")

    return {
        "members": len(present),
        "miss_members": miss_members,
        "miss_steps": miss_steps,
        "empty": empty,
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "ok": not miss_members and not miss_steps and not empty,
    }


def describe(date, r):
    """一行摘要。"""
    if r["ok"]:
        return f"{date}  OK     {r['members']:>3} 成员 x {STEPS} 步"

    bits = []
    if r["miss_members"]:
        bits.append(f"缺 {len(r['miss_members'])} 个成员目录")
    if r["miss_steps"]:
        total = sum(len(v) for v in r["miss_steps"].values())
        bits.append(f"{len(r['miss_steps'])} 个成员共缺 {total} 个步文件")
    if r["empty"]:
        bits.append(f"{len(r['empty'])} 个 0 字节文件")
    return f"{date}  残缺   {r['members']:>3} 成员 —— " + "；".join(bits)


def detail(date, r, limit=6):
    """残缺日期的缺口明细（-v 时打印）。"""
    lines = []
    if r["miss_members"]:
        head = ", ".join(r["miss_members"][:limit])
        more = f" …（共 {len(r['miss_members'])} 个）" if len(r["miss_members"]) > limit else ""
        lines.append(f"        缺成员目录：{head}{more}")
    for m, gone in list(r["miss_steps"].items())[:limit]:
        head = ", ".join(gone[:limit])
        more = f" …（共 {len(gone)} 个）" if len(gone) > limit else ""
        lines.append(f"        {m} 缺步文件：{head}{more}")
    if len(r["miss_steps"]) > limit:
        lines.append(f"        …另有 {len(r['miss_steps']) - limit} 个成员也有缺口")
    if r["empty"]:
        head = ", ".join(r["empty"][:limit])
        more = f" …（共 {len(r['empty'])} 个）" if len(r["empty"]) > limit else ""
        lines.append(f"        0 字节：{head}{more}")
    return lines


def report_runs(dates, title):
    """把日期排好，标出中间的断档。"""
    if not dates:
        print(f"\n{title}：无")
        return
    ds = sorted(datetime.strptime(d, "%Y%m%d") for d in dates)
    runs = []
    start = prev = ds[0]
    for d in ds[1:]:
        if (d - prev).days != 1:
            runs.append((start, prev))
            start = d
        prev = d
    runs.append((start, prev))

    print(f"\n{title}：{len(ds)} 个，{ds[0]:%Y%m%d} .. {ds[-1]:%Y%m%d}")
    if len(runs) == 1:
        print(f"  连续无断档（{(ds[-1] - ds[0]).days + 1} 天）")
        return
    print(f"  分成 {len(runs)} 段（段间是断档）：")
    for a, b in runs:
        print(f"    {a:%Y%m%d} .. {b:%Y%m%d}   共 {(b - a).days + 1} 天")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--copy", action="store_true", help="完整日期复制到 DEST")
    g.add_argument("--move", action="store_true", help="完整日期移动到 DEST")
    ap.add_argument("--dest", default=DEST, help=f"聚合目标目录（缺省 {DEST}）")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="残缺日期也列出缺口明细")
    args = ap.parse_args()

    srcs = [Path(d) for d in SRC_DIRS]
    for d in srcs:
        if not d.is_dir():
            sys.exit(f"源目录不存在：{d}")

    complete = {}      # date -> [src, ...]（该日期在哪些源目录里是完整的）
    incomplete = {}    # date -> [src, ...]

    print("=" * 74)
    for src in srcs:
        dates = sorted(p.name for p in src.iterdir()
                       if p.is_dir() and DATE_RE.match(p.name))
        print(f"\n{src}")
        print(f"  共 {len(dates)} 个起报日")
        if not dates:
            continue
        for date in dates:
            r = scan_date(src / date)
            print("  " + describe(date, r))
            if r["ok"]:
                complete.setdefault(date, []).append(src)
            else:
                incomplete.setdefault(date, []).append(src)
                if args.verbose:
                    for line in detail(date, r):
                        print(line)

    # ── 跨目录重复 ────────────────────────────────────────────────────────
    dup_ok = {d: s for d, s in complete.items() if len(s) > 1}
    dup_bad = {d: s for d, s in incomplete.items() if len(s) > 1}
    if dup_ok or dup_bad:
        print("\n" + "!" * 74)
        print("同一个起报日出现在多份目录里（聚合前先确认是不是重复跑）：")
        for d, s in sorted({**dup_bad, **dup_ok}.items()):
            tag = "完整" if d in dup_ok else "残缺"
            print(f"  {d}  [{tag}]  " + "  ".join(str(p) for p in s))
        print("!" * 74)

    # ── 汇总 ──────────────────────────────────────────────────────────────
    all_dates = set(complete) | set(incomplete)
    print("\n" + "=" * 74)
    print(f"起报日合计（去重）：{len(all_dates)} 个 "
          f"—— 完整 {len(complete)}，残缺 {len(incomplete)}")
    if incomplete:
        print("残缺（原样留在原地，不会被动）：")
        for d in sorted(incomplete):
            print(f"  {d}  ({', '.join(p.name for p in incomplete[d])})")

    report_runs(sorted(complete), "完整日期")
    if incomplete:
        report_runs(sorted(all_dates), "全部日期（含残缺）")

    if not (args.copy or args.move):
        print("\n（只体检，没有动任何文件。要聚合请加 --copy 或 --move）")
        return 0

    # ── 聚合 ──────────────────────────────────────────────────────────────
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    verb = "移动" if args.move else "复制"

    todo = []
    for date in sorted(complete):
        srcs_for_date = complete[date]
        if len(srcs_for_date) > 1:
            print(f"  跳过 {date}：{len(srcs_for_date)} 份里都有，无法自动挑一份")
            continue
        if (dest / date).exists():
            print(f"  跳过 {date}：{dest / date} 已存在")
            continue
        todo.append((date, srcs_for_date[0]))

    print(f"\n将{verb} {len(todo)} 个完整日期 -> {dest}")
    if args.move:
        print("  ⚠ --move：源目录里这些日期会消失")
    done = 0
    for i, (date, src) in enumerate(todo, 1):
        s, d = src / date, dest / date
        try:
            if args.move:
                shutil.move(str(s), str(d))
            else:
                shutil.copytree(s, d)
        except Exception as exc:                      # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {date} 失败：{exc}")
            continue
        done += 1
        print(f"  [{i}/{len(todo)}] {date} <- {src.name}")

    print(f"\n完成：{verb} {done}/{len(todo)} 个日期 -> {dest}")
    if done < len(todo):
        print("  有失败项，见上面。源目录未动的日期仍然是完整的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
