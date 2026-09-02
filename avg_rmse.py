#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 evaluate.log 抽「指定 lead 的指定变量 RMSE」，各算一个平均。

日志行（evaluate.py 的 log.debug 格式，见 evaluate.py 的
`log.debug("  %-5s lead %3dh valid %s  RMSE=%.4f ...")`）形如：

    [10:26:30] DEBUG     z500  lead   6h valid 2025010306  RMSE=17.4336 MAE=13.5648 CRPS=- SSR=-

同一变量在同一 lead 下，每个起报时间各打一行；本脚本把这些行的 RMSE 平均起来
（跨起报取平均，只看指定的那一个 lead，不混入其他 lead）。

用法：
    python avg_rmse.py evaluate.log                                   # 默认 lead=360h、z500 tp
    python avg_rmse.py evaluate.log --lead 360 --vars z500 tp
    python avg_rmse.py evaluate.log --lead 360 --vars z500,tp         # 也支持逗号
    python avg_rmse.py evaluate.log --lead 6 --vars z500 u850 v850
"""
import argparse
import re

# 匹配一行评测日志里的 var / lead(小时) / RMSE。RMSE 由 %.4f 输出，恒为普通小数。
LINE_RE = re.compile(
    r"DEBUG\s+(?P<var>\S+)\s+lead\s+(?P<lead>\d+)h\s+valid\s+\S+\s+"
    r"RMSE=(?P<rmse>[0-9.]+)"
)


def main():
    ap = argparse.ArgumentParser(description="抽某 lead 的某变量平均 RMSE")
    ap.add_argument("log", help="evaluate.log 路径")
    ap.add_argument("--lead", type=int, default=360, help="lead 小时数（默认 360）")
    ap.add_argument("--vars", nargs="+", default=["z500", "tp"],
                    help="要统计的变量，空格或逗号分隔（默认 z500 tp）")
    args = ap.parse_args()

    # 支持 "--vars z500 tp" 与 "--vars z500,tp" 两种写法，拍平成一个列表
    vars_ = []
    for v in args.vars:
        vars_.extend(x.strip() for x in v.split(",") if x.strip())

    want = set(vars_)
    acc = {v: [] for v in vars_}

    with open(args.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            var = m.group("var")
            if var not in want or int(m.group("lead")) != args.lead:
                continue
            acc[var].append(float(m.group("rmse")))

    for v in vars_:            # 按用户给定顺序输出
        vals = acc[v]
        if not vals:
            print(f"{v:<6} lead {args.lead:>3}h: 无数据")
            continue
        mean = sum(vals) / len(vals)
        print(f"{v:<6} lead {args.lead:>3}h: 平均 RMSE = {mean:.4f}"
              f"  (N={len(vals)}, min={min(vals):.4f}, max={max(vals):.4f})")


if __name__ == "__main__":
    main()
