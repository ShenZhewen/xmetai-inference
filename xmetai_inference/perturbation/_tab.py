# -*- coding: utf-8 -*-
"""CSV 落盘的公共样板：只有「开文件 + 写表头 + 写行」这一段。

spread 与 verdict 都要写 CSV，但**行的构造逻辑完全不同**——一个是标定曲线的字段字典，
一个是臂 x 时效的 r/k 剖面。所以这里只收这段样板，不强行统一两个 writer。
"""
from __future__ import annotations

import csv


def write_table(path: str, header, rows) -> str:
    """写一张 CSV：`header` 是列名序列，`rows` 是「每行一个序列」。返回 path。"""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
    return path

