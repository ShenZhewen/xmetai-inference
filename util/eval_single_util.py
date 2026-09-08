#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""确定性预报评测：纬度加权 RMSE、MAE 和 Bias。

实况走 ``xmetai_inference.data`` 的 ``processed_multi_zarr``（与推理同源）：同一套
``unit_convert`` + ``geometry`` 预处理，把 ERA5 store 读成和预测输出同网格、同单位
的二维场，再做纬度加权误差。发现/读取/指标/落盘的公共逻辑都在 ``eval_common``，
确定性与集合两条路径共用一份。

**默认零参数可跑**：forecast 目录、变量、预报步长、实况 store、单位换算全部从
``EVAL_CONFIG``（默认 ``fengqing``）读取，与推理配置天然同步。CLI 参数仍可覆盖。

用法：
    python util/eval_single_util.py                              # 直接评 config 的输出
    XMETAI_EVAL_CONFIG=fuxi21 python util/eval_single_util.py    # 换 config
    python util/eval_single_util.py --forecast /path/to/output --vars z500,t850
    python util/eval_single_util.py --inits 20250102 --steps 20
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.eval_common import (
    _fmt_dur,
    add_common_arguments,
    available_steps,
    build_context,
    defaults_from_config,
    deterministic_metrics,
    ensure_matching_grid,
    load_single_predictions,
    read_observations,
    resolve_stores,
    write_results,
)

log = logging.getLogger(__name__)

# 评测哪个 config 的输出：换模型/换跑法只改这里（或设 XMETAI_EVAL_CONFIG）。
EVAL_CONFIG = os.environ.get("XMETAI_EVAL_CONFIG", "fengqing")


def main(argv=None):
    defaults, config_error = defaults_from_config(EVAL_CONFIG)
    parser = argparse.ArgumentParser(
        description="确定性气象预报评测（基于 xmetai_inference.data）")
    add_common_arguments(parser, ensemble=False, defaults=defaults)
    args = parser.parse_args(argv)
    context = build_context(args, ensemble=False)
    if config_error is not None:
        log.warning("读取 config %r 失败（%s），默认值已退回硬编码",
                    EVAL_CONFIG, config_error)
    else:
        log.info("默认值来自 config %r：forecast=%s interval=%dh",
                 EVAL_CONFIG, defaults["forecast"], defaults["interval"])

    # 1) 展开全部 (起报, step) 工作项，同时收集每个 step 对应的实况时次
    work = []
    valid_set = set()
    total_inits = len(context.init_dirs)
    for init_index, (init_time, init_dir) in enumerate(context.init_dirs, start=1):
        steps = available_steps(context, init_dir, ensemble=False)
        log.info("发现起报 %s（%d/%d），已有 %d 步",
                 init_dir, init_index, total_inits, len(steps))
        for step in steps:
            lead_hour = step * context.interval
            valid_time = init_time + pd.Timedelta(hours=lead_hour)
            work.append((init_time, init_dir, step, lead_hour, valid_time))
            valid_set.add(valid_time)

    # 2) 用 data 层一次读完所有实况时次
    obs_map, obs_lat, obs_lon = read_observations(
        sorted(valid_set),
        context.variables,
        resolve_stores(args, defaults),
        context.interval,
        defaults.get("scales"),
    )

    # 3) 逐工作项评测
    rows = []
    total = len(work)
    log.info("开始评测：工作项=%d（起报=%d）", total, len(context.init_dirs))
    t_start = time.monotonic()
    for index, item in enumerate(work, start=1):
        init_time, init_dir, step, lead_hour, valid_time = item
        if valid_time not in obs_map:
            raise KeyError(
                f"实况缺少时次 {valid_time:%Y%m%d%H}（store 覆盖不足）")
        predictions, pred_lat, pred_lon = load_single_predictions(
            context, init_dir, step)
        ensure_matching_grid(pred_lat, pred_lon, obs_lat, obs_lon)
        observations = obs_map[valid_time]
        for variable in context.variables:
            rows.append({
                "init": init_time.strftime("%Y%m%d%H"),
                "lead_hour": lead_hour,
                "valid_time": valid_time.strftime("%Y%m%d%H"),
                "var": variable,
                **deterministic_metrics(
                    predictions[variable],
                    observations[variable],
                    pred_lat,
                ),
            })
        if index == 1 or index == total or index % 10 == 0:
            elapsed = time.monotonic() - t_start
            rate = index / elapsed                          # 项/秒
            eta = (total - index) / rate                    # 剩余秒
            log.info(
                "已评测 %d/%d（%.1f%%）｜init=%s step=%03d lead=%dh｜"
                "已用 %s，剩余约 %s（%.3f it/s）",
                index, total, 100.0 * index / total,
                init_dir, step, lead_hour,
                _fmt_dur(elapsed), _fmt_dur(eta), rate,
            )

    write_results(rows, context.output_dir, "eval_single")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
