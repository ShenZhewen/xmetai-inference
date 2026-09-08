#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集合预报评测：集合均值误差、CRPS、Spread 和 Spread/RMSE。

实况走 ``xmetai_inference.data`` 的 ``processed_multi_zarr``（与推理同源），
不再用已删除的 ``xmetai_inference.loaders``。

**默认零参数可跑**：forecast 目录、变量、预报步长、实况 store、单位换算全部从
``EVAL_CONFIG``（默认 ``fuxi_ens``）读取。所有 CLI 参数仍可覆盖。

用法：
    python util/eval_ens_util.py
    XMETAI_EVAL_CONFIG=fgvp python util/eval_ens_util.py
    python util/eval_ens_util.py --forecast /path/to/output --members 10
"""

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
    add_common_arguments,
    available_steps,
    build_context,
    defaults_from_config,
    ensemble_metrics,
    ensure_matching_grid,
    load_ensemble_predictions,
    read_observations,
    resolve_stores,
    write_results,
)

log = logging.getLogger(__name__)


def _fmt_dur(seconds: float) -> str:
    """把秒数转成 '1h02m05s' / '2m26s' / '35s' 的可读时长。"""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# 评测哪个 config 的输出：换模型/换跑法只改这里（或设 XMETAI_EVAL_CONFIG）。
EVAL_CONFIG = os.environ.get("XMETAI_EVAL_CONFIG", "fuxi_ens")


def main(argv=None):
    defaults, config_error = defaults_from_config(EVAL_CONFIG)
    parser = argparse.ArgumentParser(description="集合气象预报评测")
    add_common_arguments(parser, ensemble=True, defaults=defaults)
    args = parser.parse_args(argv)
    context = build_context(args, ensemble=True)
    if config_error is not None:
        log.warning("读取 config %r 失败（%s），默认值已退回硬编码",
                    EVAL_CONFIG, config_error)
    else:
        log.info("默认值来自 config %r：forecast=%s interval=%dh",
                 EVAL_CONFIG, defaults["forecast"], defaults["interval"])

    # 先展开全部 (起报, step) 工作项并收集实况时次，再一次性批量读实况。旧实现
    # 在循环里逐 valid_time 调 load_observations（每次重开 store），慢且与推理
    # 不同源。
    work = []
    valid_set = set()
    total_inits = len(context.init_dirs)
    for init_index, (init_time, init_dir) in enumerate(context.init_dirs, start=1):
        steps = available_steps(context, init_dir, ensemble=True)
        log.info(
            "发现起报 %s（%d/%d），已有 %d 步、%d 个成员",
            init_dir, init_index, total_inits, len(steps), len(context.members))
        for step in steps:
            lead_hour = step * context.interval
            work.append((
                init_time,
                init_dir,
                step,
                lead_hour,
                init_time + pd.Timedelta(hours=lead_hour),
            ))
            valid_set.add(init_time + pd.Timedelta(hours=lead_hour))

    obs_map, obs_lat, obs_lon = read_observations(
        sorted(valid_set),
        context.variables,
        resolve_stores(args, defaults),
        context.interval,
        defaults.get("scales"),
    )

    rows = []
    total = len(work)
    log.info("开始评测：工作项=%d（起报=%d × 步数，成员=%d）",
             total, total_inits, len(context.members))
    t_start = time.monotonic()
    for index, item in enumerate(work, start=1):
        init_time, init_dir, step, lead_hour, valid_time = item
        if valid_time not in obs_map:
            raise KeyError(
                f"实况缺少时次 {valid_time:%Y%m%d%H}（store 覆盖不足）")
        predictions, pred_lat, pred_lon = load_ensemble_predictions(
            context, init_dir, step)
        ensure_matching_grid(pred_lat, pred_lon, obs_lat, obs_lon)
        observations = obs_map[valid_time]
        for variable in context.variables:
            rows.append({
                "init": init_time.strftime("%Y%m%d%H"),
                "lead_hour": lead_hour,
                "valid_time": valid_time.strftime("%Y%m%d%H"),
                "var": variable,
                "members": len(context.members),
                **ensemble_metrics(
                    predictions[variable],
                    observations[variable],
                    pred_lat,
                    pred_lon,
                    variable,
                ),
            })
        if index == 1 or index == total or index % 10 == 0:
            elapsed = time.monotonic() - t_start
            rate = index / elapsed                         # 项/秒
            eta = (total - index) / rate                   # 剩余秒
            log.info(
                "已评测 %d/%d（init=%s step=%03d lead=%dh）｜"
                "已用 %s，剩余约 %s（%.3f it/s）",
                index, total, init_dir, step, lead_hour,
                _fmt_dur(elapsed), _fmt_dur(eta), rate,
            )

    write_results(rows, context.output_dir, "eval_ens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
