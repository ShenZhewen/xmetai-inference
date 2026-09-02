#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集合预报评测：集合均值误差、CRPS、Spread 和 Spread/RMSE。"""

import argparse
import logging
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xmetai.util.eval_common import (
    add_common_arguments,
    available_steps,
    build_context,
    ensure_matching_grid,
    ensemble_metrics,
    load_ensemble_predictions,
    load_observations,
    write_results,
)

log = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="集合气象预报评测")
    add_common_arguments(parser, ensemble=True)
    args = parser.parse_args(argv)
    context = build_context(args, ensemble=True)

    rows = []
    total_inits = len(context.init_dirs)
    for init_index, (init_time, init_dir) in enumerate(context.init_dirs, start=1):
        steps = available_steps(context, init_dir, ensemble=True)
        log.info(
            "开始评测起报 %s（%d/%d），已有 %d 步、%d 个成员",
            init_dir, init_index, total_inits, len(steps), len(context.members))
        for step_index, step in enumerate(steps, start=1):
            lead_hour = step * context.interval
            valid_time = init_time + pd.Timedelta(hours=lead_hour)
            predictions, pred_lat, pred_lon = load_ensemble_predictions(
                context, init_dir, step)
            observations, obs_lat, obs_lon = load_observations(
                context, valid_time)
            ensure_matching_grid(pred_lat, pred_lon, obs_lat, obs_lon)
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
            if step_index == 1 or step_index == len(steps) or step_index % 5 == 0:
                log.info(
                    "起报 %s 已完成 %d/%d 步（lead=%dh）",
                    init_dir, step_index, len(steps), lead_hour)

    write_results(rows, context.output_dir, "eval_ens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
