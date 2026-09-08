# -*- coding: utf-8 -*-
"""确定性与集合预报评测的共用发现、读取、指标和输出逻辑。"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from xmetai_inference.data import create_dataset_source
from xmetai_inference.logging_util import configure_logging
from xmetai_inference.models import GRID_025


log = logging.getLogger(__name__)
_INIT_PATTERN = re.compile(r"^\d{8}(?:\d{2})?$")
_STEP_PATTERN = re.compile(r"^(\d{3})\.nc$")
PRECIP_THRESHOLDS = (0.1, 4.0, 13.0, 25.0)
CHINA_LAT_RANGE = (15.0, 55.0)
CHINA_LON_RANGE = (70.0, 140.0)


ERA5_ROOT = os.environ.get(
    "ERA5_STORE_ROOT", "/workspace/data/liujunjie/era5_foundation_store2"
)
# 仅当 config 读不到时的兜底 store 名。
_FALLBACK_STORE_NAMES = [
    "era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    "era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
]


def _fmt_dur(seconds):
    """秒 -> '1h02m05s' / '2m26s' / '35s' 的可读时长。"""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class _TruthModel:
    """create_dataset_source 的最小 model_cls：只声明网格与通道，不做推理。

    input_channels 在发现评测变量后动态赋值；grid 复用 0.25° 全球网格。
    """

    grid = GRID_025
    input_channels: tuple = ()


def defaults_from_config(config_name):
    """读 config 的 output_dir/vars/hour_interval/dataset 作为评测默认值。

    返回 (defaults, error)。error 非 None 时 defaults 是硬编码兜底 —— 调用方应
    在 configure_logging 之后把它打出来（这里 log 还没配好，warning 会被丢掉）。

    默认值从 config 读而不是在评测脚本里抄一份：抄一份的下场已经踩过 —— config
    的 output_dir 写 ``fengqing_single_output2``、评测默认值写
    ``fengqing_single_output``（少个 2），不显式传 --forecast 就静默评到另一个
    目录里的旧结果，指标看着「没变化」。单位换算抄错更危险：推理按 tp×6000 跑、
    评测按别的读，预测与实况不在同一单位上，RMSE 全错且不报错。
    """
    fallback = {
        "forecast": None,
        "vars": None,
        "interval": 6,
        "stores": None,
        "scales": None,
    }
    try:
        from xmetai_inference.configs.base import load_config
        from xmetai_inference.processing.tensor_processors import (
            unit_scales_from_specs,
        )

        cfg = load_config(config_name)
    except Exception as error:  # noqa: BLE001
        return fallback, error

    stores = None
    scales = None
    dataset_spec = getattr(cfg, "dataset", None)
    if isinstance(dataset_spec, dict):
        paths = dataset_spec.get("paths")
        if paths:
            stores = [os.fspath(path) for path in paths]
        specs = dataset_spec.get("processors") or ()
        found = unit_scales_from_specs(specs)
        if found:
            scales = found
    return {
        "forecast": cfg.output_dir or None,
        "vars": cfg.vars or None,
        "interval": cfg.hour_interval or 6,
        "stores": stores,
        "scales": scales,
    }, None


def resolve_stores(args, defaults=None):
    """优先级：--stores > --data-root > config 的 dataset.paths > 硬编码兜底。"""
    if getattr(args, "stores", None):
        return [os.fspath(path) for path in args.stores]
    if getattr(args, "data_root", None):
        return [
            os.path.join(args.data_root, name)
            for name in _FALLBACK_STORE_NAMES
        ]
    from_config = (defaults or {}).get("stores")
    if from_config:
        return list(from_config)
    return [
        os.path.join(ERA5_ROOT, name) for name in _FALLBACK_STORE_NAMES
    ]


@dataclass
class EvaluationContext:
    forecast_root: str
    output_dir: str
    init_dirs: list[tuple[pd.Timestamp, str]]
    variables: list[str]
    interval: int
    step_limit: int | None
    members: list[int]


def add_common_arguments(parser: argparse.ArgumentParser, *, ensemble: bool, defaults=None) -> None:
    """Add evaluation arguments; ``defaults`` can pre-fill common values.

    ``defaults`` keys use the argparse destination names, e.g.
    ``forecast``, ``loader``, ``steps``, ``vars``, ``interval``.
    """
    defaults = dict(defaults or {})

    parser.add_argument(
        "--forecast",
        # 按「有没有可用值」判定，不能按「键在不在」——config 读到但 output_dir
        # 为空时值是 None，argparse 不会要求传参，最后在 os.path.isdir(None) 崩。
        required=defaults.get("forecast") is None,
        default=defaults.get("forecast"),
        help="预测输出根目录（缺省读 config 的 output_dir）",
    )
    parser.add_argument(
        "--stores",
        nargs="*",
        default=defaults.get("stores"),
        help="实况 zarr store 路径（可多个）；缺省用 --data-root 拼默认 pl/sfc",
    )
    parser.add_argument(
        "--data-root",
        default=defaults.get("data_root"),
        help="实况 store 根目录（与 --stores 二选一）",
    )
    parser.add_argument(
        "--inits",
        default=defaults.get("inits"),
        help="可选：只评指定起报，逗号分隔 YYYYMMDD/ YYYYMMDDHH",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=defaults.get("steps"),
        help="可选：最多评前 N 步；缺省评测目录中所有已有步骤",
    )
    parser.add_argument(
        "--vars",
        default=defaults.get("vars"),
        help="可选：只评指定变量，逗号分隔；缺省读取预测文件中的全部变量",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=defaults.get("interval", 6),
        help="相邻预测步的小时数（默认 6）",
    )
    parser.add_argument(
        "--out",
        default=defaults.get("out"),
        help="CSV 和日志目录；缺省为预测目录下的 evaluation",
    )
    if ensemble:
        parser.add_argument(
            "--members",
            type=int,
            default=defaults.get("members"),
            help="可选：只使用前 N 个成员；缺省自动发现全部 member_*",
        )
    parser.add_argument(
        "--log-level",
        default=defaults.get("log_level", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )


def _parse_init_dir(name):
    if not _INIT_PATTERN.fullmatch(name):
        return None
    fmt = "%Y%m%d" if len(name) == 8 else "%Y%m%d%H"
    return pd.to_datetime(name, format=fmt)


def _discover_inits(forecast_root, requested):
    discovered = {}
    for name in os.listdir(forecast_root):
        path = os.path.join(forecast_root, name)
        init_time = _parse_init_dir(name)
        if init_time is not None and os.path.isdir(path):
            discovered[name] = init_time
    if not discovered:
        raise SystemExit(
            f"{forecast_root} 下没有 YYYYMMDD 或 YYYYMMDDHH 起报目录")

    if requested:
        selected = []
        for token in requested.split(","):
            token = token.strip()
            if not token:
                continue
            if token in discovered:
                selected.append((discovered[token], token))
                continue
            if len(token) == 10 and token.endswith("00") and token[:8] in discovered:
                selected.append((discovered[token[:8]], token[:8]))
                continue
            raise SystemExit(f"预测目录中找不到起报 {token}")
        if not selected:
            raise SystemExit("--inits 没有包含有效起报时间")
        return sorted(set(selected))
    return sorted((value, key) for key, value in discovered.items())


def _discover_members(forecast_root, init_dirs, limit):
    common = None
    for _, init_dir in init_dirs:
        root = os.path.join(forecast_root, init_dir)
        current = {
            int(match.group(1))
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
            and (match := re.fullmatch(r"member_(\d{3})", name))
        }
        common = current if common is None else common & current
    members = sorted(common or ())
    if limit is not None:
        if limit < 2:
            raise SystemExit("--members 必须 >= 2")
        members = members[:limit]
    if len(members) < 2:
        raise SystemExit("集合预测目录中至少需要两个共同的 member_* 目录")
    return members


def _discover_steps(directory):
    steps = []
    for name in os.listdir(directory):
        match = _STEP_PATTERN.fullmatch(name)
        if match and os.path.isfile(os.path.join(directory, name)):
            steps.append(int(match.group(1)))
    return sorted(steps)


def available_steps(context, init_dir, *, ensemble):
    if ensemble:
        common = None
        for member in context.members:
            directory = os.path.join(
                context.forecast_root, init_dir, f"member_{member:03d}")
            current = set(_discover_steps(directory))
            common = current if common is None else common & current
        steps = sorted(common or ())
    else:
        steps = _discover_steps(os.path.join(context.forecast_root, init_dir))
    if context.step_limit is not None:
        steps = [step for step in steps if step <= context.step_limit]
    if not steps:
        raise FileNotFoundError(f"起报目录 {init_dir} 没有可评测的预测步骤")
    return steps


def _first_forecast_file(forecast_root, init_dirs, members):
    for _, init_dir in init_dirs:
        directory = os.path.join(forecast_root, init_dir)
        if members:
            directory = os.path.join(directory, f"member_{members[0]:03d}")
        steps = _discover_steps(directory)
        if steps:
            return os.path.join(directory, f"{steps[0]:03d}.nc")
    raise SystemExit("预测目录中没有可读取的 NetCDF 文件")


def _discover_variables(path, requested):
    with xr.open_dataset(path) as dataset:
        available = {
            str(name).lower(): str(name)
            for name in dataset.data_vars
            if dataset[name].ndim == 2
        }
    if not available:
        raise SystemExit(f"预测文件没有二维气象变量：{path}")
    if not requested:
        return list(available)
    variables = [
        value.strip().lower()
        for value in requested.split(",")
        if value.strip()
    ]
    missing = [value for value in variables if value not in available]
    if missing:
        raise SystemExit(
            f"预测文件缺少变量：{', '.join(missing)}；"
            f"现有 {', '.join(available)}")
    return variables


def build_context(args, *, ensemble: bool) -> EvaluationContext:
    if not os.path.isdir(args.forecast):
        raise SystemExit(f"预测目录不存在：{args.forecast}")
    if args.interval <= 0:
        raise SystemExit("--interval 必须 > 0")
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps 必须 > 0")

    init_dirs = _discover_inits(args.forecast, args.inits)
    members = _discover_members(
        args.forecast, init_dirs, args.members) if ensemble else []
    first_file = _first_forecast_file(args.forecast, init_dirs, members)
    variables = _discover_variables(first_file, args.vars)

    output_dir = args.out or os.path.join(args.forecast, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    configure_logging(
        level=args.log_level,
        log_file=os.path.join(
            output_dir, "eval_ens.log" if ensemble else "eval_single.log"),
    )
    log.info(
        "评测发现：起报=%d 变量=%s%s",
        len(init_dirs),
        ",".join(variables),
        f" 成员={len(members)}" if ensemble else "",
    )
    return EvaluationContext(
        forecast_root=args.forecast,
        output_dir=output_dir,
        init_dirs=init_dirs,
        variables=variables,
        interval=args.interval,
        step_limit=args.steps,
        members=members,
    )


def _align_regular_grid(values, latitudes, longitudes):
    values = np.asarray(values, dtype=np.float64)
    latitudes = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    longitudes = np.asarray(longitudes, dtype=np.float64).reshape(-1)
    if values.shape[-2:] != (latitudes.size, longitudes.size):
        raise ValueError(
            f"数据 shape {values.shape[-2:]} 与坐标 "
            f"{(latitudes.size, longitudes.size)} 不一致")
    if latitudes.size > 1 and latitudes[0] < latitudes[-1]:
        latitudes = latitudes[::-1]
        values = np.flip(values, axis=-2)
    normalized_lon = np.mod(longitudes, 360.0)
    order = np.argsort(normalized_lon)
    return (
        np.take(values, order, axis=-1),
        latitudes,
        normalized_lon[order],
    )


def read_observations(valid_times, variables, stores, interval, scales):
    """用 processed_multi_zarr 一次读完所有实况时次（确定性/集合共用）。

    处理链只有 unit_convert + geometry：不 normalize，读出来就是与预测同单位的
    物理量；geometry 保证北->南、lon 0:360。``scales`` 必须与推理 config 的
    ``dataset.processors[unit_convert].scales`` 一致 —— 推理把 store 换算成什么
    单位，评测读实况就得用同一套，否则预测与实况不在同一单位上，RMSE 全错且
    不报错。

    返回 (obs_map, lat, lon)：obs_map 以 pd.Timestamp 为键，值是
    {变量名: (lat, lon) float64 数组}。

    这里是批量读（一次构造 Dataset 覆盖全部时次），不是逐时次按需读 —— 旧的
    loader 路径是后者，每个 valid_time 重新开一次 store，慢且与推理不同源。
    """
    if not valid_times:
        raise SystemExit("没有可评测的实况时次")
    _TruthModel.input_channels = tuple(variables)
    try:
        source = create_dataset_source(
            {
                "type": "processed_multi_zarr",
                "paths": list(stores),
                "processors": [
                    {"name": "unit_convert", "scales": dict(scales or {})},
                    {"name": "geometry"},
                ],
            },
            model_cls=_TruthModel,
            init_times=list(valid_times),
            history_steps=1,
            hour_interval=interval,
            dataloader={"batch_size": 1, "num_workers": 0, "pin_memory": False},
        )
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"构建实况数据源失败（检查 --stores/--data-root 与 store 通道命名）：{error}"
        ) from error

    channels = list(source.channel_names)
    obs = {}
    total = len(valid_times)
    done = 0
    t_start = time.monotonic()
    # 起报多时这一段批量读实况是耗时大头，之前没任何输出，看着像卡死，加进度 + ETA。
    report_every = max(1, total // 20)
    log.info(
        "开始读实况：%d 个时次（store：%s）",
        total,
        ", ".join(os.path.basename(path) for path in stores),
    )
    for batch in source:
        inputs = batch["inputs"]
        times = batch["times"]
        if hasattr(inputs, "detach"):
            inputs = inputs.detach().cpu().numpy()
        inputs = np.asarray(inputs, dtype=np.float64)
        times = (
            times.detach().cpu().numpy()
            if hasattr(times, "detach")
            else np.asarray(times)
        )
        for index in range(inputs.shape[0]):
            timestamp = pd.Timestamp(int(times[index]))
            frame = inputs[index, 0]  # [C, H, W]，history_steps=1 取单帧
            obs[timestamp] = {
                channel: frame[c]
                for c, channel in enumerate(channels)
            }
        done += inputs.shape[0]
        if done == total or done % report_every == 0:
            elapsed = time.monotonic() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else 0.0
            log.info(
                "读实况 %d/%d（%.0f%%）｜已用 %s，剩余约 %s（%.2f it/s）",
                done, total, 100.0 * done / total,
                _fmt_dur(elapsed), _fmt_dur(eta), rate,
            )

    if len(obs) != len(valid_times):
        log.warning(
            "实况只读到 %d/%d 个时次（部分有效时次在 store 中缺失）",
            len(obs), len(valid_times),
        )
    log.info("实况读取完成：%d 个时次（共 %d）", len(obs), total)
    return (
        obs,
        np.asarray(source.latitudes, dtype=np.float64),
        np.asarray(source.longitudes, dtype=np.float64),
    )


def _dataset_variable(dataset, variable):
    names = {str(name).lower(): name for name in dataset.data_vars}
    if variable not in names:
        raise KeyError(f"预测文件缺少变量 {variable!r}")
    return names[variable]


def load_prediction_file(path, variables):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"预测文件不存在：{path}")
    with xr.open_dataset(path) as dataset:
        if "lat" not in dataset.coords or "lon" not in dataset.coords:
            raise ValueError(f"{path} 缺少 lat/lon 坐标")
        predictions = {}
        expected_lat = expected_lon = None
        for variable in variables:
            name = _dataset_variable(dataset, variable)
            values = np.asarray(dataset[name].values, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"{path} 的 {name} 不是二维规则网格")
            values, current_lat, current_lon = _align_regular_grid(
                values, dataset["lat"].values, dataset["lon"].values)
            predictions[variable] = values
            expected_lat, expected_lon = current_lat, current_lon
        return predictions, expected_lat, expected_lon


def load_single_predictions(context, init_dir, step):
    path = os.path.join(
        context.forecast_root, init_dir, f"{step:03d}.nc")
    return load_prediction_file(path, context.variables)


def load_ensemble_predictions(context, init_dir, step):
    arrays = {variable: [] for variable in context.variables}
    expected_lat = expected_lon = None
    for member in context.members:
        path = os.path.join(
            context.forecast_root,
            init_dir,
            f"member_{member:03d}",
            f"{step:03d}.nc",
        )
        values, latitudes, longitudes = load_prediction_file(
            path, context.variables)
        if expected_lat is None:
            expected_lat, expected_lon = latitudes, longitudes
        elif not (
            np.array_equal(latitudes, expected_lat)
            and np.array_equal(longitudes, expected_lon)
        ):
            raise ValueError(f"集合成员网格不一致：{path}")
        for variable in context.variables:
            arrays[variable].append(values[variable])
    return {
        variable: np.stack(member_values)
        for variable, member_values in arrays.items()
    }, expected_lat, expected_lon


def ensure_matching_grid(pred_lat, pred_lon, obs_lat, obs_lon):
    """比对预测与实况的网格坐标。

    容差 1e-4（不是 1e-6）：两侧坐标来自不同来源 —— 预测是 NetCDF 里存的值
    （落盘时经 float32 往返），实况是 Dataset 从 store 读的。0.25° 网格上 1e-4
    远小于格距，不会掩盖真实的网格错配（那种情况通常差 0.25 或整个翻转/滚动），
    但能容忍 float32 往返的舍入。
    """
    if pred_lat.shape != obs_lat.shape or pred_lon.shape != obs_lon.shape:
        raise ValueError("预测与实况的网格坐标 shape 不一致")
    if not (
        np.allclose(pred_lat, obs_lat, rtol=0.0, atol=1e-4)
        and np.allclose(pred_lon, obs_lon, rtol=0.0, atol=1e-4)
    ):
        raise ValueError("预测与实况的网格坐标不一致")


def _common_finite(prediction, observation):
    if prediction.ndim == 2:
        finite = np.isfinite(prediction) & np.isfinite(observation)
    else:
        finite = np.all(np.isfinite(prediction), axis=0) & np.isfinite(observation)
    if not finite.any():
        raise ValueError("预测和实况没有共同的有限网格点")
    return finite


def _weighted_mean(field, latitudes, finite):
    weights = np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64)))
    weights = np.broadcast_to(weights[:, np.newaxis], field.shape)
    selected_weights = np.where(finite, weights, 0.0)
    denominator = selected_weights.sum()
    if denominator <= 0:
        raise ValueError("有效网格点的纬度权重之和为 0")
    return float(np.where(finite, field * weights, 0.0).sum() / denominator)


def deterministic_metrics(prediction, observation, latitudes):
    finite = _common_finite(prediction, observation)
    error = prediction - observation
    return {
        "rmse": np.sqrt(_weighted_mean(error ** 2, latitudes, finite)),
        "mae": _weighted_mean(np.abs(error), latitudes, finite),
        "bias": _weighted_mean(error, latitudes, finite),
    }


def _crps_field(ensemble, observation):
    members = ensemble.shape[0]
    term1 = np.mean(
        np.abs(ensemble - observation[np.newaxis, ...]), axis=0)
    ordered = np.sort(ensemble, axis=0)
    index = np.arange(members, dtype=np.float64)
    coefficient = (2.0 * index + 1.0 - members) / (members ** 2)
    coefficient = coefficient.reshape((-1,) + (1,) * (ensemble.ndim - 1))
    term2 = np.sum(ordered * coefficient, axis=0)
    return np.maximum(term1 - term2, 0.0)


def _brier_skill_score(probability, observed, weights):
    denominator = float(weights.sum())
    if denominator <= 0:
        return float("nan")
    score = float((weights * (probability - observed) ** 2).sum() / denominator)
    climatology = float((weights * observed).sum() / denominator)
    reference = climatology * (1.0 - climatology)
    return 1.0 - score / reference if reference > 0 else float("nan")


def _roc_area(probability, observed, weights):
    observed_weight = float((weights * observed).sum())
    non_observed_weight = float((weights * (1.0 - observed)).sum())
    if observed_weight <= 0 or non_observed_weight <= 0:
        return float("nan")
    points = [(0.0, 0.0)]
    for threshold in np.unique(probability[probability > 0]):
        predicted = probability >= threshold
        hit_rate = float(
            (weights * (predicted & (observed == 1))).sum()
            / observed_weight
        )
        false_alarm_rate = float(
            (weights * (predicted & (observed == 0))).sum()
            / non_observed_weight
        )
        points.append((false_alarm_rate, hit_rate))
    points.append((1.0, 1.0))
    points.sort(key=lambda point: point[0])
    area = sum(
        (points[index][0] - points[index - 1][0])
        * (points[index][1] + points[index - 1][1])
        / 2.0
        for index in range(1, len(points))
    )
    return float(np.clip(area, 0.0, 1.0))


def ensemble_metrics(ensemble, observation, latitudes, longitudes, variable):
    finite = _common_finite(ensemble, observation)
    mean_prediction = np.mean(ensemble, axis=0)
    metrics = deterministic_metrics(mean_prediction, observation, latitudes)
    metrics.update({
        "crps": float("nan"),
        "spread": float("nan"),
        "ssr": float("nan"),
        "bss": float("nan"),
        "aroc": float("nan"),
    })
    if variable != "tp":
        spread = np.sqrt(_weighted_mean(
            np.var(ensemble, axis=0, ddof=1), latitudes, finite))
        metrics.update({
            "crps": _weighted_mean(
                _crps_field(ensemble, observation), latitudes, finite),
            "spread": spread,
            "ssr": (
                spread / metrics["rmse"]
                if metrics["rmse"] > 0
                else float("nan")
            ),
        })
        return metrics

    latitude_grid, longitude_grid = np.meshgrid(
        latitudes, longitudes, indexing="ij")
    china = (
        (latitude_grid >= CHINA_LAT_RANGE[0])
        & (latitude_grid <= CHINA_LAT_RANGE[1])
        & (longitude_grid >= CHINA_LON_RANGE[0])
        & (longitude_grid <= CHINA_LON_RANGE[1])
    )
    weights = np.cos(np.deg2rad(latitude_grid))
    weights = np.where(china & finite, weights, 0.0)
    bss_values = []
    aroc_values = []
    for threshold in PRECIP_THRESHOLDS:
        probability = np.mean(ensemble >= threshold, axis=0)
        observed = (observation >= threshold).astype(np.float64)
        bss_values.append(
            _brier_skill_score(probability, observed, weights))
        aroc_values.append(_roc_area(probability, observed, weights))
    finite_bss = [value for value in bss_values if np.isfinite(value)]
    finite_aroc = [value for value in aroc_values if np.isfinite(value)]
    metrics["bss"] = (
        float(np.mean(finite_bss)) if finite_bss else float("nan"))
    metrics["aroc"] = (
        float(np.mean(finite_aroc)) if finite_aroc else float("nan"))
    return metrics


def write_results(rows, output_dir, prefix):
    if not rows:
        raise RuntimeError("评测没有产生任何结果")
    detail = pd.DataFrame(rows)
    detail_path = os.path.join(output_dir, f"{prefix}_detail.csv")
    detail.to_csv(detail_path, index=False, float_format="%.8g")
    metric_columns = [
        column
        for column in (
            "rmse", "mae", "bias", "crps", "spread", "ssr", "bss", "aroc"
        )
        if column in detail.columns
    ]
    by_lead = (
        detail.groupby(["var", "lead_hour"], as_index=False)
        .agg(
            n_inits=("init", "nunique"),
            **{
                column: (column, "mean")
                for column in metric_columns
            },
        )
        .sort_values(["var", "lead_hour"])
    )
    by_lead_path = os.path.join(output_dir, f"{prefix}_by_lead.csv")
    by_lead.to_csv(by_lead_path, index=False, float_format="%.8g")

    summary = (
        detail.groupby("var", as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )
    summary_path = os.path.join(output_dir, f"{prefix}_summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.8g")
    log.info(
        "评测完成：明细 %s；逐 step 平均 %s；整体平均 %s",
        detail_path,
        by_lead_path,
        summary_path,
    )
    return detail_path, by_lead_path, summary_path
