# -*- coding: utf-8 -*-
"""确定性与集合预报评测的共用发现、读取、指标和输出逻辑。"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from xmetai.loaders import create_loader
from xmetai.logging_util import configure_logging


log = logging.getLogger(__name__)
_INIT_PATTERN = re.compile(r"^\d{8}(?:\d{2})?$")
_STEP_PATTERN = re.compile(r"^(\d{3})\.nc$")
PRECIP_THRESHOLDS = (0.1, 4.0, 13.0, 25.0)
CHINA_LAT_RANGE = (15.0, 55.0)
CHINA_LON_RANGE = (70.0, 140.0)


@dataclass
class EvaluationContext:
    loader: object
    forecast_root: str
    output_dir: str
    init_dirs: list[tuple[pd.Timestamp, str]]
    variables: list[str]
    interval: int
    step_limit: int | None
    members: list[int]


def add_common_arguments(parser: argparse.ArgumentParser, *, ensemble: bool) -> None:
    parser.add_argument("--forecast", required=True, help="预测输出根目录")
    parser.add_argument(
        "--loader",
        required=True,
        choices=["era5_store", "zarr", "zarr_normalized"],
        help="实况数据 Loader",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="可选：覆盖 Loader 默认数据地址；zarr 类型必须指定",
    )
    parser.add_argument(
        "--inits",
        default=None,
        help="可选：只评指定起报，逗号分隔 YYYYMMDD/ YYYYMMDDHH",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="可选：最多评前 N 步；缺省评测目录中所有已有步骤",
    )
    parser.add_argument(
        "--vars",
        default=None,
        help="可选：只评指定变量，逗号分隔；缺省读取预测文件中的全部变量",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=6,
        help="相邻预测步的小时数（默认 6）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="CSV 和日志目录；缺省为预测目录下的 evaluation",
    )
    if ensemble:
        parser.add_argument(
            "--members",
            type=int,
            default=None,
            help="可选：只使用前 N 个成员；缺省自动发现全部 member_*",
        )
    parser.add_argument(
        "--log-level",
        default="INFO",
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
    loader = create_loader(args.loader, path=args.data_root)
    log.info(
        "评测发现：loader=%s 起报=%d 变量=%s%s",
        args.loader,
        len(init_dirs),
        ",".join(variables),
        f" 成员={len(members)}" if ensemble else "",
    )
    return EvaluationContext(
        loader=loader,
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


def _base_variable(name):
    match = re.fullmatch(r"([a-zA-Z]+?)(\d+)", name)
    return match.group(1).lower() if match else name.lower()


def load_observations(context, valid_time):
    state = context.loader.load_state(
        valid_time, channels=context.variables)
    fields = {str(name).lower(): value for name, value in state["fields"].items()}
    latitudes = state.get("latitudes")
    longitudes = state.get("longitudes")
    if latitudes is None or longitudes is None:
        raise ValueError("当前评测工具只支持带 lat/lon 坐标的规则网格实况")
    observations = {}
    expected_lat = expected_lon = None
    for variable in context.variables:
        if variable not in fields:
            raise KeyError(
                f"实况在 {valid_time:%Y%m%d%H} 缺少变量 {variable!r}")
        scale = getattr(context.loader, "SCALE", {}).get(
            _base_variable(variable), 1.0)
        values, current_lat, current_lon = _align_regular_grid(
            np.asarray(fields[variable], dtype=np.float64) * float(scale),
            latitudes,
            longitudes,
        )
        observations[variable] = values
        expected_lat, expected_lon = current_lat, current_lon
    return observations, expected_lat, expected_lon


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
    if pred_lat.shape != obs_lat.shape or pred_lon.shape != obs_lon.shape:
        raise ValueError("预测与实况的网格坐标 shape 不一致")
    if not (
        np.allclose(pred_lat, obs_lat, rtol=0.0, atol=1e-6)
        and np.allclose(pred_lon, obs_lon, rtol=0.0, atol=1e-6)
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
