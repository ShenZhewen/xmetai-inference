#!/usr/bin/env python
"""Run AIFS Single 1.1 from local ERA5 Zarr stores."""

import argparse
import inspect
import os
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import xarray as xr
from anemoi.inference.runners.simple import SimpleRunner


PRESSURE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]
PRESSURE_VARIABLES = ["z", "t", "u", "v", "w", "q"]
PRESSURE_FIELDS = [
    f"{variable}_{level}"
    for variable in PRESSURE_VARIABLES
    for level in PRESSURE_LEVELS
]

SURFACE_FIELD_MAP = {
    "10u": "u10m",
    "10v": "v10m",
    "2d": "d2m",
    "2t": "t2m",
    "msl": "msl",
    "skt": "skt",
    "sp": "sp",
    "tcw": "tcw",
}
SOIL_FIELD_MAP = {
    "swvl1": "vsw1",
    "swvl2": "vsw2",
    "stl1": "sot1",
    "stl2": "sot2",
}
STATIC_FIELD_MAP = {
    "lsm": "lsm",
    "z": "z_sfc",
    "slor": "slor",
    "sdor": "sdor",
}

OUTPUT_FIELD_MAP = {
    "z500": ("pl", "z_500"),
    "q700": ("pl", "q_700"),
    "t700": ("pl", "t_700"),
    "t850": ("pl", "t_850"),
    "u850": ("pl", "u_850"),
    "v850": ("pl", "v_850"),
    "u10": ("sfc", "u10m"),
    "v10": ("sfc", "v10m"),
    "t2m": ("sfc", "t2m"),
    "d2m": ("sfc", "d2m"),
    "msl": ("sfc", "msl"),
    "tp": ("sfc", "tp"),
}
MODEL_OUTPUT_NAMES = {
    "z500": "z_500",
    "q700": "q_700",
    "t700": "t_700",
    "t850": "t_850",
    "u850": "u_850",
    "v850": "v_850",
    "u10": "10u",
    "v10": "10v",
    "t2m": "2t",
    "d2m": "2d",
    "msl": "msl",
    "tp": "tp",
}
OUTPUT_CHANNELS = list(OUTPUT_FIELD_MAP)
SPECIFIC_HUMIDITY_SCALE = np.float32(0.001)


def normalise_names(values):
    return [
        str(value).strip("\x00 ").lower()
        for value in values
    ]


def open_store(path, require_time=True):
    if not path.is_dir():
        raise FileNotFoundError(f"Zarr store not found: {path}")

    dataset = xr.open_zarr(path, consolidated=None)
    if "data" not in dataset.data_vars:
        dataset.close()
        raise ValueError(f"{path} does not contain a 'data' variable")

    data = dataset["data"]
    required_dims = {"channel", "lat", "lon"}
    if require_time:
        required_dims.add("time")
    if not required_dims.issubset(data.dims):
        dataset.close()
        raise ValueError(
            f"Invalid dimensions in {path}: {data.dims}; "
            f"required: {sorted(required_dims)}"
        )

    ordered_dims = (
        ("time", "channel", "lat", "lon")
        if require_time
        else ("channel", "lat", "lon")
    )
    return dataset, data.transpose(*ordered_dims)


def channel_positions(data, required_names, store_name):
    available = {
        name: index
        for index, name in enumerate(
            normalise_names(data.channel.values)
        )
    }
    missing = [
        name
        for name in required_names
        if name not in available
    ]
    if missing:
        raise ValueError(
            f"{store_name} is missing fields: {missing}"
        )
    return [available[name] for name in required_names]


def canonical_grid(latitudes, longitudes):
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.mod(
        np.asarray(longitudes, dtype=np.float64),
        360.0,
    )

    lat_order = np.argsort(latitudes)[::-1]
    lon_order = np.argsort(longitudes)
    latitudes = latitudes[lat_order]
    longitudes = longitudes[lon_order]

    if latitudes.size != 721 or longitudes.size != 1440:
        raise ValueError(
            "Expected a 0.25-degree 721x1440 source grid, "
            f"got {latitudes.size}x{longitudes.size}"
        )
    if not np.allclose(np.diff(latitudes), -0.25):
        raise ValueError("Latitude coordinate is not a regular 0.25-degree grid")
    if not np.allclose(np.diff(longitudes), 0.25):
        raise ValueError("Longitude coordinate is not a regular 0.25-degree grid")

    return latitudes, longitudes, lat_order, lon_order


def reorder_grid(values, lat_order, lon_order):
    values = np.asarray(values, dtype=np.float32)
    values = np.take(values, lat_order, axis=-2)
    values = np.take(values, lon_order, axis=-1)
    return values


def impute_for_aifs(field, model_name):
    field = np.asarray(field, dtype=np.float32)
    invalid = ~np.isfinite(field)
    invalid_count = int(invalid.sum())
    if invalid_count == 0:
        return field

    valid = field[~invalid]
    if valid.size == 0:
        raise ValueError(
            f"Field {model_name} contains no finite values"
        )

    if model_name in {"swvl1", "swvl2"}:
        fill_value = np.min(valid)
        method = "minimum"
    elif model_name in {"stl1", "stl2"}:
        fill_value = np.mean(valid, dtype=np.float64)
        method = "mean"
    else:
        raise ValueError(
            f"Unexpected non-finite values in field {model_name}: "
            f"{invalid_count} grid points"
        )

    result = field.copy()
    result[invalid] = np.float32(fill_value)
    print(
        f"Imputed field={model_name}, method={method}, "
        f"points={invalid_count}, value={float(fill_value):.6g}"
    )
    return result


def checkpoint_grid(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        lat_name = next(
            name
            for name in names
            if name.endswith("anemoi-metadata/latitudes.numpy")
        )
        lon_name = next(
            name
            for name in names
            if name.endswith("anemoi-metadata/longitudes.numpy")
        )
        latitudes = np.frombuffer(
            archive.read(lat_name),
            dtype=np.float64,
        )
        longitudes = np.frombuffer(
            archive.read(lon_name),
            dtype=np.float64,
        )
    return latitudes, longitudes


def bilinear_interpolator(
    source_lat,
    source_lon,
    target_lat,
    target_lon,
):
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.mod(
        np.asarray(target_lon, dtype=np.float64),
        360.0,
    )

    lat_position = (source_lat[0] - target_lat) / 0.25
    lon_position = (target_lon - source_lon[0]) / 0.25

    lat0 = np.clip(
        np.floor(lat_position).astype(np.int64),
        0,
        len(source_lat) - 1,
    )
    lat1 = np.clip(lat0 + 1, 0, len(source_lat) - 1)
    lon0 = (
        np.floor(lon_position).astype(np.int64)
        % len(source_lon)
    )
    lon1 = (lon0 + 1) % len(source_lon)

    lat_weight = np.clip(
        lat_position - lat0,
        0,
        1,
    ).astype(np.float32)
    lon_weight = (
        lon_position - np.floor(lon_position)
    ).astype(np.float32)

    def interpolate(field):
        field = np.asarray(field, dtype=np.float32)
        result = (
            (1 - lat_weight)
            * (1 - lon_weight)
            * field[lat0, lon0]
            + lat_weight
            * (1 - lon_weight)
            * field[lat1, lon0]
            + (1 - lat_weight)
            * lon_weight
            * field[lat0, lon1]
            + lat_weight
            * lon_weight
            * field[lat1, lon1]
        )
        if not np.isfinite(result).all():
            raise ValueError(
                "Non-finite values found after interpolation to N320"
            )
        return result.astype(np.float32, copy=False)

    return interpolate


def reduced_grid_interpolator(
    source_lat,
    source_lon,
    target_lat,
    target_lon,
):
    source_lat = np.asarray(source_lat, dtype=np.float64)
    source_lon = np.mod(
        np.asarray(source_lon, dtype=np.float64),
        360.0,
    )
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.mod(
        np.asarray(target_lon, dtype=np.float64),
        360.0,
    )

    if source_lat.ndim != 1 or source_lon.shape != source_lat.shape:
        raise ValueError("Invalid reduced-grid coordinates")

    starts = np.flatnonzero(
        np.r_[True, np.diff(source_lat) != 0]
    )
    ends = np.r_[starts[1:], source_lat.size]
    row_latitudes = source_lat[starts]
    row_order = np.argsort(row_latitudes)
    sorted_latitudes = row_latitudes[row_order]

    south_position = np.searchsorted(
        sorted_latitudes,
        target_lat,
        side="right",
    ) - 1
    south_position = np.clip(
        south_position,
        0,
        len(row_order) - 1,
    )
    north_position = np.clip(
        south_position + 1,
        0,
        len(row_order) - 1,
    )
    south_rows = row_order[south_position]
    north_rows = row_order[north_position]

    south_latitudes = row_latitudes[south_rows]
    north_latitudes = row_latitudes[north_rows]
    latitude_span = north_latitudes - south_latitudes
    latitude_weight = np.divide(
        target_lat - south_latitudes,
        latitude_span,
        out=np.zeros_like(target_lat),
        where=latitude_span != 0,
    )
    latitude_weight = np.clip(
        latitude_weight,
        0,
        1,
    ).astype(np.float32)

    row_left = []
    row_right = []
    row_weight = []
    for start, end in zip(starts, ends):
        indices = np.arange(start, end, dtype=np.int64)
        order = np.argsort(source_lon[start:end])
        indices = indices[order]
        longitudes = source_lon[indices]

        right = np.searchsorted(
            longitudes,
            target_lon,
            side="right",
        ) % len(indices)
        left = (right - 1) % len(indices)
        left_longitudes = longitudes[left]
        right_longitudes = longitudes[right]
        wrapped = right_longitudes <= left_longitudes
        right_longitudes = (
            right_longitudes + wrapped.astype(np.float64) * 360.0
        )
        adjusted_target = target_lon.copy()
        adjusted_target[adjusted_target < left_longitudes] += 360.0
        longitude_weight = (
            (adjusted_target - left_longitudes)
            / (right_longitudes - left_longitudes)
        )

        row_left.append(indices[left])
        row_right.append(indices[right])
        row_weight.append(longitude_weight.astype(np.float32))

    south_left = np.stack(
        [row_left[row] for row in south_rows]
    )
    south_right = np.stack(
        [row_right[row] for row in south_rows]
    )
    south_weight = np.stack(
        [row_weight[row] for row in south_rows]
    )
    north_left = np.stack(
        [row_left[row] for row in north_rows]
    )
    north_right = np.stack(
        [row_right[row] for row in north_rows]
    )
    north_weight = np.stack(
        [row_weight[row] for row in north_rows]
    )
    latitude_weight = latitude_weight[:, None]

    def interpolate(values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != source_lat.shape:
            raise ValueError(
                f"Expected reduced-grid shape {source_lat.shape}, "
                f"got {values.shape}"
            )

        south = (
            values[south_left] * (1 - south_weight)
            + values[south_right] * south_weight
        )
        north = (
            values[north_left] * (1 - north_weight)
            + values[north_right] * north_weight
        )
        result = (
            south * (1 - latitude_weight)
            + north * latitude_weight
        ).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError(
                "Non-finite values found after interpolation "
                "from N320 to the regular grid"
            )
        return result

    return interpolate


def check_dynamic_alignment(stores):
    reference_name, reference = stores[0]
    for name, data in stores[1:]:
        if not np.array_equal(
            reference.time.values,
            data.time.values,
        ):
            raise ValueError(
                f"Time coordinates differ between "
                f"{reference_name} and {name}"
            )
        if not np.array_equal(
            reference.lat.values,
            data.lat.values,
        ):
            raise ValueError(
                f"Latitude coordinates differ between "
                f"{reference_name} and {name}"
            )
        if not np.array_equal(
            reference.lon.values,
            data.lon.values,
        ):
            raise ValueError(
                f"Longitude coordinates differ between "
                f"{reference_name} and {name}"
            )

    times = pd.to_datetime(reference.time.values)
    if len(times) < 2:
        raise ValueError("At least two source times are required")
    intervals = np.diff(times.values)
    expected = np.timedelta64(6, "h")
    if not np.all(intervals == expected):
        raise ValueError("Dynamic stores must use a continuous 6-hour interval")


def check_static_alignment(static, dynamic):
    if not np.array_equal(
        static.lat.values,
        dynamic.lat.values,
    ):
        raise ValueError(
            "Static and dynamic latitude coordinates differ"
        )
    if not np.array_equal(
        static.lon.values,
        dynamic.lon.values,
    ):
        raise ValueError(
            "Static and dynamic longitude coordinates differ"
        )


def load_dynamic_fields(
    data,
    model_to_source,
    time_indices,
    lat_order,
    lon_order,
    interpolate,
    store_name,
):
    source_names = list(model_to_source.values())
    positions = channel_positions(
        data,
        source_names,
        store_name,
    )
    raw = data.isel(
        time=time_indices,
        channel=positions,
    ).values
    raw = reorder_grid(raw, lat_order, lon_order)

    fields = {}
    for field_index, model_name in enumerate(model_to_source):
        fields[model_name] = np.stack(
            [
                interpolate(
                    impute_for_aifs(
                        raw[time_index, field_index],
                        model_name,
                    )
                )
                for time_index in range(2)
            ]
        )
    return fields


def load_pressure_fields(
    data,
    time_indices,
    lat_order,
    lon_order,
    interpolate,
    store_name,
):
    positions = channel_positions(
        data,
        PRESSURE_FIELDS,
        store_name,
    )
    raw = data.isel(
        time=time_indices,
        channel=positions,
    ).values
    raw = reorder_grid(raw, lat_order, lon_order)

    fields = {}
    for field_index, name in enumerate(PRESSURE_FIELDS):
        values = np.stack(
            [
                interpolate(raw[time_index, field_index])
                for time_index in range(2)
            ]
        )
        if name.startswith("q_"):
            values *= SPECIFIC_HUMIDITY_SCALE
        fields[name] = values
    return fields


def load_static_fields(
    data,
    lat_order,
    lon_order,
    interpolate,
    store_name,
):
    source_names = list(STATIC_FIELD_MAP.values())
    positions = channel_positions(
        data,
        source_names,
        store_name,
    )
    raw = data.isel(channel=positions).values
    raw = reorder_grid(raw, lat_order, lon_order)

    fields = {}
    for field_index, model_name in enumerate(STATIC_FIELD_MAP):
        values = interpolate(raw[field_index])
        fields[model_name] = np.stack([values, values])
    return fields


def find_time_index(times, target_date):
    target = pd.Timestamp(target_date)
    positions = np.flatnonzero(times == target)
    if len(positions) != 1:
        raise ValueError(
            f"Initial time {target} is not present exactly once"
        )
    index = int(positions[0])
    if index < 1:
        raise ValueError(
            f"Initial time {target} has no t-6h history"
        )
    if times[index] - times[index - 1] != pd.Timedelta(hours=6):
        raise ValueError(
            f"Initial time {target} has no continuous t-6h history"
        )
    return index


def build_input_state(
    initial_time,
    pl,
    sfc,
    soil,
    static,
    times,
    lat_order,
    lon_order,
    interpolate,
    store_names,
):
    index = find_time_index(times, initial_time)
    time_indices = [index - 1, index]

    fields = load_pressure_fields(
        pl,
        time_indices,
        lat_order,
        lon_order,
        interpolate,
        store_names["pl"],
    )
    fields.update(
        load_dynamic_fields(
            sfc,
            SURFACE_FIELD_MAP,
            time_indices,
            lat_order,
            lon_order,
            interpolate,
            store_names["sfc"],
        )
    )
    fields.update(
        load_dynamic_fields(
            soil,
            SOIL_FIELD_MAP,
            time_indices,
            lat_order,
            lon_order,
            interpolate,
            store_names["soil"],
        )
    )
    fields.update(
        load_static_fields(
            static,
            lat_order,
            lon_order,
            interpolate,
            store_names["static"],
        )
    )

    expected_fields = (
        set(PRESSURE_FIELDS)
        | set(SURFACE_FIELD_MAP)
        | set(SOIL_FIELD_MAP)
        | set(STATIC_FIELD_MAP)
    )
    if set(fields) != expected_fields:
        raise RuntimeError(
            "Constructed AIFS input fields do not match requirements"
        )

    return {
        "date": pd.Timestamp(initial_time).to_pydatetime(),
        "fields": fields,
    }, index


def model_field(state, name):
    fields = state.get("fields")
    if fields is None or name not in fields:
        available = sorted(fields) if fields else []
        raise KeyError(
            f"Model output field {name!r} not found; "
            f"available fields: {available}"
        )

    values = np.asarray(fields[name], dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 1:
        raise ValueError(
            f"Expected N320 field {name} to be one-dimensional, "
            f"got shape {values.shape}"
        )
    return values


def interpolate_output_to_regular(values, interpolate):
    regular = interpolate(values)
    if regular.shape != (721, 1440):
        raise ValueError(
            f"Unexpected regular output shape: {regular.shape}"
        )
    return regular


def extract_prediction(state, interpolate):
    prediction = np.empty(
        (len(OUTPUT_CHANNELS), 721, 1440),
        dtype=np.float32,
    )
    for channel_index, channel in enumerate(OUTPUT_CHANNELS):
        values = model_field(
            state,
            MODEL_OUTPUT_NAMES[channel],
        )
        prediction[channel_index] = (
            interpolate_output_to_regular(values, interpolate)
        )
    return prediction


def extract_truth(
    valid_time,
    pl,
    sfc,
    times,
    lat_order,
    lon_order,
    store_names,
):
    positions = np.flatnonzero(times == valid_time)
    if len(positions) != 1:
        raise ValueError(
            f"Truth time {valid_time} is not present exactly once"
        )
    time_index = int(positions[0])

    truth = np.empty(
        (len(OUTPUT_CHANNELS), 721, 1440),
        dtype=np.float32,
    )
    grouped = {"pl": [], "sfc": []}
    for output_index, channel in enumerate(OUTPUT_CHANNELS):
        store_kind, source_name = OUTPUT_FIELD_MAP[channel]
        grouped[store_kind].append(
            (output_index, source_name)
        )

    for store_kind, entries in grouped.items():
        data = pl if store_kind == "pl" else sfc
        positions = channel_positions(
            data,
            [source_name for _, source_name in entries],
            store_names[store_kind],
        )
        raw = data.isel(
            time=time_index,
            channel=positions,
        ).values
        raw = reorder_grid(raw, lat_order, lon_order)
        for field_index, (output_index, source_name) in enumerate(entries):
            values = raw[field_index]
            if source_name.startswith("q_"):
                values = values * SPECIFIC_HUMIDITY_SCALE
            truth[output_index] = values

    if not np.isfinite(truth).all():
        raise ValueError(
            f"Non-finite truth values found at {valid_time}"
        )
    return truth


def save_dataarray(
    values,
    variable_name,
    path,
    initial_time,
    valid_time,
    step,
    latitudes,
    longitudes,
):
    data = xr.DataArray(
        values[np.newaxis, np.newaxis],
        name=variable_name,
        dims=["time", "step", "channel", "lat", "lon"],
        coords={
            "time": [pd.Timestamp(initial_time)],
            "step": [step],
            "channel": OUTPUT_CHANNELS,
            "lat": latitudes,
            "lon": longitudes,
        },
        attrs={
            "initial_time": str(pd.Timestamp(initial_time)),
            "valid_time": str(pd.Timestamp(valid_time)),
            "hour_interval": 6,
            "tp_definition": "6-hour interval accumulation",
            "q_units": "kg kg-1",
            "grid": "regular 0.25 degree",
        },
    )
    encoding = {
        variable_name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
        }
    }
    data.to_netcdf(path, encoding=encoding)


def run_forecast(runner, input_state, lead_time):
    parameters = inspect.signature(runner.run).parameters
    if "input_state" in parameters:
        return runner.run(
            input_state=input_state,
            lead_time=lead_time,
        )
    if "input_states" in parameters:
        return runner.run(
            input_states=input_state,
            lead_time=lead_time,
        )
    raise RuntimeError(
        "Unsupported SimpleRunner.run() signature: "
        f"{list(parameters)}"
    )


def run_date(
    runner,
    initial_time,
    args,
    pl,
    sfc,
    soil,
    static,
    times,
    latitudes,
    longitudes,
    lat_order,
    lon_order,
    interpolate,
    interpolate_output,
    store_names,
):
    date_name = pd.Timestamp(initial_time).strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir) / date_name
    target_dir = Path(args.target_dir) / date_name
    success_marker = output_dir / "_SUCCESS"

    if success_marker.exists():
        print(f"Skipping completed date: {date_name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    input_state, initial_index = build_input_state(
        initial_time,
        pl,
        sfc,
        soil,
        static,
        times,
        lat_order,
        lon_order,
        interpolate,
        store_names,
    )
    if initial_index + args.steps >= len(times):
        raise ValueError(
            f"Not enough truth data for {date_name} and "
            f"{args.steps} forecast steps"
        )

    print(
        f"Starting AIFS Single 1.1: initial={initial_time}, "
        f"steps={args.steps}, device=cuda"
    )
    start = perf_counter()
    produced_steps = 0

    for step, state in enumerate(
        run_forecast(
            runner,
            input_state,
            args.steps * 6,
        ),
        start=1,
    ):
        if step > args.steps:
            break

        valid_time = pd.Timestamp(initial_time) + pd.Timedelta(
            hours=step * 6
        )
        if pd.Timestamp(times[initial_index + step]) != valid_time:
            raise ValueError(
                f"Truth time mismatch for step {step}: "
                f"{times[initial_index + step]} != {valid_time}"
            )

        prediction = extract_prediction(state, interpolate_output)
        truth = extract_truth(
            valid_time,
            pl,
            sfc,
            times,
            lat_order,
            lon_order,
            store_names,
        )

        save_dataarray(
            prediction,
            "output",
            output_dir / f"{step:03d}.nc",
            initial_time,
            valid_time,
            step,
            latitudes,
            longitudes,
        )
        save_dataarray(
            truth,
            "target",
            target_dir / f"{step:03d}.nc",
            initial_time,
            valid_time,
            step,
            latitudes,
            longitudes,
        )

        produced_steps = step
        if (
            step == 1
            or step % args.log_every_steps == 0
            or step == args.steps
        ):
            print(
                f"Progress initial={date_name}: "
                f"steps={step}/{args.steps}"
            )

    if produced_steps != args.steps:
        raise RuntimeError(
            f"Runner produced {produced_steps} steps; "
            f"expected {args.steps}"
        )

    success_marker.write_text(
        (
            f"initial_time={initial_time}\n"
            f"steps={args.steps}\n"
            f"channels={','.join(OUTPUT_CHANNELS)}\n"
        ),
        encoding="utf-8",
    )
    print(
        f"Completed initial={date_name} in "
        f"{perf_counter() - start:.2f}s"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument(
        "--pl_store",
        default="era5_pl_2025.01-2025.03.c84.p25.h6.zarr",
    )
    parser.add_argument(
        "--sfc_store",
        default="era5_sfc_2025.01-2025.03.c15.p25.h6.zarr",
    )
    parser.add_argument(
        "--soil_store",
        default="era5_soil_2025.01-2025.03.c4.p25.h6.zarr",
    )
    parser.add_argument(
        "--static_store",
        default="era5_static.c4.p25.zarr",
    )
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.log_every_steps < 1:
        parser.error("--log_every_steps must be at least 1")
    return args


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; CPU model inference is prohibited"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each worker must see exactly one GPU through "
            f"CUDA_VISIBLE_DEVICES; visible GPUs: {torch.cuda.device_count()}"
        )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    input_dir = Path(args.input_dir)
    store_names = {
        "pl": args.pl_store,
        "sfc": args.sfc_store,
        "soil": args.soil_store,
        "static": args.static_store,
    }

    pl_dataset, pl = open_store(
        input_dir / args.pl_store
    )
    sfc_dataset, sfc = open_store(
        input_dir / args.sfc_store
    )
    soil_dataset, soil = open_store(
        input_dir / args.soil_store
    )
    static_dataset, static = open_store(
        input_dir / args.static_store,
        require_time=False,
    )

    try:
        channel_positions(
            pl,
            PRESSURE_FIELDS,
            args.pl_store,
        )
        channel_positions(
            sfc,
            list(SURFACE_FIELD_MAP.values()) + ["tp"],
            args.sfc_store,
        )
        channel_positions(
            soil,
            list(SOIL_FIELD_MAP.values()),
            args.soil_store,
        )
        channel_positions(
            static,
            list(STATIC_FIELD_MAP.values()),
            args.static_store,
        )

        check_dynamic_alignment(
            [
                (args.pl_store, pl),
                (args.sfc_store, sfc),
                (args.soil_store, soil),
            ]
        )
        check_static_alignment(static, pl)

        (
            latitudes,
            longitudes,
            lat_order,
            lon_order,
        ) = canonical_grid(pl.lat.values, pl.lon.values)
        target_lat, target_lon = checkpoint_grid(checkpoint)
        interpolate = bilinear_interpolator(
            latitudes,
            longitudes,
            target_lat,
            target_lon,
        )
        interpolate_output = reduced_grid_interpolator(
            target_lat,
            target_lon,
            latitudes,
            longitudes,
        )
        times = pd.to_datetime(pl.time.values)

        Path(args.output_dir).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(args.target_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"Python process: {os.getpid()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Dates: {args.dates}")
        print(f"Saved channels: {OUTPUT_CHANNELS}")

        runner = SimpleRunner(
            str(checkpoint),
            device="cuda",
        )

        for date in args.dates:
            initial_time = pd.Timestamp(date)
            if initial_time.hour != 0:
                raise ValueError(
                    f"Only 00 UTC initial times are allowed: {date}"
                )
            run_date(
                runner,
                initial_time,
                args,
                pl,
                sfc,
                soil,
                static,
                times,
                latitudes,
                longitudes,
                lat_order,
                lon_order,
                interpolate,
                interpolate_output,
                store_names,
            )
    finally:
        pl_dataset.close()
        sfc_dataset.close()
        soil_dataset.close()
        static_dataset.close()


if __name__ == "__main__":
    main()
