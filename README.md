# xmetai-inference

Inference framework for autoregressive weather forecasting models. It reads ERA5 / CRA-40
reanalysis from Zarr stores, feeds each model the channel set and history window its contract
declares, runs the rollout, and writes the forecast as NetCDF.

- Runs ONNX and `torch.export` (`.pt2`) graphs behind one interface
- Supports deterministic and 51-member ensemble models on a 0.25° global grid
- Ships a run recipe per model, selected with `--config <name>`
- Multi-GPU by sharding initialization times × members across cards

## Table of Contents

- [Quick Start](#quick-start)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Run](#run)
  - [Override Parameters](#override-parameters)
- [Supported Models](#supported-models)
  - [Channel Sets and Order](#channel-sets-and-order)
- [Supported Datasets](#supported-datasets)
  - [Dataset Spec](#dataset-spec)
  - [Store Inventory](#store-inventory)
  - [Two Store Layouts](#two-store-layouts)
  - [Units](#units)
  - [Per-Model Preprocessing Chains](#per-model-preprocessing-chains)
  - [Single-Input `.nc` Example](#single-input-nc-example)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)

## Quick Start

### Requirements

- Linux (or WSL2 / Git Bash on Windows)
- An NVIDIA GPU with CUDA
- Model weights and the input Zarr stores (both excluded from git)

### Install

```bash
pip install -e .
```

Add the extras for what you run:

```bash
pip install -e ".[netcdf,zarr,onnx]"   # every model except FuXi-2.1 (ONNX)
pip install -e ".[netcdf,zarr]"        # NetCDF I/O only: FuXi-2.1 / evaluation
```

`torch` comes in with the base install. The `netcdf` extra brings `netCDF4` / `h5netcdf`
(reading `mean.nc`/`std.nc`, writing output), `zarr` powers store reads in `xmetai-core`, and
`onnx` pulls in `onnxruntime`.

> PyPI's `onnxruntime` is the **CPU** build. For GPU inference install `onnxruntime-gpu`
> instead (the `onnx` extra and `onnxruntime-gpu` conflict; use one or the other). The loader
> logs a warning when it has to fall back to the CPU provider.

### Run

Every supported model has a ready-made recipe (`configs/<name>.py`):

```bash
xmetai-infer --config weather_era_0p25_fuxi_ens
xmetai-infer --config weather_era_0p25_fuxi21
xmetai-infer --config weather_era_0p25_fgvp
xmetai-infer --config weather_era_0p25_fgvp_ens
xmetai-infer --config weather_era_0p25_fengqing
xmetai-infer --config weather_era_0p25_pangu
xmetai-infer --config weather_cra_0p25_tongyun_ens
xmetai-infer --config s2s_cra_0p25_fengshun20_ens
xmetai-infer --config weather_example_tongyun   # demo: single-input .nc (FGVP deterministic)
```

Each recipe declares the weight path, dataset and preprocessing chain, initialization times, step
count, GPU count, and output directory. The shipped recipes use `/path/to/...` placeholders —
point those at your own weights and stores (or set the matching environment variables) before the
first run. From a source checkout without installing:

```bash
python -m xmetai_inference --config weather_era_0p25_fuxi21
```

Output goes to `{out}/{YYYYMMDD|YYYYMMDDHH}/[member_XXX/]{step:03d}.nc`, one NetCDF per step.

`xmetai-infer --worker ...` is the internal worker subprocess entry point used for every run
(single- or multi-GPU) and should not be called manually.

### Override Parameters

```bash
xmetai-infer --config weather_era_0p25_fuxi_ens \
  --times "2025010600..2025011200:24" \
  --steps 60 --members 51 --gpus 4 --cuda-devices 0,1,2,3 \
  --vars z500,u200,v200,msl,tp \
  --out <output_dir>
```

- `--times` follows the `parse_times` syntax in `xmetai_inference/configs/base.py`: a single
  `2025010600`, a comma-separated list, a closed interval `A..B` (24 h step by default), or
  `:N` to override the step.
- `--config` takes a built-in recipe name or any external `.py` recipe; the positional argument
  is the same thing for a file path. See `configs/` for examples.
- `--model-path` swaps the weight file without touching the recipe.

## Supported Models

| Model | Recipe (`--config`) | Backend | Type | Members | Channels | Grid | Step / history | Normalization |
|------|------|------|------|------|------|------|------|--------|
| FuXi-Ens | `weather_era_0p25_fuxi_ens` | `onnx` | Ensemble | 51 | 78 | `GRID_025` | 6h / 2 frames | Baked into graph; physical units in/out |
| FuXi-2.1 | `weather_era_0p25_fuxi21` | `pt2` | Deterministic | 1 | 85 | `GRID_025` | 6h / 2 frames | Data-side `normalize`, output `denormalize` (mean.nc/std.nc next to the weights) |
| IWC FGVP GDN2 | `weather_era_0p25_fgvp` | `onnx` | Deterministic | 1 | 78 | `GRID_025` | 6h / 2 frames | Baked into graph; custom ops (`GatedDeltaNet2Fn`) require `ops_library` |
| IWC FGVP Mamba-Ens | `weather_era_0p25_fgvp_ens` | `onnx` | Ensemble | 51 | 78 | `GRID_025` | 6h / 2 frames | Baked into graph; the `.onnx.data` external-weights file must sit next to the graph |
| FengQing V1.5Beta | `weather_era_0p25_fengqing` | `onnx` | Deterministic | 1 | 70 | `GRID_025` | 6h / 2 frames | `normalize_fengqing` / `denormalize_fengqing` (per-pixel stats; graph emits a residual) |
| Pangu-Weather | `weather_era_0p25_pangu` | `onnx` | Deterministic | 1 | 69 | `GRID_025` | 6h / 1 frame (+24h jump) | Baked into graph; physical units in/out |
| Tongyun-Ens (CRA) | `weather_cra_0p25_tongyun_ens` | `onnx` | Ensemble | 51 | 71 | `GRID_025_S` | 6h / 2 frames | Store-normalized → data-side `denormalize`; graph normalizes at its head, outputs physical units |
| fengshun2.0 | `s2s_cra_0p25_fengshun20_ens` | `onnx` | Ensemble | 51 | 87 | `GRID_025` | 24h / 2 frames | Store-normalized → data-side `denormalize` (tp via `expm1`); graph normalizes at its head |

The ERA5 models share the 0.25° global regular grid (721×1440, latitude north→south, longitude
0–360°). All use a 2-frame history window except Pangu, which uses a single frame
(`history_steps=1`) and, every 4 steps (lead 24 h), switches to a 24 h model that jumps directly
from the anchor (official "Plan B").

Three exceptions worth knowing:

- **Tongyun-Ens (CRA)** uses `GRID_025_S` (latitude south→north, the store's native orientation)
  and its preprocessing **must not configure `geometry`** — that flips the field in reverse.
- **fengshun2.0** runs on a daily step (`hour_interval=24`), so `steps=60` is a 60-day forecast.
  Its store is CRA north→south, so it also skips `geometry`. The recipe asks for 51 members even
  though the model class declares `members=11` as the graph's native member axis.
- **Pangu** needs a second weight file for the 24 h jump (`PANGU_ONNX_24`); it defaults to
  `pangu_weather_24.onnx` next to the 6 h model.

### Channel Sets and Order

Every model's `input_channels` (also `output_channels`) is the flattened list below; the order is
part of the contract. A wrong order does **not** raise — it silently forecasts the wrong field.

```
upper = [f"{var}{level}" for var in PL_VARS for level in LEVELS]      # 65 channels

FuXi-Ens  (78): PL z,t,u,v,q × [50,100,150,200,250,300,400,500,600,700,850,925,1000]
                + [t2m, d2m, sst, u10m, v10m, u100m, v100m, msl, ssr, ssrd, fdir, ttr, tp]
FuXi-2.1  (85): PL z,t,u,v,q × same levels
                + [msl, t2m, d2m, sst, ws10m, ws100m, u10m, v10m, u100m, v100m,
                   lcc, mcc, hcc, tcc, ssr, ssrd, fdir, ttr, tcw, tp]
fengshun  (87): PL z,t,u,v,q × same levels
                + [msl, t2m, t2m_min, t2m_max, d2m, sst, ws10m, ws100m, u10m, v10m,
                   u100m, v100m, lcc, mcc, hcc, tcc, ssr, ssrd, fdir, ttr, tcw, tp]
FengQing  (70): PL z,q,t,u,v × [1000,925,850,700,600,500,400,300,250,200,150,100,50]
                + [tp, msl, u10m, v10m, t2m]
Pangu     (69): PL z,q,t,u,v × same descending-pressure levels
                + [msl, u10m, v10m, t2m]
Tongyun   (71): PL z,t,u,v,q × [50 … 1000] (ascending pressure)
                + [t2m, q2m, u10m, v10m, msl, tp]
```

Channel names are matched against store names by `_raw_channel_name`: exact match first, then
`z50` → `z_50`. A store may be a superset of the model channels — every requested channel must
exist, or construction fails loudly. Two processors read names differently: `denormalize`
aligns store stats by underscore-insensitive, lower-cased name and tolerates a superset;
`normalize` requires stats that match `input_channels` in count **and order** exactly.

> **Model naming.** Config names follow `<domain>_<source>_<resolution>_<model>[_ens]`:
> `domain` = `weather` (15-day, 6-hourly) or `s2s` (subseasonal, daily steps); `source` = `era`
> (ERA5) or `cra` (CRA-40); `resolution` = `0p25` (0.25°) or `1p5`. The trailing `_ens` marks an
> ensemble output when the model name doesn't already contain `ens`; deterministic configs carry
> no suffix. Examples: `weather_era_0p25_fuxi_ens`, `weather_cra_0p25_tongyun_ens`,
> `s2s_cra_0p25_fengshun20_ens`. Each maps to the eponymous recipe under `configs/`.
> **Model-class registry names** (the `model_class=` in recipes, e.g. `fuxi_ens_onnx`,
> `fuxi21_pt2`, `iwc_fgvp_gdn2_onnx`, `iwc_fgvp_mamba_ens_onnx`, `fengqing_pre_onnx`,
> `pangu_onnx`, `tongyun_cra_ens_onnx`, `fengshun20_onnx`) map to `MODEL_REGISTRY` in
> `utils/model_factory.py`.

## Supported Datasets

All datasources are ingested via `processed_multi_zarr`, backed by `xmetai-core`'s
`MultiZarrDataset`.

### Dataset Spec

```python
dataset = {
    "type": "processed_multi_zarr",
    "paths": ["<store_root>/store_a.zarr", "<store_root>/store_b.zarr"],
    "processors": [ ... ],          # executed in the DataLoader worker, in order
}
```

- `paths` may list several stores. They are merged by channel name; **the first store listed
  wins on duplicate names** (the FengQing recipe relies on this: the 6h-accumulated tp store is
  listed before the surface store whose `tp` is a 1 h accumulation).
- The framework derives `hist_frames`, `fcst_frames=1`, `freq`, `interval=1`,
  `inference_only=True`, `years`, and `in_names=[]` from the model contract and the requested
  init times. Do not repeat these keys in `dataset` (duplicate keyword arguments); any **other**
  key is forwarded to `MultiZarrDataset` as-is.
- History frames are read backwards from each init time; a requested init time without enough
  history raises.

**`example_nc`** — infer straight from one business-supplied `.nc` file instead of a Zarr store:

```python
dataset = {
    "type": "example_nc",
    "paths": ["/abs/input.nc"],
    "mode": "model_ready",   # default; or "native"
    # "processors": [...],   # native mode only; model_ready must not set it
}
```

Each file is one initialization; all history frames live inside the nc itself (the engine never
reads back into a store). The init time is the last `time` frame's value, so `cfg.times` is not
consulted for this type. Two modes:

- `model_ready` (default): the nc already holds exactly the model input — physical quantities in
  `input_channels` order, in the model's native grid orientation. No processor chain runs; only
  the channel axis is verified/rewritten.
- `native`: the nc holds raw fields in its own channel order; the model's `dataset.processors`
  chain runs as in `processed_multi_zarr`, then the channel axis is reordered to `input_channels`.

Samples are built eagerly once at dataset build time, so this type is single-process-only.

### Store Inventory

The store root names below are relative to the data roots declared at the top of each recipe in
`configs/`. File names encode channel count, resolution, and step (`c84.p25.h6` = 84 channels,
0.25°, 6-hourly).

| Store | Channels | Coverage / step | Layout | Orientation | What it holds | Used by |
|---|---|---|---|---|---|---|
| `era5_foundation_store2/era5_pl_...c84.p25.h6.zarr` | 84 | 2025-01 … 2026-07, 6h | **Raw** | north→south | Upper-air pressure-level fields | fuxi_ens, fuxi21, fgvp, fgvp_ens, fengqing, pangu |
| `.../era5_sfc_...c15.p25.h6.zarr` | 15 | same | **Raw** | north→south | Surface fields; `tp`/radiation are 1 h accumulations | same as above |
| `.../era5_cldrad_...c8.p25.h6.zarr` | 8 | same | **Raw** | north→south | Radiation and cloud-cover channels | fuxi_ens, fuxi21, fgvp, fgvp_ens |
| `.../era5_soil_...c4.p25.h6.zarr` | 4 | same | **Raw** | north→south | Soil channels | listed by fuxi21, but no soil channel is in its 85-channel set (selected channels come from pl/sfc/cldrad) |
| `.../era5_wave_...c5.p25.h6.zarr` | 5 | same | **Raw** | north→south | Wave channels | same as soil: listed by fuxi21, not selected by its channel set |
| `era5_tp_zarr/era5.202501-202607.c1.p25.h6.v1.lon180.zarr` | 1 | 2025-01 … 2026-07, 6h | **Raw** | global (`lon180` naming) | True 6h-accumulated `tp` (metres) | fengqing (listed first to override the sfc store's 1h tp) |
| `zarr_refresh_v3/cra.19790101_20260714.c71.p25.h6.v1` | 71 | 1979-01-01 … 2026-07-14, 6h | **Normalized** | south→north | mean/std arrays inside the store; `tp` stays log1p | tongyun_cra_ens |
| `zarr_refresh_v3/cra.1979_2024.c87.p25.h24.v1` | 87 | 1979-01-01 … 2024-12-31, daily 00Z, fp16 | **Normalized** | north→south | fengshun2.0 channel order; `tp` = log1p(mm per day) | fengshun20_ens |
| `zarr_refresh_v3/era5.202501_202607.c116.p25.h6.v3` | 116 | 2025-01 … 2026-07, 6h | **Normalized** (superset) | north→south | q in g/kg, radiation in W/m², `tp` = log1p(6h mm) | — (no shipped recipe; opt-in A/B store) |
| `zarr_refresh_v3/era5.1979_2024.c78.p25.h6.v1` | 78 | 1979 … 2024 backfill | **Normalized** (superset) | north→south | Same conventions as c116 | — (no shipped recipe; backfill store) |

The counts in file names are store channel counts; the authoritative list is the `channel`
coordinate inside each store. Inspect it before wiring a new model (read-only):

```bash
python - <<'PY'
import xarray as xr
store = xr.open_zarr("<store_root>/store.zarr", consolidated=True)
print(store.sizes)
print(list(store.channel.values))
PY
```

> Do **not** use `era5.1959_2025.c87`: its `tp` switches accumulation window at 2024-01-01
> (1 h → 24 h) and its `sst` carries 33.89% land NaN.

### Two Store Layouts

| Layout | What the store holds | Where normalization happens |
|---|---|---|
| **Raw** | Physical units (q=kg/kg, z=m²/s², tp/radiation = 1 h accumulations) | Baked into the graph, or framework-side `mean.nc`/`std.nc` |
| **Normalized** | Already-normalized values; `mean`/`std` arrays live inside the store | Store-side normalization, and the graph head does `Div(Sub(x,mean),std)` again; the data side must `denormalize` |

How to tell them apart:

1. Does the store contain `mean`/`std` arrays? → almost certainly normalized.
2. Is the graph head `Div(Sub(x, mean), std)`? → the graph expects physical values, so the data
   side must `denormalize` the normalized store.
3. Latitude orientation: north→south (FuXi convention) or south→north (CRA's native orientation)?
   For a south→north store use `GRID_025_S` and **do not configure `geometry`** — `geometry`
   always flips south→north to north→south, which feeds the field reversed (measured: +6 h z500
   RMSE 90 → 486).
4. For a log1p channel (`tp`), `denormalize` yields `log1p(physical value)`, not the value itself.
   Whether to `expm1` depends on the graph: c71/Tongyun expects the log1p value (**no** `expm1`);
   the v3 ERA5 stores and c87/fengshun expect physical mm (**yes**, `expm1_channels=["tp"]`).

### Units

| Quantity | store2 (raw) | zarr_refresh_v3 (after `denormalize`) | Model expectation → processor |
|---|---|---|---|
| `q` | kg/kg | **g/kg** | FuXi / FGVP / fengshun use g/kg directly; FengQing / Pangu need kg/kg ⇒ `unit_convert {"q": 1e-3}` when switching to v3 |
| `tp` | 1 h accumulation, metres (the sfc store); the separate tp store is true 6 h accumulation | **log1p(6h mm)** | FuXi / FGVP: 6 h mm ⇒ store2 `×6000`; v3 `expm1_channels=["tp"]`. FengQing overrides with the 6 h tp store and multiplies by 1000 |
| radiation (`ssr`/`ssrd`/`fdir`/`ttr`) | 1 h accumulation, J/m² | **W/m²** (step mean) | 6 h Wh/m² ⇒ store2 `×6/3600`; v3 `×6` |
| `z`, `t`, `u`, `v`, `msl`, surface fields | physical | physical | 1:1 |
| CRA c71 (Tongyun) | — | — | normalized store: no unit conversion; `tp` stays log1p (the graph expects it) |
| CRA c87 (fengshun) | — | — | normalized store: no unit conversion; `tp` is log1p at input, `expm1` applied by `denormalize` |

### Per-Model Preprocessing Chains

| Recipe | `dataset.processors` | `model_processing.recurrent` | `model_processing.output` |
|---|---|---|---|
| fuxi_ens | `unit_convert` (q×1000, rad×6/3600, tp×6000) → `geometry` | — | — |
| fuxi21 | `unit_convert` → `geometry` → `normalize` (log1p tp) → `fill_missing` (`{"*": 0.0}`, keep) | `zero_channels` (ssr, ssrd, fdir, ttr, tp) | `denormalize` (expm1 tp, nonnegative tp) |
| fgvp / fgvp_ens | `unit_convert` (same) → `geometry` → `fill_missing` (sst=287.066, error) | — | — |
| fengqing | `unit_convert` (tp×1000) → `geometry` → `normalize_fengqing` | — | `denormalize_fengqing` |
| pangu | `geometry` | — | — |
| tongyun_cra | `denormalize` (stats_path=CRA), **no geometry** | — | — |
| fengshun20 | `denormalize` (stats_path=c87, expm1 tp) → `fill_missing` (sst=287.222, error) | — | — |

ERA5 recipes default to the raw store2 root; the two CRA recipes read `zarr_refresh_v3` stores.
The v3 ERA5 stores (c116 / c78) have no shipped recipe — copy an existing ERA5 recipe and swap
`paths` (and `times`) to run them.

### Single-Input `.nc` Example

`weather_example_tongyun` is a self-contained demo recipe that infers straight from **one
business-supplied `.nc`** — no Zarr store. The graph is the FGVP deterministic model
(`tongyun_det.onnx` == `iwc_fgvp_gdn2_onnx`): **78 channels** (FGVP/ERA5 order), `GRID_025`
north→south, single member, 2-frame history, physical units in/out (normalization baked into the
graph).

1. Download the sample input (a 78-channel `input.nc`, already `model_ready`):
   https://drive.google.com/file/d/1h8jBXPEtQLfayUe8f_UpNCHWwIKDGHeD/view?usp=drive_link
2. Point the three placeholders in `configs/weather_example_tongyun.py` at your model, custom-op
   library, and the downloaded `.nc`:
   ```python
   SRC_GRAPH        = "<tongyun_det.onnx>"
   OPS_LIBRARY      = "<xmetai_onnx_plugins.ort1_24.cuda12.so>"
   TONGYUN_NC_INPUT = "<path>/input.nc"
   ```
3. Run — one command:
   ```bash
   python -m xmetai_inference --config weather_example_tongyun --out <output_dir>
   ```
   Output lands at `{out}/{YYYYMMDD}/{step:03d}.nc` (60 steps, 6 h apart; single member, so no
   `member_XXX` layer).

The input only works if it is exactly the graph's input: physical quantities in
`FUXI_ENS_CHANNELS` order, `GRID_025` orientation, `float32`, `(time=history, channel=78, 721,
1440)`, and `sst` cold-spots filled. `tools/make_tongyun_sample_nc.py` regenerates such a file
from any normalized store (denormalize in, `geometry`/`fill_missing` preserved).

## Project Structure

```
.
├── pyproject.toml           # PyPI build metadata and CLI entry
├── xmetai_inference/        # The installable Python package (package name: xmetai-inference)
│   ├── cli.py               # Config loading, multi-GPU scheduling, the autoregressive main loop, async disk-write
│   ├── data/                # Dataset implementations, grouped into subpackages by category (weather/)
│   ├── utils/               # dataloader_factory (config -> DataLoader, dispatched by dataset.type),
│   │                        #   runner (Rollout / Trajectory autoregressive orchestration),
│   │                        #   model_factory (MODEL_REGISTRY / get_model_class / create_model),
│   │                        #   model-side pipeline (recurrent/output), data-side tensor processors, logging
│   ├── backends/            # BaseInferModel + onnx / pt2 execution engines
│   └── configs/             # Run recipes (`--config <name>`) + declared model contracts;
│                            #   base.py = shared GRID/channels + InferConfig/load_config
├── tests/                   # Unit + integration tests (integration runs only with `-m realdata`)
└── model_artifacts/         # Local model artifacts; absent in a fresh clone and excluded from Git
```

## How It Works

One forecast, from store to disk:

```
zarr store  →  xmetai-core MultiZarrDataset          (read history frames, select variables by model channel)
            →  dataset.processors                    (in DataLoader worker, torch; chain depends on store layout)
                 unit_convert → geometry → normalize → fill_missing     ← raw store (ERA5)
                 denormalize (stats_path=store, ± expm1) → …            ← normalized store (CRA / c87 / v3)
            →  prepare_dataset_batch                 ([B,T,C,H,W] tensor)
            →  Rollout (autoregressive)              (state = result feedback)
                 recurrent processors (zero_channels)
            →  output processors (denormalize)       (workspace → physical units)
            →  to_dataset                            (xarray, non-negative fallback for tp)
            →  _AsyncWriter                          (single background thread writes NetCDF)
```

- **Contract-driven models**: channels, grid, history window, member semantics, and state
  representation are declared by the model class, so backends stay interchangeable.
- **Pluggable backends**: `onnx` and `pt2` execution engines are decoupled from model semantics.
- **One recipe per run**: the config declares `dataset` (incl. `processors`), `dataloader`, and
  `model_processing` (the `recurrent` and `output` stages).
- **Unified autoregression**: `utils/runner.py`'s `Rollout`/`Trajectory` chain single-step
  `forward` into a multi-trajectory loop with `state = result` feedback.
- **Async disk-write**: one background thread writes NetCDF; on failure it falls back to per-step
  `raw_step_XXX.npy`.
- **Multi-GPU data parallelism**: work is sharded by "initialization time × member"; each worker
  process is pinned to one card via `CUDA_VISIBLE_DEVICES`.

> Input preprocessing goes through `dataset.processors`; `model_processing.input` must be empty
> (misconfiguring it errors out). Normalization already fused into the model file is not
> re-executed by the framework.
