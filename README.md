# Weather Model Inference Framework

Converts reanalysis data (ERA5 / CRA-40) into model input, runs autoregressive weather forecasting, and saves the results as NetCDF.

## Table of Contents

- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
  - [Dataset Processors](#dataset-processors)
  - [Adding a New Model / Dataset](#adding-a-new-model--dataset)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Architecture & Data Flow](#architecture--data-flow)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [License](#license)

## Supported Models

| Model | Recipe (`--model`) | Backend | Type | Members | Channels | Grid | Step / history | Normalization |
|------|------|------|------|------|------|------|------|--------|
| FuXi-Ens | `weather_era_0p25_fuxi_ens` | `onnx` | Ensemble | 51 | 78 | `GRID_025` | 6h / 2 frames | Baked into graph; physical units in/out |
| FuXi-2.1 | `weather_era_0p25_fuxi21` | `pt2` | Deterministic | 1 | 85 | `GRID_025` | 6h / 2 frames | Data-side `normalize`, output `denormalize` (mean.nc/std.nc next to the weights) |
| IWC FGVP GDN2 | `weather_era_0p25_fgvp` | `onnx` | Deterministic | 1 | 78 | `GRID_025` | 6h / 2 frames | Baked into graph; custom ops (`GatedDeltaNet2Fn`) require `ops_library` |
| IWC FGVP Mamba-Ens | `weather_era_0p25_fgvp_ens` | `onnx` | Ensemble | 51 | 78 | `GRID_025` | 6h / 2 frames | Baked into graph; the `.onnx.data` external-weights file must sit next to the graph |
| FengQing V1.5Beta | `weather_era_0p25_fengqing` | `onnx` | Deterministic | 1 | 70 | `GRID_025` | 6h / 2 frames | `normalize_fengqing` / `denormalize_fengqing` (per-pixel stats; graph emits a residual) |
| Pangu-Weather | `weather_era_0p25_pangu` | `onnx` | Deterministic | 1 | 69 | `GRID_025` | 6h / 1 frame (+24h jump) | Baked into graph; physical units in/out |
| Tongyun-Ens (CRA) | `weather_cra_0p25_tongyun_ens` | `onnx` | Ensemble | 51 | 71 | `GRID_025_S` | 6h / 2 frames | Store-normalized → data-side `denormalize`; graph normalizes at its head, outputs physical units |
| fengshun2.0 | `s2s_cra_0p25_fengshun20_ens` | `onnx` | Ensemble | 51 | 87 | `GRID_025` | 24h / 2 frames | Store-normalized → data-side `denormalize` (tp via `expm1`); graph normalizes at its head |

The ERA5 models share the 0.25° global regular grid (721×1440, latitude north→south, longitude 0–360°).
All use a 2-frame history window except Pangu; Pangu uses a single history frame (`history_steps=1`), 6-hour steps,
but every 4 steps (lead 24h) switches to a 24h model that jumps 24h directly from the anchor (official "Plan B").
**Tongyun-Ens (CRA) is the exception**: it uses `GRID_025_S` (latitude south→north, the store's native orientation),
and its preprocessing **must not configure `geometry`** (doing so feeds the field in reversed).
**fengshun2.0** runs on a daily step (`hour_interval=24`); `steps=60` means a 60-day forecast, and its store is
CRA north→south, so it also does not use `geometry`. Its recipe asks for 51 members even though the model class
declares `members=11` as the graph's native member axis.

### Channel Sets and Order

Every model's `input_channels` (also `output_channels`) is the flattened list below; the order is part of the
contract. A wrong order does **not** raise — it silently forecasts the wrong field.

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

Channel names are matched against store names by `_raw_channel_name`: exact match first, then `z50` → `z_50`.
Normalization files/certain processors align names with underscores stripped (case-insensitive), so a store may be
a superset of the model channels — but every requested channel must exist, or construction fails loudly.

> There are two model-naming schemes — don't mix them: **config names** (for `--model` / `--config`) follow
> `<domain>_<source>_<resolution>_<model>[_ens]`: `domain` = `weather` (15-day, 6-hourly) or `s2s`
> (subseasonal, daily steps); `source` = `era` (ERA5) or `cra` (CRA-40); `resolution` = `0p25` (0.25°) or `1p5`.
> The trailing `_ens` marks an ensemble output when the model name doesn't already contain `ens`
> (deterministic configs carry no suffix).
> Examples: `weather_era_0p25_fuxi_ens`, `weather_cra_0p25_tongyun_ens`, `s2s_cra_0p25_fengshun20_ens`,
> `weather_era_0p25_fgvp_ens`. Each maps to the eponymous recipe file under `configs/`;
> **model-class registry names** (the `model_class=` in recipes, e.g. `fuxi_ens_onnx`, `fuxi21_pt2`,
> `iwc_fgvp_gdn2_onnx`, `iwc_fgvp_mamba_ens_onnx`, `fengqing_pre_onnx`, `pangu_onnx`,
> `tongyun_cra_ens_onnx`, `fengshun20_onnx`) map to `MODEL_REGISTRY` in `utils/model_factory.py`.

## Supported Datasets

All datasources are ingested via `processed_multi_zarr`, backed by `xmetai-core`'s `MultiZarrDataset`.

### Dataset Spec

```python
dataset = {
    "type": "processed_multi_zarr",
    "paths": ["/abs/path/to/store_a.zarr", "/abs/path/to/store_b.zarr"],
    "processors": [ ... ],          # executed in the DataLoader worker, in order
}
```

- `paths` may list several stores. They are merged by channel name; **the first store listed wins on duplicate
  names** (the FengQing recipe relies on this: the 6h-accumulated tp store is listed before the surface store
  whose `tp` is a 1h accumulation).
- The framework derives, from the model contract and the requested init times: `hist_frames` (= `history_steps`),
  `fcst_frames=1`, `freq` (= `hour_interval`), `interval=1`, `inference_only=True`, `years`, and `in_names=[]`.
  Do not repeat these keys in `dataset` (they would be duplicate keyword arguments); any **other** key is
  forwarded to `MultiZarrDataset` as-is.
- History frames are read backwards from each init time; a requested init time without enough history raises.

### Store Inventory

Roots: `era5_foundation_store2` → `/workspace/data/liujunjie/era5_foundation_store2`; the tp store →
`/workspace/data/liujunjie/era5_tp_zarr`; all `zarr_refresh_v3/...` entries → `/workspace/data/zarr_refresh_v3`.
File names encode channel count, resolution, and step (`c84.p25.h6` = 84 channels, 0.25°, 6-hourly).

| Store | Channels | Coverage / step | Layout | Orientation | What it holds | Used by |
|---|---|---|---|---|---|---|
| `era5_foundation_store2/era5_pl_...c84.p25.h6.zarr` | 84 | 2025-01 … 2026-07, 6h | **Raw** | north→south | Upper-air pressure-level fields | fuxi_ens, fuxi21, fgvp, fgvp_ens, fengqing, pangu |
| `.../era5_sfc_...c15.p25.h6.zarr` | 15 | same | **Raw** | north→south | Surface fields; `tp`/radiation are 1h accumulations | same as above |
| `.../era5_cldrad_...c8.p25.h6.zarr` | 8 | same | **Raw** | north→south | Radiation and cloud-cover channels | fuxi_ens, fuxi21, fgvp, fgvp_ens |
| `.../era5_soil_...c4.p25.h6.zarr` | 4 | same | **Raw** | north→south | Soil channels | listed by fuxi21, but no soil channel is in its 85-channel set (selected channels come from pl/sfc/cldrad) |
| `.../era5_wave_...c5.p25.h6.zarr` | 5 | same | **Raw** | north→south | Wave channels | same as soil: listed by fuxi21, not selected by its channel set |
| `era5_tp_zarr/era5.202501-202607.c1.p25.h6.v1.lon180.zarr` | 1 | 2025-01 … 2026-07, 6h | **Raw** | global (`lon180` naming) | True 6h-accumulated `tp` (metres) | fengqing (listed first to override the sfc store's 1h tp) |
| `zarr_refresh_v3/cra.19790101_20260714.c71.p25.h6.v1` | 71 | 1979-01-01 … 2026-07-14, 6h | **Normalized** | south→north | mean/std arrays inside the store; `tp` stays log1p | tongyun_cra_ens |
| `zarr_refresh_v3/cra.1979_2024.c87.p25.h24.v1` | 87 | 1979-01-01 … 2024-12-31, daily 00Z, fp16 | **Normalized** | north→south | fengshun2.0 channel order; `tp` = log1p(mm per day) | fengshun20_ens |
| `zarr_refresh_v3/era5.202501_202607.c116.p25.h6.v3` | 116 | 2025-01 … 2026-07, 6h | **Normalized** (superset) | north→south | q in g/kg, radiation in W/m², `tp` = log1p(6h mm) | fgvp_v3 (template for the other four ERA5 models) |
| `zarr_refresh_v3/era5.1979_2024.c78.p25.h6.v1` | 78 | 1979 … 2024 backfill | **Normalized** (superset) | north→south | Same conventions as c116 | fgvp_v3 with `paths` and `times` swapped |

The counts in file names are store channel counts; the authoritative list is the `channel` coordinate inside each
store. Inspect it before wiring a new model (this is a read-only command, safe to run anywhere the store is mounted):

```bash
python - <<'PY'
import xarray as xr
store = xr.open_zarr("/workspace/data/.../store.zarr", consolidated=True)
print(store.sizes)
print(list(store.channel.values))
PY
```

> Do **not** use `era5.1959_2025.c87`: its `tp` switches accumulation window at 2024-01-01 (1h → 24h) and its
> `sst` carries 33.89% land NaN.

### Two Store Layouts

| Layout | What the store holds | Where normalization happens | How to feed the model |
|---|---|---|---|
| **Raw** | Physical units (q=kg/kg, z=m²/s², tp/radiation = 1h accumulations) | Baked into the graph, or framework-side `mean.nc`/`std.nc` | `unit_convert` → `geometry` → (optional `normalize` / `fill_missing`) |
| **Normalized** | Already-normalized values; `mean`/`std` arrays live inside the store | Store-side normalization, and the graph head does `Div(Sub(x,mean),std)` again | Data-side `denormalize` (`stats_path`=that store), **no `unit_convert`**, then model-specific units |

How to tell them apart:

1. Does the store contain `mean`/`std` arrays? → almost certainly normalized.
2. Is the graph head `Div(Sub(x, mean), std)`? → the graph expects physical values, so the data side must
   `denormalize` the normalized store.
3. Latitude orientation: north→south (FuXi convention) or south→north (CRA's native orientation)? For a south→north
   store use `GRID_025_S` and **do not configure `geometry`** — `geometry` always flips south→north to north→south,
   which feeds the field reversed (measured: +6h z500 RMSE 90 → 486).
4. For a log1p channel (`tp`), `denormalize` yields `log1p(physical value)`, not the value itself. Whether to
   `expm1` depends on the graph: CRA's graph expects the log1p value (**no** `expm1`); the v3 ERA5 stores and
   fengshun expect physical mm (**yes**, `expm1_channels=["tp"]`).

### Units

| Quantity | store2 (raw) | zarr_refresh_v3 (after `denormalize`) | Model expectation → processor |
|---|---|---|---|
| `q` | kg/kg | **g/kg** | FuXi / FGVP / fengshun use g/kg directly; FengQing / Pangu need kg/kg ⇒ `unit_convert {"q": 1e-3}` when switching to v3 |
| `tp` | 1h accumulation, metres (the sfc store); the separate tp store is true 6h accumulation | **log1p(6h mm)** | FuXi / FGVP: 6h mm ⇒ store2 `×6000`; v3 `expm1_channels=["tp"]`. FengQing overrides with the 6h tp store and multiplies by 1000 |
| radiation (`ssr`/`ssrd`/`fdir`/`ttr`) | 1h accumulation, J/m² | **W/m²** (step mean) | 6h Wh/m² ⇒ store2 `×6/3600`; v3 `×6` |
| `z`, `t`, `u`, `v`, `msl`, surface fields | physical | physical | 1:1 |
| CRA c71 / c87 | — | — | normalized store: no unit conversion; `tp` intentionally stays log1p |

### Dataset Processors

Data-side processors run per sample in the DataLoader worker, in the declared order:
`utils/dataset_processors.py`.

| Processor | Parameters | What it does | Pitfalls |
|---|---|---|---|
| `unit_convert` | `scales={"<base name>": factor}` | Multiplies each channel by its base-name factor (`z500`→`z`); unlisted channels ×1 | FuXi/FGVP want g/kg (`q`: 1000); FengQing/Pangu take store2's kg/kg and need no factor. A missed factor does not raise — the whole field is off by that factor |
| `geometry` | `expected_shape` (optional) | Flips latitude to north→south and rolls longitude to 0–360, lazily | Never on a south→north store (CRA) |
| `normalize` | `mean_file="mean.nc"`, `std_file="std.nc"`, `log1p_channels=[]` | `(log1p(x) - mean) / std` using stats next to `model_path` | Channel count/order must equal `input_channels`; needs `model_path` |
| `denormalize` | `stats_path`, `mean_name="mean"`, `std_name="std"`, `expm1_channels`, `exp_clip_max=20` | `x*std + mean`, then optional `expm1` with clamping; stats read from the store and aligned by channel name (`_`-insensitive) | Stats must match the graph head bit-for-bit; CRA: no `expm1`; v3/fengshun: `expm1` tp |
| `normalize_fengqing` | `mean_std_dir` (optional) | FengQing's per-pixel mean `(70,H,W)` and per-channel std `(70,1)`, then NaN→0 | Only accepts exactly 70 channels; finds `mean_std/` relative to `model_path` unless overridden |
| `fill_missing` | `rules={"<channel>": {"method": "constant", "value": v}, "*": ...}`, `unconfigured="error"\|"keep"` | Replaces non-finite values with constants | Only `constant` is supported. Its position relative to `normalize` decides whether you fill physical or normalized values (fuxi21 fills normalized `0.0` after `normalize`) |

Model-side processors run in the main process (`utils/model_processors.py`):

| Processor | Stage | Parameters | What it does |
|---|---|---|---|
| `zero_channels` | `recurrent` | `channels=[...]` | Zeroes diagnostic channels in normalized space before feeding the state back |
| `denormalize` | `output` | `mean_file`, `std_file`, `expm1_channels`, `nonnegative`, `exp_clip_max` | Workspace → physical units using stats next to `model_path` |
| `denormalize_fengqing` | `output` | — | FengQing per-pixel stats and residual reconstruction |

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
| fgvp_v3 | `denormalize` (expm1 tp) → `unit_convert` (rad×6) → `geometry` → `fill_missing` (sst) | — | — |

Configs still default to the raw store2 root; `configs/weather_era_0p25_fgvp_v3.py` is an opt-in A/B recipe for the
v3 stores.

### Adding a New Model / Dataset

1. Declare the model contract (channels, grid, history window, member semantics) with
   `OnnxInferModel.declare(...)` / `Pt2InferModel.declare(...)` and register it in `MODEL_REGISTRY`
   (`utils/model_factory.py`).
2. Pick the store from the inventory, decide raw vs normalized, then write the `dataset.processors` chain from the
   tables above.
3. Write the recipe under `configs/<name>.py` with a module-level `cfg = InferConfig(...)`, following the naming
   convention `<domain>_<source>_<resolution>_<model>[_ens]`.
4. Run one init time with `--steps 1` first to confirm shapes, units, and orientation, then scale up.

The contract points that are easy to get silently wrong are channel order, units, diagnostic channels, NaN channels,
and output representation.

## Key Features

- **Datasources uniformly go through `processed_multi_zarr`**: backed by `xmetai-core`'s
  `MultiZarrDataset`, it reads history frames by initialization time and selects variables in model channel order.
  Input preprocessing (unit conversion / geometry alignment / normalization / fill missing) is declared by
  `dataset.processors` and executed in a DataLoader worker via `dataset_processors.py`.
- **Pluggable backends**: the `onnx` / `pt2` execution engines are decoupled from model semantics.
- **Contract-driven models**: channels, grid, history window, member semantics, and state representation are declared by the model class.
- **Complete run recipes**: a config declares `dataset` (incl. `processors`), `dataloader`, and
  `model_processing` (the `recurrent` and `output` stages).
- **Unified autoregressive orchestration**: `utils/runner.py`'s `Rollout`/`Trajectory` chain single-step `forward`
  into a multi-trajectory loop, where `state = result` feedback and `windowed_output` output shapes converge.
- **Asynchronous disk-write**: a single background thread writes NetCDF; on write failure, it falls back to per-step `raw_step_XXX.npy`.
- **Multi-GPU data parallelism**: work is sharded over cards by "initialization time × member"; each process is isolated to one card via `CUDA_VISIBLE_DEVICES`.

## Directory Structure

```
.
├── pyproject.toml           # PyPI build metadata and CLI entry
├── xmetai_inference/        # The installable Python package (package name: xmetai-inference)
│   ├── cli.py               # Config loading, multi-GPU scheduling, the autoregressive main loop, async disk-write
│   ├── data/                # Dataset 实现，按分类放子包（weather/）
│   ├── utils/               # dataloader_factory（配置 → DataLoader，按 dataset.type 分发）,
│   │                        #   runner（Rollout / Trajectory 自回归编排）,
│   │                        #   model_factory (MODEL_REGISTRY / get_model_class / create_model),
│   │                        #   model-side pipeline (recurrent/output), data-side tensor processors, logging
│   ├── backends/            # BaseInferModel + onnx / pt2 execution engines
│   └── configs/             # Run recipes (`--config <name>`) + declared model contracts;
│                            #   base.py = shared GRID/channels + InferConfig/load_config
├── tests/                   # Unit + integration tests (integration runs only with `-m realdata`)
└── model_artifacts/         # Local model artifacts; the whole directory is excluded from Git
```

## Architecture & Data Flow

The full chain from data to disk for one forecast:

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

| Step | Owner | What it does |
|------|--------|--------|
| Data read | `xmetai-core` | Opens Zarr, selects history frames, maps variables by model channel names |
| Input preprocessing | `dataset.processors` | Raw: physical units → model workspace (unit conversion/geometry alignment/normalization/fill missing); normalized: `denormalize` recovers physical units |
| Autoregression | `Rollout` | `state = result` feedback; `windowed_output` decides where the sliding window is done |
| Recurrent Processor | `model_processing.recurrent` | In-place edits before feedback (e.g. `zero_channels` sets diagnostic channels to normalized 0) |
| Output Processor | `model_processing.output` | Model workspace → physical units (`denormalize` / `denormalize_fengqing`) |
| Disk-write | `_AsyncWriter` / `to_dataset` | Physical units → xarray/NetCDF; `tp` gets a non-negative fallback here |

> Input preprocessing goes through `dataset.processors`; `model_processing.input` must be empty (misconfiguring
> it errors out). Normalization already fused into the model file is not re-executed by the framework.

## Quick Start

### Prerequisites

- Linux environment (driver scripts are bash; on Windows prefer WSL2 or Git Bash)
- NVIDIA GPU + CUDA (ONNX / PT2 inference)
- Model weight files and Zarr store input data (ERA5 / CRA; both are excluded from git)

### Installing Dependencies

```bash
pip install -e .
```

Install optional extras depending on what you run:

```bash
pip install -e ".[netcdf,zarr,onnx]"   # ONNX models (FuXi-Ens / FGVP / FengQing) need onnxruntime
pip install -e ".[netcdf,zarr]"        # NetCDF read/write engines only (FuXi-2.1 / evaluation)
```

> `torch` is a core dependency installed along with `pip install -e .`. Writing output NetCDF and reading
> `mean.nc`/`std.nc` requires `netCDF4` or `h5netcdf` (`netcdf` extra); the underlying Zarr storage is read by
> `xmetai-core` (`zarr` extra); ONNX model runtime needs `onnxruntime` (`onnx` extra).

### One-Command Inference

```bash
xmetai-infer --model weather_era_0p25_fuxi_ens
xmetai-infer --model weather_era_0p25_fuxi21
xmetai-infer --model weather_era_0p25_fgvp
xmetai-infer --model weather_era_0p25_fgvp_ens
xmetai-infer --model weather_era_0p25_fengqing
xmetai-infer --model weather_era_0p25_pangu
xmetai-infer --model weather_cra_0p25_tongyun_ens
xmetai-infer --model s2s_cra_0p25_fengshun20_ens
```

Both `--model` and `--config` select a built-in run recipe (`configs/<name>.py`); the recipe declares the weight path,
dataset, preprocessing chain, initialization times, step count, card count, and output directory. Initialization times,
steps, member count, GPU count, and output directory can be overridden on the command line
(see [Configuration Reference](#configuration-reference)).

When running from the source root without installation:

```bash
python -m xmetai_inference --model weather_era_0p25_fuxi21
```

`xmetai-infer --worker ...` is the internal subprocess entrypoint for multi-GPU scheduling and should not be called manually.

### Overriding Parameters

```bash
xmetai-infer --model weather_era_0p25_fuxi_ens \
  --times "2025010600..2025011200:24" \
  --steps 60 --members 51 --gpus 4 --cuda-devices 0,1,2,3 \
  --vars z500,u200,v200,msl,tp \
  --out /workspace/data/shenzw/fuxi_ens_output
```

- `--model-path` replaces the recipe's weight file; `--model` selects the recipe.
- `--times` syntax is documented in [configs/base.py `parse_times`](xmetai_inference/configs/base.py): a single
  `2025010600`, a comma-separated list, a closed interval `A..B` (24h step by default), and `:N` to override the step.
- An external `.py` recipe loads via `xmetai-infer /path/to/config.py` (or `--config`); see `configs/` for examples.

## Configuration Reference

### `xmetai-infer` Options

| Option | Default | Description |
|------|------|------|
| `config_path` | — | Positional; the external config file form of `--config` |
| `--model` | mutually exclusive with `--config` | Built-in recipe name, e.g. `fuxi_ens`, `fuxi21`, `fgvp`, `fengqing` |
| `--config` | mutually exclusive with `--model` | Built-in config name or an external `.py` config path |
| `--model-path` | recipe value | Overrides the model weight path |
| `--times` | recipe value | Overrides initialization times (`parse_times` syntax) |
| `--members` | recipe value | Overrides ensemble member count |
| `--steps` | recipe value | Overrides forecast step count |
| `--vars` | recipe value | Overrides output variables, comma-separated |
| `--out` | recipe value | Overrides output directory |
| `--gpus` | recipe value | Overrides the number of GPUs used |
| `--cuda-devices` | recipe value | Overrides physical card IDs, comma-separated |
| `--batch-size` | recipe value | Overrides DataLoader batch size |
| `--num-workers` | recipe value | Overrides DataLoader worker count |
| `--log-level` | recipe value | Overrides console log level |

### Environment Variables

| Variable | Applies to | Description |
|------|--------|------|
| `XMETAI_INFERENCE_ROOT` | All | Project root (base for config relative paths / built-in weights lookup) |
| `ERA5_STORE_ROOT` | fuxi21 / fuxi_ens / fgvp | Foundation store2 data root |
| `FUXI21_MODEL_PATH` | fuxi21 | FuXi-2.1 weight `.pt2` path |
| `FUXI_ENS_ONNX` | fuxi_ens | FuXi-Ens weight `.onnx` path |
| `FGVP_ONNX` | fgvp / fgvp_v3 | IWC FGVP GDN2 weight `.onnx` path |
| `FGVP_OPS_LIBRARY` | fgvp / fgvp_v3 | Custom ops library `.so` path |
| `XMETAI_GPU_STATE` | fuxi_ens / fgvp / fgvp_v3 | `1`/`0` overrides the persistent-GPU switch |
| `FENGQING_MEAN_STD_DIR` | fengqing | FengQing `mean_std/` directory |
| `FENGQING_MASKS_PATH` | fengqing | FengQing `constant_masks.npy` path |
| `PANGU_ONNX_24` | pangu | Pangu 24h jump model `.onnx` path (defaults to `pangu_weather_24.onnx` next to the 6h model) |

## License

This project does not yet declare its own license. `TODO:` add a `LICENSE` file if open-sourced.

The bundled Pangu-Weather model weights are licensed under **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike),
for non-commercial use only; see `model_artifacts/Pangu-Weather-main/README.md`.