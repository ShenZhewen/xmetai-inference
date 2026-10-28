# Weather Model Inference Framework

Converts ERA5 reanalysis data into model input, runs autoregressive weather forecasting, and saves the results as NetCDF.

## Table of Contents

- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Architecture & Data Flow](#architecture--data-flow)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [License](#license)

## Supported Models

| Model | Backend | Type | Members | Channels | Normalization |
|------|------|------|------|------|--------|
| FuXi-Ens | `onnx` (ONNX Runtime) | Ensemble | 51 | 78 | Baked into graph; physical units in/out |
| FuXi-2.1 | `pt2` (torch.export) | Deterministic | 1 | 85 | Input `normalize`, output `denormalize` (mean.nc/std.nc) |
| IWC FGVP GDN2 | `onnx` (ONNX Runtime) | Deterministic | 1 | 78 | Baked into graph; custom ops in backbone, `.so` registration required |
| FengQing V1.5Beta | `onnx` (ONNX Runtime) | Deterministic | 1 | 70 | Input `normalize_fengqing`, output `denormalize_fengqing` (graph emits residual) |
| Pangu-Weather | `onnx` (ONNX Runtime) | Deterministic | 1 | 69 | Baked into graph; physical units in/out |
| Tongyun-Ens (CRA) | `onnx` (ONNX Runtime) | Ensemble | 51 | 71 | Datasource stores normalized values → data-side `denormalize`; graph normalizes at its head, outputs physical units |

The first five ERA5 models share the 0.25° global regular grid (721×1440, latitude north→south, longitude 0–360°).
All use a 2-frame history window except Pangu; Pangu uses a single history frame (`history_steps=1`), 6-hour steps,
but every 4 steps (lead 24h) switches to a 24h model that jumps 24h directly from the anchor (official "Plan B").
**Tongyun-Ens (CRA) is the exception**: it uses `GRID_025_S` (latitude south→north, the store's native orientation),
and its preprocessing **must not configure `geometry`** (doing so feeds the field in reversed).

> There are two model-naming schemes — don't mix them: **config names** (for `--model` / `--config`, e.g. `fuxi_ens`,
> `fuxi21`, `fgvp`, `fengqing`, `pangu`, `tongyun_cra`) map to the eponymous recipe files under `configs/`;
> **model-class registry names** (the `model_class=` in configs, e.g. `fuxi_ens_onnx`, `fuxi21_pt2`,
> `iwc_fgvp_gdn2_onnx`, `fengqing_pre_onnx`, `pangu_onnx`, `tongyun_cra_ens_onnx`) map to
> `models/MODEL_REGISTRY`.

### Perturbation Schemes SPA / SPW (composable on any model)

SPA and SPW are two **perturbation schemes**, not standalone base models — they act on a base model to turn a
deterministic model into an ensemble (or to perturb an existing ensemble), and are **decoupled from the base model**,
applicable to any model (as long as that model's backend implements support).

| Scheme | What it does | Declared by | Key parameters | Supported backends |
|---|---|---|---|---|
| **SPA** | Latent-space AR(1) noise forcing: injects time-evolving AR(1) random noise into the model's **latent state**; members = different noise samples | `latent_forcing` | `targets` (injection points), `sigma`, `tau_days`, `rank`, `length_scale`, `seed` | ONNX (graph rewrite), PT2 (fx graph rewrite) |
| **SPW** | Weight perturbation: perturbs specified weight tensors under **energy balance** to produce a set of member weight copies; members = different weight copies | `weight_perturb` | `groups`, `sigma_full`, `balance`, `noise_struct`, `seed` | ONNX |

The two are **orthogonal and composable**: SPA rewrites the graph, SPW rewrites the weights, and both can be enabled on
the same model (currently only ONNX supports them together).
Recipe examples: `fgvp_spa.py`, `fgvp_ens_spa.py` (SPA); `fgvp_spw.py`, `fgvp_ens_spw.py` (SPW);
`fuxi_spa.py` (SPA on PT2).

## Supported Datasets

Datasources are uniformly ingested via `processed_multi_zarr`, backed by `xmetai-core`'s `MultiZarrDataset`.
There are two kinds of **store layout**; identify the layout before plugging in a new datasource
(see `skills/xmetai-inference/references/datasets.md`):

| Datasource | Layout | What it stores | Where normalization happens | `dataset.processors` chain |
|---|---|---|---|---|
| ERA5 foundation store (`era5_pl/sfc/cldrad/soil/wave`) | **Raw** | Physical units (q=kg/kg, tp/radiation = 1h accumulation) | Baked into graph, or framework-side mean.nc/std.nc | `unit_convert` → `geometry` → (optional) `normalize` / `fill_missing` |
| CRA c71 store (`cra...c71.p25.h6.v1`) | **Normalized** | Already-normalized values; mean/std arrays live inside the store | Store normalization + graph head `Div(Sub(x,mean),std)` | `denormalize` (`stats_path`=that store), **no geometry** |
| zarr_refresh_v3 ERA5 stores (`era5.202501_202607.c116...`, `era5.1979_2024.c78...`) | **Normalized** (superset) | zscore float16, all channels in one store (116 / 78); q in g/kg, radiation in W/m², tp = log1p(6h mm) | Store normalization + graph head | `denormalize` (`stats_path`=store, `expm1_channels=["tp"]`) → `unit_convert` → `geometry` → … ; ready-made per-model chains in `configs/era5_stores.py` |

The key to telling them apart: for a **raw** layout, normalization lives on the "graph/framework" side, and the store
provides physical units; for a **normalized** layout, normalization lives on the "store" side, so you must
`denormalize` on the data side to recover physical values before feeding the graph, which then normalizes again.
For a normalized store's log1p channel, denormalization yields `log1p(physical_value)`. Whether to `expm1` it depends
on the graph: CRA's graph expects the log1p value (don't), while FuXi/FGVP/FengQing expect physical mm/6h (do, via
`expm1_channels`). Configs still default to the raw store2 root; `configs/fgvp_v3.py` is an opt-in A/B recipe for the
new stores.

## Key Features

- **Datasources uniformly go through `processed_multi_zarr`**: backed by `xmetai-core`'s
  `MultiZarrDataset`, it reads history frames by initialization time and selects variables in model channel order.
  Input preprocessing (unit conversion / geometry alignment / normalization / fill missing) is declared by
  `dataset.processors` and executed in a DataLoader worker via `dataset_processors.py`.
- **Pluggable backends**: the `onnx` / `pt2` execution engines are decoupled from model semantics.
- **Contract-driven models**: channels, grid, history window, member semantics, and state representation are declared by the model class.
- **Complete run recipes**: a config declares `dataset` (incl. `processors`), `dataloader`, and
  `model_processing` (the `recurrent` and `output` stages).
- **Unified autoregressive orchestration**: `runner.py`'s `Rollout`/`Trajectory` chain single-step `forward`
  into a multi-trajectory loop, where `state = result` feedback and `windowed_output` output shapes converge.
- **Asynchronous disk-write**: a single background thread writes NetCDF; on write failure, it falls back to per-step `raw_step_XXX.npy`.
- **Multi-GPU data parallelism**: work is sharded over cards by "initialization time × member"; each process is isolated to one card via `CUDA_VISIBLE_DEVICES`.

## Directory Structure

```
.
├── pyproject.toml           # PyPI build metadata and CLI entry
├── xmetai_inference/        # The installable Python package (package name: xmetai-inference)
│   ├── cli.py               # Config loading, multi-GPU scheduling, the autoregressive main loop, async disk-write
│   ├── runner.py            # Rollout / Trajectory autoregressive orchestration layer
│   ├── data/                # Dataset 构造：loader 按 type 分发，实现按分类放子包（weather/）
│   ├── utils/               # Model-side pipeline (recurrent/output), data-side tensor processors, logging
│   ├── backends/            # BaseInferModel + onnx / pt2 execution engines
│   ├── models/              # Concrete models + the lazy-loading registry MODEL_REGISTRY
│   └── configs/             # Built-in complete run recipes
├── skills/                  # Integration guide: this skill (SKILL.md routing hub + references/ details)
├── tests/                   # Unit + integration tests (integration runs only with `-m realdata`)
└── model_artifacts/         # Local model artifacts; the whole directory is excluded from Git
```

## Architecture & Data Flow

The full chain from data to disk for one forecast:

```
zarr store  →  xmetai-core MultiZarrDataset          (read history frames, select variables by model channel)
            →  dataset.processors                    (in DataLoader worker, torch; chain depends on store layout)
                 unit_convert → geometry → normalize → fill_missing     ← raw store (ERA5)
                 denormalize (stats_path=store)                         ← normalized store (CRA)
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
xmetai-infer --model fuxi_ens
xmetai-infer --model fuxi21
xmetai-infer --model fgvp
xmetai-infer --model fengqing
xmetai-infer --model pangu
xmetai-infer --model tongyun_cra
```

Both `--model` and `--config` select a built-in run recipe (`configs/<name>.py`); the recipe declares the weight path,
dataset, preprocessing chain, initialization times, step count, card count, and output directory. Initialization times,
steps, member count, GPU count, and output directory can be overridden on the command line
(see [Configuration Reference](#configuration-reference)).

When running from the source root without installation:

```bash
python -m xmetai_inference --model fuxi21
```

`xmetai-infer --worker ...` is the internal subprocess entrypoint for multi-GPU scheduling and should not be called manually.

### Overriding Parameters

```bash
xmetai-infer --model fuxi_ens \
  --times "2025010600..2025011200:24" \
  --steps 60 --members 51 --gpus 4 --cuda-devices 0,1,2,3 \
  --vars z500,u200,v200,msl,tp \
  --out /workspace/data/shenzw/fuxi_ens_output
```

- `--model-path` replaces the recipe's weight file; `--model` selects the recipe.
- `--times` syntax is documented in [configs/base.py `parse_times`](xmetai_inference/configs/base.py): a single
  `2025010600`, a comma-separated list, a closed interval `A..B` (24h step by default), and `:N` to override the step.
- An external `.py` recipe loads via `xmetai-infer /path/to/config.py` (or `--config`); how to add a new model /
  dataset / processor is covered in [skills/xmetai-inference](skills/xmetai-inference/).

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
| `FGVP_ONNX` | fgvp | IWC FGVP GDN2 weight `.onnx` path |
| `FGVP_OPS_LIBRARY` | fgvp | Custom ops library `.so` path |
| `XMETAI_GPU_STATE` | fuxi_ens / fgvp | `1`/`0` overrides the persistent-GPU switch |
| `FENGQING_MEAN_STD_DIR` | fengqing | FengQing `mean_std/` directory |
| `FENGQING_MASKS_PATH` | fengqing | FengQing `constant_masks.npy` path |
| `PANGU_ONNX_24` | pangu | Pangu 24h jump model `.onnx` path (defaults to `pangu_weather_24.onnx` next to the 6h model) |

## License

This project does not yet declare its own license. `TODO:` add a `LICENSE` file if open-sourced.

The bundled Pangu-Weather model weights are licensed under **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike),
for non-commercial use only; see `model_artifacts/Pangu-Weather-main/README.md`.