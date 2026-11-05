# -*- coding: utf-8 -*-
from xmetai_inference.configs.base import InferConfig, ModelProcessingConfig

ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
MODEL_DIR = "/workspace/tmp/douzsh/models"
SRC_GRAPH = f"{MODEL_DIR}/20261008/iwc_gssp_im_fp16.onnx"
OPS_LIBRARY = f"{MODEL_DIR}/xmetai_onnx_plugins.ort1_24.cuda12.so"
MODEL_ONNX_DIR = "/workspace/data/shenzw/model_onnx"

FGVP_DYNAMIC_STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]

PERTURB_GRAPHS = 8
RUNS_PER_GRAPH = 6
MEMBERS = PERTURB_GRAPHS * RUNS_PER_GRAPH

BASE_EXCLUDE = ("mean", "std", "const", "mask", "attn_mask", "ones", "zeros")


cfg = InferConfig(
    name="weather_era_0p25_fgvp_ens_spw",
    model_path=SRC_GRAPH,
    ops_library=OPS_LIBRARY,
    gpu_state=False,
    model_class="iwc_fgvp_mamba_ens_onnx",
    model_batch=6,
    gpu_mem=0.9,
    weight_perturb={
        "groups": ("latent_mixer",),
        "sigma_full": 0.027,
        "balance": "energy",
        "exclude": BASE_EXCLUDE,
        "noise_struct": "channel",
        "min_elements": 64,
        "member_dir": MODEL_ONNX_DIR,
        "seed": 0,
    },
    perturb_graphs=PERTURB_GRAPHS,

    dataset={
        "type": "processed_multi_zarr",
        "paths": FGVP_DYNAMIC_STORES,
        "processors": [
            {
                "name": "unit_convert",
                "scales": {
                    "q": 1000.0,
                    "ssr": 6.0 / 3600.0,
                    "ssrd": 6.0 / 3600.0,
                    "fdir": 6.0 / 3600.0,
                    "ttr": 6.0 / 3600.0,
                    "tp": 6000.0,
                },
            },
            {"name": "geometry"},
            {
                "name": "fill_missing",
                "rules": {"sst": {"method": "constant", "value": 287.066}},
                "unconfigured": "error",
            },
        ],
    },
    dataloader={
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "drop_last": False,
    },

    model_processing=ModelProcessingConfig(input=[], recurrent=[], output=[]),

    times="2025070200..2025073000:24,2025110100..2025113000:24",
    steps=60,
    hour_interval=6,
    members=MEMBERS,
    vars="z500,u200,v200,msl,tp",
    gpus=4,
    cuda_devices="0,1,2,3",
    intra_op_threads=0,
    cudnn_algo_search="",
    resident_members=2,
    output_dir="/workspace/data/shenzw/fgvp_ens_spw_std",
)
