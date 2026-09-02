"""Public Python API."""

from __future__ import annotations

from xmetai.inference import main as inference_main


def infer(
    model,
    data=None,
    *,
    model_path=None,
    data_root=None,
    times=None,
    steps=None,
    members=None,
    variables=None,
    out=None,
    gpus=None,
    cuda_devices=None,
    log_level=None,
):
    """Run a registered model recipe with optional data and runtime overrides."""
    argv = ["--model", str(model)]
    options = (
        ("--data", data),
        ("--model-path", model_path),
        ("--data-root", data_root),
        ("--times", times),
        ("--steps", steps),
        ("--members", members),
        ("--vars", variables),
        ("--out", out),
        ("--gpus", gpus),
        ("--cuda-devices", cuda_devices),
        ("--log-level", log_level),
    )
    for option, value in options:
        if value is not None:
            argv.extend([option, str(value)])
    return inference_main(argv)
