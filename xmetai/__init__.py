"""Configuration-driven weather model inference and evaluation."""

__version__ = "0.1.0"


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
    """Run inference through the public Python API."""
    from xmetai.api import infer as _infer

    return _infer(
        model,
        data,
        model_path=model_path,
        data_root=data_root,
        times=times,
        steps=steps,
        members=members,
        variables=variables,
        out=out,
        gpus=gpus,
        cuda_devices=cuda_devices,
        log_level=log_level,
    )


__all__ = ["__version__", "infer"]
