"""Config-driven weather model inference and evaluation."""

__version__ = "0.1.0"


def main(argv=None):
    """Inference command-line entry point."""
    from .cli import main as _main

    return _main(argv)


__all__ = ["__version__", "main"]
