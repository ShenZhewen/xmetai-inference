"""配置驱动的气象模型推理与评测。"""

__version__ = "0.1.0"


def main(argv=None):
    """推理命令行入口。"""
    from .cli import main as _main

    return _main(argv)


__all__ = ["__version__", "main"]
