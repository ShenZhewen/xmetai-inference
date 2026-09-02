# -*- coding: utf-8 -*-
"""推理与评测命令共用的日志配置。"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler


class _RuntimeContextFilter(logging.Filter):
    def __init__(self, rank: int | str = "-"):
        super().__init__()
        self.rank = str(rank)

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


def configure_logging(
    *,
    level: str = "INFO",
    log_file: str | None = None,
    console: bool = True,
    rank: int | str = "-",
) -> None:
    """Configure root logging with optional rotating file output."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效日志级别 {level!r}，应为 DEBUG/INFO/WARNING/ERROR")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if log_file else numeric_level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    context_filter = _RuntimeContextFilter(rank)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s [pid=%(process)d rank=%(rank)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        root.addHandler(handler)

    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(directory, exist_ok=True)
        handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        root.addHandler(handler)

    if not root.handlers:
        root.addHandler(logging.NullHandler())

    for name in (
        "asyncio",
        "eccodes",
        "eccodeslib",
        "eckitlib",
        "findlibs",
        "fsspec",
        "matplotlib",
        "numcodecs",
        "urllib3",
        "zarr",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
