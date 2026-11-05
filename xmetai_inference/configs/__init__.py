# -*- coding: utf-8 -*-
"""Re-exports of the config base.

Each ``configs/<name>.py`` is both a recipe (``cfg``) and a model class; variant recipes define only cfg.
Common declarations live in ``configs/base.py`` and are re-exported verbatim here.
"""
from .base import *                      # noqa: F401,F403
from .base import __all__ as _base_all

__all__ = list(_base_all)
