# -*- coding: utf-8 -*-
"""配置基座的 re-export。

每个 ``configs/<name>.py`` 既是配方（``cfg``）也是模型类；变体配方只定义 cfg。
公共声明在 ``configs/base.py``，这里原样 re-export。
"""
from .base import *                      # noqa: F401,F403
from .base import __all__ as _base_all

__all__ = list(_base_all)
