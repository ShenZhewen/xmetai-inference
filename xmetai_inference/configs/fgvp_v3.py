# -*- coding: utf-8 -*-
"""FGVP 在 zarr_refresh_v3 新 ERA5 盘上的 A/B 测试配方（只换数据源）。

除 ``dataset`` / ``output_dir`` 外与 ``configs/fgvp.py`` 完全一致（权重、起报、
步数、输出变量都不动），方便同一批起报逐日对比：

    python -m xmetai_inference.cli --config fgvp        # store2（现用盘）
    python -m xmetai_inference.cli --config fgvp_v3     # 新盘

测通之前不要改 ``configs/fgvp.py`` 里的 ``ERA5_ROOT``；确认新盘出数正常后，
再把原配置的 dataset 换成 ``era5_stores.fgvp_dataset()`` 即可。

新盘 ERA5_V3_2025 覆盖 2025-01-01 … 2026-07-31。回算到 2024 及更早时，
把 store 换成 ``era5_stores.ERA5_V3_HIST``（78 通道，正好是 FuXi-Ens / FGVP
这套输入），同时把 ``times`` 改到那个时段。
"""
import copy

from xmetai_inference.configs.era5_stores import fgvp_dataset
from xmetai_inference.configs.fgvp import cfg as _store2_cfg

cfg = copy.deepcopy(_store2_cfg)
cfg.name = "fgvp_v3"
cfg.dataset = fgvp_dataset()
cfg.output_dir = "/workspace/data/shenzw/fgvp_output_v3"
