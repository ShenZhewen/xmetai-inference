# -*- coding: utf-8 -*-
"""ERA5 输入盘目录，以及 zarr_refresh_v3 新盘（归一化态）的 dataset 装配。

两套盘：

1) **现用盘** ``ERA5_STORE2_ROOT``（原始态、物理量、按变量族拆多个 store）：
   ``era5_pl/sfc/cldrad/soil/wave_2025.01-2026.07.*.zarr``。
   各 config 直接拼路径，单位约定见 ``processing/tensor_processors.py`` 顶部注释。

2) **新盘** ``ERA5_V3_*``（zarr_refresh_v3，归一化态、zscore float16、单 store 全通道）：
   ``era5.202501_202607.c116.p25.h6.v3``  116 通道，2025-01-01 … 2026-07-31
   ``era5.1979_2024.c78.p25.h6.v1``        78 通道，1979-01-01 … 2024-12-31（回算用）

   新盘读出来先 ``denormalize``（mean/std 就在 store 里），逐通道预处理与 store2
   不同，还原成物理量之后还要按模型期望换算：

   ==========================  ============================  ==========================
   通道                        新盘（还原后）                 model 期望 → 处理
   ==========================  ============================  ==========================
   ``q``                       g/kg                          FuXi/FGVP 直接用；
                                                             风清/Pangu 要 kg/kg ⇒ ×0.001
                                                             （风清的单位烧在
                                                             ``mean_std/*.npy`` 里，
                                                             少乘不报错、整场偏 1000×）
   ``ssr/ssrd/fdir/ttr``       W/m²（时段平均通量）           6h Wh/m² ⇒ ×6
   ``tp``                      log1p(6h 累积 mm)              mm/6h ⇒ ``expm1_channels``
   其余（z/t/u/v/msl/t2m/…）   物理量                         直接用
   ==========================  ============================  ==========================

   注意 ``tp`` 与 CRA 相反：CRA 的图要的就是 log1p 空间，这里要给物理量。

下面的 ``*_dataset()`` 与各 config 现有 dataset 块一一对应（processor 链的顺序、
fill_missing 规则都照抄 store2 版本）。切到新盘 = 把 config 里的 ``dataset=`` 换成
对应函数；其余字段（权重、起报、步数、输出变量）一律不动，跑完再 A/B 比。
要切回 store2 只需换回原 dataset 块。

**默认仍是 store2**：现有 config 一个都没改，这些函数只在显式调用时生效。
"""

# 现用盘根（历史约定，部分 config 仍直接引用这个常量）
ERA5_STORE2_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"

# zarr_refresh_v3 新盘
V3_ROOT = "/workspace/data/zarr_refresh_v3"
ERA5_V3_2025 = f"{V3_ROOT}/era5.202501_202607.c116.p25.h6.v3"
ERA5_V3_HIST = f"{V3_ROOT}/era5.1979_2024.c78.p25.h6.v1"

# 新盘辐射是 W/m²（6h 时段平均通量），模型输入要 6h Wh/m²。
_RADIATION_6H = {"ssr": 6.0, "ssrd": 6.0, "fdir": 6.0, "ttr": 6.0}


def _v3_processors(store, *, radiation=True, precip=True, q_scale=None):
    """归一化态 → 物理量（模型输入单位）：denormalize + 按需的 unit_convert。"""
    processors = [{
        "name": "denormalize",
        "stats_path": store,
        # tp 是 log1p 归一化，模型要物理量；辐射/q 只做线性换算，留在下一步。
        "expm1_channels": ["tp"] if precip else [],
    }]
    scales = dict(_RADIATION_6H) if radiation else {}
    if q_scale is not None:
        scales["q"] = float(q_scale)
    if scales:
        processors.append({"name": "unit_convert", "scales": scales})
    return processors


def _v3_dataset(store, processors):
    return {
        "type": "processed_multi_zarr",
        "paths": [store],
        "processors": processors,
    }


def fgvp_dataset(store=ERA5_V3_2025):
    """FGVP / FGVP-Ens（78 通道）→ 对应 ``configs/fgvp.py`` 的 dataset 块。"""
    return _v3_dataset(store, _v3_processors(store) + [
        {"name": "geometry"},
        # sst 在陆地上是缺测。原 fgvp.py 就有这条；原 fuxi_ens.py 没有，
        # 但新盘同样会给 NaN，所以这里一并补上（有值时不生效）。
        {"name": "fill_missing",
         "rules": {"sst": {"method": "constant", "value": 287.066}},
         "unconfigured": "error"},
    ])


def fuxi_ens_dataset(store=ERA5_V3_2025):
    """FuXi-Ens（78 通道）→ 对应 ``configs/fuxi_ens.py`` 的 dataset 块。"""
    return _v3_dataset(store, _v3_processors(store) + [
        {"name": "geometry"},
        {"name": "fill_missing",
         "rules": {"sst": {"method": "constant", "value": 287.066}},
         "unconfigured": "error"},
    ])


def fuxi21_dataset(store=ERA5_V3_2025):
    """FuXi-2.1（85 通道）→ 对应 ``configs/fuxi21.py`` 的 dataset 块。

    还原成物理量后照旧走框架侧 normalize（mean.nc/std.nc 在模型文件旁），
    所以「先 expm1 成 mm/6h、再由 normalize 做 log1p」正好抵消。
    """
    return _v3_dataset(store, _v3_processors(store) + [
        {"name": "geometry"},
        {
            "name": "normalize",
            "mean_file": "mean.nc",
            "std_file": "std.nc",
            "log1p_channels": ["tp"],
        },
        {
            "name": "fill_missing",
            "rules": {"*": {"method": "constant", "value": 0.0}},
            "unconfigured": "keep",
        },
    ])


def fengqing_dataset(store=ERA5_V3_2025):
    """FengQing（70 通道）→ 对应 ``configs/fengqing.py`` 的 dataset 块。

    原配置要额外拼一个 6h 累积 tp store 顶掉 1h 的 tp；新盘的 tp 本身就是
    6h 累积，单 store 就够。

    但 q 要换算：风清的逐像素统计量（``mean_std/*.npy``）是按 **kg/kg** 建的
    ——原配置直接喂 store2 的 kg/kg，没有任何 ``unit_convert``。新盘还原来的是
    g/kg，这里必须 ×0.001 换回去。`normalize_fengqing` 不校验单位，漏了不报错，
    只会整场偏 1000 倍。
    """
    return _v3_dataset(store, _v3_processors(
        store, radiation=False, q_scale=1e-3) + [
        {"name": "geometry"},
        {"name": "normalize_fengqing"},
    ])


def pangu_dataset(store=ERA5_V3_2025):
    """Pangu（69 通道）→ 对应 ``configs/pangu.py`` 的 dataset 块。

    Pangu 不吃辐射/tp，只要 q 从 g/kg 换回 kg/kg（store2 本来就是 kg/kg，
    所以原配置没有 unit_convert）。
    """
    return _v3_dataset(store, _v3_processors(
        store, radiation=False, precip=False, q_scale=1e-3) + [
        {"name": "geometry"},
    ])
