# -*- coding: utf-8 -*-
"""探测 era5_foundation_store2 三个 zarr store 的全部通道。

只做元数据层（open + 读坐标/属性），不碰数据块，秒级完成、内存近零。
输出每个通道的：名字 / 维度 / 形状 / 单位 / long_name，外加时间覆盖与网格。
"""
import xarray as xr

# 路径写死成字面量，要换根目录直接改这里
ERA5_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
STORES = [
    f"{ERA5_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
    f"{ERA5_ROOT}/era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
]


def open_store(path):
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)


def main():
    for path in STORES:
        print("=" * 96)
        print(path)
        ds = open_store(path)

        print(f"全局属性: {dict(ds.attrs) or '(无)'}")
        print(f"坐标: {', '.join(f'{c}{ds[c].shape}' for c in ds.coords)}")

        # 时间覆盖与网格
        if "time" in ds.coords:
            t = ds["time"].values
            print(f"time : {len(t)} 步, {t[0]} ~ {t[-1]}")
        for axis in ("lat", "lon"):
            if axis in ds.coords:
                a = ds[axis]
                print(f"{axis:<5}: {a.size} 点, "
                      f"[{float(a.min())}, {float(a.max())}]")

        # 通道编码在维坐标里的形态（比如 channel: 84）
        for c in ds.coords:
            if (ds[c].dims == (c,) and ds[c].size > 1
                    and c not in ("time", "lat", "lon")):
                vals = list(ds[c].values)
                shown = ", ".join(str(v) for v in vals[:40])
                more = "" if len(vals) <= 40 else f" ... 共 {len(vals)} 个"
                print(f"维度坐标 {c}（{len(vals)} 个）: {shown}{more}")

        # 每个通道一个变量的形态：逐个列出
        chans = sorted(ds.data_vars)
        if chans:
            print(f"\n数据通道共 {len(chans)} 个:")
            for name in chans:
                da = ds[name]
                units = da.attrs.get("units", "")
                desc = da.attrs.get(
                    "long_name", da.attrs.get("standard_name", "")
                )
                print(f"  {name:<14} {str(da.dims):<22} "
                      f"{str(da.shape):<22} {units:<10} {desc}")

            # 按家族归组（取下划线前的词），看通道数对不对得上 c84/c15/c8
            families = {}
            for name in chans:
                fam = name.split("_")[0]
                families[fam] = families.get(fam, 0) + 1
            print("\n按前缀归组:",
                  ", ".join(f"{k}×{v}" for k, v in sorted(families.items())))
        else:
            print("\n(没有逐变量通道——通道在维坐标里，见上方「维度坐标」)")
        ds.close()


if __name__ == "__main__":
    main()
