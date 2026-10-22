# -*- coding: utf-8 -*-
"""把 aBt15 首个起报（20250702 00Z）的模型输入和归档参考复制到 spa_test_data。

归档里只有输出（001.nc..060.nc），**没有**那次推理喂进图的输入；输入必须
用项目自己的数据管线重建，才能与归档同源（同 zarr、同 unit_convert /
geometry / fill_missing、同通道序）：

    configs/fgvp_ens_ar1state_aBt15.py -> cfg.dataset
    create_processed_multi_zarr_dataset(cfg.dataset, ..., init_times=[2025-07-02 00:00])
    -> sample["inputs"]  [2, 78, 721, 1440]  float32
       （模型入图前的物理量；图内归一化已烘焙，不需要再处理）

同时复制归档 rank0 前 4 行对应的成员 0/4/8/12 的 001..004.nc，并写
manifest.json 记录行号 <-> 成员号映射。

输出目录（脚本顶部可改）：
    /workspace/data/shenzw/inference_test_data/spa_test_data/
        fgvp_input_20250702.npy
        fgvp_input_20250702.json
        reference/20250702/member_000/001.nc ...
        manifest.json

运行（aBt15 用的那个环境，项目根目录）：
    cd /workspace/szwCode/xmetai-inference2
    python copy_spa_test_data.py

只想核对不写盘？把 DRY_RUN 改成 True。
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---- 直接改这里 ----
INIT_TIME = datetime(2025, 7, 2, 0, 0, 0)
OUT_DIR = Path(
    "/workspace/data/shenzw/inference_test_data/spa_test_data")
ARCHIVE_ROOT = Path("/workspace/data/shenzw/fgvp_ens_ar1state_aBt15")
ARCHIVE_INIT = "20250702"
ARCHIVE_MEMBERS = (0, 4, 8, 12)      # 归档 rank0 前 4 行
ARCHIVE_STEPS = (1, 2, 3, 4)         # 文件号 = code step + 1；001 = lead 6h
CONFIG_MODULE = "xmetai_inference.tmp_config.fgvp_ens_ar1state_aBt15"
DRY_RUN = False


def _check_project():
    try:
        from importlib import import_module

        from xmetai_inference.data.datasets.processed_multi_zarr import (
            create_processed_multi_zarr_dataset,
        )
        from xmetai_inference.models import get_model_class
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "导入项目失败；请在项目根目录、用 aBt15 那个环境运行：\n"
            "  cd /workspace/szwCode/xmetai-inference2\n"
            "  python copy_spa_test_data.py\n"
            f"原始错误：{exc}"
        ) from exc
    return import_module, create_processed_multi_zarr_dataset, get_model_class


def _frame_times(dataset):
    """sample 实际读的两帧时间（base.inds[position] 起的 hist_frames 帧）。"""
    try:
        import pandas as pd

        base = dataset.base
        raw_idx = int(base.inds[dataset.positions[0]])
        return [
            str(pd.Timestamp(base.id_to_time[int(raw_idx) + i]))
            for i in range(int(base.hist_frames))
        ]
    except Exception:  # noqa: BLE001 - 只是诊断信息
        return None


def build_input():
    import_module, create_dataset, get_model_class = _check_project()
    cfg = import_module(CONFIG_MODULE).cfg
    model_cls = get_model_class(cfg.model_class)

    dataset, metadata = create_dataset(
        cfg.dataset,
        model_cls=model_cls,
        init_times=[INIT_TIME],
        history_steps=model_cls.history_steps,
        hour_interval=cfg.hour_interval,
        model_path=cfg.model_path,
    )
    sample = dataset[0]
    inputs = sample["inputs"]
    arr = inputs.detach().cpu().numpy() if hasattr(inputs, "detach") \
        else np.asarray(inputs)
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    frames = int(model_cls.history_steps)
    channels = len(model_cls.input_channels)
    height = int(model_cls.grid["lat"]["size"])
    width = int(model_cls.grid["lon"]["size"])
    expected = (frames, channels, height, width)
    if arr.shape != expected:
        raise RuntimeError(f"输入形状 {arr.shape} != {expected}")

    # cfg.dataset 可能是 dict/DictConfig，也可能不是；元信息只是诊断用
    try:
        source_zarr = [str(p) for p in cfg.dataset.get("paths", ())]
        processors = cfg.dataset.get("processors")
    except (AttributeError, TypeError):
        source_zarr = None
        processors = None

    info = {
        "init_time": INIT_TIME.isoformat(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "channel_names": [str(name) for name in model_cls.input_channels],
        "frame_times": _frame_times(dataset),
        "config": CONFIG_MODULE,
        "model_class": str(cfg.model_class),
        "source_zarr": source_zarr,
        "processors": processors,
        "note": (
            "模型入图前的物理量；单位换算/几何/fill_missing 已按 cfg.dataset "
            "处理，归一化与诊断通道清零烘焙在图内。"
        ),
    }
    stats = {
        "finite_frac": float(np.isfinite(arr).mean()),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr, dtype=np.float64)),
    }
    return arr, info, stats


def copy_reference():
    copied = []
    for member in ARCHIVE_MEMBERS:
        for step in ARCHIVE_STEPS:
            src = (ARCHIVE_ROOT / ARCHIVE_INIT / f"member_{member:03d}"
                   / f"{step:03d}.nc")
            if not src.is_file():
                raise SystemExit(
                    f"归档参考不存在：{src}\n"
                    f"检查 ARCHIVE_ROOT/ARCHIVE_INIT（当前 "
                    f"{ARCHIVE_ROOT}/{ARCHIVE_INIT}）")
            dst = (OUT_DIR / "reference" / ARCHIVE_INIT
                   / f"member_{member:03d}" / src.name)
            if not DRY_RUN:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied.append({
                "archive_member": member,
                "archive_step": step,
                "source": str(src),
                "dest": str(dst.relative_to(OUT_DIR)),
                "bytes": int(src.stat().st_size),
            })
    return copied


def main():
    out_dir = OUT_DIR
    if not DRY_RUN:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("aBt15 测试数据准备")
    print("=" * 78)
    print(f"init    : {INIT_TIME.isoformat()}")
    print(f"output  : {OUT_DIR}  DRY_RUN={DRY_RUN}")
    print(f"archive : {ARCHIVE_ROOT}/{ARCHIVE_INIT} "
          f"members={list(ARCHIVE_MEMBERS)} steps={list(ARCHIVE_STEPS)}")

    arr, info, stats = build_input()
    npy_path = out_dir / "fgvp_input_20250702.npy"
    json_path = out_dir / "fgvp_input_20250702.json"
    if not DRY_RUN:
        np.save(npy_path, arr)
        json_path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[input]")
    print(f"  shape={arr.shape} dtype={arr.dtype} "
          f"bytes={int(arr.nbytes)}")
    print(f"  finite={stats['finite_frac']:.6f} "
          f"min={stats['min']:.6g} max={stats['max']:.6g} "
          f"mean={stats['mean']:.6g}")
    print(f"  frames={info['frame_times']}")
    print(f"  channels={len(info['channel_names'])} "
          f"(first={info['channel_names'][:3]}, "
          f"last={info['channel_names'][-3:]})")
    if not DRY_RUN:
        print(f"  -> {npy_path}")
        print(f"  -> {json_path}")

    print("\n[reference]")
    copied = copy_reference()
    total = sum(item["bytes"] for item in copied)
    for item in copied:
        print(f"  member_{item['archive_member']:03d}/"
              f"{item['archive_step']:03d}.nc  "
              f"{item['bytes'] / 1e6:8.1f} MB")
    print(f"  共 {len(copied)} 个文件，{total / 1e6:.1f} MB"
          + ("（DRY_RUN 未写）" if DRY_RUN else ""))

    manifest = {
        "init_time": INIT_TIME.isoformat(),
        "input_file": str(npy_path.name),
        "input_meta_file": str(json_path.name),
        "input_shape": list(arr.shape),
        "input_dtype": str(arr.dtype),
        "input_stats": stats,
        "archive": {
            "source_root": str(ARCHIVE_ROOT),
            "init": ARCHIVE_INIT,
            "members": list(ARCHIVE_MEMBERS),
            "steps": list(ARCHIVE_STEPS),
            "reference_dir": f"reference/{ARCHIVE_INIT}",
            "files": copied,
        },
        "member_mapping": (
            "example 的 row i (0..3) <-> 归档 member 4*i "
            "（rank0 的 AR(1) 流前 4 行）；归档 member id 是全局成员号。"
        ),
        "xi_note": (
            "归档 rank0 的 ξ 流是 13 行（51 成员 / 4 卡向上取整）。example "
            "自己 4 行的流在 step 0 与它相同，step>=1 起不同（advance 的 "
            "噪声形状不同）；做归档对拍必须把 example 的 ξ 换成归档流前 4 行，"
            "见 verify_example.py 的 archive-xi 那一遍。"
        ),
        "native_noise_note": (
            "FGVP 图里有 RandomNormalLike，且没有 perturb_std_scale 输入；"
            "图内 ε 按 worker 内第 k 次前向对齐，同调用序可逐位比较。归档那次"
            " batch=6、成员分块 6/6/1（每步 3 次前向），example 是 1 块 × 4 步，"
            "调用序不同 ⇒ 归档对拍只能近似；逐位一致的证明靠同 feeds、"
            "同调用序的 A/B。"
        ),
    }
    if not DRY_RUN:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"  -> {out_dir / 'manifest.json'}")

    print("\n下一步：")
    print("  1) python find_abt15_graph.py   # 拿 --model")
    print("  2) python verify_example.py \\")
    print("       --model /workspace/data/shenzw/model_onnx/"
          "iwc_gssp_im_fp16.ar1-<MATCH>.onnx \\")
    print(f"       --input {npy_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
