#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AIFS 最小闭环：读输入→插值 N320→SimpleRunner 跑 1 步(6h)→存 N320 结果 + 打印量级。

目的是**先在服务器上验证输入装配 + 插值顺序 + 单位正确**，再铺开成完整 infer_aifs。
跑通后重点看两件事：
  1. 输入自检打印里 z_500 中位数 ≈ 5.4e4（位势 m²/s²）、q_850 中位数 ≈ 5e-3（kg/kg）；
  2. 输出 z_500 中位数仍 ≈ 5.4e4，量级与输入一致、无离谱 NaN/爆点。

用法（服务器，环境已装 anemoi-inference / anemoi-models / torch-geometric / flash_attn）：
  python aifs_minimal.py --checkpoint /path/aifs_single_v1.1.ckpt --time 2025010600 --lead 6 --out aifs_smoke

注意：GPU 非确定（README 明确说明），无法与官方逐位一致，只能量级对拍。
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", "16")

import numpy as np

from build_input_aifs import build_aifs_fields, load_spec, self_check
from models.aifs_anemoi import AifsAnemoiModel


def _stats(name, arr):
    v = np.asarray(arr, dtype=np.float64).ravel()
    fin = v[np.isfinite(v)]
    if not fin.size:
        return f"{name:<8} 全 NaN（{v.size} 节点）"
    return (f"{name:<8} min={fin.min():.4g} median={np.median(fin):.4g} "
            f"max={fin.max():.4g} NaN={v.size - fin.size}")


def main():
    p = argparse.ArgumentParser(description="AIFS 1.1 最小闭环冒烟测试")
    p.add_argument("--checkpoint", required=True, help="aifs_single_v1.1.ckpt 路径")
    p.add_argument("--time", default="2025010600", help="起报时间 YYYYMMDDHH")
    p.add_argument("--lead", type=int, default=6, help="预报时长(小时)，默认 6h=1 步")
    p.add_argument("--spec", default="aifs11.json")
    p.add_argument("--out", default="aifs_smoke", help="输出目录")
    args = p.parse_args()

    spec = load_spec(args.spec)
    self_check(spec)

    print(f"[aifs] 构建输入 state（起报 {args.time}）...")
    input_state = build_aifs_fields(args.time, spec=spec, verbose=True)
    n_grid = next(iter(input_state["fields"].values())).shape[-1]

    print(f"[aifs] 加载 checkpoint：{args.checkpoint}")
    model = AifsAnemoiModel(args.checkpoint, device="cuda")

    print(f"[aifs] 跑 {args.lead}h 预报 ...")
    final = None
    for state in model.run(input_state, lead_time=args.lead):
        final = state

    assert final is not None, "没有拿到任何 step 输出"
    print(f"[aifs] 输出 date={final.get('date')} step={final.get('step')} "
          f"字段数={len(final['fields'])}")

    print("\n== 输出字段抽样（N320 节点）==")
    keys = [k for k in ("z_500", "t_850", "u_850", "2t", "msl", "sp", "tp",
                        "100u", "ssrd", "tcc", "stl1", "swvl1")
            if k in final["fields"]]
    for k in keys:
        print("  " + _stats(k, final["fields"][k]))

    # 落盘：整个输出 state 的字段 + 节点经纬度，方便进一步核对/绘图
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"state_{args.time}_lead{args.lead}.npz")
    save = {k: np.asarray(v) for k, v in final["fields"].items()}
    if "latitudes" in final:
        save["_latitudes"] = np.asarray(final["latitudes"])
    if "longitudes" in final:
        save["_longitudes"] = np.asarray(final["longitudes"])
    np.savez(out_path, **save)
    print(f"\n[完成] 已写入 {out_path}（{len(save)} 项）")


if __name__ == "__main__":
    main()
