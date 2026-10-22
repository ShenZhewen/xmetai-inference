# -*- coding: utf-8 -*-
"""在 iwc_gssp_im_fp16.ar1-*.onnx 里认出 aBt15 用的那张 SPA 图。

为什么不能靠文件名认：图名是
    <源图 stem>.ar1-<LatentForcingSpec.fingerprint(源图)>.onnx
指纹只含 targets / sigma / rank / length_scale / seed / tokens / channels /
grid / input_name / format（见 weights/spa.py 的 budgets()）。
aBt15 的 amp_profile 和 tau_days **只在驱动侧生效、不进指纹**，所以 aBt15
和 aB/s04 复用同一张图 —— 文件名不会出现 aBt15。

aBt15 的图参数（从 configs/fgvp_ens_ar1state_s04.py / _aB.py 继承）：
    targets      = add_35782, add_35787, add_35792   (decoder 入口 / main)
    sigma        = 0.04
    rank         = 16
    length_scale = 32.0
    seed         = 0
    input        = spw/xi

脚本做两件事（都只读）：
  1) 读 15 张图的 metadata，按上面的 budgets 标出 MATCH；
  2) 如果归档的 rank_*.log 里出现过 iwc_gssp_im_fp16.ar1-<hash>.onnx，
     把路径打出来 —— 那是归档那次运行真正加载的图，优先级高于推断。

运行（服务器）：
    cd /workspace/szwCode/xmetai-inference2
    python find_abt15_graph.py

没有 MATCH、日志里也没有时，才需要重建；命令见文件末尾的 BUILD_HINT。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

GRAPH_DIR = "/workspace/data/shenzw/model_onnx"
GRAPH_GLOB = "iwc_gssp_im_fp16.ar1-*.onnx"
ARCHIVE_DIR = "/workspace/data/shenzw/fgvp_ens_ar1state_aBt15"
LOG_GLOB = "rank_*.log"

EXPECTED_TARGETS = ["add_35782", "add_35787", "add_35792"]
EXPECTED = {
    "targets": EXPECTED_TARGETS,
    "sigma": 0.04,
    "rank": 16,
    "length_scale": 32.0,
    "seed": 0,
}

GRAPH_RE = re.compile(
    r"([^\s'\"`]*iwc_gssp_im_fp16(?:\.k[\d.]+)?\.ar1-[0-9a-f]{10}\.onnx)")

BUILD_HINT = """\
重建方法（只在扫描不到时用；图名由指纹决定，重复构建是幂等的）：

  改 /workspace/szwCode/xmetai-inference2/xmetai_inference/weights/spa.py 末尾：
    MODEL = "/workspace/tmp/douzsh/models/20261008/iwc_gssp_im_fp16.onnx"
    ACTION = "build"
    TARGETS = ("add_35782", "add_35787", "add_35792")
    SIGMA = 0.04
    RANK = 16
    LENGTH_SCALE = 32
    SEED = 0
    TAU_DAYS = 3.0
    HOUR_INTERVAL = 6.0
    CACHE_DIR = "/workspace/data/shenzw/model_onnx"
  然后运行：
    cd /workspace/szwCode/xmetai-inference2
    python -m xmetai_inference.weights.spa
"""


def _meta(path):
    import onnx

    model = onnx.load(path, load_external_data=False)
    try:
        return {p.key: p.value for p in model.metadata_props}
    finally:
        del model


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(value, expected, tol=1e-6):
    got = _num(value)
    return got is not None and abs(got - expected) <= tol * max(1.0, abs(expected))


def _match(meta):
    """按 budgets 判断；spec_json 是权威来源，缺失时退回 metadata 字符串。"""
    checks = []
    spec = None
    try:
        spec = json.loads(meta.get("spw.ar1.spec_json", ""))
    except (TypeError, ValueError):
        spec = None

    if isinstance(spec, dict):
        targets = [str(v) for v in (spec.get("targets") or [])]
        checks = [
            ("targets", targets == EXPECTED_TARGETS),
            ("sigma", _close(spec.get("sigma"), EXPECTED["sigma"])),
            ("rank", int(spec.get("rank") or -1) == EXPECTED["rank"]),
            ("length_scale",
             _close(spec.get("length_scale"), EXPECTED["length_scale"])),
            ("seed", spec.get("seed") == EXPECTED["seed"]),
            ("input_name", str(spec.get("input_name") or "") == "spw/xi"),
        ]
    else:
        targets = [v.strip() for v in
                   str(meta.get("spw.ar1.targets", "")).split(" | ") if v.strip()]
        checks = [
            ("targets", targets == EXPECTED_TARGETS),
            ("sigma", _close(meta.get("spw.ar1.sigma"), EXPECTED["sigma"])),
            ("rank", int(_num(meta.get("spw.ar1.rank")) or -1)
             == EXPECTED["rank"]),
            ("length_scale",
             _close(meta.get("spw.ar1.length_scale"),
                    EXPECTED["length_scale"])),
            ("seed", str(meta.get("spw.ar1.seed", "")).strip() == "0"),
        ]

    return all(ok for _, ok in checks), checks, (spec or {})


def _scan_logs(log_dir):
    refs = []
    for path in sorted(glob.glob(os.path.join(log_dir, LOG_GLOB))):
        try:
            text = Path(path).read_text(errors="replace")
        except OSError as exc:
            print(f"[log] 读不了 {path}: {exc}")
            continue
        found = sorted(set(m.group(1) for m in GRAPH_RE.finditer(text)))
        if found:
            print(f"[log] {os.path.basename(path)}:")
            for ref in found:
                print(f"       {ref}")
                refs.append((os.path.basename(path), ref))
        else:
            print(f"[log] {os.path.basename(path)}: 没有 ar1 图路径")
    return refs


def main():
    graph_dir = Path(GRAPH_DIR)
    graphs = sorted(graph_dir.glob(GRAPH_GLOB))
    if not graphs:
        raise SystemExit(f"{graph_dir} 下没有 {GRAPH_GLOB}")

    print("=" * 100)
    print(f"扫描 {len(graphs)} 张图：{graph_dir}")
    print("=" * 100)
    header = (f"{'file':46s} {'match':5s} {'sigma':>7s} {'tau':>5s} "
              f"{'rank':>4s} {'lscale':>6s} {'seed':>4s} "
              f"{'tokens':>6s} {'ch':>4s} {'grid':>8s}  targets")
    print(header)
    print("-" * 100)

    matched = []
    for path in graphs:
        try:
            meta = _meta(str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"{path.name:46s} 读取失败：{exc}")
            continue
        ok, checks, _ = _match(meta)
        name = path.name
        targets = meta.get("spw.ar1.targets", "")
        print(f"{name:46s} {'YES' if ok else '':5s} "
              f"{meta.get('spw.ar1.sigma', '?'):>7s} "
              f"{meta.get('spw.ar1.tau_days', '?'):>5s} "
              f"{meta.get('spw.ar1.rank', '?'):>4s} "
              f"{meta.get('spw.ar1.length_scale', '?'):>6s} "
              f"{meta.get('spw.ar1.seed', '?'):>4s} "
              f"{meta.get('spw.ar1.tokens', '?'):>6s} "
              f"{meta.get('spw.ar1.channels', '?'):>4s} "
              f"{meta.get('spw.ar1.grid', '?'):>8s}  {targets}")
        if ok:
            matched.append(str(path))
            print(f"    -> MATCH aBt15 budgets；spec={meta.get('spw.ar1.spec')!r}")

    print("\n" + "=" * 100)
    print("归档日志扫描（归档实际加载的图以此为准）")
    print("=" * 100)
    log_refs = _scan_logs(ARCHIVE_DIR)

    print("\n" + "=" * 100)
    if len(matched) == 1:
        print(f"推断唯一：{matched[0]}")
    elif matched:
        print("推断有多个（aBt15 的 budgets 相同、图应只有一张；看下面的日志）：")
        for path in matched:
            print(f"  {path}")
    else:
        print("没有图的 budgets 与 aBt15 匹配。")
        print(BUILD_HINT)

    if log_refs:
        print("\n日志里出现过 ar1 图，归档实际用的就是它们：")
        for log_name, ref in log_refs:
            print(f"  {log_name}: {ref}")
        print("把上面这条路径直接作为 verify_example.py 的 --model。")
    elif matched:
        print("\n日志里没抓到图路径；用上面的 MATCH 路径作为 --model。")
    else:
        print("\n日志里也没抓到；先按 BUILD_HINT 重建，再扫一遍。")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
