# -*- coding: utf-8 -*-
"""跑完之后的集合体检：读**真实输出目录**，确认「成员真的互不相同、离散度随时效增长」。

`e2e` 验的是**框架接线**（合成图，不碰真模型与自定义算子）；`check` 验的是
**真实产物**：

    {out}/{起报目录}/member_{NNN}/{预测步序号3位}.nc

逐时效给出一行：

    step   预测步序号（1 起）
    day    时效（天）= step x 单步小时 / 24
    M      该时效有文件的成员数
    uniq   内容（空间子采样后逐位）不同的成员数
    min|d| 与成员 0 的最大绝对差里最小的那个（最近的两个成员有多近）
    spread 集合标准差的空间平均（离散度）
    rel    spread / |集合平均|，跨变量可比的口径

判据只有两条，都是**不需要实况**的自洽性检查：

  1. uniq 必须等于 M。只要出现 uniq < M，就是有成员逐位重复 —— 也就是 README 反复强调的
     坑 1（各 rank 抽到同一条随机流，51 个成员里真正独立的只剩 ceil(51/卡数) 个）。
     这条不需要任何参考数据就能判定，所以放在最前面。
  2. 离散度应随时效增长（后 1/3 的平均 spread 至少是前 1/3 的 1.05 倍）。完全不增长
     说明扰动没有真正进入自回归回路（例如只扰了输出层，或噪声被某个算子吸收）。
     不要求严格单调 —— 长时效离散度饱和甚至回落是正常现象。

注意：**这不是评测**。它不碰实况、不算 CRPS/SSR，只回答「这套集合是不是真的集合」。

用法：
    python -m xmetai_inference.perturbation check /path/to/output_dir
    python -m xmetai_inference.perturbation check /path/to/out --steps all --stride 2 --var MSL
    python -m xmetai_inference.perturbation check /path/to/out --init 20250102 --expect-members 51
"""
from __future__ import annotations

import hashlib
import os
import os.path as osp
import re

import numpy as np

_STEP_RE = re.compile(r"^(\d{3})\.nc$")


def _member_dirs(init_dir: str):
    """{起报目录} 下的 member_XXX/ 列表，按成员号排序。"""
    found = []
    for name in os.listdir(init_dir):
        path = osp.join(init_dir, name)
        if name.startswith("member_") and osp.isdir(path):
            found.append((name, path))
    found.sort()
    return found


def _step_files(member_dir: str):
    """成员目录下的 {步号: 路径}。"""
    steps = {}
    for name in os.listdir(member_dir):
        match = _STEP_RE.match(name)
        if match:
            steps[int(match.group(1))] = osp.join(member_dir, name)
    return steps


def _dataset_dirs(root: str):
    """顶层输出目录 -> [(起报目录名, 路径, 成员列表)]。"""
    result = []
    for name in sorted(os.listdir(root)):
        path = osp.join(root, name)
        if not osp.isdir(path):
            continue
        members = _member_dirs(path)
        if members:
            result.append((name, path, members))
    return result


def _pick_steps(available, spec):
    """挑要体检的步号：spec 为 "all" 或 <=0 时全查，否则等距抽样（首尾必含）。"""
    available = sorted(available)
    if spec == "all":
        return available
    try:
        count = int(spec)
    except (TypeError, ValueError):
        return available
    if count <= 0 or len(available) <= count:
        return available
    index = np.linspace(0, len(available) - 1, count).round().astype(int)
    return sorted({available[i] for i in index})


def _read_probe(path: str, var, stride: int):
    """读一个成员一步的输出，空间按 stride 子采样。返回 (变量名, 数组)。"""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        names = list(ds.data_vars)
        if not names:
            raise ValueError(f"{path} 里没有数据变量")
        picked = names[0] if var is None else var
        if picked not in names:
            # 大小写不敏感：cli 的 --vars 收小写（z500），而 to_dataset 落盘时写的是
            # 大写的通道名（Z500），直接按字面比会白白报错。
            actual = {name.lower(): name for name in names}.get(str(picked).lower())
            if actual is None:
                raise KeyError(
                    f"{path} 里没有变量 {picked!r}（可选：{', '.join(names)}）")
            picked = actual
        if stride and stride > 1:
            da = ds[picked].isel(lat=slice(None, None, stride),
                                 lon=slice(None, None, stride))
        else:
            da = ds[picked]
        chunk = np.asarray(da.values)
    # 只比内容是否相同，所以要归一化 dtype 与内存布局，避免同一份数据算出两个哈希。
    return picked, np.ascontiguousarray(chunk, dtype=np.float32)


def _growth(rows):
    """离散度增长的稳健判据：前 1/3 与后 1/3 的平均值之比。

    不用「末步 > 首步」：集合离散度在长时效饱和甚至回落是正常现象，逐点比较会误判。
    也不要求严格单调。只看「完全不长」——那才是噪声没进自回归回路的信号。
    """
    third = max(1, len(rows) // 3)
    early = float(np.mean([row["spread"] for row in rows[:third]]))
    late = float(np.mean([row["spread"] for row in rows[-third:]]))
    peak = max(rows, key=lambda row: row["spread"])
    ratio = (late / early) if early else float("inf")
    return early, late, ratio, peak["step"]


def _digest(array) -> str:
    return hashlib.blake2b(array.tobytes(), digest_size=8).hexdigest()


def _inspect_init(name, members, *, steps_spec, stride, var, verbose=True):
    """体检一个起报目录，返回 (失败原因列表, 数据行列表)。"""
    failures = []
    per_member = [(label, _step_files(mdir)) for label, mdir in members]
    counts = {label: len(files) for label, files in per_member}
    common = set(per_member[0][1])
    for _label, files in per_member[1:]:
        common &= set(files)
    if not common:
        return ([f"{name}：成员之间没有共同的输出步，无法体检"], [])
    if len(set(counts.values())) > 1 and verbose:
        print(f"  提示：各成员步数不一致（继续用共同步）：{counts}")

    chosen = _pick_steps(common, steps_spec)
    picked_var = var
    rows = []
    for step in chosen:
        arrays, digests, bad = [], [], 0
        for _label, files in per_member:
            got_var, chunk = _read_probe(files[step], picked_var, stride)
            picked_var = got_var
            if not np.isfinite(chunk).all():
                bad += int((~np.isfinite(chunk)).sum())
            arrays.append(chunk)
            digests.append(_digest(chunk))
        # 统计量走 float64：float32 下「完全相同的成员」也会因为求均值的舍入
        # 显出 ~1e-8 的假离散度，会让人误判成「有点差异但不是零」。
        stack = np.stack(arrays, axis=0).astype(np.float64)
        count = int(stack.shape[0])
        unique = len(set(digests))
        if count > 1:
            diff = np.abs(stack[1:] - stack[0])
            gaps = float(diff.max(axis=tuple(range(1, diff.ndim))).min())
        else:
            gaps = float("nan")
        spread = float(stack.std(axis=0).mean())
        scale = float(np.abs(stack.mean(axis=0)).mean())
        rows.append({
            "step": step,
            "members": count,
            "unique": unique,
            "min_gap": gaps,
            "spread": spread,
            "rel": (spread / scale) if scale else float("inf"),
            "nonfinite": bad,
        })
        if unique < count:
            failures.append(
                f"{name} step {step:03d}：{count} 个成员里只有 {unique} 个内容不同"
                "（有成员逐位重复 —— 检查各 rank 的 spw-<指纹> 是否不同、"
                "LOCAL_RANK 是否生效）")
        if bad:
            failures.append(f"{name} step {step:03d}：输出里有 {bad} 个 NaN/Inf")

    if len(rows) >= 3:
        early, late, ratio, peak_step = _growth(rows)
        if ratio <= 1.05:
            failures.append(
                f"{name}：离散度基本不随时效增长（前 1/3 平均 {early:.4g} -> "
                f"后 1/3 平均 {late:.4g}，{ratio:.2f}x；峰值在 step {peak_step:03d}）"
                " —— 检查扰动是否真的进入了自回归回路")
    return failures, rows


def run_check(root, *, init=None, var=None, stride=4, steps_spec=12,
              expect_members=None, hour_interval=6.0, verbose=True):
    if not osp.isdir(root):
        print(f"输出目录不存在：{root}")
        return 1
    dirs = _dataset_dirs(root)
    if not dirs:
        print(f"{root} 下没有找到任何「起报目录/member_XXX/」结构。"
              "确定性输出（members=1）没有 member_ 子目录，本工具不适用。")
        return 1
    if init:
        wanted = {token.strip() for token in str(init).split(",") if token.strip()}
        dirs = [item for item in dirs if item[0] in wanted]
        if not dirs:
            print(f"没有匹配 --init {init!r} 的起报目录")
            return 1

    print(f"集合体检：{root}")
    print(f"  起报目录 {len(dirs)} 个；空间子采样 stride={stride}；"
          f"体检步 steps={steps_spec}（整数=等距抽这么多个，all=全查）")
    all_failures = []
    for name, path, members in dirs:
        count = len(members)
        print(f"\n[{name}] 成员目录 {count} 个")
        if expect_members is not None and count != expect_members:
            all_failures.append(
                f"{name}：成员目录 {count} 个，期望 {expect_members} 个")
        if count == 1:
            print("  只有 1 个成员，这不是集合（members=1 是确定性输出）")
            continue
        failures, rows = _inspect_init(
            name, members, steps_spec=steps_spec, stride=stride,
            var=var, verbose=verbose)
        if rows:
            print(f"  变量：{var or '（未指定，取每个文件的第一个数据变量）'}")
            print("   step   day      M  uniq     min|d|      spread      rel")
            for row in rows:
                print(f"  {row['step']:4d} {row['step']*hour_interval/24:5.1f}"
                      f" {row['members']:6d} {row['unique']:5d}"
                      f" {row['min_gap']:11.3g} {row['spread']:11.4g}"
                      f" {row['rel']:8.3g}")
            if len(rows) >= 3:
                early, late, ratio, peak_step = _growth(rows)
                print(f"  离散度：前 1/3 平均 {early:.4g} -> 后 1/3 平均 {late:.4g} "
                      f"= {ratio:.2f}x；峰值在 step {peak_step:03d}")
        all_failures.extend(failures)

    print("\n" + "=" * 72)
    if all_failures:
        print(f"结论：{len(all_failures)} 项不通过 —— 这套输出还不是可用的集合。")
        for item in all_failures:
            print(f"  - {item}")
        print("\n排查顺序：各 rank 的图文件名是否为不同的 spw-members-<指纹>；"
              "config 的 weight_perturb.member_dir 是否各 rank 一致；"
              "加载日志里有没有「成员号对不上」的报错（加载期唯一的自检）。")
        return 1
    print("结论：成员互不相同、离散度随时效增长 —— 集合产出是健康的。")
    print("      这只是自洽性检查；标定好不好要用实况评（本工具不做）。")
    return 0


__all__ = ["run_check"]
