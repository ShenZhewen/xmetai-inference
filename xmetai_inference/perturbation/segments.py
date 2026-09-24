# -*- coding: utf-8 -*-
"""分段 rollout（E3）：在**同一条轨迹**里按段更换注入权重。

为什么要有它 —— m1a 的成员是「整条轨迹冻结一组权重」，噪声在时间上没有记忆（iid、无
自相关）。论文的 SPW 主配置正是冻结权重，而它治的是**过**离散（区域平均 SSR 2.5 -> 1.9）；
我们要检验的反而是「噪声的时间结构」这个自由度在**欠**离散的一侧能做什么。把它做成
同预算、可切换的对照臂才有意义，所以有本模块。

一个机制覆盖两件事：

  * 轨迹内**刷新**（论文的「沿轨迹刷新噪声」变体）：段边界换到另一份成员权重，等价于在
    轨迹中途重新抽一次  W~ = W ⊙ (1 + sigma xi)；
  * 分段 **sigma**（论文的 sigma_N = sigma_frozen x sqrt(T/N)）：每段给不同的 sigma_full。

预算口径：每段各自按 `balance` 配平（一段一份预算），K 段之和才是总注入方差。所以要与
冻结臂**同预算**分段，每段应写 sigma_full / sqrt(K)（K 段各摊 1/K 份）；论文那种
「每段 sigma x sqrt(T/N)」是**超预算**的写法，两种都能写，但要清楚自己在花多少。

本模块只做**调度**：把 config 里的分段表解析成逐段的 WeightPerturbSpec，并给出「跑完第
几步之后该换成第几段」。真正的换权重由 cli 在段边界调用 model.load()（见 cli._member_step
/ _run_member_outer）—— 这里不碰 onnx，也不碰 session，纯逻辑，可单测。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .spec import WeightPerturbSpec

# 分段表每一项认这些键。多余键直接报错而不是忽略：写错一个键名（比如 segma_full）
# 会被静默当成「没给幅度」而落到 base 的 sigma_full 上——同预算的半段计划会白跑。
SEGMENT_KEYS = ("steps", "sigma", "sigma_full", "seed", "why")


@dataclass(frozen=True)
class Segment:
    """一段：跑 steps 步，用 spec 这份成员集（= 这一段自己的 sigma / 种子）。"""

    steps: int
    spec: WeightPerturbSpec
    note: str = ""

    def amplitude(self) -> str:
        """日志里那段「这段多大」的写法。"""
        if self.spec.sigma is not None:
            return f"sigma={self.spec.sigma:.6g}"
        return f"sigma_full={self.spec.sigma_full:.6g}"


@dataclass(frozen=True)
class SegmentPlan:
    """逐段规格 + 段边界。步号一律是**全局 1-based**（= 输出帧号 NNN）。"""

    segments: tuple

    @property
    def total_steps(self) -> int:
        return sum(segment.steps for segment in self.segments)

    def specs(self) -> tuple:
        """逐段要准备的成员集规格（cli 会在跑之前把它们都造好）。"""
        return tuple(segment.spec for segment in self.segments)

    def switch_after(self) -> "dict[int, int]":
        """{跑完第几步(全局 1-based): 该换成第几段(0-based)}。

        末段之后**不**产生换段点：整条轨迹的最后一步跑完就没有下一步了，多挂一次
        session 只是白花时间（51 成员 x 多卡上就是分钟级）。
        """
        out: dict[int, int] = {}
        done = 0
        for index, segment in enumerate(self.segments[:-1]):
            done += segment.steps
            out[done] = index + 1
        return out

    def describe(self) -> str:
        parts = []
        for index, segment in enumerate(self.segments):
            seed = "" if segment.spec.seed is None else f" seed={segment.spec.seed}"
            parts.append(f"{index}:{segment.steps}步 {segment.amplitude()}{seed}")
        return " | ".join(parts)


def _entry_mapping(entry: Any, index: int, label: str) -> dict:
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"{label} 第 {index} 段应是 dict（如 {{\"sigma_full\": 0.0231, \"steps\": 20}}），"
            f"实际 {type(entry).__name__}：{entry!r}")
    return dict(entry)


def parse_segment_plan(value, base_spec, total_steps, *, label="spw_segments"):
    """config 的分段表 -> SegmentPlan；None/False/空 表示不分段。

    逐项校验都在这里落地（而不是留给运行期）：分段写错的表现是「跑完 60 步但权重只换了
    一次」或「每段都用 base 的 sigma」，两种都不报错、只是结论不可信。
    """
    if value is None or value is False or value == "":
        return None
    if base_spec is None:
        raise ValueError(f"{label} 要配分段，就必须同时配 weight_perturb（换的就是它的权重）")
    if not isinstance(base_spec, WeightPerturbSpec):
        base_spec = WeightPerturbSpec.from_config(base_spec)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{label} 应是分段表（list of dict），实际 {type(value).__name__}：{value!r}")
    if not value:
        raise ValueError(f"{label} 是空表；不分段就直接写 None")

    segments = []
    for index, entry in enumerate(value, start=1):
        items = _entry_mapping(entry, index, label)
        unknown = [key for key in items if key not in SEGMENT_KEYS]
        if unknown:
            raise ValueError(
                f"{label} 第 {index} 段有无法识别的键 {unknown}；只认 {list(SEGMENT_KEYS)}")
        if "steps" not in items:
            raise ValueError(f"{label} 第 {index} 段缺 steps（这一段跑多少步）")
        steps = int(items["steps"])
        if steps <= 0:
            raise ValueError(f"{label} 第 {index} 段的 steps 必须 > 0，实际 {steps}")
        has_sigma = items.get("sigma") is not None
        has_full = items.get("sigma_full") is not None
        if has_sigma == has_full:
            # 两个都给时 spec 里 sigma 会压过 sigma_full（见 resolve_sigmas），静默改变
            # 预算；两个都不给则这一段退回 base 的幅度，分段就只剩「换成员」一种效果。
            raise ValueError(
                f"{label} 第 {index} 段必须**恰好**给一个 sigma 或 sigma_full，"
                f"实际 sigma={items.get('sigma')!r} / sigma_full={items.get('sigma_full')!r}")
        overrides = ({"sigma": float(items["sigma"]), "sigma_full": base_spec.sigma_full}
                     if has_sigma else
                     {"sigma": None, "sigma_full": float(items["sigma_full"])})
        if items.get("seed") is not None:
            overrides["seed"] = int(items["seed"])
        # replace 会重跑 __post_init__，所以幅度/结构的校验（>0、合法枚举）在这里就生效。
        segments.append(Segment(steps=steps,
                                spec=replace(base_spec, **overrides),
                                note=str(items.get("why") or "")))

    plan = SegmentPlan(tuple(segments))
    if plan.total_steps != int(total_steps):
        raise ValueError(
            f"{label} 各段 steps 之和 {plan.total_steps} != 本次预报的 steps "
            f"{int(total_steps)}。分段表必须正好铺满整条轨迹（--steps 覆盖会让它们对不上，"
            "改了 --steps 就要一起改分段表）")
    return plan


__all__ = ["SEGMENT_KEYS", "Segment", "SegmentPlan", "parse_segment_plan"]
