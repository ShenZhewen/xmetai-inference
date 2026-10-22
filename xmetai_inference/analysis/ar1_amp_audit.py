# -*- coding: utf-8 -*-
"""a(t) 包络离线审计：**跑的到底是不是设计的那条曲线**。

为什么要单独一个脚本
--------------------
``amp_profile`` 只在驱动侧乘出口 ξ，不进 ``budgets()``，所以它**不留下任何图痕迹**：
图名一样、日志一样、推理一路正常。曲线抄错一个折点、或者 step 口径差一格，
产出的臂照样跑完、照样出数，只是**它是另一条臂**。这类错误只能靠对着代码算一遍
来抓，抓不到就是静默错臂。

本脚本跑几秒、不要 GPU、不碰数据，纯读 config + 复用 ``LatentForcingSpec.amp_at``，
回答三件事：

1. **十条臂的轴值对不对**（σ / τ / 折数 / demean）—— 与跑批闸里那张预登记表比。
2. **每一步实际乘的倍数是多少**，以及**它落在哪个时效上**（step → lead 的映射）。
3. **时均等效 σ** 是多少 —— aA/aB 的意义全在它与配对平坦臂的关系上
   （``s08 ↔ aA``、``s15 ↔ aB``）。配不上，"形状 vs 大小"这个对照就不成立。

⚠ 第 2 条里有一个**已知的口径差**，是本脚本的主要产出，见下面 ``STEP_CONVENTION``。

step → lead 的三处证据（都在盘上，可复核）
------------------------------------------
* ``cli.py:658``      ``step_idx = step + 1`` ⇒ 落盘文件名 ``001.nc`` 对应的 **code step 是 0**。
* ``gridded.py:121``  ``lead_time = (position + 1) * step_hours`` ⇒ ``001.nc`` = **6h**。
* ``runner.py:189``   ``valid_times = init + (step + offset) * hour_interval``，fuxi 系
  ``offset = 0`` ⇒ step 0 的输入窗末帧是起报时刻本身，**它的预报是 +6h**。

合起来：``NNN.nc`` = 提前 ``NNN × 6h`` = **code step ``NNN − 1``**。

而 ``tmp_config/fgvp_ens_ar1state_aB.py`` 的 docstring 那张 σ*(t) 表的 step 列写的是
1 / 4 / 8 / 12 / 20 / 28 / 40 / 60 —— 那是 ``lead_h / 6``，**1 基**。config 里把第一个
折点抄成了 0、其余照抄 1 基值，于是整条曲线相对设计表**晚一格（6h）**。
（第一个折点两端同值 1.76，所以 6h 那一档看不出来；后面每一档都晚 6h。）

用法
----
在仓库根（``D:\\output`` 或服务器上的包根上一层）跑::

    python -m xmetai_inference.analysis.ar1_amp_audit

要改臂表/步数，**直接改下面的字面量**（不用环境变量、不用命令行）。
"""
from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

# 仓库根 = analysis/ 的上两层（analysis → xmetai_inference → 根）
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from xmetai_inference.weights.spa import LatentForcingSpec  # noqa: E402

# ── 可调项：直接改这里 ────────────────────────────────────────────────────
# 第 14 轮那十条臂（run-a: 前五条 / run-b: 后五条）。跨 Job 不许横比，本脚本只
# 审计「配置说了什么」，不做跨臂读数，所以这里合在一起看没问题。
ARMS = (
    "fgvp_ens_ar1state_s00", "fgvp_ens_ar1state_s04", "fgvp_ens_ar1state_s08",
    "fgvp_ens_ar1state_s12", "fgvp_ens_ar1state_s15", "fgvp_ens_ar1state_aA",
    "fgvp_ens_ar1state_aB", "fgvp_ens_ar1state_zm15", "fgvp_ens_ar1state_zmaB",
    "fgvp_ens_ar1state_t005s15",
)

#: 一步多少小时 / 一步跑几步 —— 与 config 的 steps 一致（本批 60 步 = 360h）。
HOUR_INTERVAL = 6.0
STEPS = 60

#: 预登记轴值表（与 run-a 闸里的 EXP 逐字同源）：臂 -> (σ, τ天, a(t) 折数, demean 成员数)
EXPECTED = {
    "s00": (0.0, 3.0, 0, 0), "s04": (0.04, 3.0, 0, 0), "s08": (0.08, 3.0, 0, 0),
    "s12": (0.12, 3.0, 0, 0), "s15": (0.15, 3.0, 0, 0), "aA": (0.04, 3.0, 2, 0),
    "aB": (0.04, 3.0, 8, 0), "zm15": (0.15, 3.0, 0, 51), "zmaB": (0.04, 3.0, 8, 51),
    "t005s15": (0.15, 0.05, 0, 0),
}

#: aB 的设计表：**lead_h -> σ*(t)/0.04**，逐字抄自 aB 的 config docstring（1 基 step）。
#: 只给 aB，因为只有它把这张表写进了文档；别的臂没有可对的设计曲线。
A_B_DESIGN = {6: 1.76, 24: 2.62, 48: 2.34, 72: 2.36,
              120: 2.68, 168: 3.21, 240: 4.74, 360: 6.13}

#: 设计表与之配对的平坦臂（config docstring 明写）：时均等效 σ 应当相等。
PAIRING = {"aA": "s08", "aB": "s15"}

#: 时均等效 σ 的宣称值（config docstring）：aA 0.0793 / aB 0.1507
CLAIMED_MEAN_SIGMA = {"aA": 0.0793, "aB": 0.1507}

SHOW_LEADS = (6, 24, 48, 72, 120, 168, 240, 360)


def _load_spec(arm: str) -> LatentForcingSpec:
    """把 config 的 latent_forcing 字典灌进真正的 LatentForcingSpec。

    用**真的类**而不是自己复刻 amp_at：``__post_init__`` 会把折点归一化并校验
    （严格递增、非负），这正是要一起测的东西。字典里可能有类不认的键，按
    dataclass 字段名过滤掉——这样 config 加新字段时本脚本不会先崩。
    """
    cfg = importlib.import_module("xmetai_inference.tmp_config." + arm).cfg
    raw = cfg.latent_forcing
    if not isinstance(raw, dict):
        raise TypeError("%s 的 latent_forcing 不是 dict（%r）" % (arm, raw))
    known = {f.name for f in dataclasses.fields(LatentForcingSpec)}
    unknown = sorted(set(raw) - known)
    if unknown:
        print("      ⚠ 配置里有 LatentForcingSpec 不认识的键（已忽略）：%s" % unknown)
    return LatentForcingSpec(**{k: v for k, v in raw.items() if k in known})


def _axis_tuple(spec: LatentForcingSpec) -> tuple:
    return (round(float(spec.sigma), 4), round(float(spec.tau_days), 4),
            len(spec.amp_profile or ()), int(spec.demean_members))


def _mean_amp(spec: LatentForcingSpec) -> float:
    """跑批实际用的时均倍数：step 0..STEPS-1 上取平均（**不是**折点的平均）。"""
    return sum(spec.amp_at(s) for s in range(STEPS)) / STEPS


def main() -> int:
    print("=" * 96)
    print("a(t) 包络离线审计   步长 %.0fh × %d 步 = %.0fh"
          % (HOUR_INTERVAL, STEPS, STEPS * HOUR_INTERVAL))
    print("step → lead：lead_h = (step + 1) × %.0fh     （证据见本文件 docstring）"
          % HOUR_INTERVAL)
    print("=" * 96)

    specs, bad = {}, 0

    print("\n【1】轴值 vs 预登记表")
    print("   %-6s %-40s %8s %6s %5s %7s   %s"
          % ("臂", "config.name", "σ", "τ(d)", "折数", "demean", "判定"))
    for arm in ARMS:
        key = arm.replace("fgvp_ens_ar1state_", "")
        try:
            spec = _load_spec(arm)
        except Exception as exc:                       # noqa: BLE001
            print("   %-6s !!!! 加载失败：%s: %s" % (key, type(exc).__name__, exc))
            bad += 1
            continue
        specs[key] = spec
        got = _axis_tuple(spec)
        want = EXPECTED.get(key)
        if want is None:
            why = "本臂不在预登记表里（新臂忘了登记）"
        elif got != want:
            why = "轴值 %s != 期望 %s" % (got, want)
        else:
            why = "OK"
        if why != "OK":
            bad += 1
        print("   %-6s %-40s %8.4f %6.2f %5d %7d   %s"
              % (key, arm, got[0], got[1], got[2], got[3], why))

    print("\n【2】a(t) 曲线：设计 vs 实际交付（只有 aB 有设计文字表）")
    ab = specs.get("aB")
    if ab is None or not ab.amp_profile:
        print("   （aB 没加载到或没有 amp_profile，跳过）")
    else:
        print("   折点（config 写法，step 是 **0 基 code step**）：%s"
              % (tuple(ab.amp_profile),))
        print("   设计表（docstring，step 是 **1 基 = lead_h/6**）：%s"
              % (tuple(int(L / HOUR_INTERVAL) for L in A_B_DESIGN),))
        print()
        print("   %-7s %-7s %10s %10s %10s   %s"
              % ("lead_h", "code step", "设计倍数", "实际倍数", "实际σ_eff", "差"))
        worst = 0.0
        for lead in SHOW_LEADS:
            step = int(lead / HOUR_INTERVAL) - 1        # 文件 NNN.nc = lead NNN*6h = code step NNN-1
            got = ab.amp_at(step)
            design = A_B_DESIGN[lead]
            rel = 100.0 * (got - design) / design
            worst = max(worst, abs(rel))
            print("   %-7d %-10d %10.2f %10.2f %10.4f   %+6.1f%%"
                  % (lead, step, design, got, ab.sigma * got, rel))
        print("\n   最大偏离 %.1f%%（整条曲线相对设计表晚一格 = 6h；折点连同值处看不出来）" % worst)
        step60 = ab.amp_at(STEPS)                       # 设计曲线的最后一个折点
        print("   设计终点 step 60 → %.3f；但跑批只到 code step %d ⇒ 360h 实际拿到 %.3f"
              % (step60, STEPS - 1, ab.amp_at(STEPS - 1)))

    print("\n【3】时均等效 σ —— aA/aB 的全部意义在这里")
    print("   %-6s %8s %10s %12s %12s   %s"
          % ("臂", "σ", "时均倍数", "时均σ_eff", "配对平坦臂", "判定"))
    for key, flat in PAIRING.items():
        spec = specs.get(key)
        if spec is None or not spec.amp_profile:
            print("   %-6s （没加载到或没有 amp_profile）" % key)
            continue
        m = _mean_amp(spec)
        eff = spec.sigma * m
        other = specs.get(flat)
        if other is None:
            print("   %-6s %8.4f %10.4f %12.4f %12s   （配对臂没加载到）"
                  % (key, spec.sigma, m, eff, flat))
            continue
        gap = 100.0 * (eff - other.sigma) / other.sigma
        claim = CLAIMED_MEAN_SIGMA.get(key)
        note = ""
        if claim is not None:
            note = "；docstring 宣称 %.4f（差 %+.2f%%）" % (claim, 100.0 * (eff - claim) / claim)
        verdict = "OK" if abs(gap) <= 2.0 else "!!!! 配对失效（相差 %+.1f%%，>2%%）" % gap
        print("   %-6s %8.4f %10.4f %12.4f %12s   %s%s"
              % (key, spec.sigma, m, eff, "%s(%.4f)" % (flat, other.sigma), verdict, note))
    print("   注：时均是按 **step 0..%d** 算的（跑批实际用的那一段），不是按折点平均。"
          % (STEPS - 1))
    print("       设计表若按 1 基 step 1..%d 算，均值会高 (a(60)-a(0))/%d ≈ %.4f。"
          % (STEPS, STEPS, (ab.amp_at(STEPS) - ab.amp_at(0)) / STEPS if ab else float("nan")))

    print("\n【4】曲线不进图指纹（所以换曲线不重导、也不再有任何痕迹）")
    for key, spec in specs.items():
        b = spec.budgets()
        leaked = sorted(set(b) & {"amp_profile", "tau_days", "demean_members", "hour_interval"})
        if leaked:
            print("   %-6s !!!! budgets() 里出现了 %s —— 换曲线会重导图" % (key, leaked))
            bad += 1
    print("   %d 条臂的 budgets() 都不含 amp_profile/tau_days/demean_members/hour_interval  ✓"
          % len(specs))
    print("   ⇒ 这四条轴全靠「配置对不对」保证，跑完没有第二道防线。")

    print("\n" + "=" * 96)
    print("结论：%s（%d 项不符）" % ("全部通过" if bad == 0 else "有 %d 项要人看一眼" % bad, bad))
    print("=" * 96)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
