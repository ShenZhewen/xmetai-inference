# -*- coding: utf-8 -*-
"""SPW 端到端预检（e2e）：在**真实框架路径**上把「确定性 ONNX -> N 成员集合」跑一遍。

self-test 验的是「选张量/配平/落盘」本身（纯 onnx / onnxruntime 层）；正式跑之前真正会
翻车的是**框架接线**。本模块把下面五条逐条钉死，任何一条不过就说明先别上服务器：

  1. 规格解析：config 里的 weight_perturb dict -> WeightPerturbSpec -> create_model；
     非 ONNX 后端必须被明确拒绝（ValueError），而不是悄悄忽略扰动。
  2. 加载路径：给了 member_index 时，框架加载的确实是**那一份成员图**（不是悄悄加载
     源图），源图字节不变，成员图引用的 base/perturbed 外部数据都在成员集目录内。
  3. 成员可用：N 个成员**两两不同**（扰动没生效时 N 个成员逐位相同、且不报错——这是
     本方案唯一的静默失败模式），成员内两次 rollout **逐位相同**（冻结）、离散度随时效
     增长、session 用完即释放。
  4. 成员集的 cli 编排：直接驱动 `cli._run_member_outer`（服务器上真正跑的就是它）——
     逐成员 load/release、每个成员各跑遍全部起报、输出落到 member_NNN/ 子目录、跑完
     session 已释放。第 3 节验的是语义，这一节验的是**落地那段代码**。
  5. 分段 rollout（E3）：在段边界换一份冻结权重，边界前逐位相同、边界后全部不同。

为什么用合成图：正式模型（iwc_fgvp_gdn2）依赖 Linux 侧自定义算子库 + CUDA，本地/CI
建不了 session。所以这里用一张结构同构的小图 + 一个最小的 OnnxInferModel 子类。
**它验证的是框架接线，不是 FGVP 的数值。**

用法：python -m xmetai_inference.perturbation e2e [--members 51] [--steps 6] [--keep]
"""
from __future__ import annotations

import hashlib
import os.path as osp
import tempfile

import numpy as np
import pandas as pd

from ..backends.onnx import OnnxInferModel
from ..models import create_model
from ..runner import Rollout, Trajectory
from .graph import (
    _onnx,
    external_tensors,
    load_model,
    resolve_effective_path,
    resolve_sigmas,
    select_initializers,
)
from .selftest import _Checker
from .spec import DEFAULT_EXCLUDE, WeightPerturbSpec
from .weights import build_member_set, member_model_path, set_dir_for

# 合成「小天气模型」的几何：2 帧 x 6 通道，20x20 网格，64 隐层通道。
# 组名照抄 FGVP 的三个架构组，让「按架构组选张量」这条路径也被走到。
_IN_SHAPE = (1, 2, 6, 20, 20)
_HID = 64
_OUT_CH = 6

# 默认 spec 下**应该**被选中的张量（名字排序后）。
_EXPECT_SELECTED = (
    "decoder_group.head.conv.weight",
    "latent_mixer.layers.0.mixer.bias",
    "latent_mixer.layers.0.mixer.weight",
    "vae_encoder_group.blocks.0.conv.bias",
    "vae_encoder_group.blocks.0.conv.weight",
)

# **不该**被选中的张量 -> 挡下它的规则（子串匹配 skipped 里的原因）。
_EXPECT_SKIPPED = (
    ("decoder_group.head.conv.bias", "min_elements"),
    ("vae_encoder_group.norm.mean", "exclude"),
    ("vae_encoder_group.norm.const", "exclude"),
    ("slice.starts", "非浮点"),
)

_TOTAL_SECTIONS = 5


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cfg(**overrides):
    """与 configs/fgvp_spw_ab.py 里那个 dict 同形（只是规模小）。"""
    cfg = {
        "groups": ("all",),
        "sigma_full": 0.01,
        "balance": "energy",
        "exclude": DEFAULT_EXCLUDE,
        "min_elements": 64,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# 合成图与最小模型类
# ---------------------------------------------------------------------------
def _write_mini(path: str) -> str:
    """一张单步型（windowed_output=False）的小图：Conv x3 + Sub(统计量) + Slice。

    结构上照着 FGVP 的骨架裁：三个 Conv 分别挂 vae_encoder_group / latent_mixer /
    decoder_group 三个组；归一化统计量走 Sub（证明 exclude 在数值上也成立）；Slice 的
    三元组是 int64（非浮点，必须被挡下）；head bias 只有 6 个元素（< min_elements）。
    """
    onnx = _onnx()
    helper, TensorProto = onnx.helper, onnx.TensorProto
    rng = np.random.default_rng(0)

    def rand(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    def init(name, array):
        return onnx.numpy_helper.from_array(np.asarray(array), name)

    initializers = [
        init("vae_encoder_group.blocks.0.conv.weight", rand(_HID, 6, 1, 3, 3)),
        init("vae_encoder_group.blocks.0.conv.bias", rand(_HID)),
        init("latent_mixer.layers.0.mixer.weight", rand(_HID, _HID, 1, 3, 3)),
        init("latent_mixer.layers.0.mixer.bias", rand(_HID)),
        init("decoder_group.head.conv.weight", rand(_OUT_CH, _HID, 1, 3, 3)),
        init("decoder_group.head.conv.bias", rand(_OUT_CH)),
        # 统计量挂在隐层上（64 个元素 >= min_elements），才能走到 exclude 那条规则；
        # 太小会先被 min_elements 挡下，就测不到「排除统计量」这件事。
        init("vae_encoder_group.norm.mean", rand(1, _HID, 1, 1, 1)),
        init("vae_encoder_group.norm.const", rand(1, _HID, 1, 1, 1)),
        init("slice.starts", np.array([1], dtype=np.int64)),
        init("slice.ends", np.array([2], dtype=np.int64)),
        init("slice.axes", np.array([2], dtype=np.int64)),
    ]
    pads = [0, 1, 1, 0, 1, 1]
    nodes = [
        # 框架的 state 是 (N, 帧, 通道, H, W)，而 Conv 认的是 (N, C, D, H, W)。
        # 单步型天气模型要么在导出时把这一步烘进图里，要么在这里转一次 —— 转一次更直白，
        # 也让「帧」这个维度真的参与卷积（相当于一个最小的时域混合器）。
        helper.make_node("Transpose", ["input"], ["frames"],
                         name="vae_encoder_group/input/Transpose", perm=[0, 2, 1, 3, 4]),
        helper.make_node(
            "Conv",
            ["frames", "vae_encoder_group.blocks.0.conv.weight",
             "vae_encoder_group.blocks.0.conv.bias"],
            ["enc"], name="vae_encoder_group/blocks.0/Conv", pads=pads),
        helper.make_node("Relu", ["enc"], ["enc_r"],
                         name="vae_encoder_group/blocks.0/Relu"),
        helper.make_node(
            "Conv",
            ["enc_r", "latent_mixer.layers.0.mixer.weight",
             "latent_mixer.layers.0.mixer.bias"],
            ["mix"], name="latent_mixer/layers.0/GatedDeltaNet2Fn", pads=pads),
        # 归一化统计量（名字命中 exclude，必须保持原值不被扰动）
        helper.make_node("Sub", ["mix", "vae_encoder_group.norm.mean"], ["mix_n"],
                         name="vae_encoder_group/norm/Sub"),
        helper.make_node("Sub", ["mix_n", "vae_encoder_group.norm.const"], ["mix_n2"],
                         name="vae_encoder_group/norm/SubConst"),
        helper.make_node("Relu", ["mix_n2"], ["mix_r"],
                         name="latent_mixer/layers.0/Relu"),
        helper.make_node(
            "Conv",
            ["mix_r", "decoder_group.head.conv.weight",
             "decoder_group.head.conv.bias"],
            ["dec"], name="decoder_group/head/Conv", pads=pads),
        helper.make_node(
            "Slice", ["dec", "slice.starts", "slice.ends", "slice.axes"], ["dec1"],
            name="decoder_group/head/Slice"),
        helper.make_node("Transpose", ["dec1"], ["pred"],
                         name="decoder_group/head/Transpose", perm=[0, 2, 1, 3, 4]),
    ]
    graph = helper.make_graph(
        nodes, "spw_e2e_mini",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(_IN_SHAPE))],
        [helper.make_tensor_value_info("pred", TensorProto.FLOAT,
                                       [1, 1, _OUT_CH, 20, 20])],
        initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _mini_model_class():
    """最小的 OnnxInferModel 子类：声明单步型契约即可，不碰任何 FGVP 专属逻辑。"""

    class _MiniSpwModel(OnnxInferModel):
        model_name = "spw_e2e_mini"
        input_channels = tuple(f"c{index}" for index in range(_OUT_CH))
        output_channels = tuple(f"c{index}" for index in range(_OUT_CH))
        history_steps = 2
        hour_interval = 6
        forecast_type = "deterministic"
        members = 1
        # 图只出预报那一帧，滑窗归 runner.Rollout（与 iwc_fgvp_gdn2 同款契约）。
        windowed_output = False
        valid_time_step_offset = 1
        # 图要 4 个键、输入输出不同形时不能开 GPU 常驻；这里是 1 个键但同理保守。
        gpu_state = False

    return _MiniSpwModel


def _rollout(model, members: int, steps: int, *, seed: int = 0):
    """N 条轨迹共享同一初值（成员即轨迹），扰动来自图内随机算子。

    返回 (members, steps, C, H, W)。
    """
    base = np.random.default_rng(seed).standard_normal(_IN_SHAPE).astype(np.float32)
    init = pd.Timestamp("2025-01-02 00:00")
    trajectories = [
        Trajectory(state=base.copy(), init_time=init, label=f"member_{index:03d}")
        for index in range(members)
    ]
    return Rollout(model, None).run(
        trajectories, steps=steps, hour_interval=6, progress=False)


def _rollout_one(model, steps, *, seed: int = 0):
    """单个成员的一条轨迹，返回 (steps, C, H, W)。

    m1a 的成员不是「同一 session 里的并行轨迹」，而是各自一个模型；所以要一条一条跑。
    """
    base = np.random.default_rng(seed).standard_normal(_IN_SHAPE).astype(np.float32)
    init = pd.Timestamp("2025-01-02 00:00")
    return Rollout(model, None).run(
        [Trajectory(state=base.copy(), init_time=init, label="member")],
        steps=steps, hour_interval=6, progress=False)[0]


def _select(work, spec):
    """在合成图上跑一遍选组（纯静态，不建 session）。"""
    path = osp.join(work, "mini.onnx")
    return select_initializers(load_model(path), spec, osp.dirname(path))


def _load_mini(work: str):
    path = osp.join(work, "mini.onnx")
    if not osp.isfile(path):
        _write_mini(path)
    return path, load_model(path)


# ---------------------------------------------------------------------------
# 1/6 规格与选组
# ---------------------------------------------------------------------------
def _section_spec(work, check, metrics):
    check.section(1, _TOTAL_SECTIONS, "规格解析与选组（config dict -> spec -> 选中/排除）")
    from ..backends.base import BaseInferModel

    cfg = _cfg()
    spec = WeightPerturbSpec.from_config(cfg)
    check(isinstance(spec, WeightPerturbSpec), "config dict 能解析成 WeightPerturbSpec")

    cls = _mini_model_class()
    model = create_model(cls, weight_perturb=cfg)
    check(isinstance(getattr(model, "weight_perturb", None), WeightPerturbSpec),
          "create_model 把 spec 挂到了模型实例上（cli 的透传路径）")
    check(getattr(model, "supports_weight_perturb", False) is True,
          "ONNX 后端声明 supports_weight_perturb=True")

    class _NoSupport(BaseInferModel):
        backend = "fake"

        def load(self, path):
            raise NotImplementedError

        def forward(self, x, step, valid_time):
            raise NotImplementedError

    try:
        create_model(_NoSupport, weight_perturb=cfg)
        rejected, detail = False, "没有报错（会静默跑成确定性模型）"
    except ValueError as exc:
        rejected, detail = "不支持权重扰动" in str(exc), str(exc)[:0]
    check(rejected, "非 ONNX 后端被明确拒绝（ValueError）", detail)

    selection = _select(work, spec)
    got = tuple(sorted(selection.selected))
    check(got == _EXPECT_SELECTED,
          f"选中集与预期一致（{len(_EXPECT_SELECTED)} 个张量）", f"实际 {got}")
    for name, token in _EXPECT_SKIPPED:
        reason = selection.skipped.get(name, "")
        check(token in reason, f"{name} 被挡下（{token}）", f"reason={reason!r}")

    # groups=("all",) 是**单组**，配平退化成 sigma=sigma_full；要看配平必须给显式的
    # 架构组，这也正是论文「按张量组消融」的用法。
    grouped = WeightPerturbSpec.from_config(
        _cfg(groups=("latent_mixer", "decoder_group", "vae_encoder_group")))
    grouped_selection = _select(work, grouped)
    check(tuple(sorted(grouped_selection.selected)) == _EXPECT_SELECTED,
          "显式三组时选中集不变（组划分只影响配平，不改变谁被扰动）")
    sigmas = resolve_sigmas(grouped_selection, grouped)
    metrics["sigmas"] = sigmas
    check(sigmas["latent_mixer"] < sigmas["decoder_group"],
          "能量配平：能量最大的组拿到最小的 σ",
          f"{ {k: round(v, 5) for k, v in sigmas.items()} }")
    flat = resolve_sigmas(selection, spec)
    check(abs(flat["all"] - spec.sigma_full) < 1e-12,
          "groups=all 时 σ 就等于 σ_full（单组不配平）", f"{flat}")
    return spec


# ---------------------------------------------------------------------------
# 2/5 加载路径
# ---------------------------------------------------------------------------
def _section_load(work, check, metrics):
    check.section(2, _TOTAL_SECTIONS, "加载路径（走的是成员图 / 源图不变 / 外部数据在集内）")
    path, _source = _load_mini(work)
    before = _sha256(path)
    cfg = _cfg(groups=("all",), sigma_full=0.01)
    spec = WeightPerturbSpec.from_config(cfg)

    report = build_member_set(path, spec, 2, verbose=False)
    set_dir = report["set_dir"]
    check(set_dir == set_dir_for(path, spec),
          "成员集目录由 spec 指纹决定（同一 spec 复用同一目录）",
          osp.basename(set_dir))

    resolved = resolve_effective_path(path, spec, 1)
    check(resolved == member_model_path(path, spec, 1),
          "resolve_effective_path 把源图换成**第 index 个成员图**",
          osp.basename(resolved))
    check(osp.abspath(resolved) != osp.abspath(path), "成员图与源图不同路径")
    check(_sha256(path) == before, "源图字节未被改动")

    # 成员图对外只引用成员集目录内的 base.data / perturbed.mNNN.data：onnxruntime 会校验
    # 外部数据的**真实路径**必须落在模型所在目录内，跑到目录外就直接建不了会话。
    member = load_model(resolved)
    refs = {entry.value for tensor in external_tensors(member)
            for entry in tensor.external_data if entry.key == "location"}
    outside = sorted(
        ref for ref in refs
        if osp.dirname(osp.abspath(osp.join(set_dir, ref))) != osp.abspath(set_dir))
    check(refs and not outside,
          f"成员图的外部数据引用都在成员集目录内（{len(refs)} 个）", str(outside))

    # 框架侧真的加载了那一份成员图：图里的 spw.member 必须与请求的成员号对上。
    model = create_model(_mini_model_class(), weight_perturb=cfg)
    model.member_index = 1
    model.load(path)
    meta = dict(model.session.get_modelmeta().custom_metadata_map)
    check(meta.get("spw.member") == "1" and meta.get("spw.mechanism") == "m1a",
          "会话加载的确实是成员 1 的图（spw.member 对上，不是源图）",
          f"metadata keys={sorted(meta)}")
    metrics["loaded_member"] = meta.get("spw.member")

    feeds = np.random.default_rng(7).standard_normal(_IN_SHAPE).astype(np.float32)
    start = pd.Timestamp("2025-01-02 06:00")
    first = model.forward(feeds, 0, start)
    second = model.forward(feeds, 0, start)
    check(np.array_equal(first, second),
          "成员内冻结：同一 session 连续两次前向逐位相同")


# ---------------------------------------------------------------------------
# 3/5 成员可用性
# ---------------------------------------------------------------------------
def _section_members(work, check, metrics, members, steps):
    check.section(3, _TOTAL_SECTIONS,
                  f"成员可用性（{members} 个成员各自一份模型 / 成员内冻结 / 跨成员发散）")
    path = osp.join(work, "mini.onnx")
    cfg = _cfg(groups=("all",), sigma_full=0.01)
    spec = WeightPerturbSpec.from_config(cfg)
    report = build_member_set(path, spec, members, verbose=False)
    check(report["members"] == members and len(report["member_reports"]) == members,
          f"成员集生成 {members} 个", str(report["members"]))
    check(all(item["files"] for item in report["member_reports"])
          and all(item["dir"] == report["set_dir"] for item in report["member_reports"]),
          "成员共用成员集目录（扁平布局），每个成员都能列出自己的文件清单")

    # 逐成员加载 = 一个成员一个 session；跑完立即 release，所以任何时刻只驻留 1 份权重。
    # 这正是 cli 的 _run_member_outer 做的事，这里把它按同样的顺序走一遍。
    model = create_model(_mini_model_class(), weight_perturb=cfg)
    per_member, matched, frozen, released = [], 0, 0, 0
    for index in range(members):
        model.member_index = index
        model.load(path)
        meta = dict(model.session.get_modelmeta().custom_metadata_map)
        matched += int(meta.get("spw.mechanism") == "m1a"
                       and meta.get("spw.member") == str(index))
        first = _rollout_one(model, steps)
        second = _rollout_one(model, steps)
        frozen += int(np.array_equal(first, second))
        per_member.append(first)
        model.release()
        released += int(model.session is None)

    check(matched == members,
          "每个成员加载到的都是自己那一份权重（图里的 spw.member 逐一对上）")
    check(frozen == members,
          "成员内冻结：同一成员连跑两次 rollout 逐位相同（与图内重抽的语义分水岭）")
    check(released == members, "成员之间释放了 session（51 份权重不会同时挂住）")

    ensemble = np.stack(per_member)
    check(tuple(ensemble.shape) == (members, steps, _OUT_CH, 20, 20),
          "输出形状 = (成员, 步, C, H, W)", str(ensemble.shape))
    check(np.isfinite(ensemble).all(), "输出无 NaN/Inf")

    gaps = []
    for step in range(steps):
        block = ensemble[:, step].reshape(members, -1)
        gaps.append(float(np.abs(block[1:] - block[0]).max(axis=1).min()))
    metrics["m1a_min_gap"] = gaps
    check(min(gaps) > 0.0,
          f"每一步的成员都互不相同（最小 max|Δ|={min(gaps):.3e}）",
          str([f"{value:.2e}" for value in gaps]))

    spread = ensemble.std(axis=0).mean(axis=(1, 2, 3))
    metrics["m1a_spread"] = [float(value) for value in spread]
    check(spread[-1] > spread[0],
          f"离散度随时效增长（首步 {spread[0]:.3e} -> 末步 {spread[-1]:.3e}）")

    # 确定性对照：不加扰动时 N 条轨迹必须逐位相同——上面的差异确实来自扰动，而不是
    # 框架里别的地方带了随机性。
    deterministic = create_model(_mini_model_class())
    deterministic.load(path)
    flat = _rollout(deterministic, members, steps)
    check(all(np.array_equal(flat[0], flat[index]) for index in range(1, members)),
          "确定性对照：不加扰动时 N 个成员逐位相同")


class _StubWriter:
    """只记文件名与 raw 的最小 writer（替代 cli 的 _AsyncWriter，不落盘）。"""

    def __init__(self):
        self.names = []
        self.raw = []

    def put(self, name, ds, raw=None):
        self.names.append(name)
        self.raw.append((name, raw[0] if raw else None))

    def flush(self):
        pass

    def close(self):
        return []


def _stub_batch_model_class():
    """最小模型类 + cli 编排要用的桩：短路取数据、记下每次 load 的成员号与 release 次数。"""

    class _BatchModel(_mini_model_class()):
        """把「取数据」这一层短路掉，只留 cli 编排要用的接口。"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen = []
            self.released = 0

        def load(self, model_path):
            self.seen.append(int(self.member_index))
            return super().load(model_path)

        def release(self):
            self.released += 1
            return super().release()

        def prepare_dataset_batch(self, batch, verbose=False):
            return (np.stack([item["state"] for item in batch]),
                    [item["init"] for item in batch])

        def to_dataset(self, state, save_names=None, lat=None, lon=None):
            return {"state": np.asarray(state), "names": save_names}

    return _BatchModel


def _run_cli_m1a(path, spec, count, steps, inits, frame, *, segments=None):
    """用 cli 的 m1a 编排跑一遍，返回 (返回码, 模型桩, {输出名: 该帧的预报数组})。

    两条路径（segments=None / 分段）用的是同一份成员集与同一个初始态，所以逐帧可比：
    只要第 0 段与基线同规格，`边界之前逐位相同` 就是一条可断言的强命题。
    """
    import argparse

    from ..cli import _run_member_outer

    model = _stub_batch_model_class()()
    model.weight_perturb = spec
    writer = _StubWriter()
    args = argparse.Namespace(steps=steps, out="", model=path)
    batches = [[{"state": frame, "init": init} for init in inits]]
    code = _run_member_outer(model, None, batches, writer, args, inits,
                             list(range(count)), ["c0"], np.zeros(1), np.zeros(1),
                             6, 0, 1, segments=segments)
    return code, model, {name: array for name, array in writer.raw if array is not None}


def _section_segments(work, check, metrics, members, steps):
    check.section(5, _TOTAL_SECTIONS,
                  "分段 rollout（E3：段边界换权重 / 帧号连续 / 边界前后逐帧可对照）")
    from .segments import parse_segment_plan

    if steps < 2:
        raise ValueError("分段预检至少要 2 步")

    path = osp.join(work, "mini.onnx")
    cfg = _cfg(groups=("all",), sigma_full=0.01)
    spec = WeightPerturbSpec.from_config(cfg)
    count = min(members, 4)
    build_member_set(path, spec, count, verbose=False)

    head = steps // 2
    # 第 0 段故意与基线**同规格同种子**：于是「边界之前逐位相同」变成可断言的强命题
    # （换段不会回头污染已经算完的帧）。第 1 段换 σ 与种子，保证边界之后必然不同。
    plan = parse_segment_plan(
        [{"sigma_full": spec.sigma_full, "steps": head},
         {"sigma_full": spec.sigma_full * 2.0, "steps": steps - head, "seed": 1}],
        spec, steps)
    check([round(item.sigma_full, 6) for item in plan.specs()] == [0.01, 0.02],
          "各段 σ 按分段表落地（0.01 -> 0.02）",
          str([item.sigma_full for item in plan.specs()]))
    build_member_set(path, plan.specs()[1], count, verbose=False)

    inits = [pd.Timestamp("2025-01-02 00:00")]
    frame = np.random.default_rng(11).standard_normal(_IN_SHAPE[1:]).astype(np.float32)

    base_code, base_model, base_raw = _run_cli_m1a(path, spec, count, steps, inits, frame)
    seg_code, seg_model, seg_raw = _run_cli_m1a(path, spec, count, steps, inits, frame,
                                                segments=plan)
    check(base_code == 0 and seg_code == 0, "不分段 / 分段两条路径都正常返回 0",
          f"base={base_code} segments={seg_code}")
    check(set(seg_raw) == set(base_raw) and base_raw,
          f"分段不改输出布局：帧号仍然连续到 {steps}（{len(seg_raw)} 个文件）",
          str(sorted(set(base_raw) ^ set(seg_raw))[:4]))

    same, before = 0, 0
    diff, after = 0, 0
    for name, array in base_raw.items():
        step = int(osp.basename(name)[:-3])
        equal = np.array_equal(array, seg_raw[name])
        if step <= head:
            before += 1
            same += int(equal)
        else:
            after += 1
            diff += int(not equal)
    check(before > 0 and same == before,
          f"边界之前（前 {head} 步）逐位相同：换段没有回头改写已经算完的帧",
          f"{same}/{before}")
    check(after > 0 and diff == after,
          f"边界之后（后 {steps - head} 步）全部不同：段边界确实换了权重",
          f"{diff}/{after}")

    check(base_model.seen == list(range(count)),
          "不分段时每个成员只 load 一次", str(base_model.seen))
    expect = [index for index in range(count) for _ in plan.segments]
    check(seg_model.seen == expect,
          f"分段时每个成员 load {len(plan.segments)} 次（段边界各换一次；这是分段唯一的额外开销）",
          str(seg_model.seen))
    metrics["segments_loads"] = len(seg_model.seen)
    metrics["segments_same_before"] = same
    metrics["segments_diff_after"] = diff


def _section_cli_m1a(work, check, metrics, members, steps):
    check.section(4, _TOTAL_SECTIONS,
                  "成员集的 cli 编排（成员外层 / 逐成员 release / 输出落 member_NNN/）")
    import argparse

    from ..cli import _run_member_outer

    path = osp.join(work, "mini.onnx")
    cfg = _cfg(groups=("all",), sigma_full=0.01)
    spec = WeightPerturbSpec.from_config(cfg)
    count = min(members, 4)
    build_member_set(path, spec, count, verbose=False)

    inits = [pd.Timestamp("2025-01-02 00:00"), pd.Timestamp("2025-01-03 00:00")]
    # 真实管线里 prepare_dataset_batch 返回的 state 是 (B, 帧, C, H, W)（batch 维在最前），
    # 所以单样本这里是 (帧, C, H, W)，stack 之后才是 rank-5。
    frame = np.random.default_rng(3).standard_normal(_IN_SHAPE[1:]).astype(np.float32)

    _BatchModel = _stub_batch_model_class()
    member_ids = list(range(count))

    # 增量：已有的成员不重造（判据是 mtime 不变）。这条是 cli._prepare_member_set 里那个
    # 「只造分片里还不在盘上的」过滤器的回归——build_member_set 自己不做存在性检查，
    # 会把已有的成员整份重写一遍（FGVP 上每组几十 GB）。
    from ..cli import _prepare_member_set

    before = osp.getmtime(member_model_path(path, spec, 0))
    _prepare_member_set(path, spec, count, member_ids, 0, 1)      # 全都在盘上 -> 跳过
    _prepare_member_set(path, spec, count + 1, [0, count], 0, 1)  # 只缺 member_count
    check(osp.getmtime(member_model_path(path, spec, 0)) == before,
          "成员已在盘上时不重造（mtime 不变）")
    check(osp.isfile(member_model_path(path, spec, count)),
          f"增量补齐：member_{count:03d} 已就位")

    model = _BatchModel()
    model.weight_perturb = spec          # 让 load() 真的走 m1a 分支（成员解析 + 自检）
    writer = _StubWriter()
    # args.model 必须是**源确定性图**的路径：cli 的编排靠它算成员模型路径（真实的
    # worker 里这个值是 cfg.model_path_abs()，即绝对 onnx 路径）。
    args = argparse.Namespace(steps=steps, out=work, model=path)
    # 一个 batch 装两个起报：这是成员循环**内层**再套一层起报循环的情况，最容易写错。
    batches = [[{"state": frame, "init": init} for init in inits]]

    code = _run_member_outer(model, None, batches, writer, args, inits, member_ids,
                             ["c0"], np.zeros(1), np.zeros(1), 6, 0, 1)
    check(code == 0, "编排正常返回 0", str(code))
    check(model.seen == member_ids,
          "逐成员按顺序 load 了一次（成员号接得上）", str(model.seen))
    check(model.released == count, f"每个成员跑完都 release 了（{model.released}/{count}）")
    check(model.session is None, "跑完之后没有 session 挂住（51 份权重不会同时驻留）")

    expected = {f"{init:%Y%m%d}/member_{index:03d}/{step:03d}.nc"
                for init in inits for index in member_ids
                for step in range(1, steps + 1)}
    check(set(writer.names) == expected,
          f"输出路径 = 起报/member_NNN/步号.nc（{len(writer.names)} 个文件）",
          str(sorted(set(writer.names) ^ expected)[:4]))
    metrics["cli_files"] = len(writer.names)

    last_by_member = {}
    for name, array in writer.raw:
        if name.endswith("/%03d.nc" % steps):
            last_by_member[name.split("/")[1]] = array
    stack = np.stack([last_by_member[f"member_{index:03d}"] for index in member_ids])
    gaps = float(np.abs(stack[1:] - stack[0]).max())
    check(gaps > 0.0, f"末步各成员的内容确实不同（跨成员 max|Δ|={gaps:.3e}）")


def run_e2e(members=51, steps=6, *, verbose=True, keep=False):
    work = tempfile.mkdtemp(prefix="spw_e2e_")
    check = _Checker(verbose=verbose)
    metrics = {}
    print("SPW 端到端预检：真实框架路径 "
          "（create_model -> OnnxInferModel.load -> Rollout）")
    print(f"工作目录：{work}")
    try:
        _write_mini(osp.join(work, "mini.onnx"))
        _section_spec(work, check, metrics)
        _section_load(work, check, metrics)
        _section_members(work, check, metrics, members, steps)
        _section_cli_m1a(work, check, metrics, members, steps)
        _section_segments(work, check, metrics, members, steps)

        print("\n" + "=" * 72)
        print("关键数字")
        print(f"  选中张量 {len(_EXPECT_SELECTED)} 个；配平后 σ = "
              f"{ {k: round(v, 5) for k, v in metrics.get('sigmas', {}).items()} }")
        spread = metrics.get("m1a_spread") or []
        if spread:
            print(f"  {members} 成员离散度（逐步）：" +
                  " ".join(f"{value:.2e}" for value in spread))
        if metrics.get("cli_files"):
            print(f"  cli 编排产出 {metrics['cli_files']} 个输出文件"
                  "（按 起报/member_NNN/步号 命名）")
        if metrics.get("segments_loads"):
            print(f"  分段 rollout（E3）：{metrics['segments_loads']} 次 load；"
                  f"边界前逐位相同 {metrics['segments_same_before']} 帧、"
                  f"边界后全部不同 {metrics['segments_diff_after']} 帧")
        print(f"合计：{check.passed} 条通过，{len(check.failures)} 条不通过")
        if check.failures:
            print("不通过项：")
            for failure in check.failures:
                print(f"  - {failure}")
            print("结论：端到端接线有问题，先别上服务器跑 51 成员。")
            return 1
        print("结论：框架接线通过——成员各自一份冻结权重、加载的是自己那一份、"
              "成员内逐位可复现、成员间发散、cli 编排落地正确。")
        print("      这只说明**框架**没问题；FGVP 本身的推理仍要在服务器上跑 "
              "离散度体检。")
        return 0
    finally:
        if keep:
            print(f"已保留工作目录：{work}")
        else:
            import shutil

            shutil.rmtree(work, ignore_errors=True)


__all__ = ["run_e2e"]
