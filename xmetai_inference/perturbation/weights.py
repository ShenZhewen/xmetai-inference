# -*- coding: utf-8 -*-
"""M1a：把「冻结的逐成员扰动权重」物理落盘（对应论文的主配置）。

论文出处：Adamov et al. 2026（arXiv:2609.08412）——每个成员抽一次
``W~ = W ⊙ (1 + σ ξ)``，然后**整条轨迹用同一组权重**跑完 360 h。

为什么是这条（它当年是跟「图内随机算子」那条路比出来的，后者 2026-09-23 淘汰、
2026-09-24 删码）：

  * 语义。图内重抽等价论文的「沿轨迹刷新噪声」变体；逐成员冻结才是论文的
    生产配置。实测图内重抽的 spread/RMSE 随时效**下降**（0.23 -> 0.09），而健康集合是
    上升的（fgvp_hou 0.38 -> 1.23）——成员不是一个「系统性不同的模型」，扰动不累积。
  * 开销。图内重抽每次前向都要在 CPU 上生成全部被扰动元素的随机数（FGVP 规模 731M 元素，
    实测约 6 s/前向）；这里的扰动在生成阶段算一次，**推理零开销**。
  * 可排查。每个成员就是一份**普通的确定性 onnx**，可以单独跑、单独审计。这一轮
    失败最难的地方就是成员不可分解，偏差一路静默跑到生产。

落盘布局（一个成员集 = 一个**扁平**目录）::

    <set_dir>/base.data                  # 源 .data 的硬链接（零字节）；不可行时整份复制一次
    <set_dir>/<stem>.m000.onnx           # 第 0 个成员的图（外部引用都是本目录内的裸文件名）
    <set_dir>/perturbed.m000.data        # 第 0 个成员被扰动的那些张量
    <set_dir>/spw_members.json           # 清单：规格、逐组 σ、种子、逐成员文件指纹

为什么是扁平目录、而不是「每成员一个子目录」：onnxruntime（新版本）会校验外部数据的
**真实路径**必须落在模型所在目录内。一个成员就是一张模型，所以 base/perturbed 必须与它
同目录；而 base 的字节还得被所有成员共享——只有硬链接能让「另一个名字的真实路径仍在
目录内」，符号链接会被解析到目录外而被直接拒绝（2026-09-23 服务器实测，见 README 坑 6）。
扁平布局顺带把「硬链接不可用」的代价从「每成员复制 2.98 GB」降到「整个成员集复制一次」。

三条实测约束（更早的坑见 README 第 5 节）：

  * **图逐成员各写一份**（85 MB x 51 ≈ 4.3 GB），不做硬链接：被扰动张量里有 64 元素这种
    小张量，它们必须留在图内（onnxruntime 会话初始化时要读，外置会加载失败），所以图内容
    逐成员不同。4.3 GB 相对权重字节数可以忽略。
  * **噪声必须逐 (成员, 张量) 独立**：用 numpy 的 SeedSequence 从 (seed, 成员号) 派生，
    每个张量一条独立子流；顺序固定为 initializer 顺序，保证可复现。
  * **字节序必须核对**：ONNX 的外部数据没有「序」标记，读取一律按 C 序解释。若某个导出器
    写的是 Fortran 序，「读出来再写回去」会把权重悄悄转置——不报错，只让预报变差。所以
    每个张量读回来都和文件里的原始字节逐位比对，不一致直接报错（见 _assert_c_order）。
"""
from __future__ import annotations

import json
import os
import os.path as osp
import shutil

import numpy as np
from dataclasses import replace

from .graph import (
    _check_model_path,
    _external_data_outside,
    _onnx,
    _verify_external_refs,
    external_tensors,
    load_model,
    resolve_sigmas,
    select_initializers,
    tensor_array,
)
from .spec import WeightPerturbSpec

BASE_DATA = "base.data"                      # 成员集里共享的源 .data（硬链接，或复制一次）
PERTURBED_DATA_FMT = "perturbed.{tag}.data"  # 逐成员的扰动权重
MEMBER_MODEL_FMT = "{stem}.{tag}.onnx"       # 逐成员的图
MANIFEST_NAME = "spw_members.json"
_BASE_PREFIX = "base"                        # base.data / base_N.data
_PERTURBED_PREFIX = "perturbed"
_MEMBER_TAG_FMT = "m{:03d}"


def _base_spec(spec: WeightPerturbSpec) -> WeightPerturbSpec:
    """归一成落盘用的规范形态：种子显式化（None -> 0）。

    **与进程 rank 无关**——成员集是**多卡共享**的落盘产物，逐成员的独立性来自
    ``SeedSequence([seed, index])``。历史上这里还兼着「丢掉 m3 的 rank 偏移」，
    那个语义会同时坏掉两件事：

      * **目录指纹带上 rank** -> 每个 rank 各造一份成员集，磁盘翻 world_size 倍；更糟的是
        分片生成时「等其它 rank 把剩下的造完」永远等不到（大家都在自己的目录里造），
        最后以 1 小时超时报错收场。
      * **各 rank 用不同基础 seed 造成员** -> 同一个成员的内容取决于「是谁造的」，而且
        相邻成员的随机流会重叠。

    m3 已删（2026-09-24），`seed=None -> 0` 是这里唯一还剩的归一。
    """
    seed = 0 if spec.seed is None else int(spec.seed)
    if spec.seed == seed:
        return spec
    return replace(spec, seed=seed)


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def set_dir_for(src_path: str, spec: WeightPerturbSpec, *, out_dir: str | None = None) -> str:
    """成员集目录：<目录>/<模型名>.spw-members-<指纹>/。

    目录优先级 out_dir -> member_dir -> cache_dir -> XMETAI_SPW_CACHE_DIR -> 源模型同目录，
    文件名同样带 spec+源文件的指纹，所以换 σ、
    换组、权重更新都会自然落到新目录（目录名**不含成员数**，所以先生成 5 个再补到 51 个
    会落在同一个目录里并把清单合并）。

    指纹里的种子取**基础 seed**，不含进程 rank（见 _base_spec）——多卡共享同一份成员集。
    """
    spec = _base_spec(spec)
    directory = (out_dir or spec.member_dir or spec.cache_dir
                 or os.environ.get("XMETAI_SPW_CACHE_DIR")
                 or osp.dirname(osp.abspath(src_path)))
    stem = osp.splitext(osp.basename(src_path))[0]
    name = f"{stem}.spw-members-{spec.fingerprint(src_path)}"
    return osp.join(osp.abspath(directory), name)


def member_tag(index: int) -> str:
    """成员号在文件名里的写法（``m000``）：成员共用成员集目录，靠它区分。"""
    return _MEMBER_TAG_FMT.format(int(index))


def _perturbed_name(index: int) -> str:
    return PERTURBED_DATA_FMT.format(tag=member_tag(index))


def member_dir_for(src_path: str, spec: WeightPerturbSpec, index: int,
                   *, out_dir: str | None = None) -> str:
    """成员图与它依赖的 base/perturbed 数据所在目录 —— 即**成员集目录本身**。

    不再逐成员开子目录（原因见模块 docstring）：onnxruntime 会校验外部数据的真实路径
    必须落在模型所在目录内，而 base 的字节又要被所有成员共享——只有硬链接同时满足这两
    条。保留 index 参数是为了不动调用方签名。
    """
    return set_dir_for(src_path, spec, out_dir=out_dir)


def member_model_path(src_path: str, spec: WeightPerturbSpec, index: int,
                      *, out_dir: str | None = None) -> str:
    """第 index 个成员的 .onnx 路径。框架侧就靠它把 model_path 换成成员模型。"""
    stem = osp.splitext(osp.basename(src_path))[0]
    name = MEMBER_MODEL_FMT.format(stem=stem, tag=member_tag(index))
    return osp.join(member_dir_for(src_path, spec, index, out_dir=out_dir), name)


def member_perturbed_path(src_path: str, spec: WeightPerturbSpec, index: int,
                          *, out_dir: str | None = None) -> str:
    """第 index 个成员自己那份被扰动权重的落盘路径。"""
    return osp.join(member_dir_for(src_path, spec, index, out_dir=out_dir),
                    _perturbed_name(index))


def member_base_path(src_path: str, spec: WeightPerturbSpec, *,
                     out_dir: str | None = None) -> str:
    """成员集里共享的源 .data（硬链接，或唯一的那份复制）。"""
    return osp.join(set_dir_for(src_path, spec, out_dir=out_dir), BASE_DATA)

def member_exists(src_path: str, spec: WeightPerturbSpec, index: int, *,
                  out_dir: str | None = None) -> bool:
    """该成员的 .onnx 是否已经在盘上（只查文件，不触发任何构建）。"""
    return osp.isfile(member_model_path(src_path, spec, index, out_dir=out_dir))


def resolve_member_path(src_path: str, spec: WeightPerturbSpec, index: int, *,
                        out_dir: str | None = None) -> str:
    """框架侧入口：返回第 index 个成员的模型路径；不在盘上就抛出可执行的构建命令。

    这里**不自动构建**：一次要造 count 个成员、
    写几十 GB，只能由调用方显式触发（cli 会在跑推理前按 rank 分片造好）。挂在「加载
    模型」这条必然会跑的路径上自动触发，等于把磁盘和时间的账单藏进加载里。
    """
    path = member_model_path(src_path, spec, index, out_dir=out_dir)
    if osp.isfile(path):
        return path
    raise FileNotFoundError(
        "M1a 成员模型不存在：" + path + "\n"
        "  成员是预先落盘的冻结权重，需要先生成（当前规格：" + spec.describe() + "）。"
        "单卡可以先跑：\n"
        "    python -m xmetai_inference.perturbation members "
        + osp.abspath(src_path) + " --members <成员数> --groups "
        + ",".join(spec.groups) + "\n"
        "  直接跑推理也行：cli 会在建 session 之前按 rank 分片自动生成。")


# ---------------------------------------------------------------------------
# 源布局与字节序
# ---------------------------------------------------------------------------
def _external_location(tensor):
    for entry in tensor.external_data:
        if entry.key == "location" and entry.value:
            return entry.value
    return None


def _location_closure(proto) -> dict:
    """成员图里的外部数据引用分布：location -> 引用它的张量个数。

    口径是 graph.external_tensors（含被外置的 Constant 节点张量）。verify_member_set 拿它
    断言「所有引用都落在成员目录内」——只查可达性是不够的：漏改的 location 会退化成源
    .data 的名字，而源 .data 只要还在原地就照样可达，于是问题一路静默到建会话才炸。
    """
    counts: dict = {}
    for tensor in external_tensors(proto):
        location = _external_location(tensor)
        counts[location] = counts.get(location, 0) + 1
    return counts



def _source_layout(onnx, model) -> dict:
    """源图里出现过的外部数据位置 -> 成员目录里的 base 文件名。

    统一改名（base.data / base_N.data）而不是沿用原名：源图的 location 可能是绝对路径或
    指向别的目录，统一成固定名字后成员目录才是真正自包含的（图里的相对路径一定能解析）。

    口径是 external_tensors（含 Constant 节点的张量属性），不是 graph.initializer：
    被外置的常量同样按「相对图文件的路径」解析，漏掉它们成员目录就不自包含。
    """
    locations: list[str] = []
    for tensor in external_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        location = _external_location(tensor)
        if location and location not in locations:
            locations.append(location)
    if not locations:
        return {}
    if len(locations) == 1:
        return {locations[0]: "base.data"}
    return {location: f"base_{i}.data" for i, location in enumerate(locations)}


def _raw_bytes(tensor, src_dir: str):
    """外部数据里的原始字节（不解释、不重排）。inline 张量返回 None。"""
    onnx = _onnx()
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        return None
    info = {entry.key: entry.value for entry in tensor.external_data}
    location = info.get("location") or ""
    if not location:
        return None
    path = location if osp.isabs(location) else osp.join(src_dir, location)
    offset = int(info.get("offset") or 0)
    length = int(info.get("length") or 0)
    if length <= 0:
        return None
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(length)


def _assert_c_order(tensor, src_dir: str, array, name: str):
    """确认源数据就是 C 序：memmap 读回来的字节必须与文件里的原始字节逐位相同。

    不做这一步的风险：Fortran 序的外部数据被按 C 序读出来、再按 C 序写回去，等于把权重
    转置。它不会报错、不会 NaN，只会让预报变差——正是这一轮最该避免的失败方式。
    """
    raw = _raw_bytes(tensor, src_dir)
    if raw is None:
        return
    if np.asarray(array).tobytes() != raw:
        raise RuntimeError(
            f"张量 {name} 的外部数据不是 C 序（按 C 序读回来的字节与文件里的原始字节不同）。"
            "按现有实现改写会把权重转置，已中止。请先用 tensor_array 的口径核对这份导出。")


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def _noise_shape(shape, struct: str, block: int):
    """按结构给出**独立抽样**的形状；用它广播/铺开就得到 ξ。

    四种结构都保证每个元素 E[ξ]=0、Var(ξ)=1，所以 `σ^2 x ‖W‖^2` 的注入方差定律
    与组间 energy 配平在四种结构下都成立 —— 这是「同预算、换结构」可比的前提。
    ndim<2 的张量（bias / norm 权重 / dt_bias）在 channel/block 下退化为 iid：它们本来
    就是逐通道的向量，再「按通道共享」就等于全张量共享一个 ξ —— 那种「整张量共享」已
    作为 scalar 结构单独实现（见 spec.NOISE_STRUCTS），不必靠 channel 的退化去近似。
    """
    if struct == "scalar":
        # 必须在 `len(shape) < 2` 那道退化之前判：scalar 的意义就是**整张量一个 xi**，
        # 对 bias / dt_bias 这类一维张量也一样（channel 对向量退化成 iid，scalar 不退）。
        return (1,) * len(shape)
    if struct == "iid" or len(shape) < 2:
        return shape
    if struct == "channel":
        return (shape[0],) + (1,) * (len(shape) - 1)
    if struct == "block":
        rows = shape[0]
        cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        return ((rows + block - 1) // block, (cols + block - 1) // block)
    raise ValueError(f"未知的 noise_struct：{struct!r}")


def _expand_noise(shape, struct: str, block: int, rng):
    """按结构抽 ξ 并展开到 shape。"""
    draw_shape = _noise_shape(shape, struct, block)
    if draw_shape == shape:
        return rng.standard_normal(shape)
    if struct in ("channel", "scalar"):
        return np.broadcast_to(rng.standard_normal(draw_shape), shape)
    rows = shape[0]
    cols = int(np.prod(shape[1:]))
    tiled = np.repeat(np.repeat(rng.standard_normal(draw_shape), block, axis=0),
                      block, axis=1)[:rows, :cols]
    return tiled.reshape(shape)


def _draw(array, sigma: float, rng, struct: str = "iid", block: int = 8) -> np.ndarray:
    """W~ = W ⊙ (1 + σ ξ)。fp32 走 fp32 快路，其它 dtype 在 float64 上算再落回。

    struct 决定 ξ 在张量内部的相关结构（见 spec.NOISE_STRUCTS）。四种结构的每元素方差都是 1，
    所以它只改「能量注到哪些自由度」，不改「注入多少能量」。
    """
    base = np.asarray(array)
    if base.dtype == np.float32:
        noise = _expand_noise(base.shape, struct, block, rng).astype(np.float32)
        out = base.astype(np.float32, copy=True)
        out *= (np.float32(1.0) + np.float32(sigma) * noise)
        return out
    work = base.astype(np.float64)
    work *= (1.0 + float(sigma) * _expand_noise(base.shape, struct, block, rng))
    return work.astype(base.dtype)


def _share_data(src: str, dst: str, *, link: bool) -> tuple[str, str]:
    """把源 .data 放进成员集目录：硬链接（首选，零字节）-> 整份复制。

    **刻意不用符号链接**：新版本 onnxruntime 会把外部数据解析成**真实路径**，再校验它
    是否落在模型所在目录内；符号链接指到目录外会被直接拒绝（2026-09-23 服务器实测）：

        External data path: "base.data" (resolved path: ".../models/<stem>.data")
        escapes both model directory: ".../member_001" and real model directory: ...

    硬链接没有这个问题：另一个名字的真实路径仍在目录内。

    硬链接会在两种真实场景下失败：跨文件系统（EXDEV），以及 fs.protected_hardlinks=1
    （多数发行版的默认值）时**源文件不属于当前进程用户**（EPERM——容器里挂载一份别人
    拥有的只读权重目录就是这么撞上的，2026-09-23 服务器正是这个）。失败就整份复制，
    因为成员集是扁平布局，这份复制只发生**一次**。

    返回 (方式, 失败原因)。方式 ∈ link / copy / exist。
    """
    if osp.lexists(dst) and not osp.exists(dst):
        os.unlink(dst)                      # 断掉的符号链接：当成不存在，重做
    elif osp.exists(dst):
        return "exist", ""
    reason = ""
    if link:
        try:
            os.link(src, dst)
            return "link", ""
        except OSError as error:
            reason = f"硬链接失败（{error.strerror or error}）"
    else:
        reason = "调用方要求不共享（link=False）"
    shutil.copy2(src, dst)
    return "copy", reason


def _rebind_unperturbed(onnx, proto, base_map, selected):
    """把**未被扰动**的外部张量也改到成员目录里的 base_*.data。

    源图的 location 可能是绝对路径、也可能带子目录；不改的话成员目录就不是自包含的，
    图会去源目录找 .data（本地测试时表现为「引用不存在」，换台机器就是加载失败）。
    offset/length 不用动——base 文件是源 .data 的硬链接，字节布局逐位相同。

    范围是 external_tensors（含 Constant 节点的张量属性）：2026-09-23 服务器上就是因为
    只扫 initializer，漏掉了 4 个被外置的 attention 常量，成员图改完仍指向 <stem>.data，
    onnxruntime 建会话时报 cannot get file size。改这里时别再退回 initializer。
    """
    for tensor in external_tensors(proto):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        if tensor.name in selected:
            continue                      # 稍后会被改写成 PERTURBED_DATA
        location = _external_location(tensor)
        base_name = base_map.get(location)
        if not base_name:
            raise RuntimeError(
                f"张量 {tensor.name} 的外部数据位置 {location!r} 不在源布局里"
                f"（源布局 {sorted(base_map)}）——成员目录无法自包含，已中止")
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = base_name


def _rewrite_member(onnx, proto, src_dir: str, member_dir: str, order, sigma_of,
                    seed: int, index: int, base_map, perturbed_name: str,
                    struct: str = "iid", block: int = 8):
    """把 proto 里被选中的张量换成扰动后的值；外部数据写进成员目录的 perturbed.*.data。"""
    by_name = {tensor.name: tensor for tensor in proto.graph.initializer}
    _rebind_unperturbed(onnx, proto, base_map, set(order))
    data_path = osp.join(member_dir, perturbed_name)
    children = np.random.SeedSequence([int(seed), int(index)]).spawn(len(order))
    records = []
    cursor = 0
    with open(data_path, "wb") as handle:
        for name, child in zip(order, children):
            tensor = by_name[name]
            is_external = tensor.data_location == onnx.TensorProto.EXTERNAL
            elements = int(np.prod(tensor.dims)) if tensor.dims else 1
            array = tensor_array(tensor, src_dir)
            _assert_c_order(tensor, src_dir, array, name)
            new = _draw(array, sigma_of[name], np.random.default_rng(child),
                        struct=struct, block=block)
            raw = new.tobytes()
            if is_external:
                del tensor.external_data[:]
                for key, value in (("location", perturbed_name),
                                   ("offset", str(cursor)),
                                   ("length", str(len(raw)))):
                    entry = tensor.external_data.add()
                    entry.key, entry.value = key, value
                handle.write(raw)
                cursor += len(raw)
            else:
                # 小张量必须留在图内：onnxruntime 会话初始化时要读它们，外置会加载失败。
                tensor.CopyFrom(onnx.numpy_helper.from_array(new, name=name))
            records.append({"name": name, "sigma": sigma_of[name],
                            "elements": elements, "external": is_external,
                            "bytes": len(raw)})
    return records, cursor


def _stamp_member(onnx, proto, src_path: str, spec, index: int, sigmas, tensors: int,
                  member_dir: str):
    props = {entry.key: entry.value for entry in proto.metadata_props}
    props.update({
        "spw.method": "m1a-frozen-per-member-weights",
        # **落盘 schema，不要改名**：磁盘上那批成员集（S2B 的 51 个 / 76 GB）都写着这个
        # 键，backends/onnx.py 与 selftest 靠它认「这张图是成员图」。spec 里那个同名的
        # mechanism 字段已经删了，这里留着的是历史数据的标记。
        "spw.mechanism": "m1a",
        "spw.member": str(index),
        "spw.spec": spec.describe(),
        # 用 " | " 连接各条目：并集 group token 内部本来含逗号（见 graph.group_alternatives），
        # 再用逗号连接会读不出条目边界。
        "spw.sigma_per_group": " | ".join(
            f"{key}={value:.6g}" for key, value in sorted(sigmas.items())),
        "spw.tensors": str(tensors),
        "spw.seed": str(spec.seed if spec.seed is not None else ""),
        "spw.source": osp.basename(src_path),
        "spw.source_fingerprint": spec.fingerprint(src_path),
        "spw.spec_json": json.dumps(spec.budgets(), sort_keys=True, ensure_ascii=False),
        "spw.member_dir": member_dir,
    })
    onnx.helper.set_metadata_props(proto, props)


def _merge_manifest(manifest: str, report: dict) -> dict:
    """把本次生成的成员并进已有清单（多 rank 分片生成时各写各的，不能互相抹掉）。"""
    try:
        with open(manifest, encoding="utf-8") as handle:
            old = json.load(handle)
    except (OSError, ValueError):
        return report
    if not isinstance(old, dict) or old.get("fingerprint") != report.get("fingerprint"):
        return report                       # 规格或源模型变了：旧清单作废
    merged = {int(item["index"]): item for item in (old.get("member_reports") or [])}
    merged.update({int(item["index"]): item for item in report["member_reports"]})
    report["member_reports"] = [merged[key] for key in sorted(merged)]
    report["members"] = max(int(old.get("members") or 0), int(report["members"]))
    warnings = list(old.get("warnings") or [])
    warnings += [w for w in report["warnings"] if w not in warnings]
    report["warnings"] = warnings
    report["built"] = sorted(set(old.get("built") or []) | set(report.get("built") or []))
    return report


def build_member_set(src_path: str, spec: WeightPerturbSpec, count: int, *,
                     indices: "list[int] | None" = None,
                     out_dir: str | None = None, link: bool = True,
                     verbose: bool = True) -> dict:
    """生成 count 个成员中的一部分（indices），返回报告 dict（含逐成员路径）。

    已存在的文件按同名同内容复用，所以重复调用是幂等的。

    indices 缺省 = 全部成员。多卡并行时每个 rank 造自己那一份
    （``range(rank, count, world_size)``）：每个成员的 ``perturbed.mNNN.data`` 是直接以
    "wb" 打开的，两个 rank 同时造同一个成员会写出交错的字节，所以分片必须互不相交。清单
    （spw_members.json）会与已有内容按成员号合并，分片生成也是自洽的。

    ``base.data`` 只建**一次**（与成员号无关）：分片生成时谁先到谁建，其余调用看到的是
    "exist"——不是竞态，内容逐位相同。
    """
    onnx = _onnx()
    src_path = osp.abspath(src_path)
    src_dir = osp.dirname(src_path)
    spec = _base_spec(spec)
    if count <= 0:
        raise ValueError(f"build_member_set 需要 count > 0，实际 {count}")
    todo = list(range(count)) if indices is None else sorted({int(i) for i in indices})
    if not todo:
        raise ValueError("build_member_set 的 indices 为空，没有要生成的成员")
    out_of_range = [i for i in todo if not 0 <= i < count]
    if out_of_range:
        raise ValueError(
            f"build_member_set 的 indices 越界（count={count}）：{out_of_range[:8]}")

    model = load_model(src_path)
    selection = select_initializers(model, spec, src_dir)
    sigmas = resolve_sigmas(selection, spec)
    order = [tensor.name for tensor in model.graph.initializer
             if tensor.name in selection.selected]
    sigma_of = {name: float(sigmas[selection.selected[name].group]) for name in order}
    base_map = _source_layout(onnx, model)

    set_dir = set_dir_for(src_path, spec, out_dir=out_dir)
    os.makedirs(set_dir, exist_ok=True)

    # 成员图与 base/perturbed 必须同在 set_dir：onnxruntime 会校验外部数据的真实路径要落在
    # 模型所在目录内，所以「共享一份 base」只能靠硬链接；硬链接不可行时退到复制，而扁平
    # 布局让这份复制**只发生一次**（不是每成员一次）。
    share: dict[str, str] = {}
    warnings: list[str] = []
    for location, base_name in base_map.items():
        source = location if osp.isabs(location) else osp.join(src_dir, location)
        mode, reason = _share_data(osp.abspath(source),
                                   osp.join(set_dir, base_name), link=link)
        share[base_name] = mode
        if mode == "copy":
            warnings.append(
                f"{base_name}: 无法与源 .data 共享字节（{reason}），已整份复制一次"
                f"（{osp.getsize(source) / 2**30:.1f} GB）——所有成员共用这一份，不会按"
                "成员数翻倍。想省掉这份复制就得让硬链接可用（与源文件同卷、且源文件属于"
                "当前用户），排查清单见 README 第 10 节第 5 条。")
    if any(mode == "link" for mode in share.values()):
        warnings.insert(0, "base_*.data 用硬链接共享源 .data，onnx 的 checker 会因此报"
                           "「multiple hard links」并拒绝——已按既有做法放宽该类定位"
                           "校验，引用可达性由自校验承担。")

    members = []
    for index in todo:
        perturbed_name = _perturbed_name(index)
        proto = onnx.ModelProto()
        proto.CopyFrom(model)
        records, perturbed_bytes = _rewrite_member(
            onnx, proto, src_dir, set_dir, order, sigma_of,
            int(spec.seed or 0), index, base_map, perturbed_name,
            struct=spec.noise_struct, block=spec.noise_block)
        _stamp_member(onnx, proto, src_path, spec, index, sigmas, len(order), set_dir)
        out_path = member_model_path(src_path, spec, index, out_dir=out_dir)
        tmp = f"{out_path}.tmp{os.getpid()}"
        onnx.save(proto, tmp)
        os.replace(tmp, out_path)
        broken = _verify_external_refs(onnx, proto, set_dir)
        if broken:
            raise RuntimeError(f"成员 {index} 的外部数据引用有问题：" + "; ".join(broken[:5]))
        if _external_data_outside(onnx, proto, set_dir):
            raise RuntimeError(f"成员 {index} 有外部数据落在成员集目录之外")
        # 放宽 checker 的「外部数据定位」那几条：base.data 是源 .data 的硬链接，而 onnx 的
        # checker 会以「多个硬链接 = 潜在硬链接攻击」为由拒绝这样的文件。我们不用复制
        # 2.8 GB 来换它闭嘴——引用可达性与字节范围由 _verify_external_refs 自校验顶上，
        # location 又都是本目录内的裸文件名（相对路径天然成立）。
        _check_model_path(onnx, out_path, relax_locations=True)
        members.append({
            "index": index, "dir": set_dir, "model": out_path,
            "perturbed": osp.join(set_dir, perturbed_name),
            "perturbed_bytes": perturbed_bytes,
            "files": sorted(os.listdir(set_dir)),
        })
        if verbose:
            print(f"  {osp.basename(out_path)}: {len(records)} 个张量 / "
                  f"{perturbed_bytes / 2**20:.1f} MB 被扰动权重")

    report = {
        "source": src_path,
        "set_dir": set_dir,
        "spec": spec.describe(),
        "fingerprint": spec.fingerprint(src_path),
        "seed": int(spec.seed or 0),
        "members": count,
        "tensors_per_member": len(order),
        "sigmas": {key: float(value) for key, value in sorted(sigmas.items())},
        "base_map": base_map,
        "share": share,
        "perturbed_elements": sum(selection.selected[name].elements for name in order),
        "member_reports": members,
        "warnings": warnings,
        "built": todo,
    }
    manifest = osp.join(set_dir, MANIFEST_NAME)
    report = _merge_manifest(manifest, report)
    with open(manifest, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    report["manifest"] = manifest
    if verbose:
        total = sum(item["perturbed_bytes"] for item in members)
        shard = "全部" if len(todo) == count else f"本次 {len(todo)}/{count} 个"
        print(f"成员集就绪（{shard}）-> {set_dir}")
        print(f"  被扰动元素 {report['perturbed_elements'] / 1e6:.2f}M / 成员，"
              f"本次生成 {len(members)} 个 x {total / 2**30:.2f} GB")
        if share:
            print("  共享的源 .data：" + ", ".join(
                f"{name}={mode}" for name, mode in sorted(share.items())))
        for group, sigma in sorted(sigmas.items()):
            print(f"  sigma[{group}] = {sigma:.6g}")
        for warning in warnings:
            print(f"  [警告] {warning}")
    return report


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def verify_member_set(src_path: str, spec: WeightPerturbSpec, count: int, *,
                      out_dir: str | None = None, sample: int = 8) -> dict:
    """独立复核成员集：文件齐全、引用可达、以及**成员之间的权重真的不同**。

    还会静态量一遍「权重级相对扰动」rms(W~-W)/rms(W)，应当≈该组配平后的 σ。这是能在
    本地、不跑模型就看出来的量级证据；探针（probe）量的是它在输出上的放大。
    """
    onnx = _onnx()
    src_path = osp.abspath(src_path)
    src_dir = osp.dirname(src_path)
    spec = _base_spec(spec)
    model = load_model(src_path)
    selection = select_initializers(model, spec, src_dir)
    sigmas = resolve_sigmas(selection, spec)

    problems: list[str] = []
    ratios: dict[str, list[float]] = {}
    names = list(selection.selected)
    step = max(1, len(names) // max(1, sample))
    picked = names[::step][:sample]

    source_values = {}
    for name in picked:
        tensor = next(t for t in model.graph.initializer if t.name == name)
        source_values[name] = np.asarray(tensor_array(tensor, src_dir))

    members = []
    set_dir = member_dir_for(src_path, spec, 0, out_dir=out_dir)
    real_dir = osp.realpath(set_dir)
    for index in range(count):
        model_path = member_model_path(src_path, spec, index, out_dir=out_dir)
        if not osp.isfile(model_path):
            problems.append(f"member_{index:03d}: 缺少 {model_path}")
            continue
        proto = load_model(model_path)
        for broken in _verify_external_refs(onnx, proto, set_dir):
            problems.append(f"member_{index:03d}: {broken}")
        # 可达还不够，位置还必须落在成员集目录内（裸文件名）。漏改一行 location，引用就
        # 退化成源 .data 的名字；而源 .data 只要还在原地，可达性检查什么都发现不了。
        closure = _location_closure(proto)
        stray = sorted(str(location) for location in closure
                       if not location or osp.dirname(location)
                       or not (location.startswith(_BASE_PREFIX)
                               or location.startswith(_PERTURBED_PREFIX)))
        if stray:
            problems.append(
                f"member_{index:03d}: 外部数据引用不在成员集目录内：{stray}"
                "（成员不是自包含的，onnxruntime 建会话时会去找源 .data）")
        # 新版 onnxruntime 会校验外部数据的**真实路径**必须落在模型所在目录内，于是符号
        # 链接（以及带 .. 的引用）会被直接拒绝——这里提前拦下，别等建会话才报。
        for location in sorted((key for key in closure if key), key=str):
            real = osp.realpath(osp.join(set_dir, location))
            if real != osp.join(real_dir, location):
                problems.append(
                    f"member_{index:03d}: {location} 的真实路径在成员集目录之外"
                    f"（{real}）——新版 onnxruntime 的外部数据路径校验会拒绝它")
        by_name = {tensor.name: tensor for tensor in proto.graph.initializer}
        for name in picked:
            tensor = by_name.get(name)
            if tensor is None:
                problems.append(f"member_{index:03d}: 少了张量 {name}")
                continue
            got = np.asarray(tensor_array(tensor, set_dir))
            want = source_values[name]
            if got.shape != want.shape:
                problems.append(f"member_{index:03d}: {name} 形状变了")
                continue
            if np.array_equal(got, want):
                problems.append(f"member_{index:03d}: {name} 与源权重逐位相同（扰动没生效）")
                continue
            ratio = float(np.sqrt(np.mean((got - want) ** 2))
                          / (np.sqrt(np.mean(want ** 2)) or 1.0))
            ratios.setdefault(selection.selected[name].group, []).append(ratio)
        perturbed_path = member_perturbed_path(src_path, spec, index, out_dir=out_dir)
        members.append({"index": index, "model": model_path,
                        "locations": closure,
                        "perturbed_bytes": osp.getsize(perturbed_path)
                        if osp.isfile(perturbed_path) else 0})

    measured = {}
    for group, values in sorted(ratios.items()):
        values.sort()
        measured[group] = {
            "measured_median": values[len(values) // 2],
            "expected_sigma": float(sigmas[group]),
            "ratio": values[len(values) // 2] / float(sigmas[group]),
        }
    return {"problems": problems, "members": members, "amplitude": measured,
            "sampled_tensors": picked, "count": count}


def format_member_report(info: dict) -> str:
    lines = []
    lines.append(f"成员校验：{len(info['members'])}/{info['count']} 个成员就位")
    for item in info["members"]:
        lines.append(f"  member_{item['index']:03d}  被扰动权重 "
                     f"{item['perturbed_bytes'] / 2**20:.1f} MB")
        for location, number in sorted((item.get("locations") or {}).items(),
                                       key=lambda pair: str(pair[0])):
            lines.append(f"      外部数据引用 {str(location):16s} x{number}")
    if info["amplitude"]:
        lines.append("权重级相对扰动 rms(W~-W)/rms(W)（抽样，应当≈该组配平后的 σ）：")
        lines.append(f"  {'group':26s} {'measured':>10s} {'sigma':>10s} {'ratio':>7s}")
        for group, item in info["amplitude"].items():
            lines.append(f"  {group:26s} {item['measured_median']:10.5f} "
                         f"{item['expected_sigma']:10.5f} {item['ratio']:7.3f}")
    if info["problems"]:
        lines.append(f"问题 {len(info['problems'])} 条：")
        for problem in info["problems"][:20]:
            lines.append(f"  - {problem}")
    else:
        lines.append("引用、形状、逐成员差异：全部通过")
    return "\n".join(lines)


__all__ = [
    "BASE_DATA",
    "MANIFEST_NAME",
    "PERTURBED_DATA_FMT",
    "build_member_set",
    "format_member_report",
    "member_base_path",
    "member_dir_for",
    "member_exists",
    "member_model_path",
    "member_perturbed_path",
    "member_tag",
    "resolve_member_path",
    "set_dir_for",
    "verify_member_set",
]
