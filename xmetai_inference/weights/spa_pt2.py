# -*- coding: utf-8 -*-
"""潜空间 AR(1) 强迫 —— pt2（torch.export）后端的注入实现。

与 weights/spa.py（ONNX 图改写）**语义同构**：同一份 LatentForcingSpec / build_basis，
以及 backends/spa.py 的 Ar1Driver / LatentStepStream；公式同一条
`z' = z ⊙ (1 + σ·U diag(ξ) Vᵀ)`，σ 折进
通道基、τ 留在驱动侧（换 τ 不改图，两个臂共图）。差别只在「把强迫接进图」的手段：

1. **不落盘**：改写发生在加载后的内存图上（gm.graph 插节点 + gm.recompile()）。
   没有缓存文件、没有外部数据落位问题；spec 的指纹只进日志，用于口径对账。
2. **目标选择**：fx 图里计算节点名是泛型 aten 名（add_2 / mul_7 …），FGVP 那套
   「节点名后缀」不可用。改用**从权重占位符回溯**：`p_model_decoder_layers_{k}_mlp_w2_weight`
   → 它喂的 linear → 吸收其输出的残差 add → 该 add 的输出 = 第 k 层末残差流。
   （权重占位符名保留原模块路径，是 torch.export 图里唯一带语义的名字。）
3. **xi 走模块属性而不是图输入**：torch.export 的图签名固定（args_0..3），塞不进
   新输入；属性 `spw_xi` 由驱动侧每步覆写，图用 get_attr 读取。属性名必须用
   **下划线**——fx codegen 会把 get_attr 的名字直接写进生成代码，`/` 会生成非法
   Python（ONNX 版的 spw/xi 命名在这里不能照抄）。

FuXi-2.1 实测布局（2026-09-30 探针，fuxi-2.1.pt2 / torch 2.11）：
tokens=28800（120×240）、channels=1536、30 个 decoder 层、全程 fp16、
潜状态 [1, 28800, 1536]（3-D，与 FGVP 同构，build_basis 直接可用）。

每目标插入 5 个节点：

    xi_t = select(spw_xi [T,R], 0, t)      # 本目标的 ξ 列
    n    = mm(spw_V [N,R] * xi_t, spw_U [R,C])   # n[tok,c] = Σ_r V[tok,r]·ξ[r]·U[r,c]
    z'   = z * (n + 1)                      # 广播 [N,C] ⊙ [1,N,C]

（ONNX 版是 6 节点：多一个 Reshape/Transpose；这边 aten.mm 直接吃 [N,R]@[R,C]。）
"""
from __future__ import annotations

import logging
import re

import numpy as np

from .spa import LatentForcingSpec, build_basis

log = logging.getLogger(__name__)

# 潜空间强迫在 GraphModule 上的属性名（下划线，见模块头第 3 条）。
ATTR_V = "spw_V"      # 空间基 [N, R]
ATTR_U = "spw_U"      # 通道基 [R, C]（σ 与归一化已折叠，同 ONNX 版）
ATTR_XI = "spw_xi"    # 驱动侧每步覆写：[T, R]（T = 目标数）

# 「第 k 层末残差流」的回溯锚点：FuXi-2.1 导出图里 MLP 输出投影的权重占位符。
# 名字来自原模块路径 model/decoder/layers.{k}/mlp/w2/weight —— torch.export 把参数
# 名变成 p_ 前缀占位符时保留了这个路径，是扁平 aten 图里唯一带语义的名字族。
LAYER_ANCHOR = re.compile(r"^p_model_decoder_layers_(\d+)_mlp_w2_weight$")

# 从 MLP 输出投影走到残差 add 之间**允许穿过**的算子：它们只包一层（clone / dtype 拷贝）
# 或乘一个广播因子（缩放），不改变「这是第 k 层末残差流」这个语义位置。
# 2026-09-30 真图实测（fuxi-2.1.pt2）：链路是 linear → clone → mul(×[1,1,1536] 调制向量)
# → add，**不是**最初假设的 linear → add。只看一步会在真图上直接报「0 个 add 消费者」。
_TRANSPARENT_OPS = ("aten.clone", "aten._to_copy", "aten.view", "aten.reshape",
                    "aten.mul", "aten.div")
#: 单链最多走多少步（防御性上限：真图是 3 步，走太远说明结构不是我们想的那样）。
_MAX_CHAIN = 8


def _shape_of(node):
    """节点输出形状（torch.export 的节点 meta['val'] 带 FakeTensor；没有则 None）。"""
    val = node.meta.get("val") if hasattr(node, "meta") else None
    return tuple(val.shape) if hasattr(val, "shape") else None


def _dtype_of(node):
    val = node.meta.get("val") if hasattr(node, "meta") else None
    return getattr(val, "dtype", None)


def _is_aten(node, prefix: str) -> bool:
    return node.op == "call_function" and str(node.target).startswith(prefix)


def _layer_indices(gm) -> list[int]:
    """图里可作锚点的 decoder 层号（有 mlp_w2 权重占位符的层）。"""
    found = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            match = LAYER_ANCHOR.match(node.name)
            if match:
                found.add(int(match.group(1)))
    return sorted(found)


def _walk_layer(gm, layer: int):
    """回溯第 layer 层的残差流：返回链路 [linear, …, 残差 add]（节点列表）。

    必须沿**单链**走，且每一步都得是已知的透明算子（见 `_TRANSPARENT_OPS`）—— 宁可在这里
    报错，也不能猜一个节点去注：注错层的所有结构检查都会是全绿的（形状一样、节点数一样），
    只有数值参考能发现，而那时候已经在跑推理了。
    """
    anchor = "p_model_decoder_layers_%d_mlp_w2_weight" % layer
    weight = next(
        (n for n in gm.graph.nodes if n.op == "placeholder" and n.name == anchor), None)
    if weight is None:
        raise ValueError(
            f"pt2 图里没有第 {layer} 层的锚点占位符（{anchor}）。"
            f"可用层号见 p_model_decoder_layers_{{k}}_mlp_w2_weight 系占位符。")
    linears = [n for n in weight.users if _is_aten(n, "aten.linear")]
    if len(linears) != 1:
        raise ValueError(
            f"层 {layer} 的 mlp_w2 权重有 {len(linears)} 个 linear 消费者"
            f"（应为 1）——图结构与此回溯规则不匹配，别硬注。")
    nodes = [linears[0]]
    want_shape = _shape_of(nodes[0])
    names = [nodes[0].name]
    while len(nodes) <= _MAX_CHAIN:
        users = list(nodes[-1].users)
        if len(users) != 1:
            raise ValueError(
                f"层 {layer} 的链路在 {nodes[-1].name}（{nodes[-1].target}）分叉："
                f"{len(users)} 个消费者（链路 {' → '.join(names)}）。"
                "回溯要求残差流是一条单链，别硬注。")
        node = users[0]
        nodes.append(node)
        names.append(node.name)
        if _is_aten(node, "aten.add"):
            if _shape_of(node) != want_shape:
                raise ValueError(
                    f"层 {layer} 找到的 add {node.name} 形状是 {_shape_of(node)}，与 MLP 输出 "
                    f"{want_shape} 不符（链路 {' → '.join(names)}）——这不像残差 add。")
            return nodes
        if not any(str(node.target).startswith(prefix) for prefix in _TRANSPARENT_OPS):
            raise ValueError(
                f"层 {layer} 的链路在 {node.name}（{node.target}）断了：它不在已知的透明算子里"
                f"（链路 {' → '.join(names)}），无法确认后面那个 add 是不是残差位置。"
                "请核对这层的实际结构后更新 _TRANSPARENT_OPS，或改用显式节点名。")
    raise ValueError(
        f"层 {layer} 从 mlp_w2 往下走了 {_MAX_CHAIN} 步还没遇到残差 add"
        f"（链路 {' → '.join(names)}）——结构与此回溯规则不匹配。")


def _layer_out_node(gm, layer: int):
    """第 layer 层末的残差流节点（= `_walk_layer` 链尾）。"""
    return _walk_layer(gm, layer)[-1]


def layer_chain(gm, layer: int) -> list:
    """该层的回溯链路（节点名），给日志/验收看「到底注到了哪个节点」。"""
    return [node.name for node in _walk_layer(gm, layer)]


def _parse_layer_alias(token: str):
    """'layer_stream' / 'layer_stream:0,10,20' -> (True, [层号] | None)；否则 (False, None)。"""
    text = str(token).strip()
    if text == "layer_stream":
        return True, None
    if text.startswith("layer_stream:"):
        layers = [int(part) for part in text.split(":", 1)[1].split(",") if part.strip()]
        if not layers:
            raise ValueError(f"layer_stream 冒号后面没有层号：{token!r}")
        return True, layers
    return False, None


def resolve_fx_targets(gm, spec: LatentForcingSpec) -> list:
    """把 spec.targets 解析成 fx 节点列表（保持顺序、去重）。

    支持两种写法（可混用）：
      * 别名 "layer_stream"（全部层）或 "layer_stream:0,10,20"（指定层）；
      * 显式节点名（先精确、再后缀匹配）——回溯规则覆盖不了的结构用它兜底。
    """
    nodes = {n.name: n for n in gm.graph.nodes}
    targets = []
    for token in spec.targets:
        is_alias, layers = _parse_layer_alias(token)
        if is_alias:
            available = _layer_indices(gm)
            if not available:
                raise ValueError(
                    "pt2 图里没有任何 layer_stream 锚点"
                    f"（{LAYER_ANCHOR.pattern}）——这个图不是 FuXi 式 decoder 结构，"
                    "请用显式节点名指定注入点。")
            want = layers if layers is not None else available
            for layer in want:
                if layer not in available:
                    raise ValueError(
                        f"layer_stream 层号 {layer} 不在图里（可用 {available[0]}..{available[-1]}，"
                        f"共 {len(available)} 层）。")
                targets.append(_layer_out_node(gm, layer))
        else:
            if token.isdigit():
                # 裸数字几乎只可能来自「layer_stream:0,10,20 被逗号拆开」——它作为节点名后缀
                # 极可能**碰巧**匹配到某个无关节点（实测踩过：静默注到了 channels 不同的节点上），
                # 所以宁可直接报错，也不做这种后缀匹配。CLI 侧另有一层合并，这里是兜底。
                raise ValueError(
                    f"注入点 {token!r} 是裸数字，疑似 layer_stream 的层号被逗号拆开了"
                    "（配置里请写在一个 token 里：('layer_stream:0,10,20',)）。"
                    "裸数字做后缀匹配会静默命中无关节点，这里拒绝。")
            node = nodes.get(token)
            if node is None:
                node = next(
                    (n for n in gm.graph.nodes if n.name.endswith(token)), None)
            if node is None:
                raise ValueError(
                    f"latent_forcing（pt2）：注入点 {token!r} 在图里找不到"
                    f"（既不是 layer_stream 别名，也不是节点名/后缀）。")
            targets.append(node)
    unique = []
    for node in targets:
        if node not in unique:
            unique.append(node)
    if not unique:
        raise ValueError("latent_forcing（pt2）：一个注入点都没解析到。")
    return unique


def detect_fx_layout(gm, target_nodes, spec: LatentForcingSpec) -> dict:
    """从目标节点的 meta 推 (tokens, channels)，grid 必须显式给。"""
    observed = set()
    for node in target_nodes:
        shape = _shape_of(node)
        if shape is None or len(shape) != 3:
            raise ValueError(
                f"pt2 注入目标 {node.name} 的输出不是 3-D（{shape}）。"
                "layer_stream 只适用于 [batch, tokens, channels] 的潜状态；"
                "4-D（B,C,H,W）的潜张量是另一套基构造，本实现明确不支持。")
        observed.add(tuple(shape[1:]))
    if len(observed) != 1:
        raise ValueError(
            f"各注入目标的 (tokens, channels) 不一致：{sorted(observed)}。"
            "共享同一组基要求所有目标同形。")
    tokens, channels = observed.pop()
    tokens, channels = int(tokens), int(channels)
    for name, want, got in (
            ("tokens", spec.tokens, tokens), ("channels", spec.channels, channels)):
        if want is not None and int(want) != got:
            raise ValueError(
                f"latent_forcing.{name} 显式给的是 {want}，图里实测 {got} —— 配置与模型不符，"
                "以图为准请删掉该字段，改配置请三思。")
    if spec.grid is None:
        raise ValueError(
            "pt2 图推不出空间网格：latent_forcing.grid 必须显式给（FuXi-2.1 是 (120, 240)）。"
            "空间基的低通平滑必须落在真实网格上，网格猜错会把场静默搅乱"
            "（fx 图里没有可靠的 rows×cols 来源，不像 ONNX 能从 patch_embed 推）。")
    grid = tuple(int(v) for v in spec.grid)
    if grid[0] * grid[1] != tokens:
        raise ValueError(
            f"grid {grid} 的乘积 {grid[0] * grid[1]} != tokens {tokens}。")
    return {"tokens": tokens, "channels": channels, "grid": grid, "observed": True}


def inject_fx_forcing(gm, target_nodes, basis, spec: LatentForcingSpec, *,
                      dtype=None, device=None) -> int:
    """把 `z' = z ⊙ (1 + σ·U diag(ξ) Vᵀ)` 接到每个目标节点上，返回新增节点数。

    基挂在 GraphModule 属性上（spw_V / spw_U / spw_xi），图用 get_attr 读；
    spw_xi 由驱动侧每步覆写。改完 recompile()，**不落盘**。
    """
    import torch

    graph = gm.graph
    v_np = np.ascontiguousarray(basis["Vt"].T)   # [N, R]
    u_np = np.ascontiguousarray(basis["Ut"])      # [R, C]，σ 已折叠
    if dtype is None:
        dtype = _dtype_of(target_nodes[0])
    kwargs = {"dtype": dtype}
    if device is not None:
        kwargs["device"] = device
    if getattr(gm, ATTR_XI, None) is not None:
        raise RuntimeError(
            "这个 GraphModule 已经注入过潜空间强迫（spw_* 属性已存在）。"
            "重复注入会双重扰动，这里直接拒绝。")
    setattr(gm, ATTR_V, torch.from_numpy(v_np).to(**kwargs))
    setattr(gm, ATTR_U, torch.from_numpy(u_np).to(**kwargs))
    setattr(gm, ATTR_XI, torch.zeros((len(target_nodes), int(spec.rank)), **kwargs))

    select = torch.ops.aten.select.int
    mul = torch.ops.aten.mul.Tensor
    mm = torch.ops.aten.mm
    add_scalar = torch.ops.aten.add.Scalar
    added = 0
    for index, node in enumerate(target_nodes):
        # ⚠ fx 的 inserting_after 是**一个插入点**，不是「从这里往后追加」：同一个 with 块里
        # 连续 create_node 全都插在同一个位置 ⇒ 后建的反而排在前面。实测（torch 2.11）
        #
        #     with graph.inserting_after(anchor):
        #         a = call(...); b = call(...); c = call(...)
        #     # 图里顺序是 c, b, a —— 逆序
        #
        # 逆序的图**照样 recompile 成功**，但代码生成会「先用后定义」，而且 ep.module() 的
        # unlift（deepcopy 整张图）会直接 KeyError: <先出现的那个节点名>。所以每建一个节点
        # 就把锚点挪到它身上，接力着往下插 —— 这条是 2026-09-30 写自测时才踩出来的。
        anchor = node

        def _emit(target, args):
            nonlocal anchor
            with graph.inserting_after(anchor):
                created = graph.call_function(target, args)
            anchor = created
            return created

        def _attr(name):
            """get_attr 也受插入点约束（它同样是 create_node），必须显式落位。"""
            nonlocal anchor
            with graph.inserting_after(anchor):
                created = graph.get_attr(name)
            anchor = created
            return created

        # 常量锚点**每目标各建一份**：跨目标复用会把它们集中在第一个目标旁边，而目标之间
        # 未必按图层先后排（"layer_stream:5,3" 就是反的），那样后面那条链会引用到更靠后的
        # 节点。3 个 get_attr 的代价可以忽略。
        xi_attr, v_attr, u_attr = _attr(ATTR_XI), _attr(ATTR_V), _attr(ATTR_U)
        xi_t = _emit(select, (xi_attr, 0, index))
        scaled = _emit(mul, (v_attr, xi_t))
        noise = _emit(mm, (scaled, u_attr))
        gain = _emit(add_scalar, (noise, 1.0))
        out = _emit(mul, (node, gain))
        # 只换除新节点外的消费者（含新节点会把 out 的输入改指向自己，成环）。
        for user in [u for u in node.users if u is not out]:
            user.replace_input_with(node, out)
        added += 5
    gm.recompile()
    return added


def audit_fx_forcing(gm) -> dict:
    """注入后的结构自检：属性在、每目标一条 select(spw_xi) 链、属性形状自洽。"""
    report = {"attrs": all(hasattr(gm, name) for name in (ATTR_V, ATTR_U, ATTR_XI))}
    chains = 0
    if report["attrs"]:
        v_t = getattr(gm, ATTR_V)
        u_t = getattr(gm, ATTR_U)
        x_t = getattr(gm, ATTR_XI)
        report["V_shape"] = tuple(v_t.shape)
        report["U_shape"] = tuple(u_t.shape)
        report["xi_shape"] = tuple(x_t.shape)
        n_tokens, rank = int(v_t.shape[0]), int(v_t.shape[1])
        report["layout"] = {"tokens": n_tokens,
                             "channels": int(u_t.shape[1]),
                             "rank": rank}
        report["consistent"] = (
            int(u_t.shape[0]) == rank and int(x_t.shape[1]) == rank)
        for node in gm.graph.nodes:
            if node.op != "call_function" or str(node.target) != "aten.select.int":
                continue
            first = node.args[0] if node.args else None
            if first is not None and getattr(first, "op", None) == "get_attr" \
                    and first.target == ATTR_XI:
                chains += 1
        report["targets"] = chains
        # **拓扑序**单独查一遍（fx 的 lint）：逆序的图照样能 recompile 成功、节点数也对，
        # 但代码生成是「先用后定义」，而且 ep.module() 的 unlift 会直接 KeyError —— 2026-09-30
        # 写自测时踩的正是这个：所有结构性的检查全绿，一跑 module() 就炸。所以把它变成断言。
        try:
            gm.graph.lint()
            report["topological"] = True
        except Exception as error:  # noqa: BLE001 —— lint 的异常类型随版本变，只看能不能过
            report["topological"] = False
            report["lint_error"] = str(error).strip().splitlines()[-1][:200]
        report["ok"] = bool(report["consistent"] and chains == int(x_t.shape[0])
                            and report.get("topological"))
    else:
        report["ok"] = False
    return report


def attach_forcing(ep, spec: LatentForcingSpec, *, device=None,
                   src_path=None, verbose: bool = True) -> dict:
    """pt2 后端的总入口：在**已加载**的 ExportedProgram 上就地注入（改图不落盘）。

    必须在 ep.module() 之前调用（module() 包装的是注入后的图）。返回报告 dict，
    其中 n_targets 是驱动侧要喂的目标数（sigma=0 空转对照时为 0 —— 一个节点都不插）。
    src_path 给了就按「规格 + 源文件身份」算指纹（与 ONNX 版同一套口径，日志对账用）；
    没给就只对规格本身算（s- 前缀区分）。
    """
    if spec.sigma <= 0:
        report = {"n_targets": 0, "sigma": float(spec.sigma), "skipped": True}
        log.info(
            "潜空间 AR(1)（pt2）：sigma=0，空转对照 —— 不改图，成员间差异只来自"
            "（GPU 非确定性等）注入之外的来源。%s", spec.describe())
        return report
    gm = ep.graph_module
    targets = resolve_fx_targets(gm, spec)
    layout = detect_fx_layout(gm, targets, spec)
    basis = build_basis(layout["tokens"], layout["channels"], int(spec.rank),
                        float(spec.length_scale), layout["grid"], spec.seed,
                        float(spec.sigma))
    dtype = _dtype_of(targets[0])
    if device is None:
        try:
            device = next(iter(ep.state_dict.values())).device
        except Exception:  # noqa: BLE001 —— state_dict 拿不到就交给默认设备
            device = None
    added = inject_fx_forcing(gm, targets, basis, spec, dtype=dtype, device=device)
    if src_path:
        import hashlib
        import json

        fingerprint = spec.fingerprint(src_path)
    else:
        import hashlib
        import json

        blob = json.dumps(spec.budgets(), sort_keys=True, ensure_ascii=False)
        fingerprint = "s-" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]
    report = audit_fx_forcing(gm)
    # ‖Vt‖ 与基的隐含 sigma：与 ONNX 版（build_forcing_model 的打印）同一套口径，用于对账
    # 「σ 真的折进基里了」。**先转 float32 再求范数** —— FuXi-2.1 全程 fp16，V 有 28800x16、
    # U 有 16x1536 个元素，fp16 的平方和在范数里会直接溢出（Inf）或下溢成 0。
    v_np32 = gm.spw_V.detach().float().cpu().numpy()
    u_np32 = gm.spw_U.detach().float().cpu().numpy()
    # 验收用：把 layer_stream 别名点到的那几层的**回溯链路**记进报告。「注到了哪个节点」
    # 是这个后端最需要人眼确认的一件事 —— 真图上它既不是 linear 直连 add，节点名（add_12）
    # 也看不出是哪一层。
    chains = {}
    for token in spec.targets:
        is_alias, layers = _parse_layer_alias(token)
        if not is_alias:
            continue
        for layer in (layers if layers is not None else _layer_indices(gm)):
            chains[f"layer_{layer}"] = layer_chain(gm, layer)
    report.update({
        "n_targets": len(targets), "nodes": added, "layout": layout,
        "basis": basis["report"], "fingerprint": fingerprint,
        "chains": chains, "dtype": str(dtype), "skipped": False,
        "norm_v": float(np.linalg.norm(v_np32)),
        "sigma_implied": float(np.linalg.norm(u_np32))
        / float(np.sqrt(layout["channels"] * layout["tokens"])),
        "targets": [node.name for node in targets[:6]],
    })
    if verbose:
        log.info(
            "潜空间 AR(1)（pt2）已注入：%s；目标 %d 个（%s%s），新增节点 %d，"
            "布局 tokens=%d channels=%d grid=%s，dtype=%s，指纹 %s",
            spec.describe(), len(targets), " ".join(report["targets"]),
            "…" if len(targets) > 6 else "", added, layout["tokens"],
            layout["channels"], layout["grid"], report["dtype"],
            report["fingerprint"])
    if not report["ok"]:
        if getattr(spec, "verify_on_load", True):
            raise RuntimeError(
                f"潜空间 AR(1)（pt2）注入后审计不过：{report}。图未被使用前请先修好。")
        log.warning("潜空间 AR(1)（pt2）审计不过但 verify_on_load=False，继续：%s", report)
    if report["basis"].get("fallbacks"):
        log.warning("空间基有 %d 个方向退回白噪声（见 build_basis 的警告）。",
                    report["basis"]["fallbacks"])
    return report


__all__ = [
    "ATTR_U",
    "ATTR_V",
    "ATTR_XI",
    "LAYER_ANCHOR",
    "attach_forcing",
    "audit_fx_forcing",
    "detect_fx_layout",
    "inject_fx_forcing",
    "layer_chain",
    "resolve_fx_targets",
]
