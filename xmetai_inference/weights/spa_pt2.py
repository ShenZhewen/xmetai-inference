# -*- coding: utf-8 -*-
"""潜空间 AR(1) 强迫 —— pt2（torch.export）后端的注入实现。

与 ONNX 版语义同构，差别在接线：改写发生在加载后的内存图上（不落盘），注入目标从
mlp_w2 权重占位符回溯到层末残差流。ξ 走 GraphModule 属性 ``spw_xi``（图签名固定，
塞不进新输入）；属性名必须用下划线，否则 fx codegen 会写出非法 Python。
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
# torch.export 把参数名变成 p_ 前缀时保留了原模块路径，是扁平 aten 图里唯一带语义的名字族。
LAYER_ANCHOR = re.compile(r"^p_model_decoder_layers_(\d+)_mlp_w2_weight$")

# 从 MLP 输出投影走到残差 add 之间允许穿过的算子：只包一层或乘广播因子，
# 不改变「第 k 层末残差流」的语义。真图链路是 linear -> clone -> mul -> add。
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

    必须沿单链走且每步都是已知透明算子；宁可报错也不猜节点——注错层时形状、节点数
    全对，只有数值对照能发现。
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

    支持 "layer_stream"（全部层）、"layer_stream:0,10,20"（指定层）与显式节点名/后缀。
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
                # 裸数字多半是「layer_stream:0,10,20 被逗号拆开」，做后缀匹配会静默
                # 命中无关节点，所以直接报错。
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
        # ⚠ fx 的 inserting_after 是「一个插入点」而非「往后追加」：同一 with 块里连续
        # create_node 会逆序（c, b, a）。逆序图照样 recompile 成功，但代码生成先用后定义，
        # ep.module() 的 unlift 会 KeyError。所以每建一个节点就把锚点挪到它身上。
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

        # 常量锚点每目标各建一份：目标未必按图序排列（如 "layer_stream:5,3"），
        # 跨目标复用会让链引用到更靠后的节点。
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
        # 单独检查拓扑序：逆序图能 recompile、节点数也对，但 module() 会 KeyError。
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
    """pt2 后端总入口：在已加载的 ExportedProgram 上就地注入（改图不落盘）。

    必须在 ep.module() 之前调用。n_targets 是驱动侧要喂的目标数（sigma=0 时为 0）。
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
    # 基的隐含 sigma（与 ONNX 版同一口径）。先转 float32 再求范数：
    # fp16 的平方和在范数里会溢出或下溢。
    v_np32 = gm.spw_V.detach().float().cpu().numpy()
    u_np32 = gm.spw_U.detach().float().cpu().numpy()
    # 验收用：记下 layer_stream 别名实际点到的回溯链路，便于人眼确认注到了哪一层。
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
