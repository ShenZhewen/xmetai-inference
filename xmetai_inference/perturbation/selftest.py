# -*- coding: utf-8 -*-
"""SPW 自检：在一张合成的小 ONNX 上，把「图改写是否正确」跑成可执行的断言。

为什么需要它：本方案的实现方式是**改图 + 逐成员落盘**，而出错几乎全是静默的——权重
没被真正扰动（51 个成员一模一样）、扰动挂到了别的张量上、外部数据被复制或丢了引用
（服务器上 2.8 GB 变 51 份 / 改完找不到 .data）。正式模型（FGVP）依赖 Linux 侧的自
定义算子库 xmetai_onnx_plugins.so，本地和 CI 都建不了 session，所以这里用一张结构同
构的小图把这几件事逐条钉死：

  1. 无损改写：sigma 取到 1e-30 时（float32 下 1+sigma*xi 恒等于 1.0），成员里的权重
     与源权重**逐位相同**；任何位置改错都会在这里露出来；
  2. 扰动冻结：同一成员两次前向逐位相同，而跨成员的注入方差满足
     Var(W'-W) = sigma^2*||W||^2（这是 sigma 无量纲、组间按能量配平的前提）；
  3. 选择集：mean / const / 掩码 / 非浮点 / 无消费者 / 过小的张量确实没被扰动，
     哪一条规则把它挡下来的都记在 skipped 里，一并断言；
  4. 噪声结构：iid / 逐通道 / 标量三种结构的抽样形状、行内常数性、标量只抽一个 xi，
     以及三种结构下的方差定律都成立；
  5. 流独立性：同一张图里不同张量的噪声流不相关（共用一条流会把整个模型退化成
     rank-1 方向扰动）；
  6. M1a：逐成员冻结权重落盘——成员目录自包含、成员之间不同、**成员内两次前向逐位
     相同**（这是与「图内随机算子」那版的语义分水岭：那版每次前向重抽，这版整条轨迹
     冻结）、按索引分片生成时清单会合并、后端加载时的成员号自检能拦住「51 个成员加载
     同一个文件」；以及**被外置的 Constant 节点张量也必须被改名**
     （2026-09-23 服务器事故的回归：只扫 graph.initializer 会漏掉它们，
     成员图会继续指向源 .data）；
  7. 分段编排与判据（segments.py / verdict.py）；
  8. 潜空间 AR(1)（latent.py）：sigma=0 时与源图逐位相同、非零时与 numpy 参考一致、
     同 xi 两次逐位相同而换 xi 才变（图自身无随机性）、目标的消费者全部改接且只改
     一次、基正交且无 DC、rho/平稳方差/成员独立性、尺度旋钮真的接上了。

用法：python -m xmetai_inference.perturbation self-test [--keep]
"""
from __future__ import annotations

import math
import os
import os.path as osp
import shutil
import tempfile

import numpy as np

from .graph import (
    _onnx,
    _verify_external_refs,
    external_tensors,
    iter_graphs,
    group_alternatives,
    resolve_sigmas,
    load_model,
    resolve_effective_path,
    select_initializers,
    tensor_array,
)
from .latent import (Ar1Driver, LatentForcingSpec, LatentStepStream,
                     _grid_from_patch_embed, build_basis, build_forcing_model,
                     detect_layout, lag1_autocorr, resolve_targets)
from .spec import NOISE_STRUCTS, WeightPerturbSpec
from .weights import (
    build_member_set,
    member_base_path,
    member_model_path,
    member_perturbed_path,
    resolve_member_path,
    set_dir_for,
    verify_member_set,
)

# 合成图里**应该**被扰动的张量（默认 spec 下），顺序即排序后的顺序。
EXPECTED_INJECTED = (
    "backbone.conv.bias",
    "backbone.conv.weight",
    "latent_mixer.head.dt_bias",
    "latent_mixer.head.weight",
)

# 合成图里**不该**被扰动的张量 -> 挡下它的规则（子串匹配 skipped 里的原因）。
EXPECTED_SKIPPED = (
    ("vae_encoder_group.norm.mean", "exclude"),
    ("vae_encoder_group.norm.const", "exclude"),
    ("attn_mask", "exclude"),
    ("tiny.weight", "min_elements"),
    ("dead.weight", "无消费者"),
    ("latent_mixer.index", "非浮点"),
)

_INPUT_SHAPE = (1, 8, 16, 16)
_CONV_OUT = 64
_HEAD_OUT = 128

# 合成的「被外置的 Constant 节点」张量名：复刻 FGVP 的 /<group>/attn/Constant_NN。
_CONST_OUT = "latent_mixer.attn.const_kv"


class _Checker:
    """收集断言结果。不抛异常，跑完所有条目再给一个总结，方便一次看到全部问题。"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.passed = 0
        self.failures = []

    def check(self, ok, label, detail=""):
        if ok:
            self.passed += 1
            if self.verbose:
                print(f"    [通过] {label}")
        else:
            self.failures.append(f"{label}：{detail}" if detail else label)
            if self.verbose:
                print(f"    **[不通过]** {label}" + (f" —— {detail}" if detail else ""))
        return bool(ok)

    def __call__(self, ok, label, detail=""):
        """让用例里可以直接 `check(...)`，不必写 `check.check(...)`。"""
        return self.check(ok, label, detail)

    def section(self, index, total, title):
        if self.verbose:
            print(f"\n[{index}/{total}] {title}")


def _write_synthetic(path, *, external=False, constant=False):
    """写一张结构同构于 FGVP 的小图：Conv 权重槽、Add 消费的 bias、Gemm 权重，
    外加 mean/const/掩码/非浮点/无消费者/过小这几类「必须被挡住」的张量。

    constant=True 再挂一个**大张量的 Constant 节点**：导出器开 convert_attribute=True
    时它会被外置，和 initializer 一样按「相对图文件的 location」解析（FGVP 就有 4 个
    这样的 attention 常量）。少了这一味，夹具测不出「只扫 initializer」的漏改。"""
    onnx = _onnx()
    helper, TensorProto = onnx.helper, onnx.TensorProto
    rng = np.random.default_rng(0)

    def init(name, array):
        return onnx.numpy_helper.from_array(np.asarray(array), name)

    def rand(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.1)

    initializers = [
        # --- 应被扰动的四类：Conv weight 槽 / Conv bias / Gemm 权重 / Add 消费的 *_bias
        init("backbone.conv.weight", rand(_CONV_OUT, 8, 3, 3)),
        init("backbone.conv.bias", rand(_CONV_OUT)),
        init("latent_mixer.head.weight", rand(_CONV_OUT, _HEAD_OUT)),
        init("latent_mixer.head.dt_bias", rand(_HEAD_OUT)),
        # --- 不该被扰动的：归一化统计量与烘焙常量（量级故意做大，模拟真实模型）
        init("vae_encoder_group.norm.mean", np.full((1, _CONV_OUT, 1, 1), 1234.5, np.float32)),
        init("vae_encoder_group.norm.const", np.full((1, _CONV_OUT, 1, 1), 4321.0, np.float32)),
        init("attn_mask", np.ones((1, _CONV_OUT, 1, 1), np.float32)),
        # --- 不该被扰动的：过小 / 无消费者 / 非浮点
        init("tiny.weight", np.ones(8, np.float32)),
        init("dead.weight", np.ones(70, np.float32)),
        init("latent_mixer.index", np.arange(_HEAD_OUT, dtype=np.int64)),
        # --- Reshape 的形状（int64，非浮点）
        init("reshape.shape", np.asarray([1, _CONV_OUT], dtype=np.int64)),
    ]

    nodes = [
        helper.make_node("Conv", ["x", "backbone.conv.weight", "backbone.conv.bias"],
                         ["conv"], name="/backbone/conv/Conv"),
        helper.make_node("Relu", ["conv"], ["relu"], name="/backbone/conv/Relu"),
        helper.make_node("Sub", ["relu", "vae_encoder_group.norm.mean"], ["centred"],
                         name="/backbone/norm/Sub"),
        helper.make_node("Add", ["centred", "vae_encoder_group.norm.const"], ["shifted"],
                         name="/backbone/norm/Add"),
        helper.make_node("Mul", ["shifted", "attn_mask"], ["masked"],
                         name="/backbone/attn/Mul"),
        helper.make_node("GlobalAveragePool", ["masked"], ["pooled"],
                         name="/backbone/pool/GlobalAveragePool"),
        helper.make_node("Reshape", ["pooled", "reshape.shape"], ["flat"],
                         name="/head/Reshape"),
        helper.make_node("Gemm", ["flat", "latent_mixer.head.weight"], ["logits"],
                         name="/latent_mixer/head/Gemm"),
    ]
    if constant:
        # 名字取成 FGVP 那种形式，且**不是 initializer**：只扫 graph.initializer 的
        # 实现完全看不见它。
        nodes += [
            helper.make_node(
                "Constant", [], [_CONST_OUT], name="/latent_mixer/attn/Constant_15",
                # 必须用 numpy_helper.from_array（落进 raw_data）：onnx 只外置带 raw_data
                # 的张量，helper.make_tensor(float_data) 造出来的会被静默跳过，夹具就假绿了。
                value=onnx.numpy_helper.from_array(
                    (rng.standard_normal((_HEAD_OUT, _HEAD_OUT)).astype(np.float32) * 0.1),
                    _CONST_OUT + ".value")),
            helper.make_node("MatMul", ["logits", _CONST_OUT], ["mixed"],
                             name="/latent_mixer/attn/MatMul"),
        ]
        head_source = "mixed"
    else:
        head_source = "logits"
    nodes.append(
        helper.make_node("Add", [head_source, "latent_mixer.head.dt_bias"], ["y"],
                         name="/latent_mixer/head/Add"))
    graph = helper.make_graph(
        nodes, "spw_selftest",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(_INPUT_SHAPE))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, _HEAD_OUT])],
        initializers,
        value_info=[
            helper.make_tensor_value_info("conv", TensorProto.FLOAT,
                                          [1, _CONV_OUT, 14, 14]),
            helper.make_tensor_value_info("masked", TensorProto.FLOAT,
                                          [1, _CONV_OUT, 14, 14]),
            helper.make_tensor_value_info("pooled", TensorProto.FLOAT,
                                          [1, _CONV_OUT, 1, 1]),
            helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, _CONV_OUT]),
        ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    if external:
        # size_threshold=1024：只有大权重外置，形状/掩码这类**小张量留在图内**。
        # 这不是细节——onnxruntime 在会话初始化时要读那些小张量做形状推导，把它们也
        # 外置会让模型直接加载失败（实测：Reshape 的 shape 走外部数据 -> NO_SUCHFILE /
        # Cannot parse data from external tensors），真实导出的模型也都只外置大权重。
        onnx.save(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=osp.splitext(osp.basename(path))[0] + ".data",
                  size_threshold=1024, convert_attribute=constant)
    else:
        onnx.save(model, path)
    return path


def _write_probe(path, *, count=1, size=256, seed=1):
    """N 个互不相关的权重 -> Identity，输出就是它们各自被扰动后的值。

    用来量「同一张图里不同张量的随机流是否独立」——这件事在图结构上完全看不出来，
    只能靠数值。
    """
    onnx = _onnx()
    helper, TensorProto = onnx.helper, onnx.TensorProto
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal(size).astype(np.float32) * 0.1 for _ in range(count)]
    inits = [onnx.numpy_helper.from_array(weights[i], f"probe{i}.weight")
             for i in range(count)]
    nodes = [helper.make_node("Identity", [f"probe{i}.weight"], [f"Y{i}"],
                              name=f"/probe{i}/Identity") for i in range(count)]
    graph = helper.make_graph(
        nodes, "spw_probe", [],
        [helper.make_tensor_value_info(f"Y{i}", TensorProto.FLOAT, [size])
         for i in range(count)], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path, weights


def _session(path):
    """建一个可反复 run 的 session（模拟「一个 rank = 一个进程 = 一个 session」）。"""
    from .probe import make_session

    return make_session(path)


def _write_weight_passthrough(path, *, size=256, seed=1):
    """一张「输出 = 被扰动的权重本身」的最小图（W -> Mul(W, 1+sigma*xi) -> Y）。

    为什么要有它：合成大图里有 Conv/Gemm，onnxruntime 会按「权重是不是普通
    initializer」选不同的 kernel 和融合策略，累加顺序随之变化——sigma->0 时两张图
    只差几 ulp，逐位相同这个判据在真实图上是达不到的（这也是实测发现的）。这张
    小图没有下游算子，输出就是扰动后的权重，于是可以拿**逐位相同**和**注入方差
    定律**这两个最锋利的判据去卡注入公式本身。
    """
    onnx = _onnx()
    helper, TensorProto = onnx.helper, onnx.TensorProto
    weights = np.random.default_rng(seed).standard_normal(size).astype(np.float32) * 0.1
    tensor = onnx.numpy_helper.from_array(weights, "probe.weight")
    # 权重必须**有消费者**才会被 select_initializers 选中（注入要给消费者换输入），
    # 所以挂一个 Identity：它不改数值，输出仍然等于被扰动后的权重本身。
    node = helper.make_node("Identity", ["probe.weight"], ["Y"], name="/probe/Identity")
    graph = helper.make_graph(
        [node], "spw_probe", [],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [size])], [tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path, weights


class _FakeMetaSession:
    """只实现 get_modelmeta() 的最小替身。

    m1a 的后端自检（OnnxInferModel._verify_member_identity）只读图元数据、不跑前向，
    所以不必为此建真实 session（也就能在没有 CUDA / 自定义算子的机器上断言它）。
    """

    def __init__(self, path):
        class _Meta:
            pass

        meta = _Meta()
        meta.custom_metadata_map = {
            entry.key: entry.value for entry in load_model(path).metadata_props}
        self._meta = meta

    def get_modelmeta(self):
        return self._meta


def _feeds(seed=0):
    return {"x": np.random.default_rng(seed).standard_normal(_INPUT_SHAPE).astype(np.float32)}


def _run(path, feeds):
    from .probe import make_session

    # 合成图故意留了「无消费者」的 initializer 来测跳过规则，ORT 会逐条告警；
    # 自检的输出只需要我们自己的断言结果，make_session 已把日志压到只报 error。
    return make_session(path).run(None, feeds)


# ---------------------------------------------------------------------------
# 各检查项
# ---------------------------------------------------------------------------
def _member_weights(src, spec, index):
    """读第 index 个成员里**被扰动后**的权重：{张量名: array}。

    成员图把这些张量外置到 perturbed.mNNN.data，所以引用要按**成员集目录**解析
    （而不是源图目录——那正是 2026-09-23 那次事故的盲区）。
    """
    directory = set_dir_for(src, spec)
    model = load_model(member_model_path(src, spec, index))
    return {tensor.name: np.asarray(tensor_array(tensor, directory))
            for tensor in model.graph.initializer}


def _case_equivalence(work, check):
    """sigma -> 0 时扰动后的权重必须等价于源权重。

    两个层次：
      * 纯权重透传图（无下游算子）：必须**逐位相同**。float32 下 sigma=1e-30 时
        1+sigma*xi 会舍入回 1.0，于是 W ⊙ (1+sigma*xi) 逐位等于 W——逐位相同才说明
        扰动确实是**乘性、且乘在权重上**（写成加性、或乘到别处，这里都会露出来）。
      * 完整合成图：只能是**浮点容差内**相同。权重一旦不再驻留图内，onnxruntime 会换
        kernel / 换融合（实测 Conv+Gemm 图上差几 ulp），逐位相同在真实图上是达不到的
        ——这条经验值直接决定了服务器上「等价性」该怎么判。
    """
    sigma = 1e-30
    probe_src, weights = _write_weight_passthrough(osp.join(work, "probe.onnx"))
    probe_spec = WeightPerturbSpec(groups=("all",), sigma=sigma,
                                   member_dir=osp.join(work, "probe_zero_set"))
    build_member_set(probe_src, probe_spec, 1, verbose=False)
    got = _member_weights(probe_src, probe_spec, 0)["probe.weight"]
    check(np.array_equal(weights, got),
          "sigma->0 时扰动后的权重与原始权重逐位相同",
          f"max|delta|={float(np.abs(got - weights).max()):.3g}")

    src = _write_synthetic(osp.join(work, "src.onnx"))
    feeds = _feeds()
    reference = _run(src, feeds)[0]
    spec = WeightPerturbSpec(groups=("all",), sigma=sigma,
                             member_dir=osp.join(work, "zero_set"))
    build_member_set(src, spec, 1, verbose=False)
    got = _run(member_model_path(src, spec, 0), feeds)[0]
    check(got.shape == reference.shape, "输出形状与源图一致")
    if got.shape != reference.shape:
        return
    rel = float(np.abs(got - reference).max() / max(np.abs(reference).max(), 1e-12))
    check(rel < 1e-5,
          "sigma->0 时完整图输出与源图的相对差异在浮点容差内", f"max rel={rel:.3g}")


def _case_randomness(work, check):
    """成员权重必须**冻结**，且跨成员的注入方差满足 Var(W'-W) = sigma^2 * ||W||^2。

    这是与「图内随机算子」的语义分水岭：那一版的噪声每次前向重抽，这一版的噪声在**生成
    阶段**就烘进了逐成员的权重里，成员内整条轨迹冻结。所以「多次抽样」在这里是**多个
    成员**，而不是同一张图的多次前向。
    """
    src = _write_synthetic(osp.join(work, "rand_src.onnx"))
    feeds = _feeds()
    spec = WeightPerturbSpec(groups=("all",), sigma_full=0.01,
                             member_dir=osp.join(work, "rand_set"))
    build_member_set(src, spec, 2, verbose=False)
    session = _session(member_model_path(src, spec, 0))
    first, second = session.run(None, feeds)[0], session.run(None, feeds)[0]
    check(np.array_equal(first, second),
          "同一成员两次前向逐位相同（权重冻结，图里没有随机算子）")

    # 注入方差定律：逐成员量 ||W'_m - W||^2 / ||W||^2，再对成员取平均。这是 SPW 全部分析
    # （sigma 无量纲、组间按能量配平）的基础，用透传图直接量一次比只看「输出变了」有信息量。
    probe_src, weights = _write_weight_passthrough(osp.join(work, "probe_var.onnx"))
    sigma, draws = 0.05, 8
    spec = WeightPerturbSpec(groups=("all",), sigma=sigma,
                             member_dir=osp.join(work, "probe_var_set"))
    build_member_set(probe_src, spec, draws, verbose=False)
    arrays = [_member_weights(probe_src, spec, index)["probe.weight"]
              for index in range(draws)]
    check(not np.array_equal(arrays[0], arrays[1]),
          "不同成员抽到不同的扰动（不是同一条随机流）")
    ratios = [float(np.sum((array - weights) ** 2)) / float(np.sum(weights ** 2))
              for array in arrays]
    measured = float(np.mean(ratios))
    # 带子取 [0.6, 1.6]：256 元素 x 8 个成员的等效自由度约 85x8，均值比的标准差约 5%，
    # 所以这条带在 ~7σ 上，既不会假红，又能抓住「方差定律差 1.6 倍」这种量级的真错误
    # （原来的 [0.3, 3.0] 要错到 3 倍才报，等于没测）。
    check(0.6 < measured / sigma ** 2 < 1.6,
          f"注入方差 = sigma^2 * ||W||^2（{draws} 个成员平均）",
          f"实测 {measured:.4g} vs sigma^2={sigma ** 2:.4g}")
    bias = float(np.mean(np.abs(np.mean(arrays, axis=0) - weights))
                 / np.mean(np.abs(weights)))
    check(bias < 0.2,
          f"扰动无系统偏置（{draws} 个成员平均趋近原权重）", f"rel={bias:.3g}")


def _case_noise_struct(work, check):
    """噪声在张量内部的**相关结构**：四条不可分割的性质。

    为什么要单独测：结构化噪声的全部意义在于「同预算、换结构」。如果它顺手把注入方差
    也改了，那就变成了又一次 σ 扫描，前面 A/B 的结论一个都用不上。所以「独立样本数」、
    「每元素方差」、「行内常数性」、「方差定律」四条都要钉死。纯数值，不需要 onnx。
    """
    from .weights import _draw, _noise_shape

    shape = (12, 5, 3, 3)               # (out, in, kh, kw)，模仿一个卷积权重
    want = {
        "iid": (12, 5, 3, 3),           # 逐元素独立
        "channel": (12, 1, 1, 1),       # 每个输出通道一个
        "scalar": (1, 1, 1, 1),                     # 整张量只有**一个**随机数
    }
    for struct in NOISE_STRUCTS:
        got = _noise_shape(shape, struct)
        check(got == want[struct],
              f"[{struct}] 独立样本数正确（iid 逐元素 / channel 逐通道 / scalar 整张量）",
              f"got={got} want={want[struct]}")

    # 因子 1：每元素 E[xi]=0、Var(xi)≈1 —— 所有结构都必须成立（这才叫「同预算」）
    # scalar 每次抽样只产出**一个**随机数，元素级方差统计量的自由度就等于抽样次数、相对
    # 误差 ~sqrt(2/draws)：200 次下约三分之一的概率掉出 ±10% 的带（会变成随机假红）。所以
    # scalar 单独抬到 4000 次（误差 ~2.2%，±10% 相当于 4.5 sigma）。
    for struct in NOISE_STRUCTS:
        rng = np.random.default_rng(7)
        acc = np.zeros(shape)
        squares = np.zeros(shape)
        draws = 4000 if struct == "scalar" else 200
        for _ in range(draws):
            noise = _draw(np.ones(shape), 1.0, rng, struct=struct) - 1.0
            acc += noise
            squares += noise ** 2
        mean = acc / draws
        var = squares / draws - mean ** 2
        check(abs(float(mean.mean())) < 0.05,
              f"[{struct}] 噪声无系统偏置（{draws} 次抽样的元素均值）",
              f"mean={float(mean.mean()):.4g}")
        check(0.9 < float(var.mean()) < 1.1,
              f"[{struct}] 每元素方差 ≈ 1（注入方差定律的前提）",
              f"var={float(var.mean()):.4g}")

    # 因子 2：结构本身 —— channel 逐行常数、iid 不是
    rng = np.random.default_rng(11)
    ones = np.ones(shape)
    iid = _draw(ones, 1.0, rng, struct="iid") - 1.0
    chan = _draw(ones, 1.0, rng, struct="channel") - 1.0
    scal = _draw(ones, 1.0, rng, struct="scalar") - 1.0

    def row_scatter(a):
        """行内元素与「本行第一个元素」的最大偏离：0 表示整行常数。

        不能用 std：45 个**完全相同**的 float 的样本标准差不是 0，而是 ~1e-16
        （均值先舍入了一次），会把「逐通道一致」误判成不一致。
        """
        rows = a.reshape(shape[0], -1)
        return float(np.abs(rows - rows[:, :1]).max())

    check(row_scatter(chan) == 0.0 and float(np.abs(chan).std()) > 1e-6,
          "[channel] 同一输出通道内逐元素相同，且不同通道不同",
          f"row_scatter={row_scatter(chan):.3g} overall_std={float(np.abs(chan).std()):.3g}")
    check(row_scatter(iid) > 0.5,
          "[iid] 行内逐元素独立（对照组：行内明显不是常数）",
          f"row_scatter={row_scatter(iid):.3g}")

    scal_gap = float(np.abs(scal - scal.reshape(-1)[0]).max())
    check(scal_gap == 0.0 and abs(float(scal.reshape(-1)[0])) > 1e-6,
          "[scalar] 整张量只有一个 xi（张量内逐元素相同，且不为 0）",
          f"张量内最大偏离={scal_gap:.3g} xi={float(scal.reshape(-1)[0]):.3g}")
    check(scal_gap == 0.0 and row_scatter(chan) == 0.0
          and float(np.abs(scal - chan).max()) > 1e-6,
          "[scalar] 与 channel 不同：channel 逐行各异，scalar 连行间也相同",
          f"两者最大偏离={float(np.abs(scal - chan).max()):.3g}")

    # 因子 3：方差定律在所有结构下都成立 —— 这是「同预算」的操作定义
    w = np.random.default_rng(3).standard_normal((64, 32))
    for struct in NOISE_STRUCTS:
        sigma = 0.05
        # 同样是自由度问题：iid/channel 一次抽样就有上千个独立元素，40 次平均已经
        # 远超需要；scalar 一次只有一个随机数，40 次的相对误差 ~22%（相对 ±30% 的带就是
        # 1.4 sigma），所以它抽 400 次（~7%）。
        draws = 400 if struct == "scalar" else 40
        ratios = [float(np.sum((_draw(w, sigma, np.random.default_rng(s),
                                      struct=struct) - w) ** 2) / np.sum(w ** 2))
                  for s in range(draws)]
        measured = float(np.mean(ratios))
        check(0.7 < measured / sigma ** 2 < 1.4,
              f"[{struct}] 注入方差 = sigma^2 x ||W||^2（{draws} 次抽样平均）",
              f"实测 {measured:.4g} vs sigma^2={sigma ** 2:.4g}")


def _case_m1a_noise_struct(work, check):
    """结构化噪声必须**真的落到成员权重上**，而且不改变注入方差。

    为什么不能只靠 _case_noise_struct：那一节测的是 _draw 这个函数。
    从 config 到 spec.noise_struct 到 build_member_set 到写进 perturbed.mNNN.data
    这条链上任何一环漏传，函数级测试都还是绿的，而服务器上跑的
    却是 iid ——本仓库已经被这类「静默退回默认值」咬过一次（m3 的 seed 纪律）。
    """
    # 自己开一个子目录：同批用例会断言「工作目录里没有多余 .data」，
    # 把自己的 .data 丢进 work 会把那条弄成假红。
    box = osp.join(work, "m1a_ns")
    os.makedirs(box, exist_ok=True)
    src = _write_synthetic(osp.join(box, "ns_src.onnx"), external=True, constant=True)
    spec_of, member_of = {}, {}
    for struct in ("iid", "channel", "scalar"):
        parent = osp.join(box, "ns_set_" + struct)
        os.makedirs(parent, exist_ok=True)
        spec = WeightPerturbSpec(groups=("all",), sigma_full=0.01,
                                 member_dir=parent, noise_struct=struct)
        build_member_set(src, spec, 2, verbose=False)
        spec_of[struct] = spec
        member_of[struct] = load_model(member_model_path(src, spec, 0))

    source = {tensor.name: tensor for tensor in load_model(src).graph.initializer}

    def arrays(struct, name):
        anchor = osp.dirname(member_model_path(src, spec_of[struct], 0))
        member = {t.name: t for t in member_of[struct].graph.initializer}
        base = tensor_array(source[name], box).astype(np.float64)
        return base, tensor_array(member[name], anchor).astype(np.float64)

    def ratio(struct, name):
        base, pert = arrays(struct, name)
        return pert / base - 1.0

    def row_ptp(values):
        """行内最大偏离（行 = 第一个轴）：0 表示整行常数。"""
        rows = values.reshape(values.shape[0], -1)
        return float(np.abs(rows - rows[:, :1]).max())

    # 1. 二维 / 四维张量：channel 结构 = 每个输出通道一个 xi，行内必须是常数
    for name in ("latent_mixer.head.weight", "backbone.conv.weight"):
        iid_ptp, chan_ptp = row_ptp(ratio("iid", name)), row_ptp(ratio("channel", name))
        check(chan_ptp < 1e-5,
              f"[{name}] channel 结构：同一输出通道内 xi 相同（行内常数）",
              f"行内最大偏离={chan_ptp:.3g}")
        check(iid_ptp > 1e-3,
              f"[{name}] iid 对照组：行内明显不是常数",
              f"行内最大偏离={iid_ptp:.3g}")

    # 1b. scalar：整张量只有一个 xi（二维 / 四维都一样），且与 iid / channel 都不同
    for name in ("latent_mixer.head.weight", "backbone.conv.weight"):
        scal_values = ratio("scalar", name)
        iid_gap = float(np.abs(scal_values - ratio("iid", name)).max())
        chan_gap = float(np.abs(scal_values - ratio("channel", name)).max())
        scal_ptp = float(np.abs(scal_values - scal_values.reshape(-1)[0]).max())
        # 容差不用 == 0：pert/base 是在 float64 里做的除法，逐元素舍入差 ~1e-8 是正常的
        # （权重本身是 fp32，pert 已经含了 sigma*xi 的舍入）。
        check(scal_ptp < 1e-6 and abs(float(scal_values.reshape(-1)[0])) > 1e-6,
              f"[{name}] scalar 结构：整张量只有一个 xi",
              f"张量内最大偏离={scal_ptp:.3g} xi={float(scal_values.reshape(-1)[0]):.3g}")
        check(iid_gap > 1e-4 and chan_gap > 1e-4,
              f"[{name}] scalar 与 iid / channel 都不同（三种结构互不重合）",
              f"vs iid={iid_gap:.3g} vs channel={chan_gap:.3g}")

    # 2. 一维张量（bias / dt_bias）：channel **退化为 iid**，必须与 iid 图逐位相同
    for name in ("latent_mixer.head.dt_bias", "backbone.conv.bias"):
        delta = float(np.abs(ratio("iid", name) - ratio("channel", name)).max())
        check(delta == 0.0,
              f"[{name}] 一维张量在 channel 结构下退化为 iid（与 iid 图逐位相同）",
              f"最大偏离={delta:.3g}")

    # 2b. 一维张量在 scalar 下**不**退化：整个向量共享一个 xi（这正是 scalar 与 channel 的分界）
    for name in ("latent_mixer.head.dt_bias", "backbone.conv.bias"):
        values = ratio("scalar", name)
        gap = float(np.abs(values - values.reshape(-1)[0]).max())
        check(values.size > 1 and gap < 1e-6,
              f"[{name}] 一维张量在 scalar 下整向量共享一个 xi（不退化成 iid）",
              f"元素数={values.size} 张量内最大偏离={gap:.3g}")

    # 3. 同预算：注入方差比都必须 ≈ sigma^2（否则「换结构」变成了又一次 sigma 扫描）
    for name in ("latent_mixer.head.weight", "backbone.conv.weight"):
        for struct in ("iid", "channel"):
            base, pert = arrays(struct, name)
            budget = float(np.sum((pert - base) ** 2) / np.sum(base ** 2))
            check(0.4 < budget / 1e-4 < 1.8,
                  f"[{name}] {struct}：单个成员的注入方差比 ≈ sigma^2",
                  f"实测 {budget:.4g} vs sigma^2=1e-4")
        # scalar 的**单个张量**注入方差比 = sigma^2 x xi^2（整个张量只有一个随机数 = chi^2_1），
        # 套 iid 那种 [0.4, 1.8] 的带会有约 70% 概率假红 —— 这是统计限制，不是实现问题。
        # 所以这里只粗查「量级没跑偏」（|xi| < 6 对应 2e-9 的假红概率），而「每元素方差 = 1」
        # 由 _case_noise_struct 用 4000 次抽样钉死（那里的自由度够）。
        # 顺带记一笔：这条统计限制本身就是 scalar 臂的结论之一 —— 一个成员只有 210 个随机数，
        # 离散度统计量天然比 iid 抖，N 个成员张成的扰动子空间秩 <= #被扰张量。
        base, pert = arrays("scalar", name)
        budget = float(np.sum((pert - base) ** 2) / np.sum(base ** 2))
        check(0.0 < budget / 1e-4 < 36.0,
              f"[{name}] scalar：单个张量的注入方差比 = sigma^2 x xi^2，量级没跑偏",
              f"实测 {budget:.4g}（xi^2={budget / 1e-4:.3g}）vs sigma^2=1e-4")

    # 4. 指纹必须认 noise_struct：否则 channel 会靆合命中 iid 的成员集目录，静默复用错权重
    _dirs = {struct: set_dir_for(src, spec) for struct, spec in spec_of.items()}
    check(len(set(_dirs.values())) == len(_dirs),
          "noise_struct 进指纹：iid / channel / scalar 各自一个成员集目录",
          " / ".join(f"{struct}={osp.basename(path)}" for struct, path in _dirs.items()))


def _case_selection(work, check):
    """该扰的扰、不该扰的不扰，且每条跳过都有明确原因。"""
    src = _write_synthetic(osp.join(work, "src.onnx"))
    spec = WeightPerturbSpec(groups=("all",), sigma_full=0.01,
                             member_dir=osp.join(work, "sel_set"))
    build_member_set(src, spec, 1, verbose=False)

    # 「扰动到了谁」以**成员里的数值**为准，不数图上的节点：成员权重与源权重不同的那些
    # 张量，就是真正被改写过的集合。这比读图更贴近事实——选错了张量、选中了却没真正换掉
    # 值，都会在这里露出来。
    source = {tensor.name: np.asarray(tensor_array(tensor, work))
              for tensor in load_model(src).graph.initializer}
    member = _member_weights(src, spec, 0)
    touched = sorted(name for name, value in source.items()
                     if name in member and not np.array_equal(value, member[name]))
    check(touched == sorted(EXPECTED_INJECTED),
          "被改写的权重集合与预期一致", f"实际 {touched}")
    check(all(np.array_equal(value, member[name])
              for name, value in source.items() if name not in EXPECTED_INJECTED),
          "其余 initializer 一个字节都没动")

    selection = select_initializers(load_model(src), spec, work)
    for name, fragment in EXPECTED_SKIPPED:
        reason = selection.skipped.get(name, "<没有 skip 记录>")
        check(fragment in reason, f"{name} 被正确跳过（{fragment}）", reason)

    # 分组：只选 latent_mixer 时应当只剩它自己的两个张量
    grouped = select_initializers(
        load_model(src), WeightPerturbSpec(groups=("latent_mixer",)), work)
    check(sorted(grouped.selected) == ["latent_mixer.head.dt_bias", "latent_mixer.head.weight"],
          "按组前缀筛选只命中该组", f"实际 {sorted(grouped.selected)}")

    # 并集 token：一个 token 里写多个前缀（`,` 分隔）必须合成**一个组**。这条是给
    # 「按块类型注入」兜底的 —— FGVP 的 12 个 block 交错（偶数是 GatedDeltaNet、
    # 奇数是 attn+mlp），只能写成并集；写成 6 个 token 会逐组配平、总注入方差变 6 倍
    # （G 已经踩过 2 倍的那个坑，见 perturbation/README 的 2E.1）。
    spec_split = WeightPerturbSpec(groups=("latent_mixer.head.weight",
                                           "latent_mixer.head.dt_bias"), sigma_full=0.01)
    spec_union = WeightPerturbSpec(
        groups=("latent_mixer.head.weight,latent_mixer.head.dt_bias",), sigma_full=0.01)
    split = select_initializers(load_model(src), spec_split, work)
    union = select_initializers(load_model(src), spec_union, work)

    check(sorted(union.selected) == sorted(split.selected),
          "并集 token 选中的张量集与分写两个 token 相同", f"{sorted(union.selected)}")
    check(sorted(union.selected) == ["latent_mixer.head.dt_bias", "latent_mixer.head.weight"],
          "并集 token 命中两个张量（含 dt_bias 这种小张量）")
    check(len({item.group for item in union.selected.values()}) == 1,
          "并集 token 只产生一个组（配平只吃一份预算）",
          f"{sorted({item.group for item in union.selected.values()})}")
    check(len({item.group for item in split.selected.values()}) == 2,
          "分写成两个 token 就是两个组（这就是 N 倍预算的来源）")

    # 预算守恒：并集的总注入方差 = 1 份，分写 = 2 份
    def _budget(selection, spec):
        sigmas = resolve_sigmas(selection, spec)
        return sum(sigmas[group] ** 2 * energy
                   for group, energy in selection.group_energy.items())

    full = selection.candidate_energy * 0.01 ** 2
    near = lambda a, b: abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0)
    check(near(_budget(union, spec_union), full),
          "并集组的注入方差 = 1 份预算 sigma_full^2 * E_candidate",
          f"{_budget(union, spec_union):.6g} vs {full:.6g}")
    check(near(_budget(split, spec_split), 2 * full),
          "分写两组 = 2 份预算（并集写法要避免的就是这个）",
          f"{_budget(split, spec_split):.6g} vs {2 * full:.6g}")
    check(group_alternatives("a.b,c/d") == (("a", "b"), ("c", "d")),
          "group_alternatives 按 `,`/`|` 展开并复用 key_parts 的分割规则",
          f"{group_alternatives('a.b,c/d')}")
    check(group_alternatives("latent_mixer") == (("latent_mixer",),),
          "不含分隔符时退化成单前缀（与旧写法逐位兼容）")


def _case_stream_independence(work, check):
    """不同张量的噪声流必须不相关：SPW 的逐张量独立假设全建在这上面。

    实现上每个张量从同一条 SeedSequence spawn 出一条独立子流（weights._draw）。
    若图省事让所有张量共用一条流，整个模型就被一个 rank-1 方向扰动——实测相关系数
    0.9999999，而组间按能量配平、逐通道归因这些分析都会随之下错结论。
    """
    count = 3
    src, weights = _write_probe(osp.join(work, "probe_stream.onnx"),
                                count=count, size=256)
    spec = WeightPerturbSpec(groups=("all",), sigma=0.05,
                             member_dir=osp.join(work, "probe_stream_set"))
    build_member_set(src, spec, 1, verbose=False)
    drawn = _member_weights(src, spec, 0)
    deltas = [np.asarray(drawn[f"probe{i}.weight"]) - np.asarray(weights[i])
              for i in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            correlation = float(np.corrcoef(deltas[left], deltas[right])[0, 1])
            check(abs(correlation) < 0.3,
                  f"张量 {left} 与张量 {right} 的噪声流不相关",
                  f"corr={correlation:.3f}")


def _case_m1a(work, check):
    """M1a：逐成员冻结权重的落盘布局、冻结语义、成员差异、分片生成与后端自检。"""
    src = _write_synthetic(osp.join(work, "m1a_src.onnx"), external=True, constant=True)
    src_data = osp.join(work, "m1a_src.data")
    check(osp.isfile(src_data), "M1a 源图是外部数据形式（成员共享源 .data）")

    # 被外置的 Constant 节点也要算进「外部数据引用」：它和 initializer 一样按相对图文件
    # 的路径解析，只看 initializer 的实现在这里全盲。
    const_tensors = [attribute.t for graph in iter_graphs(load_model(src).graph)
                     for node in graph.node for attribute in node.attribute
                     if attribute.type == 4]
    check(len(const_tensors) == 1 and const_tensors[0].data_location == 1,
          "M1a 源图里也有 1 个被外置的 Constant 节点张量",
          str([(tensor.name, tensor.data_location) for tensor in const_tensors]))

    set_parent = osp.join(work, "m1a_set")
    os.makedirs(set_parent, exist_ok=True)
    spec = WeightPerturbSpec(groups=("all",), sigma_full=0.01,
                             member_dir=set_parent)
    report = build_member_set(src, spec, 3, verbose=False)
    check(report["members"] == 3 and len(report["member_reports"]) == 3,
          "生成 3 个成员", str(report["members"]))
    check(osp.dirname(report["set_dir"]) == osp.abspath(set_parent),
          "成员集落在 spec.member_dir", report["set_dir"])

    member0 = member_model_path(src, spec, 0)
    set_dir = report["set_dir"]
    base_path = member_base_path(src, spec)
    check(osp.dirname(member0) == set_dir,
          "成员图与 base/perturbed 同在成员集目录（扁平布局，不逐成员开子目录）",
          osp.dirname(member0))
    check(sorted(os.listdir(set_dir)) == [
        "base.data", "m1a_src.m000.onnx", "m1a_src.m001.onnx", "m1a_src.m002.onnx",
        "perturbed.m000.data", "perturbed.m001.data", "perturbed.m002.data",
        "spw_members.json"],
        "成员集目录 = 共享 base + 逐成员图/扰动数据 + 清单",
        str(sorted(os.listdir(set_dir))))
    check(osp.isfile(src_data), "源 .data 没有被搬走或改写")
    check(osp.samefile(base_path, src_data),
          "base.data 是源 .data 的硬链接（零字节复制）")
    check(not osp.islink(base_path),
          "base.data 是硬链接而非符号链接（符号链接会被 ORT 的真实路径校验拒绝）")
    check(report["share"] == {"base.data": "link"},
          "报告记下共享方式：硬链接", str(report["share"]))
    check(0 < osp.getsize(member_perturbed_path(src, spec, 0))
          < osp.getsize(src_data) + 1,
          "perturbed.m000.data 只装被扰动的张量，不是整份权重的副本")

    # 成员图里**所有**外部数据引用（含被外置的 Constant 节点）都必须落在成员目录内。
    # 这是 2026-09-23 服务器事故的回归：只扫 initializer 的实现漏改了 4 个 attention
    # 常量的 location，成员图仍指向 <stem>.data，onnxruntime 建会话时报
    # filesystem error: cannot get file size: No such file or directory。
    member = load_model(member0)
    closure: dict = {}
    for tensor in external_tensors(member):
        for entry in tensor.external_data:
            if entry.key == "location":
                closure.setdefault(entry.value, []).append(tensor.name or "<匿名>")
    check(sorted(closure) == ["base.data", "perturbed.m000.data"],
          "成员图的外部数据引用只有 base.data / perturbed.m000.data（含 Constant 节点）",
          str({key: len(value) for key, value in closure.items()}))
    check(all(osp.isfile(osp.join(osp.dirname(member0), name)) for name in closure),
          "这些引用都指向成员目录里的文件", str(sorted(closure)))
    check("m1a_src.data" not in closure,
          "没有任何引用还指着源 .data 的名字（源 .data 就在隔壁，漏改时看着一切正常）")
    node_tensors = [attribute.t for graph in iter_graphs(member.graph)
                    for node in graph.node for attribute in node.attribute
                    if attribute.type == 4]
    check(len(node_tensors) == 1 and
          {entry.key: entry.value for entry in node_tensors[0].external_data}
          .get("location") == "base.data",
          "被外置的 Constant 节点张量也改到了 base.data（只扫 initializer 会漏掉它）",
          str([{entry.key: entry.value for entry in tensor.external_data}
               for tensor in node_tensors]))

    # 硬链接不可用时的退化路径（跨卷 EXDEV、或 fs.protected_hardlinks=1 而源文件不属于
    # 当前用户 EPERM——2026-09-23 服务器就是后者）：整份复制**一次**进成员集目录，且
    # **不能**是符号链接。onnxruntime 会把外部数据解析成真实路径再校验它落在模型目录
    # 内，符号链接指到目录外会被直接拒绝（External data path ... escapes both ...）。
    copy_parent = osp.join(work, "m1a_set_copy")
    os.makedirs(copy_parent, exist_ok=True)
    copy_spec = WeightPerturbSpec(groups=("all",), sigma_full=0.01,
                                  member_dir=copy_parent)
    copied = build_member_set(src, copy_spec, 2, link=False, verbose=False)
    copy_base = member_base_path(src, copy_spec)
    check(copied["share"] == {"base.data": "copy"},
          "link=False 时报告记下共享方式：复制", str(copied["share"]))
    check(not osp.islink(copy_base) and osp.isfile(copy_base),
          "退化路径产出的是真实文件，绝不是符号链接")
    check(not osp.samefile(copy_base, src_data),
          "这份 base 是复制，不是硬链接")
    check(len(os.listdir(copied["set_dir"])) == 2 * 2 + 2,
          "base 只复制一次：2 成员 x(图+扰动) + base + 清单",
          str(sorted(os.listdir(copied["set_dir"]))))

    feeds = _feeds()
    outputs = []
    for index in range(3):
        path = member_model_path(src, spec, index)
        props = {entry.key: entry.value for entry in load_model(path).metadata_props}
        check(props.get("spw.mechanism") == "m1a"
              and props.get("spw.member") == str(index),
              f"member_{index:03d} 图里带着机制与成员号",
              f"{props.get('spw.mechanism')}/{props.get('spw.member')}")
        session = _session(path)
        first = session.run(None, feeds)[0]
        second = session.run(None, feeds)[0]
        check(np.array_equal(first, second),
              f"member_{index:03d} 同一输入两次前向逐位相同（整条轨迹冻结）")
        outputs.append(first)

    base = _run(src, feeds)[0]
    check(not np.array_equal(base, outputs[0]), "成员与源模型不同（扰动确实写进了权重）")
    check(not np.array_equal(outputs[0], outputs[1])
          and not np.array_equal(outputs[1], outputs[2]),
          "成员之间互不相同（逐成员独立抽噪声）")

    info = verify_member_set(src, spec, 3, sample=4)
    check(not info["problems"], "verify_member_set 逐成员复核无问题",
          "; ".join(info["problems"][:3]))
    ratios = [item["ratio"] for item in info["amplitude"].values()]
    check(bool(ratios) and all(0.8 < ratio < 1.2 for ratio in ratios),
          "抽样权重级相对扰动 rms(W~-W)/rms(W) ≈ 配平后的 sigma",
          str(info["amplitude"]))

    # 分片生成（多卡时每个 rank 造自己那一份）+ 清单按成员号合并
    shard = build_member_set(src, spec, 4, indices=[3], verbose=False)
    check(osp.isfile(member_model_path(src, spec, 3)),
          "按索引生成：indices=[3] 造出 member_003")
    check(shard["members"] == 4 and len(shard["member_reports"]) == 4,
          "清单合并：原 3 个 + 新增 1 个 = 4 个", str(len(shard["member_reports"])))
    check(sorted(shard["built"]) == [0, 1, 2, 3],
          "清单记下累计生成过的成员号", str(shard["built"]))

    # rank 无关性：成员集是**多卡共享**的落盘产物，目录与内容都不能随 LOCAL_RANK 变。
    # 目录名带 rank 会让每个 rank 各造一份（磁盘翻 world_size 倍），而且分片生成时
    # 「等其它 rank 把剩下的造完」永远等不到——所以这一条必须钉住（见 weights._base_spec）。
    previous = os.environ.get("LOCAL_RANK")
    try:
        dirs = []
        for rank in ("0", "3"):
            os.environ["LOCAL_RANK"] = rank
            dirs.append(set_dir_for(src, spec))
    finally:
        if previous is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = previous
    check(dirs[0] == dirs[1], "成员集目录不随 LOCAL_RANK 变（多卡共享同一份）", str(dirs))
    check(dirs[0] == report["set_dir"], "与刚才生成时落在同一个目录")

    check(resolve_effective_path(src, spec, 1) == member_model_path(src, spec, 1),
          "resolve_effective_path 按成员号换到对应的成员模型")
    check(resolve_effective_path(src, None, 1) == src,
          "没有 spec 时原样返回源模型（非扰动档不受影响）")

    def raises(function, exception):
        try:
            function()
        except exception:
            return True
        return False

    missing = resolve_member_path.__doc__ and raises(
        lambda: resolve_member_path(src, spec, 99), FileNotFoundError)
    check(missing, "成员缺失时报 FileNotFoundError")
    try:
        resolve_member_path(src, spec, 99)
    except FileNotFoundError as error:
        check("perturbation members" in str(error),
              "成员缺失时的报错里带着可直接执行的构建命令")

    # 后端加载期的成员号自检（防「51 个成员其实加载了同一个文件」）
    from xmetai_inference.backends.onnx import OnnxInferModel

    model = OnnxInferModel()
    model.session = _FakeMetaSession(member_model_path(src, spec, 1))
    model.member_index = 1
    model._verify_perturbation(spec)
    check(True, "后端自检：成员号与图元数据一致时放行")
    model.member_index = 2
    check(raises(lambda: model._verify_perturbation(spec), RuntimeError),
          "后端自检：成员号对不上时报错")
    # 没有 spw.member 标记的图（源模型、或任何非成员图）一律拦下：这正是「51 个成员
    # 其实加载了同一个文件」的落盘形态，也是本机制最难发现的静默失败。
    model.session = _FakeMetaSession(src)
    model.member_index = 0
    check(raises(lambda: model._verify_perturbation(spec), RuntimeError),
          "后端自检：把没有成员标记的图（源模型）当成员加载会被拦下")

    # 负例（反向验证自校验真的看得见）：手工把 member_002 里 Constant 节点的 location
    # 改回源名字——这正是老实现漏改后的落盘状态——self-test 与 check 都必须报出来。
    onnx = _onnx()
    tampered_path = member_model_path(src, spec, 2)
    tampered = load_model(tampered_path)
    for graph in iter_graphs(tampered.graph):
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == 4:
                    for entry in attribute.t.external_data:
                        if entry.key == "location":
                            entry.value = "m1a_src.data"
    onnx.save(tampered, tampered_path)
    broken = _verify_external_refs(onnx, load_model(tampered_path),
                                   osp.dirname(tampered_path))
    check(any("m1a_src.data" in item for item in broken),
          "自校验看得见 Constant 节点的失效引用（老实现只扫 initializer，全盲）",
          "; ".join(broken[:2]))
    check(any("不在成员集目录内" in item
              for item in verify_member_set(src, spec, 3)["problems"]),
          "check 命令能报出「引用不在成员集目录内」（这类事故原来全盲）")

    # 负例：把 base.data 换成指向目录外的符号链接（正是老实现的退化路径）——静态复核必须
    # 报出「真实路径在成员集目录之外」。没有建符号链接的权限（Windows 未开开发者模式）时跳过。
    try:
        os.remove(base_path)
        os.symlink(src_data, base_path)
    except (OSError, NotImplementedError) as error:
        print(f"    （跳过符号链接负例：{error}）")
    else:
        stray = verify_member_set(src, spec, 3)["problems"]
        check(any("真实路径在成员集目录之外" in item for item in stray),
              "符号链接形式的 base.data 会被自校验拦下（ORT 也会拒绝）",
              "; ".join(stray[:2]))
        os.remove(base_path)
    build_member_set(src, spec, 3, verbose=False)
    check(not verify_member_set(src, spec, 3)["problems"],
          "重新生成后成员集恢复干净（生成是确定性的）")


def _case_segment_plan(work, check):
    """分段 rollout 的**计划层**：预算、继承、指纹、段边界四类。

    为什么要单测这一层：分段写错不会报错，只会「跑完 60 步但权重只换了一次」或者「每段都用
    base 的 σ」，两种都只让结论不可信 —— 正是这一版最贵的一类 bug。真正换 session 的动作在
    cli 里，这里钉的是「规格算得对不对、边界在第几步、错例会不会被拦下」。纯逻辑，不碰 onnx。
    """
    from .segments import parse_segment_plan

    base = WeightPerturbSpec(groups=("latent_mixer",), sigma_full=0.04,
                             balance="energy", seed=0,
                             noise_struct="channel", member_dir=osp.join(work, "seg_dir"))
    sigma_seg = base.sigma_full / 3 ** 0.5
    plan = parse_segment_plan(
        [{"sigma_full": sigma_seg, "steps": 20},
         {"sigma_full": sigma_seg, "steps": 20, "seed": 1},
         {"sigma_full": sigma_seg, "steps": 20, "seed": 2}], base, 60)
    check(plan is not None and plan.total_steps == 60 and len(plan.segments) == 3,
          "3 段 x 20 步：总步数与段数正确",
          f"total={plan.total_steps} 段数={len(plan.segments)}")

    specs = plan.specs()
    check(all(spec.groups == base.groups and spec.balance == base.balance
              and spec.exclude == base.exclude and spec.include == base.include
              and spec.noise_struct == base.noise_struct
              and spec.member_dir == base.member_dir
              and spec.min_elements == base.min_elements for spec in specs),
          "各段规格：组 / 预算口径 / 排除 / include / 噪声结构 / 落盘目录 全部继承 base",
          specs[0].describe())
    check([spec.sigma_full for spec in specs] == [sigma_seg] * 3
          and [spec.seed for spec in specs] == [0, 1, 2],
          "各段的幅度与种子按分段表逐段落地（不是都被 base 的 sigma 抹平）",
          f"sigma_full={[spec.sigma_full for spec in specs]} seed={[spec.seed for spec in specs]}")

    # 预算：每段 sigma_full/sqrt(K) -> 一段摊 1/K 份，K 段和起来正好是冻结臂那一份
    total = sum(spec.sigma_full ** 2 for spec in specs)
    check(abs(total / base.sigma_full ** 2 - 1.0) < 1e-12,
          "同预算分段：各段 (sigma_full/sqrt(K))^2 之和 = 冻结臂的 sigma_full^2",
          f"比值={total / base.sigma_full ** 2:.9f}")

    switch = plan.switch_after()
    check(switch == {20: 1, 40: 2},
          "段边界：跑完第 20 / 40 步之后换成第 1 / 2 段",
          f"switch_after={switch}")
    check(plan.total_steps not in switch and max(switch) < plan.total_steps,
          "末段之后不产生换段点（最后一步跑完不必再挂一次 session）",
          f"switch_after={switch} total={plan.total_steps}")

    # 指纹：段规格不同 -> 各自一个成员集目录。探针图放在子目录里，避免给同批用例添乱
    # （它们会断言工作目录里没有多余的 .data）。
    box = osp.join(work, "seg_case")
    os.makedirs(box, exist_ok=True)
    src = osp.join(box, "seg_probe.onnx")
    with open(src, "wb") as handle:
        handle.write(b"seg")
    dirs = [set_dir_for(src, spec) for spec in specs]
    check(len(set(dirs)) == 3,
          "分段指纹：三段各自一个成员集目录（seed / 幅度不同就不共用）",
          " / ".join(osp.basename(path) for path in dirs))
    twin = parse_segment_plan([{"sigma_full": sigma_seg, "steps": 30},
                               {"sigma_full": sigma_seg, "steps": 30}], base, 60)
    check(set_dir_for(src, twin.specs()[0]) == set_dir_for(src, twin.specs()[1]),
          "同幅度 + 同种子的两段 = 同一份权重（要真刷新必须在分段表里给不同的 seed）",
          "两段目录相同，这是预期语义而不是 bug")

    for bad, label in (
        ([{"sigma_full": 0.02, "steps": 30}], "各段步数之和 != steps"),
        ([{"steps": 60}], "缺 sigma / sigma_full"),
        ([{"sigma_full": 0.02, "sigma": 0.02, "steps": 60}], "同时给了 sigma 与 sigma_full"),
        ([{"sigma_full": 0.02, "steps": 60, "segma_full": 0.1}], "键名写错（segma_full）"),
        ([{"sigma_full": 0.02, "steps": 0}], "某段步数为 0"),
        ([], "空表"),
        ("sigma_full=0.02", "不是列表"),
    ):
        try:
            parse_segment_plan(bad, base, 60)
        except ValueError:
            check(True, f"错例被拦下：{label}")
        else:
            check(False, f"错例被拦下：{label}", "竟然没报错")
    check(parse_segment_plan(None, base, 60) is None,
          "None = 不分段（默认路径逐位不变）")
    check("0:20步" in plan.describe() and "seed=2" in plan.describe(),
          "describe() 里读得到每段步数 / 幅度 / 种子（服务器上核对日志用）",
          plan.describe())


def _case_verdict(work, check):
    """判决页（verdict）的**判据层**：p 插值 / k 折算 / 平坦度 / 判据分类 / 门槛。

    为什么要单测这一层：判决页输出的就是「判负 / 不判负」，算错了不会抛异常，只会把整整
    一族配置的结论翻个面 —— 而这一页正是要拿去给别人看的。纯逻辑，不碰 onnx、不需要
    评测归档（归档缺失时 run_verdict 会带着「可用臂」列表显式报错）。
    """
    import math

    import pandas as pd

    from .verdict import (BANDS, P_TABLE, _band_symbol, bin_mean, criteria, flatness,
                          k_needed, k_profile, p_at)

    # 1) p 在对数时效上插值：端点原样、越界夹住、单调不增
    check(abs(p_at(P_TABLE, 6) - 1.007) < 1e-12 and abs(p_at(P_TABLE, 342) - 0.466) < 1e-12,
          "p_at：表格两个端点的时效原样返回实测值")
    check(p_at(P_TABLE, 3) == 1.007 and p_at(P_TABLE, 720) == 0.466,
          "p_at：超出表格范围时夹到端点（**不外推**——外推出来的 p 没有实测支撑）")
    middle = p_at(P_TABLE, 96)
    check(0.946 < middle < 0.990, "p_at：+96 h 落在 +54/+102 h 之间（对数插值）",
          f"p(96h)={middle:.4f}")
    profile = [p_at(P_TABLE, lead) for lead in range(6, 361, 6)]
    check(all(first >= second for first, second in zip(profile, profile[1:])),
          "p(lead) 随时效单调不增（这就是「单一 sigma 补不了两头」的来源）")

    # 2) k 的定义：k^p 必须还原 r
    reference = pd.Series({6: 100.0, 102: 200.0, 342: 430.0})
    doubled = pd.Series({6: 150.0, 102: 300.0, 342: 645.0})
    rows = k_profile(doubled, reference, P_TABLE)
    check(len(rows) == 3 and all(abs(row["k"] ** row["p"] - row["r"]) < 1e-9 for row in rows),
          "k = r^(1/p) 的定义（k^p 逐点还原 r）",
          " ".join(f"{row['k']:.4f}" for row in rows))

    # 3) 平坦度：平的 = 纯幅度；上升的 = 形状红利；样本不足 = 判不了（nan）
    same = flatness([{"k": 1.5}] * 8)
    check(abs(same["span"] - 1.0) < 1e-12 and abs(same["trend"] - 1.0) < 1e-12,
          "flatness：全平的 k -> 极差 1.00、趋势 1.00（= 纯幅度）")
    rising = flatness([{"k": 1.0 + 0.1 * index} for index in range(8)])
    check(rising["trend"] > 1.25 and rising["span"] > 1.5,
          "flatness：随时效上升的 k -> 趋势 > 1.25（= 形状真被治好）",
          f"trend={rising['trend']:.3f} span={rising['span']:.3f}")
    check(math.isnan(flatness([{"k": 1.0}] * 3)["trend"]),
          "flatness：样本 < 4 个时效时返回 nan（判不了，不是判过）")

    # 4) 分箱均值：判据 1 / 2 就是拿它比的
    series = pd.Series({6: 1.0, 48: 3.0, 240: 5.0, 360: 7.0})
    check(abs(bin_mean(series, 6, 48) - 2.0) < 1e-12
          and abs(bin_mean(series, 240, 360) - 6.0) < 1e-12,
          "bin_mean：闭区间均值（+6..48 与 +240..360 两个分箱）")

    # 5) 判据 3 的分类：五个合成图案，逐一核对到 README 2L.11 / 2L.12 的实际判决
    leads = [6, 48, 240, 360]
    ref_spread = pd.Series({lead: 100.0 + 2.0 * lead for lead in leads})
    ref_bands = {band: pd.Series({lead: 1000.0 for lead in leads}) for band in BANDS}
    ref_curves = {"spread": ref_spread}

    def classify(relative):
        arm_bands = {band: ref_bands[band] * relative[band] for band in BANDS}
        return criteria({"spread": ref_spread * 1.5}, ref_curves, arm_bands, ref_bands)

    patterns = (
        ("选择性（只有 5-20 涨，且过了 8% 的可辨识门槛）",
         {"1_4": 1.04, "5_20": 1.12, "21_40": 1.00, "41_64": 1.00, "65_128": 1.00},
         "选择性"),
        ("净增益太弱：只有 5-20 抬了 6%（低于可辨识门槛，与 S 同类）",
         {"1_4": 1.02, "5_20": 1.06, "21_40": 1.00, "41_64": 1.00, "65_128": 1.00},
         "没有可辨识的净增益"),
        ("S 型：都几乎没动，21-40 反而更暗",
         {"1_4": 1.006, "5_20": 1.039, "21_40": 0.845, "41_64": 1.064, "65_128": 1.024},
         "没有可辨识的净增益"),
        ("T / E3 型：5-20 与 21-40 一起亮（整体锐化）",
         {"1_4": 0.991, "5_20": 1.399, "21_40": 1.610, "41_64": 1.074, "65_128": 1.028},
         "整体锐化"),
        ("V 型：增益跑到 41-64 / 65-128（谱尾变脏）",
         {"1_4": 0.947, "5_20": 0.645, "21_40": 0.966, "41_64": 5.512, "65_128": 7.482},
         "增益落在更小尺度"),
        ("只涨被抑制的 21-40",
         {"1_4": 0.99, "5_20": 0.99, "21_40": 1.30, "41_64": 1.00, "65_128": 1.00},
         "增益落在被抑制的通道"),
    )
    for label, relative, expect in patterns:
        crit = classify(relative)
        check(expect in crit["band_kind"], f"判据 3 分类：{label}",
              f"期望含 {expect!r}，实得 {crit['band_kind']!r}")
    check(_band_symbol({"pass_band": True}) == "✅"
          and _band_symbol({"band_kind": "整体锐化（5-20 与 21-40 一起变亮，非尺度选择性）"}) == "➖"
          and _band_symbol({"band_kind": "增益落在更小尺度（谱尾部变脏）=> 判负"}) == "❌",
          "判据 3 的记号：选择性 = ✅ / 整体锐化 = ➖ / 判负 = ❌")

    # 6) 门槛：从基准臂自己的缺口算出「需要多少 k」，且随缺口加深而变大
    # 缺口形状照抄基准臂 F 的实测（最深 0.548 在 +324 h）：门槛是**从数据里长出来的**，不是手写常数
    gap = pd.Series({6: 1.08, 240: 0.700, 300: 0.594, 324: 0.548, 360: 0.608})
    needs = {row["lead"]: row for row in k_needed(gap, P_TABLE)}
    expect = (1.0 / 0.594) ** (1.0 / p_at(P_TABLE, 300))
    check(abs(needs[300]["k_needed"] - expect) < 1e-9,
          "k_needed = (1/标定比)^(1/p)：门槛是从基准臂**自己的缺口**算出来的，不用手写",
          f"k_needed(+300h)={needs[300]['k_needed']:.3f}")
    check(needs[240]["k_needed"] < needs[300]["k_needed"] < needs[324]["k_needed"],
          "k_needed：缺口越深、需要的 k 越大（所以门槛取长时效分箱里的最大值）",
          f"+240h={needs[240]['k_needed']:.2f} < +300h={needs[300]['k_needed']:.2f}"
          f" < +324h={needs[324]['k_needed']:.2f}")
    threshold = max(row["k_needed"] for row in needs.values() if row["lead"] >= 240)
    check(threshold > 3.0,
          "k_needed：FGVP 基准臂的长时效门槛要 k > 3（五个臂给到的最大是 2.4）",
          f"门槛={threshold:.2f}")
    del work


# ---------------------------------------------------------------------------
# 潜空间 AR(1) 强迫（latent.py）
# ---------------------------------------------------------------------------
_FORCE_TOKENS = 64
_FORCE_CHANNELS = 8
_FORCE_GRID = (8, 8)
_FORCE_RANK = 4
# 合成图里那个「潜状态」张量的名字：**与 FGVP 对齐**，所以别名表能直接命中它。
_FORCE_TARGET = "/latent_mixer/Add_output_0"


def _write_forcing_fixture(path, *, patch_embed=False, fp16=False):
    """写一张「有潜空间可切」的小图。

    非 patch_embed：`x [B,tokens,channels] --Identity--> 目标张量 --Add(head.bias)--> y`。
    目标张量是 FGVP 那种「潜状态」出口（名字对齐 TARGET_ALIASES 里的第一个），Add 是它的
    消费者 —— 注入链必须插在「目标自己的第一个消费者」之前，这条夹具正好能钉住。

    patch_embed=True：再加一个 kernel == stride 的 Conv（名字对齐 FGVP 的 patch_embed）+
    Reshape，用来测「从卷积与输入空间维推 tokens/grid」的那条探测路径；同时把 Conv 的
    4 维输出 `patch` 显式写进 value_info —— 它是「非 3 维目标必须被拒绝」的负例。

    fp16=True：全图 float16。**这一档必须单列**：生产用的集合图就是 fp16
    （iwc_gssp_im_fp16.onnx），而注入链的常量是 fp32，两边直接相乘会被 ORT 判为
    「Type parameter (T) of Optype (Mul) bound to different types」而**建不出会话**。
    修复是链尾往目标类型 Cast 一次（latent.py 的 CastGain）。这个 bug 从 fp32 夹具上
    完全看不见，所以夹具本身要能出 fp16。
    """
    onnx = _onnx()
    helper, TensorProto = onnx.helper, onnx.TensorProto
    rng = np.random.default_rng(11)
    tokens, channels = _FORCE_TOKENS, _FORCE_CHANNELS
    rows, cols = _FORCE_GRID
    dtype = np.float16 if fp16 else np.float32
    elem = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT

    inits = [onnx.numpy_helper.from_array(
        (rng.standard_normal(channels).astype(dtype) * 0.1), "head.bias")]
    value_info = []
    if patch_embed:
        inits += [
            onnx.numpy_helper.from_array(
                (rng.standard_normal((channels, 3, 2, 2)).astype(dtype) * 0.1),
                "backbone.patch_embed.weight"),
            onnx.numpy_helper.from_array(
                np.asarray([0, tokens, channels], dtype=np.int64), "flatten.shape"),
        ]
        nodes = [
            helper.make_node("Conv", ["img", "backbone.patch_embed.weight"], ["patch"],
                             name="/backbone/patch_embed/Conv",
                             kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("Reshape", ["patch", "flatten.shape"], [_FORCE_TARGET],
                             name="/latent_mixer/Add"),
        ]
        inputs = [helper.make_tensor_value_info(
            "img", elem, [1, 3, rows * 2, cols * 2])]
        value_info.append(helper.make_tensor_value_info(
            "patch", elem, [1, channels, rows, cols]))
    else:
        nodes = [helper.make_node("Identity", ["x"], [_FORCE_TARGET],
                                  name="/latent_mixer/Add")]
        inputs = [helper.make_tensor_value_info(
            "x", elem, ["B", tokens, channels])]
    nodes.append(helper.make_node(
        "Add", [_FORCE_TARGET, "head.bias"], ["y"], name="/latent_mixer/head/Add"))
    graph = helper.make_graph(
        nodes, "spw_latent_selftest", inputs,
        [helper.make_tensor_value_info("y", elem, ["B", tokens, channels])],
        inits, value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def _forcing_spec(**overrides):
    """夹具用的规格：显式给 tokens/channels/grid（不依赖形状推断），秩 <= 通道数。"""
    kwargs = {"targets": (_FORCE_TARGET,), "sigma": 0.04, "tau_days": 3.0,
              "rank": _FORCE_RANK, "length_scale": 1.0, "tokens": _FORCE_TOKENS,
              "channels": _FORCE_CHANNELS, "grid": _FORCE_GRID, "seed": 0}
    kwargs.update(overrides)
    return LatentForcingSpec(**kwargs)


def _case_latent(work, check):
    """潜空间 AR(1) 强迫：等价性 / 数值口径 / 结构 / 驱动侧随机流 / 基的性质。

    与 m3 的分水岭：**图本身没有随机性** —— 同一个 xi 跑两次必须逐位相同，随机性全部来自
    驱动侧每步喂进去的 xi。这条钉不住的话，「时间相关」就无从谈起（图内随机算子根本给不了
    跨步记忆，那正是 m3 失败的原因）。
    """
    onnx = _onnx()
    src = _write_forcing_fixture(osp.join(work, "latent_src.onnx"))
    model0 = load_model(src)
    channels, tokens = _FORCE_CHANNELS, _FORCE_TOKENS

    # 1) 目标解析
    check(resolve_targets(model0, _forcing_spec(rank=_FORCE_RANK)) == [_FORCE_TARGET],
          "resolve_targets：显式节点名原样命中")
    try:
        resolve_targets(model0, LatentForcingSpec(targets=("latent_state",)))
        ok = False
    except ValueError as error:
        ok = "latent_state" in str(error)
    check(ok, "resolve_targets：别名在这个骨架上解析不出节点时给可读报错")
    check(LatentForcingSpec(targets=()).targets == ("latent_state",),
          "targets 留空时回落到默认别名 latent_state（config 里不写 = 用默认）")

    # 2) 布局探测
    layout = detect_layout(model0, [_FORCE_TARGET], _forcing_spec())
    check((layout["tokens"], layout["channels"], layout["grid"]) == (tokens, channels, _FORCE_GRID),
          "detect_layout：显式值优先，且回带观测到的形状",
          f"得到 {layout}")
    try:
        detect_layout(model0, [_FORCE_TARGET],
                      _forcing_spec(grid=(_FORCE_GRID[0], _FORCE_GRID[1] + 1)))
        ok = False
    except ValueError as error:
        ok = "不一致" in str(error)
    check(ok, "detect_layout：grid 乘积与 tokens 不一致时报错")
    pe = _write_forcing_fixture(osp.join(work, "latent_pe.onnx"), patch_embed=True)
    model_pe = load_model(pe)
    guess = _grid_from_patch_embed(model_pe)
    check(guess == _FORCE_GRID,
          "patch_embed 探测：kernel == stride 的 Conv + 静态输入空间维 -> (rows, cols)",
          f"得到 {guess}")
    try:
        detect_layout(model_pe, ["patch"], _forcing_spec(targets=("patch",), tokens=None,
                                                         grid=None))
        ok = False
    except ValueError as error:
        ok = "3 维" in str(error)
    check(ok, "4 维目标被明确拒绝（只有 [batch, tokens, channels] 能注入）")

    # 3) sigma=0 必须与源图逐位相同（等价性回归）
    feeds = {"x": np.random.default_rng(3).standard_normal(
        (1, tokens, channels)).astype(np.float32)}
    source_out = _run(src, feeds)[0]
    zero = build_forcing_model(src, _forcing_spec(sigma=0.0),
                               osp.join(work, "latent_zero.onnx"), verbose=False)
    zero_out = _run(zero, {"x": feeds["x"],
                           "spw/xi": np.zeros((1, 1, _FORCE_RANK), dtype=np.float32)})[0]
    check(np.array_equal(zero_out, source_out),
          "sigma=0 时强迫图与源图**逐位相同**（空转对照成立）")

    # 4) sigma>0：与 numpy 参考一致（读图里**烘焙**的 Vt/Ut，不信别的口径）
    sigma = 0.3
    forced = build_forcing_model(src, _forcing_spec(sigma=sigma),
                                 osp.join(work, "latent_forced.onnx"), verbose=False)
    model_f = load_model(forced)
    baked = {entry.name: entry for entry in model_f.graph.initializer}
    basis_v = onnx.numpy_helper.to_array(baked["spw/ar1/t0/Vt"])      # [R, N]
    basis_u = onnx.numpy_helper.to_array(baked["spw/ar1/t0/Ut"])      # [R, C]
    check(basis_v.shape == (_FORCE_RANK, tokens) and basis_u.shape == (_FORCE_RANK, channels),
          "烘焙进图的基就是 [rank, tokens] 与 [rank, channels]",
          f"得到 {basis_v.shape} / {basis_u.shape}")
    xi = np.random.default_rng(5).standard_normal((1, 1, _FORCE_RANK)).astype(np.float32)
    field = np.einsum("br,rn,rc->bnc", xi[:, 0, :], basis_v, basis_u)
    bias = onnx.numpy_helper.to_array(baked["head.bias"])
    expected = (source_out - bias) * (1.0 + field) + bias
    got = _run(forced, {"x": feeds["x"], "spw/xi": xi})[0]
    check(np.allclose(got, expected, rtol=1e-4, atol=1e-5),
          "图内算的场 == numpy 参考 n = sum_r xi_r * v_r[tok] * u_r[chan]",
          f"最大偏差 {np.abs(got - expected).max():.3e}")

    # 5) 图自身无随机性：同 xi 两次逐位相同、换 xi 才变（与 m3 的分水岭）
    again = _run(forced, {"x": feeds["x"], "spw/xi": xi})[0]
    other = _run(forced, {"x": feeds["x"], "spw/xi": xi + 0.5})[0]
    check(np.array_equal(got, again),
          "同一个 xi 两次前向**逐位相同**（图里没有随机算子 —— 与 m3 的分水岭）")
    check(not np.allclose(got, other),
          "换一个 xi 输出就变（随机性来自驱动侧喂进来的 xi）")

    # 5b) fp16 图：注入链的常量是 fp32，必须在链尾 Cast 一次才建得出会话。
    #     生产用的集合图就是 fp16，而这个 bug 在 fp32 夹具上完全看不见 —— 所以夹具
    #     得有 fp16 档，且要同时钉住「会话建得起来」与「数值真的乘上了增益」。
    src16 = _write_forcing_fixture(osp.join(work, "latent_fp16.onnx"), fp16=True)
    forced16 = build_forcing_model(src16, _forcing_spec(sigma=sigma),
                                   osp.join(work, "latent_fp16_forced.onnx"),
                                   verbose=False)
    model16 = load_model(forced16)
    casts = [node for node in model16.graph.node if node.name == "spw/ar1/t0/CastGain"]
    check(len(casts) == 1
          and next(a.i for a in casts[0].attribute if a.name == "to") == onnx.TensorProto.FLOAT16,
          "fp16 图的注入链尾部有且只有一次 Cast(to=fp16)"
          "（链全程 fp32，只在乘目标之前降下来）",
          f"找到 {len(casts)} 个 CastGain")
    x16 = feeds["x"].astype(np.float16)
    zero16 = build_forcing_model(src16, _forcing_spec(sigma=0.0),
                                 osp.join(work, "latent_fp16_zero.onnx"), verbose=False)
    source16 = _run(src16, {"x": x16})[0]
    check(np.array_equal(_run(zero16, {"x": x16,
                                       "spw/xi": np.zeros((1, 1, _FORCE_RANK),
                                                         dtype=np.float32)})[0], source16),
          "fp16 图的 sigma=0 空转也与源图**逐位相同**（Cast 不改变恒等的增益 1）")
    baked16 = {entry.name: entry for entry in model16.graph.initializer}
    field16 = np.einsum("br,rn,rc->bnc", xi[:, 0, :],
                        onnx.numpy_helper.to_array(baked16["spw/ar1/t0/Vt"]),
                        onnx.numpy_helper.to_array(baked16["spw/ar1/t0/Ut"]))
    # 参考按图里的**每一步**都舍入到 fp16 重算（乘增益 -> Cast -> 加偏置），这样比的是
    # 同一条算术链，而不是「fp32 算完再降精度」——后者会把 fp16 的舍入差记到实现头上。
    gain16 = (1.0 + field16).astype(np.float16)
    bias16 = onnx.numpy_helper.to_array(baked16["head.bias"])
    expected16 = ((source16 - bias16) * gain16 + bias16).astype(np.float16)
    got16 = _run(forced16, {"x": x16, "spw/xi": xi})[0]
    check(got16.dtype == np.float16,
          "fp16 图的前向仍是 fp16 进出（Cast 只作用于增益，不动主干精度）",
          f"得到 {got16.dtype}")
    check(np.allclose(got16, expected16, rtol=5e-3, atol=1e-3),
          "fp16 图上乘进去的增益 == numpy 参考（增益没被 Cast 丢掉或算成 1）",
          f"最大偏差 {np.abs(got16.astype(np.float32) - expected16.astype(np.float32)).max():.3e}")

    # 6) 结构：目标的所有消费者都改接到强迫链，且只改一次
    check("spw/xi" in [value.name for value in model_f.graph.input],
          "图上多了 spw/xi 输入")
    left = [node for node in model_f.graph.node if _FORCE_TARGET in list(node.input)]
    check([node.name for node in left] == ["spw/ar1/t0/MulZ"],
          "目标张量只剩强迫链自己读它（原来的消费者全部改接到强迫链的输出）",
          f"得到 {[node.name for node in left]}")
    stem_out = "spw/ar1/t0/out"
    readers = [node for node in model_f.graph.node if stem_out in list(node.input)]
    check(len(readers) == 1 and readers[0].op_type == "Add",
          "强迫链的输出恰好被一个消费者读取（改接只做一次）",
          f"得到 {[node.name for node in readers]}")
    injected = [node for node in model_f.graph.node if node.name.startswith("spw/ar1/t0")]
    check(len(injected) == 6,
          "每个目标注入 6 个节点（Reshape/MulV/Transpose/MatMulU/AddOne/MulZ）",
          f"得到 {len(injected)}")
    shared = [node for node in model_f.graph.node
              if node.name.startswith("spw/ar1") and node.name.count("/") == 2]
    check(len(shared) == 1 and shared[0].op_type == "Split",
          "共享节点只有 1 个（Split：每个目标各切走一份 xi）",
          f"得到 {[node.name for node in shared]}")

    # 7) 基的性质：正交、无 DC、sigma 语义没被缩放带偏、length_scale 真的接上了
    basis = build_basis(tokens, channels, _FORCE_RANK, 2.0, _FORCE_GRID, 0, 0.04)
    gram = basis["Vt"] @ basis["Vt"].T
    check(np.abs(gram - np.eye(_FORCE_RANK)).max() < 1e-5,
          "空间基列正交且单位范数（V V^T = I）")
    basis_dc = float(np.abs(basis["Vt"].mean(axis=1)).max())
    check(basis_dc < 1e-6,
          "空间基去掉 DC 分量（防论文的主失效模式「整场一致偏移」）",
          f"最大 |列均值| = {basis_dc:.3e}")
    degenerate = build_basis(tokens, channels, _FORCE_RANK, 32.0, _FORCE_GRID, 0, 0.04)
    degenerate_dc = float(np.abs(degenerate["Vt"].mean(axis=1)).max())
    check(degenerate_dc < 1e-6 and degenerate["report"]["fallbacks"] == _FORCE_RANK,
          "length_scale 远大于网格时退回白噪声（否则低通把非 DC 模压成精确 0，"
          "QR 会造出**带均匀分量**的基）",
          f"最大 |列均值| = {degenerate_dc:.3e}，退化方向 "
          f"{degenerate['report']['fallbacks']}/{_FORCE_RANK}")
    implied = float(np.linalg.norm(basis["Ut"]) / np.sqrt(channels * tokens))
    check(abs(implied - 0.04) < 1e-6,
          "基隐含的 sigma 等于设定值（sigma 的无量纲语义没被缩放带偏）",
          f"隐含 {implied:.8f}")
    try:
        build_basis(tokens, _FORCE_RANK - 1, _FORCE_RANK, 1.0, _FORCE_GRID, 0, 0.04)
        ok = False
    except ValueError as error:
        ok = "不能大于通道数" in str(error)
    check(ok, "rank 超过通道数被拒绝（通道基最多只有 channels 个正交方向）")
    big_grid, big_tokens = (64, 64), 4096

    def roughness(row):
        plane = np.asarray(row, dtype=np.float64).reshape(big_grid)
        return float(np.abs(np.diff(plane, axis=1)).mean())

    smooth = build_basis(big_tokens, channels, _FORCE_RANK, 8.0, big_grid, 0, 0.04)
    rough = build_basis(big_tokens, channels, _FORCE_RANK, 1.0, big_grid, 0, 0.04)
    check(smooth["report"]["fallbacks"] == 0 and rough["report"]["fallbacks"] == 0,
          "正常的 length_scale 不触发退化兜底（兜底只在 length_scale 相对网格太大时生效）")
    check(roughness(smooth["Vt"][0]) < 0.5 * roughness(rough["Vt"][0]),
          "length_scale 大 = 空间基明显更平滑（尺度旋钮真的接上了）",
          f"平滑/白噪 = {roughness(smooth['Vt'][0]):.4f} / {roughness(rough['Vt'][0]):.4f}")

    # 8) 驱动侧：rho / 平稳方差 / lag-1 / 成员与 batch 的独立性 / reset
    spec = _forcing_spec()
    driver = Ar1Driver(spec, 1, member_index=3, batch=1)
    check(abs(driver.rho - math.exp(-(6.0 / 24.0) / 3.0)) < 1e-12,
          "rho = exp(-dt/tau)：6 h 步长、tau=3 天 -> 0.9200",
          f"得到 {driver.rho:.6f}")
    check(abs(LatentForcingSpec(tau_days=3.0).rho(12.0)
              - math.exp(-(12.0 / 24.0) / 3.0)) < 1e-12,
          "rho 支持按调用方给的 hour_interval 覆盖（模型类说了算）")
    series = []
    for _ in range(4000):
        series.append(float(driver.value(1)[0, 0, 0]))
        driver.advance()
    check(abs(np.var(series) - 1.0) < 0.15,
          "AR(1) 是平稳过程：4000 步的方差 ~ 1（不是无界随机游走）",
          f"方差 {np.var(series):.4f}")
    check(abs(lag1_autocorr(series) - driver.rho) < 0.05,
          "lag-1 自相关 ≈ rho（时间相关真的存在，且由 tau 决定）",
          f"lag-1 {lag1_autocorr(series):.4f} vs rho {driver.rho:.4f}")
    slow = Ar1Driver(_forcing_spec(tau_days=10.0), 1, batch=1)
    fast = Ar1Driver(_forcing_spec(tau_days=0.5), 1, batch=1)
    slow_series, fast_series = [], []
    for _ in range(600):
        slow_series.append(float(slow.value(1)[0, 0, 0]))
        fast_series.append(float(fast.value(1)[0, 0, 0]))
        slow.advance()
        fast.advance()
    check(lag1_autocorr(slow_series) > lag1_autocorr(fast_series) + 0.15,
          "tau 越大时间相关越强（tau -> 0 退化回「每步重抽」的 m3 端点）",
          f"tau=10d {lag1_autocorr(slow_series):.3f} vs tau=0.5d "
          f"{lag1_autocorr(fast_series):.3f}")
    first = Ar1Driver(spec, 1, member_index=0, batch=1).value(1)
    second = Ar1Driver(spec, 1, member_index=1, batch=1).value(1)
    check(not np.allclose(first, second),
          "不同成员号的 xi 互不相同（成员独立性来自 member_index 这条熵）")
    check(not np.allclose(Ar1Driver(spec, 1, member_index=0, slot=0, batch=1).value(1),
                          Ar1Driver(spec, 1, member_index=0, slot=1, batch=1).value(1)),
          "同一成员号、不同 slot（进程 rank）的 xi 也互不相同"
          "（多卡按成员切时 member_index 恒为 0，靠 rank 才不重复）")
    batched = Ar1Driver(spec, 1, member_index=0, batch=3)
    values = batched.value(3)
    check(values.shape == (3, 1, _FORCE_RANK),
          "xi 的形状是 [batch, 目标数, rank]（每步只喂这么点数字）",
          f"得到 {values.shape}")
    check(not np.allclose(values[0], values[1]),
          "同一步里不同 batch 元素的 xi 互不相同（一个 session 服务多个成员）")
    before = values.copy()
    batched.reset()
    check(int(batched.steps) == 0 and not np.allclose(before, batched.value(3)),
          "reset 重抽 xi_0 并把步数归零（新轨迹 = 新的噪声相位）")

    # 8b) 驱动状态机的**块宽**：rollout_batch>1 时一次 forward 堆 k 条轨迹，驱动必须
    #     一次吃掉 k 行。宽度恒 1 的旧接线会被 ORT 静默广播（[1,T,R] 的 ξ 配 batch=k 的
    #     state 不报错），块内 k 条轨迹共用相邻行 ⇒ 集合静默欠散。这里只查驱动侧发出的
    #     行号序列——它不依赖模型，是这条链上唯一能在合成图上钉死的环节。
    def row_plan(rb, steps=3, n_traj=12):
        """按 runner 的调用序列（外层 step、内层按 rb 切连续块）回放，记下每块拿到的行号。"""
        stream = LatentStepStream()
        plan = np.zeros((steps, n_traj), dtype=int)
        for step in range(steps):
            for start in range(0, n_traj, rb):
                width = min(rb, n_traj - start)
                index = stream(spec, step, n_targets=1, member_index=0, width=width)
                plan[step, start:start + width] = np.arange(index, index + width)
        return plan

    plan_one, plan_six = row_plan(1), row_plan(6)
    check(np.array_equal(plan_one, plan_six),
          "块宽 6 与逐轨迹（rb=1）拿到的行号序列逐位相同"
          "（行按需从同一条流顺序补出 ⇒ rollout_batch 对 ξ 中性）",
          f"rb=6 首步行号 {plan_six[0][:8]}...")
    check(np.array_equal(plan_six[0], np.arange(12)),
          "rb=6 时同一步内 12 条轨迹拿到 12 个互不相同的行号（没有共用行 = 成员没重复）",
          f"得到 {plan_six[0]}")
    stream = LatentStepStream()
    starts = (stream(spec, 0, n_targets=1, member_index=0, width=6),
              stream(spec, 0, n_targets=1, member_index=0, width=6))
    check(starts == (0, 6), "同一步内第二块的起始行号 = 首块宽度（槽位按宽度前进）",
          f"得到 {starts}")
    check(stream.ar1.row(6, 6).shape == (6, 1, _FORCE_RANK),
          "row(index, width) 直接给出 [width, 目标数, rank]，可整块喂进图"
          "（图里 spw/xi 的 batch 轴是符号维）",
          f"得到 {stream.ar1.row(6, 6).shape}")
    try:
        stream(spec, 0, n_targets=1, member_index=0, width=0)
        ok = False
    except ValueError as error:
        ok = "width" in str(error)
    check(ok, "width=0 被明确拒绝（发不出行却不报错 = 静默错位）")

    # 9) 注入到「没有消费者」的目标上必须报错，不能静默落一张坏图
    try:
        build_forcing_model(src, _forcing_spec(targets=("y",)),
                            osp.join(work, "latent_bad.onnx"), verbose=False)
        ok = False
    except ValueError as error:
        ok = "没有任何消费者" in str(error)
    check(ok, "把噪声挂在图的输出上会被明确拒绝（注入无处安放）")


# ---------------------------------------------------------------------------
# 潜空间 AR(1) 强迫 —— pt2 后端（latent_pt2.py）
# ---------------------------------------------------------------------------
# 夹具尺寸取到「网格大于 1x1、rank < channels」的最小可用档：空间基的低通在 4x4 上
# 不会退化（length_scale=1 时非 DC 模的能量远在兜底阈值之上），也不至于让用例变慢。
_PT2_TOKENS = 16
_PT2_CHANNELS = 8
_PT2_HIDDEN = 16
_PT2_LAYERS = 3
_PT2_GRID = (4, 4)
_PT2_RANK = 4


def _write_pt2_fixture(path, kind="fuxi"):
    """写一个 FuXi 形状的迷你 decoder，导出成真 .pt2，返回 (路径, 内存里的 torch 模型)。

    `kind` 换层内结构，只给负例用："fuxi"（真图拓扑）/ "sig"（链路撞上未知算子）/
    "fanout"（链路分叉）—— 后两者要求回溯**报错**，不是硬穿过去。

    为什么必须过真正的 `torch.export` 而不是手搭一个 GraphModule：本用例要钉的恰恰是
    **回溯规则本身** —— torch.export 把参数名变成 `p_model_decoder_layers_{k}_mlp_w2_weight`、
    它的消费者是 `aten.linear`、中间隔着 clone/mul 才到残差 `aten.add`。手搭图等于把这条
    规则当成前提，真机上名字一变（torch 版本、导出选项）用例还是绿的 —— 而这里的绿正是
    「别拿真模型试」的替代品。

    结构逐层照抄 FuXi-2.1 的命名**与拓扑**：

        x --(layers.0)--> z0 --(layers.1)--> z1 --(layers.2)--> z2 --head--> y
        每层：z = z + mlp(z).clone() * scale，mlp = w1 -> relu -> w2（w2 无 bias）
        注入点 = 每层那个残差 add 的输出（= build_basis 要的 [B, tokens, channels]）。

    `clone` 与 `scale`（形状 [1,1,C]，在 token 维广播）是**照抄真图**的，不是凑数：2026-09-30
    在 fuxi-2.1.pt2 上实测的链路是 `linear → clone → mul(×[1,1,1536]) → add`，而最初的回溯
    只认 `linear → add` 一步，于是在真图上直接报「0 个 add 消费者」。夹具跟真图同构，这条
    规则才有人守 —— 否则自测测的是一个真机上不存在的结构。
    """
    import torch
    import torch.nn as nn

    class _Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(_PT2_CHANNELS, _PT2_HIDDEN, bias=False)
            self.w2 = nn.Linear(_PT2_HIDDEN, _PT2_CHANNELS, bias=False)

        def forward(self, x):
            return self.w2(torch.relu(self.w1(x)))

    class _Layer(nn.Module):
        def __init__(self, kind="fuxi"):
            super().__init__()
            self.kind = kind
            self.mlp = _Mlp()
            # [1, 1, C]：与真图的调制向量同形（在 token 维广播）。
            self.scale = nn.Parameter(torch.ones(1, 1, _PT2_CHANNELS))

        def forward(self, x):
            hidden = self.mlp(x)
            if self.kind == "fuxi":
                return x + hidden.clone() * self.scale
            if self.kind == "sig":
                # 链路上出现未知算子：回溯必须**报错**而不是硬穿过去。
                return x + torch.sigmoid(hidden)
            # "fanout"：linear 输出被用两次，链路分叉 —— 同样必须报错。
            return x + hidden + hidden

    class _Decoder(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(kind) for _ in range(_PT2_LAYERS)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class _MiniFuxi(nn.Module):
        def __init__(self, kind="fuxi"):
            super().__init__()
            # model.decoder.layers.{k}.mlp.w2 这条嵌套路径是必需的：占位符名字由它拼出来。
            self.model = nn.Module()
            self.model.decoder = _Decoder(kind)
            self.head = nn.Linear(_PT2_CHANNELS, _PT2_CHANNELS, bias=False)

        def forward(self, x, step, hour, doy):
            return self.head(self.model.decoder(x))

    torch.manual_seed(0)
    model = _MiniFuxi(kind).eval()
    args = (torch.zeros(1, _PT2_TOKENS, _PT2_CHANNELS),
            torch.zeros(1), torch.zeros(1), torch.zeros(1))
    torch.export.save(torch.export.export(model, args), path)
    return path, model


def _pt2_spec(**overrides):
    """夹具用的规格：显式给 tokens/channels/grid（fx 图推不出网格）。"""
    kwargs = {"targets": ("layer_stream:0,2",), "sigma": 0.04, "tau_days": 3.0,
              "rank": _PT2_RANK, "length_scale": 1.0, "tokens": _PT2_TOKENS,
              "channels": _PT2_CHANNELS, "grid": _PT2_GRID, "seed": 0,
              "hour_interval": 6.0}
    kwargs.update(overrides)
    return LatentForcingSpec(**kwargs)


def _pt2_state(torch, path):
    """加载一份**未注入**的 ExportedProgram（attach_forcing 是就地改写，必须逐次重载）。"""
    return torch.export.load(path)


def _pt2_reference(torch, model, x, xi, applied, basis_v, basis_u):
    """独立参考：逐层重放，在指定的层输出上乘 (1 + n)。

    `applied` 是 [(层号, xi 列号), ...] —— 列号与层号的对应由 targets 决定，**故意分开传**：
    写成 `xi[层号]` 就等于把「第 k 列接在第 k 层」这条待验证的假设塞进了参考里。

    n[tok, c] = sum_r xi[r] * V[tok, r] * U[r, c] —— 与图里那条链同一个式子，但走
    torch 的 eager 路径。两边一致才说明「改图真的把场乘在了**这一层**的残差流上」。
    """
    z = x
    for index, layer in enumerate(model.model.decoder.layers):
        z = layer(z)
        for layer_no, column in applied:
            if layer_no == index:
                field = (basis_v * xi[column].unsqueeze(0)) @ basis_u   # [N, C]
                z = z * (1.0 + field)
    return model.head(z)


def _case_latent_pt2(work, check):
    """潜空间 AR(1) 强迫的 **pt2（torch.export）后端**：回溯 / 注入 / 数值 / 列序 / 驱动。

    与 ONNX 版（`_case_latent`）的分工：那边钉的是「图改写 + 落盘 + 外部数据」，这边钉的是
    torch.fx 特有的三件事 ——

      1. **回溯规则**：目标只能从权重占位符倒着找（fx 图里计算节点全是泛型 aten 名），
         名字一变就会静默注错地方；
      2. **xi 走模块属性**：torch.export 的图签名固定，塞不进新输入，所以属性名的形状 /
         执行期可覆写性是唯一的接入口（名字还必须用下划线，`/` 会生成非法 Python）；
      3. **不落盘**：没有图文件、没有 metadata，所以「注入对不对」只能靠**数值**判，
         不像 ONNX 版还能查 spw.ar1.* 元数据。

    数值判据里最锋利的一条是 **xi 全 0 时输出与未注入模型逐位相同**：gain 恒为 1.0、
    `z * 1.0` 逐位等于 `z`，所以任何「改错了节点 / 改错了乘的位置 / 改成了加性」都会在这里露出来。
    """
    try:
        import torch  # noqa: F401
    except ImportError as error:  # noqa: BLE001 - 环境里没有 torch 就跳过整节
        check(True, f"没有 torch，跳过 pt2 后端用例：{error}")
        return

    from .latent_pt2 import (attach_forcing, audit_fx_forcing, detect_fx_layout,
                             resolve_fx_targets, _walk_layer)

    box = osp.join(work, "latent_pt2")
    os.makedirs(box, exist_ok=True)
    path, model = _write_pt2_fixture(osp.join(box, "mini_fuxi.pt2"))
    tokens, channels = _PT2_TOKENS, _PT2_CHANNELS
    x = torch.randn(1, tokens, channels, generator=torch.Generator().manual_seed(3))
    step, hour, doy = torch.zeros(1), torch.zeros(1), torch.zeros(1)

    # ---- 1) 目标回溯 ----
    gm = _pt2_state(torch, path).graph_module
    anchors = sorted(node.name for node in gm.graph.nodes if node.op == "placeholder"
                     and "mlp_w2_weight" in node.name)
    check(len(anchors) == _PT2_LAYERS,
          "导出图里每层都有 mlp_w2 权重占位符（回溯锚点存在）", str(anchors))
    two = resolve_fx_targets(gm, _pt2_spec())
    everything = resolve_fx_targets(gm, _pt2_spec(targets=("layer_stream",)))
    one = resolve_fx_targets(gm, _pt2_spec(targets=("layer_stream:1",)))
    check(len(two) == 2 and len(everything) == _PT2_LAYERS and len(one) == 1,
          "layer_stream:0,2 / layer_stream / layer_stream:1 分别解析出 2 / 3 / 1 个目标",
          f"得到 {len(two)} / {len(everything)} / {len(one)}")
    order = [node.name for node in gm.graph.nodes]
    check(all(str(node.target) == "aten.add.Tensor" for node in two)
          and order.index(two[0].name) < order.index(two[1].name),
          "解析出来的目标是残差 add，且层号顺序 = 图里的先后顺序",
          f"得到 {[node.name for node in two]}")
    check(two == [everything[0], everything[2]],
          "「0,2」就是「全部层」里的第 0、2 个（层号没有错位）",
          f"{[n.name for n in two]} vs {[n.name for n in everything]}")
    check(one[0] not in two,
          "层号 1 解析到的是另一个节点（不是所有层号都落到同一个 add 上）")
    # 回溯**链路**：真图是 linear → clone → mul → add，夹具必须同构，否则这条规则没人守。
    chains = {layer: [node.name for node in _walk_layer(gm, layer)] for layer in range(_PT2_LAYERS)}
    targets = {str(node.target) for node in two}
    check(all(len(chain) == 4 and targets == {"aten.add.Tensor"}
              for chain in chains.values())
          and all("clone" in name or "mul" in name for chain in chains.values()
                  for name in chain[1:3]),
          "回溯沿单链穿过 clone/mul 才到残差 add（与真图 fuxi-2.1.pt2 同构，不是直连）",
          f"层 0 链路 {chains[0]}")
    for kind, fragment in (("sig", "不在已知的透明算子里"), ("fanout", "分叉")):
        variant = osp.join(box, f"variant_{kind}.pt2")
        _write_pt2_fixture(variant, kind=kind)
        try:
            _walk_layer(_pt2_state(torch, variant).graph_module, 0)
            ok = False
        except ValueError as error:
            ok = fragment in str(error)
        check(ok, f"回溯错例被拦下：{fragment}（层内结构 {kind}）")

    for bad, fragment in ((_pt2_spec(targets=("layer_stream:9",)), "不在图里"),
                          (_pt2_spec(targets=("没有这个节点",)), "找不到")):
        try:
            resolve_fx_targets(gm, bad)
            ok = False
        except ValueError as error:
            ok = fragment in str(error)
        check(ok, f"错例被拦下：{fragment}（{'层号越界' if '层' in fragment else '节点名不存在'}）")
    plain_path = osp.join(box, "plain.pt2")
    torch.export.save(torch.export.export(
        torch.nn.Linear(channels, channels), (torch.zeros(1, tokens, channels),)), plain_path)
    try:
        resolve_fx_targets(_pt2_state(torch, plain_path).graph_module, _pt2_spec())
        ok = False
    except ValueError as error:
        ok = "没有任何 layer_stream 锚点" in str(error)
    check(ok, "不是 decoder 结构的图（没有锚点）被明确拒绝，而不是静默注 0 个目标")

    # 裸数字：CLI 的 `--targets layer_stream:0,2` 会被逗号拆成 ("layer_stream:0", "2")，而 "2"
    # 作为节点名后缀**碰巧**能匹配到无关节点（本夹具上就匹配到了 channels 不同的那个）⇒ 静默注错。
    # 后缀匹配是给显式节点名兜底的，裸数字一律拒绝。
    try:
        resolve_fx_targets(gm, _pt2_spec(targets=("layer_stream:0", "2")))
        ok = False
    except ValueError as error:
        ok = "裸数字" in str(error)
    check(ok, "裸数字注入点被拒绝（防 CLI 逗号把 layer_stream 层号拆开后静默后缀匹配）")
    from .__main__ import _split_targets

    check(_split_targets("layer_stream:0,10,20") == ("layer_stream:0,10,20",)
          and _split_targets("layer_stream:0,2,latent_state") == ("layer_stream:0,2", "latent_state")
          and _split_targets("a, b") == ("a", "b"),
          "CLI 的 --targets 只在别名**外**按逗号切（别名内的逗号并回同一个 token）",
          str(_split_targets("layer_stream:0,2,latent_state")))

    # ---- 2) 布局：网格必须显式给，tokens/channels 对不上要报错 ----
    layout = detect_fx_layout(gm, two, _pt2_spec())
    check((layout["tokens"], layout["channels"], layout["grid"]) == (tokens, channels, _PT2_GRID),
          "detect_fx_layout：从目标节点的 meta 读出 tokens/channels，grid 用显式值",
          f"得到 {layout}")
    for bad, fragment in ((_pt2_spec(grid=None), "必须显式给"),
                          (_pt2_spec(tokens=tokens + 1), "配置与模型不符"),
                          (_pt2_spec(channels=channels + 1), "配置与模型不符"),
                          (_pt2_spec(grid=(2, 2)), "!= tokens")):
        try:
            detect_fx_layout(gm, two, bad)
            ok = False
        except ValueError as error:
            ok = fragment in str(error)
        check(ok, f"错例被拦下：{fragment}")

    # ---- 3) sigma=0 = 一个节点都不插（空转对照必须是**真的**没改图）----
    zero_ep = _pt2_state(torch, path)
    report = attach_forcing(zero_ep, _pt2_spec(sigma=0.0), src_path=path, verbose=False)
    check(report["skipped"] and report["n_targets"] == 0
          and not hasattr(zero_ep.graph_module, "spw_xi"),
          "sigma=0 时不改图（不挂 spw_* 属性、n_targets=0 == 驱动侧不喂 xi）",
          str(report))
    check(torch.equal(zero_ep.module()(x, step, hour, doy), model(x, step, hour, doy)),
          "sigma=0 的空转模型与未注入模型逐位相同")

    # ---- 4) 注入：节点数 / 属性形状 / 审计 ----
    ep = _pt2_state(torch, path)
    spec = _pt2_spec(sigma=0.3)
    report = attach_forcing(ep, spec, src_path=path, verbose=False)
    gm = ep.graph_module
    check(report["ok"] and report["nodes"] == 5 * len(two) and report["n_targets"] == len(two),
          "每个目标 5 个节点（select/mul/mm/add/mul），审计通过",
          f"nodes={report['nodes']} n_targets={report['n_targets']} ok={report['ok']}")
    check(tuple(gm.spw_V.shape) == (tokens, _PT2_RANK)
          and tuple(gm.spw_U.shape) == (_PT2_RANK, channels)
          and tuple(gm.spw_xi.shape) == (len(two), _PT2_RANK),
          "基挂在模块属性上：V [tokens, rank] / U [rank, channels] / xi [目标数, rank]",
          f"{tuple(gm.spw_V.shape)} / {tuple(gm.spw_U.shape)} / {tuple(gm.spw_xi.shape)}")
    bare = attach_forcing(_pt2_state(torch, path), _pt2_spec(sigma=0.3), verbose=False)
    check(bare["fingerprint"].startswith("s-") and len(bare["fingerprint"]) == 12,
          "没给 src_path 时指纹退成规格自身的 s- 前缀（不像 ONNX 版含源图身份）",
          bare["fingerprint"])
    check(report["fingerprint"] == spec.fingerprint(path),
          "给了 src_path 时指纹与 ONNX 版同一套口径（spec.fingerprint(源文件)）",
          report["fingerprint"])

    # ---- 5) 数值：xi=0 逐位恒等 / 与参考一致 / 图自身无随机性 ----
    module = ep.module()
    # ⚠ unlift 陷阱（后端接线全建在这条实测上）：ep.module() **不是** ep.graph_module，
    # 它会用 deepcopy 出来的图新建一个 GraphModule；spw_* 因为是 get_attr 的目标会被
    # 一起带过去，但那是同一批张量对象的**引用** ⇒ 就地 copy_ 两边都看得见，而 setattr
    # 换一个新张量只会改到其中一个副本（另一个正是前向在读的那个，于是静默退化成确定性
    # 预报）。backends/pt2.py 因此持有 module() 的返回值并 copy_ 就地写。
    check(module is not gm, "ep.module() 与 ep.graph_module 是两个不同的 GraphModule（走了 unlift）")
    check(all(hasattr(module, name) for name in ("spw_V", "spw_U", "spw_xi")),
          "注入的 spw_* 跟着 get_attr 一起被带进了 unlift 出来的 module")
    check(module.spw_xi is gm.spw_xi and module.spw_V is gm.spw_V,
          "两个 module 共享的是**同一批张量对象**（所以就地改写在两边都生效）")
    module.spw_xi.copy_(torch.full_like(module.spw_xi, 0.5))
    check(torch.equal(module.spw_xi, gm.spw_xi),
          "在 module 上 copy_ 后 graph_module 看到同样的值（就地改写，不是换张量）")
    with torch.no_grad():
        gm.spw_xi.zero_()
    # 基线用**同一份导出图**再加载一次（未注入）：与 out0 走同一条 codegen 路径，唯一差别
    # 就是那 5xN 个新节点在 xi=0 时恒等于恒等映射 —— 拿 eager 模型比会掺进「eager 与导出图
    # 的算子等价性」这个无关变量，逐位判据就不锋利了。
    base = _pt2_state(torch, path).module()(x, step, hour, doy)
    out0 = module(x, step, hour, doy)
    check(torch.equal(out0, base),
          "xi 全 0 时输出与**未注入的同一份导出图**逐位相同（gain 恒为 1.0，注入是乘性的）")
    check(torch.equal(out0, module(x, step, hour, doy)),
          "同一个 xi 两次前向逐位相同（图里没有随机算子 —— 随机性全在驱动侧）")

    basis_v, basis_u = gm.spw_V.detach(), gm.spw_U.detach()
    # 列号↔层号按 targets=("layer_stream:0,2") 的解析顺序推定：第 0 列 = 第 0 层、第 1 列
    # = 第 2 层。**只喂一列**的两种情形是关键：若图里某一列实际接到了另一层，参考与实测
    # 会立刻分家（两列都喂的话有概率互相掩盖）。
    for applied, seed in [(((0, 0),), 1), (((2, 1),), 2), (((0, 0), (2, 1)), 3)]:
        generator = torch.Generator().manual_seed(seed)
        xi = torch.zeros(len(two), _PT2_RANK)
        for _, column in applied:
            xi[column] = torch.randn(_PT2_RANK, generator=generator)
        with torch.no_grad():
            gm.spw_xi.copy_(xi)
            got = module(x, step, hour, doy)
            expected = _pt2_reference(torch, model, x, xi, applied, basis_v, basis_u)
        gap = float((got - expected).abs().max())
        label = " / ".join(f"第 {column} 列 -> 第 {layer} 层" for layer, column in applied)
        check(torch.allclose(got, expected, rtol=1e-4, atol=1e-5),
              f"xi 只喂这几列时输出 == 参考（{label}）", f"最大偏差 {gap:.3e}")
    check(not torch.allclose(out0, module(x, step, hour, doy)),
          "换一个 xi 输出就变（属性是执行期读的，不是注册时烘死的）")

    # 列序的独立证据：把两列**互换**应当给出与「交换后的参考」一致的结果 —— 如果
    # 图里某两列实际接到了同一层，这条会与上一条同时失败。
    xi = torch.randn(len(two), _PT2_RANK, generator=torch.Generator().manual_seed(13))
    swapped = xi.flip(0)
    with torch.no_grad():
        gm.spw_xi.copy_(xi)
        straight = module(x, step, hour, doy)
        gm.spw_xi.copy_(swapped)
        flipped = module(x, step, hour, doy)
        expected = _pt2_reference(torch, model, x, swapped, ((0, 0), (2, 1)), basis_v, basis_u)
    check(not torch.allclose(straight, flipped)
          and torch.allclose(flipped, expected, rtol=1e-4, atol=1e-5),
          "两列互换后输出随之改变，且等于「按互换后的 xi 重算的参考」（列序真的分到了不同的层）",
          f"互换前后最大差 {float((straight - flipped).abs().max()):.3e}")

    # ---- 6) 重复注入必须被拦下（否则同一层被乘两次，spread 静默放大）----
    try:
        attach_forcing(ep, spec, src_path=path, verbose=False)
        ok = False
    except RuntimeError as error:
        ok = "已经注入过" in str(error)
    check(ok, "对同一个 GraphModule 重复注入会被拒绝（防止双重扰动）")
    check(audit_fx_forcing(gm)["ok"],
          "audit_fx_forcing：每目标一条 select(spw_xi) 链且形状自洽", str(audit_fx_forcing(gm)))

    # ---- 7) 驱动侧：后端共享的 LatentStepStream（A32 首跑踩的是这条接线）----
    from .latent import LatentStepStream

    stream = LatentStepStream()
    rows = [stream(spec, 0, n_targets=len(two), member_index=0, hour_interval=6.0)
            for _ in range(3)]
    check(rows == [0, 1, 2],
          "同一步内按调用次序轮流发第 0/1/2 行（runner 是逐轨迹调 forward 的）",
          str(rows))
    wide = stream(spec, 0, n_targets=len(two), member_index=0, hour_interval=6.0)
    # 发号与加宽是两步：状态机只发**行号**，加宽发生在后端真正取行时（Ar1Driver.row）。
    check(wide == 3 and stream.ar1.state.shape[0] == 1,
          "第 4 条轨迹拿到行号 3，而驱动状态此时还只有 1 行（发号不预抽）",
          f"行号 {wide}，state 行数 {stream.ar1.state.shape[0]}")
    taken = stream.ar1.row(wide)
    check(taken.shape == (1, len(two), _PT2_RANK) and stream.ar1.state.shape[0] == 4
          and not np.allclose(taken[0], stream.ar1.state[0]),
          "取行时自动加宽到 4 行，新行与已有行不同（新行独立新抽，不影响已有行的递推）",
          f"取到 {taken.shape}，state 行数 {stream.ar1.state.shape[0]}")
    check(stream(spec, 1, n_targets=len(two), member_index=0, hour_interval=6.0) == 0,
          "换一个 step -> 槽位归零（同一模式步整批只推进一次）")
    check(abs(stream.ar1.rho - math.exp(-(6.0 / 24.0) / 3.0)) < 1e-12,
          "rho 由驱动侧按 hour_interval 折算（tau 不进图，所以换 tau 复用同一份改写）",
          f"rho={stream.ar1.rho:.6f}")
    drifted = stream.ar1.state.copy()
    stream(spec, 0, n_targets=len(two), member_index=0, hour_interval=6.0)
    check(not np.allclose(drifted, stream.ar1.state),
          "step 回到 0 = 新轨迹批：全部行重抽（重抽不重置随机流）")
    other = LatentStepStream()
    first = stream(spec, 0, n_targets=len(two), member_index=1, hour_interval=6.0)
    second = other(spec, 0, n_targets=len(two), member_index=2, hour_interval=6.0)
    check(first == 0 and second == 0
          and not np.allclose(stream.ar1.state[0], other.ar1.state[0]),
          "换成员号 -> 换一条独立的 xi 流（成员独立性来自 member_index 这条熵）")
    stream.release()
    check(stream.ar1 is None and stream(spec, 0, n_targets=0, member_index=0) is None,
          "release() 丢掉驱动状态；n_targets=0（sigma=0 空转）时恒返回 None",
          "两个后端共用的早退路径")


def _case_graph_split(work, check):
    """集合模型 + 权重扰动的成员摊分（cli._graph_member_split，纯函数）。

    图外层编舞（perturb_graphs）的成员身份全靠这个摊分定义：m = 图 g x 图内序号 j。
    摊错了不是「少几个成员」——是成员号重复/丢失，评测侧看不出来。这里钉四件事：
    均分 + 余数给前面的图；全局成员号连续且无重无漏；非法参数被拒；按图切卡
    （g % 卡数 == rank）时各 rank 的图互不相交且覆盖全部图。
    """
    # 惰性导入：selftest 的运行环境不该为了一个纯函数背上 cli 的依赖面。
    from xmetai_inference.cli import _graph_member_split

    split = _graph_member_split(51, 5)
    check([len(ids) for _, ids in split] == [11, 10, 10, 10, 10],
          "51/5 摊成 11+10x4", str([len(ids) for _, ids in split]))
    all_ids = [m for _, ids in split for m in ids]
    check(all_ids == list(range(51)),
          "全局成员号连续且无重无漏", f"头 {all_ids[:3]} 尾 {all_ids[-3:]}")
    check(all(len(ids) >= 1 for _, ids in split), "每份图非空")

    even = _graph_member_split(50, 5)
    check([len(ids) for _, ids in even] == [10] * 5, "50/5 整除时均分")
    single = _graph_member_split(10, 10)
    check([len(ids) for _, ids in single] == [1] * 10,
          "10/10 允许每图 1 成员（摊分本身不设下限，「每图至少 2」是 cli 的配置校验）")

    for bad in ((51, 0), (51, -1), (3, 5)):
        try:
            _graph_member_split(*bad)
        except ValueError:
            check(True, f"非法参数 {bad} 被拒（ValueError）")
        else:
            check(False, f"非法参数 {bad} 被拒（ValueError）", "没有抛")

    # 按图切卡（cli 的 graph_members 过滤条件）：G=5、卡数=8 时 rank 5..7 空手，
    # rank 0..4 各领一图；并起来正好是全部图、互不相交。
    world_size = 8
    claims = {rank: [g for g in range(5) if g % world_size == rank]
              for rank in range(world_size)}
    covered = sorted(g for gs in claims.values() for g in gs)
    check(covered == list(range(5)), "按图切卡：并集覆盖全部图", str(covered))
    check(sum(len(gs) for gs in claims.values()) == len(covered),
          "按图切卡：各 rank 的图互不相交")
    check(all(not claims[rank] for rank in (5, 6, 7)),
          "图数 < 卡数时多余 rank 领空（cli 打「没有分到图」退出）")


CASES = (
    ("无损改写（sigma->0 与源权重逐位相同）", _case_equivalence),
    ("扰动冻结与注入方差定律", _case_randomness),
    ("噪声结构（iid / channel / scalar 同预算）", _case_noise_struct),
    ("流独立性（同图内不同张量不相关）", _case_stream_independence),
    ("选择集与跳过原因", _case_selection),
    ("M1a 逐成员冻结权重落盘", _case_m1a),
    ("结构化噪声落到成员权重（m1a + channel + scalar）", _case_m1a_noise_struct),
    ("分段 rollout 计划（spw_segments / E3）", _case_segment_plan),
    ("判决页的判据层（p 插值 / k 折算 / 平坦度 / 判据分类 / 门槛）", _case_verdict),
    ("潜空间 AR(1) 强迫（等价性 / 数值 / 结构 / 驱动侧）", _case_latent),
    ("潜空间 AR(1) 强迫的 pt2 后端（fx 回溯 / 属性喂 xi / 列序）", _case_latent_pt2),
    ("集合模型+SPW 的成员摊分与按图切卡（perturb_graphs）", _case_graph_split),
)


def run_self_test(verbose=True, keep=False):
    """跑完全部检查，返回进程退出码（0 = 全通过）。"""
    work = tempfile.mkdtemp(prefix="xmetai_spw_selftest_")
    check = _Checker(verbose=verbose)
    print(f"SPW 自检（合成图）：{work}")
    try:
        for index, (title, case) in enumerate(CASES, start=1):
            check.section(index, len(CASES), title)
            case(work, check)
    finally:
        if keep:
            print(f"\n保留现场：{work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    print(f"\n合计：{check.passed} 条通过，{len(check.failures)} 条不通过")
    for failure in check.failures:
        print(f"  - {failure}")
    if check.failures:
        print("\n结论：自检不通过。先解决上面的问题，再拿真实模型跑集合预报。")
        return 1
    print("结论：自检全部通过。图改写无损、成员权重冻结且互相独立、"
          "外部数据未被复制。")
    return 0


__all__ = ["run_self_test"]
