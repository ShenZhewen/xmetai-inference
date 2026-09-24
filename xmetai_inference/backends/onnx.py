# -*- coding: utf-8 -*-
"""ONNX Runtime 后端（CUDAExecutionProvider）。

通用 ONNX 加载 + 一步前向。这里只放后端机制（session 建立、显存调优、run），
模型特定的东西（喂料、取帧、归一化）由各模型子类覆盖。
"""
import gc
import logging
import os

import numpy as np
import onnxruntime as ort

from .base import BaseInferModel, _default_gpu_mem_limit
from .session_pool import SessionPool, normalize_budget

log = logging.getLogger(__name__)


def _as_numpy_onnx(value):
    """torch → contiguous numpy；ONNX Runtime 输入边界允许这一次转换。"""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    return np.ascontiguousarray(value)


def _preload_onnxruntime_lib():
    """预加载 libonnxruntime.so.1，让自定义算子 .so 能解析到它。

    自定义算子库（xmetai_onnx_plugins.so）编译时链接的是 SONAME
    ``libonnxruntime.so.1``，但 pip 装的 onnxruntime 在 ``capi/`` 下只有版本化文件
    （如 ``libonnxruntime.so.1.24.4``），既没有 ``libonnxruntime.so.1`` 软链、该目录
    也不在 ``LD_LIBRARY_PATH`` 上。于是 ``register_custom_ops_library`` 报
    「libonnxruntime.so.1: cannot open shared object file」。

    ``LD_LIBRARY_PATH`` 只在进程启动时被链接器读取，运行中改 ``os.environ`` 无效，
    所以那条路必须靠外部脚本（见 ``scripts/run.sh``）。这里换个办法：用
    ``RTLD_GLOBAL`` 先把版本化的库 dlopen 进本进程。dlopen 解析依赖时会先按 SONAME
    查已加载的库，命中就不再走文件系统搜索 —— 于是直接
    ``python -m xmetai_inference`` 也能跑，不必套 run.sh。

    失败一律静默：这只是让「不设 LD_LIBRARY_PATH 也能跑」，真正的报错留给
    register_custom_ops_library 抛（它的信息更准确）。
    """
    import ctypes
    import glob

    try:
        import onnxruntime as _ort

        capi = os.path.join(os.path.dirname(_ort.__file__), "capi")
        # 优先无版本号的软链（若环境已经补过），否则取版本化文件
        for pattern in ("libonnxruntime.so.1", "libonnxruntime.so.1.*"):
            matches = sorted(glob.glob(os.path.join(capi, pattern)))
            if matches:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                return matches[0]
    except Exception:  # noqa: BLE001
        pass
    return None


def _scalar_feeds(valid_time, step):
    """构造 step/hour/doy 标量输入（模型签名里没有的键会被 drop）。"""
    times = [valid_time] if np.ndim(valid_time) == 0 else list(valid_time)
    return {
        "step": np.full((len(times),), step, dtype=np.float32),
        "hour": np.asarray([value.hour / 24.0 for value in times], dtype=np.float32),
        "doy": np.asarray(
            [min(365, value.day_of_year) / 365.0 for value in times],
            dtype=np.float32,
        ),
    }


class OnnxInferModel(BaseInferModel):
    """ONNX Runtime 后端（CUDAExecutionProvider）。"""
    backend = "onnx"
    supports_weight_perturb = True
    # 潜空间 AR(1) 强迫规格（config 的 latent_forcing）；None = 不加。
    # 与 weight_perturb 共用 supports_weight_perturb 这一个能力位：两者都是「改图 / 改权重」
    # 把确定性模型变成集合的手段，且**正交、可叠加**（图改写与权重改写互不干涉）。
    # 分工：m3/m1a 拧的是「噪声幅度 sigma」这条轴，latent_forcing 拧的是噪声的
    # **时间相关**（tau_days）与**空间尺度**（length_scale）—— 见 perturbation/README.md。
    latent_forcing = None
    use_cpu_initializers = True
    # 常驻 session 池的额度（config 的 resident_members）：
    #   1 = 缺省，换模型就把上一份还给 ORT —— 与小显存假设一致的资源行为；
    #   N > 1 = 最多同时驻留 N 份 session（换回来不重建、不重新预热 cuDNN）；
    #   -1 = 不限（全常驻）。
    # 只有「逐成员换权重」的路径（m1a，以及它的分段 E3）会真的反复换模型，所以这个开关
    # 在那些配方上才有意义；确定性模型只有一份 session，池子恒为 1 份。
    resident_members = 1
    # 池子本体与「上一次 load 是否命中池」（cli 的日志要用）。池子在 load() 里惰性建。
    _pool = None
    _pool_hit = False
    # 自定义算子库路径（.so）。某些模型（如 GatedDeltaNet 骨干）用了非标准算子，
    # 必须在建 session 前注册对应 plugin；None = 标准模型无需注册。子类覆盖此属性。
    ops_library = None

    def _make_session(self, path):
        """建一个 ONNX InferenceSession（路径校验 + 选项 + provider + 自定义算子）。

        从 load 抽出，让多会话模型（如 Pangu 的 6h + 24h）能复用同一套配置分别建
        session。只返回会话，不改派生属性（_pred_name/_out_shape/_input_names）。
        """
        # 防御：把 JSON 误当成模型传进来时，直接给明确提示，而不是让 ORT 报晦涩的
        # "Protobuf parsing failed"。--model / 权重文件应指向 .onnx 文件。
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在：{path}")
        if path.lower().endswith(".json"):
            raise ValueError(
                f"拿到的是 JSON 文件（{path}），不是 ONNX 模型。"
                f"模型应是 .onnx 文件。")
        with open(path, "rb") as fh:
            head = fh.read(1)
        if head in (b"{", b"["):
            raise ValueError(
                f"指向的文件是 JSON（{path}），不是 ONNX 模型。"
                f"请改成 .onnx 模型路径。")
        # 关掉 ORT 默认 logger 的 WARNING：模型里那些 ScatterND 提示纯属噪音。
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # 线程预算：不设 intra_op 时 ORT 默认吃满物理核，而本框架会同时起多个 worker
        # 进程（多卡 x procs_per_gpu），不设就是 N 倍超订 —— zarr 解码、CPU 算子、
        # 拷贝都在抢同一批核。inter_op 恒为 1：ORT_SEQUENTIAL 下它本来就没有作用。
        if int(getattr(self, "intra_op_threads", 0) or 0) > 0:
            options.intra_op_num_threads = int(self.intra_op_threads)
        options.inter_op_num_threads = 1
        # ALL 在 EXTENDED 基础上再加 layout（NHWC）优化与更多算子融合，数值等价、零精度损失。
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.use_cpu_initializers:
            # 模型权重留在主机内存，按需送显存 —— 大模型省显存的关键
            options.add_session_config_entry("session.use_device_allocator_for_initializers", "0")
            options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
        options.log_severity_level = 3

        cuda_opts = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
            # 卷积算法穷举搜索：首次推理慢一点，换来每层最优 kernel（之后缓存）。零精度损失。
            # 注意这是 CUDA EP 的 provider option，之前误放成 session config entry 是无效的。
            # 每张卡跑多个进程时，每个 session 都会各搜一遍（热启动开销翻倍），
            # 这时可以把 config 的 cudnn_algo_search 改成 HEURISTIC 换启动速度。
            "cudnn_conv_algo_search": str(
                getattr(self, "cudnn_algo_search", "") or "EXHAUSTIVE"),
        }
        # 只在能拿到真实显存上限时才限制；拿不到（无 torch / 无 CUDA）就干脆不写 gpu_mem_limit。
        # 否则 fallback 成 "0.7" 这种非法值会被 ORT 当成 0 字节上限，把 CUDA 分配器直接压崩。
        limit = _default_gpu_mem_limit(self.device_id, self.gpu_mem_fraction)
        if limit is not None:
            cuda_opts["gpu_mem_limit"] = str(limit)

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(("CUDAExecutionProvider", cuda_opts))
        providers.append(("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        # 自定义算子：注册 .so 后才能建 session（否则报「is not a registered function/op」）。
        # .so 必须与当前 onnxruntime 版本 ABI 匹配，跨环境（尤其容器）运行要先确认。
        if self.ops_library:
            if not os.path.exists(self.ops_library):
                raise FileNotFoundError(f"自定义算子库不存在：{self.ops_library}")
            _preload_onnxruntime_lib()
            options.register_custom_ops_library(self.ops_library)
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    def load(self, path):
        """加载 ONNX 会话，绑定到 self.device_id 这张卡，并做显存/性能调优。

        config 里配了 weight_perturb / latent_forcing 时，先把「源模型路径」换成实际要
        加载的路径，再建 session；建完做一次自检，抓「看着注入了、其实没生效」这类静默
        失败：

          * weight_perturb（逐成员冻结权重）：路径 = self.member_index 那个成员的模型
            （必须已落盘，不自动生成——见 weights.resolve_member_path），自检 = 图里的
            `spw.member` 元数据必须等于 self.member_index（防成员号没接上、51 个成员
            其实加载了同一个文件）。这个自检不跑前向——成员图本来就该是确定性的，
            跑前向反而会为了「两次结果相同」而误判失败。
          * latent_forcing（潜空间 AR(1)）：路径 = 在「按上面解析出来的图」之上再叠加
            spw/xi 输入与强迫链（不存在则构建并缓存），自检 = 图里真的有那个输入、且
            metadata 是 latent-ar1-multiplicative。这一步也不跑前向：随机性来自驱动侧
            每步喂进去的 xi，图的权重一个字节都没动，成员图依然确定性。
        """
        spec = getattr(self, "weight_perturb", None)
        forcing = getattr(self, "latent_forcing", None)
        if spec or forcing:
            from xmetai_inference.perturbation import resolve_stack_path

            # 两步是正交的：先按 SPW 机制解析出「该加载哪份权重」（m1a 会落到成员图），
            # 再在它之上叠加潜空间强迫（只改图结构，一个权重字节都不动）。
            path = resolve_stack_path(
                path, spec, forcing, getattr(self, "member_index", 0))
            if spec:
                log.info("权重扰动已启用：%s member=%d -> %s",
                         spec.describe(), getattr(self, "member_index", 0), path)
            if forcing:
                log.info("潜空间 AR(1) 强迫已启用：%s member=%d -> %s",
                         forcing.describe(), getattr(self, "member_index", 0), path)
        self.session = self._acquire_session(path)
        self._verify_perturbation(spec)
        self._pred_name = self.session.get_outputs()[0].name
        self._out_shape = self.session.get_outputs()[0].shape
        self._input_names = set(i.name for i in self.session.get_inputs())
        self._check_gpu_state_frames()
        self._verify_forcing(forcing)
        # 输出 OrtValue 复用池：forward_gpu 把用完的输入 state 回收为下一个输出 buffer，
        # 避免每步 ortvalue_from_shape_and_type 分配/释放 618MB 显存。首个调用时才新建。
        self._spare = None
        # 潜空间强迫的驱动状态：xi 的递推在 Python 侧（ONNX 单步图没有跨步状态机制），
        # 键 = (成员号, batch, 进程 rank)。load() 里重置，所以每个成员一份、从不共享。
        self._latent_n_targets = 0
        self._ar1 = None
        self._ar1_key = None

    def _verify_perturbation(self, spec):
        """加载后的扰动自检：确认加载进来的图就是 self.member_index 那个成员。

        只有这一件事可查——权重在**生成阶段**就烘进了逐成员的图里，加载期没有任何
        可调的随机性，所以除了「成员号对不对」之外没有别的加载期风险。
        """
        if not spec:
            return
        self._verify_member_identity()

    def _verify_member_identity(self):
        """m1a：确认加载进来的图就是 self.member_index 那个成员。

        只读图里的元数据，不跑前向。抓的是「成员号没接上」——51 个成员全加载同一个
        文件时，输出会完全一样但没有任何报错，是本机制最难发现的静默失败。
        """
        want = int(getattr(self, "member_index", 0))
        meta = dict(self.session.get_modelmeta().custom_metadata_map)
        got = meta.get("spw.member")
        if meta.get("spw.mechanism") != "m1a" or got is None:
            raise RuntimeError(
                f"启用了 m1a，但加载进来的图没有 m1a 的成员元数据"
                f"（spw.mechanism={meta.get('spw.mechanism')!r}, spw.member={got!r}）。"
                "多半是路径指向了源模型或 m3 的扰动图，而不是成员模型。"
                "成员模型路径应由 perturbation.weights.member_model_path 给出。")
        if got != str(want):
            raise RuntimeError(
                f"加载到的成员图编号是 {got}，但本实例要跑的是 member "
                f"{want}。同一份成员被加载多次会产出逐位相同的成员——"
                "请检查 cli 是否在每次 load 前给 model.member_index 赋了当前成员号。")
        log.info("权重扰动自检：member %d，被扰动张量 %s 个，sigma_per_group=%s",
                 want, meta.get("spw.tensors"), meta.get("spw.sigma_per_group"))

    def _verify_forcing(self, forcing):
        """潜空间强迫的加载期自检：图里真的有 spw/xi，且 metadata 对得上。

        抓的是最难发现的那类静默失败：图没被改写（或缓存文件张冠李戴）。驱动侧每步都会把
        xi 塞进 feeds，而 onnxruntime 对**多余的输入是静默忽略**的 —— 于是集合会退化成
        51 份逐位相同的确定性预报，日志里一条错都没有。这里只读 session 的 metadata，
        不跑前向（m1a 的成员图必须是确定性的，跑前向也无法据此判断）。
        """
        self._latent_n_targets = 0
        if not forcing:
            return
        meta = dict(self.session.get_modelmeta().custom_metadata_map)
        targets = [name for name in meta.get("spw.ar1.targets", "").split(" | ") if name]
        if not getattr(forcing, "verify_on_load", True):
            self._latent_n_targets = len(targets)
            return
        if meta.get("spw.ar1.method") != "latent-ar1-multiplicative":
            raise RuntimeError(
                f"启用了 latent_forcing，但加载进来的图没有潜空间强迫的元数据"
                f"（spw.ar1.method={meta.get('spw.ar1.method')!r}）。"
                "多半是路径指向了源模型 / m3 的扰动图 / m1a 的成员图，而不是叠加了 "
                "spw/xi 输入的那份。路径应由 perturbation.latent.resolve_stack_path 给出。")
        if meta.get("spw.ar1.input") != forcing.input_name:
            raise RuntimeError(
                f"图里的强迫输入名是 {meta.get('spw.ar1.input')!r}，但本例要喂的是 "
                f"{forcing.input_name!r}；两边对不上，喂进去只会被静默忽略。")
        if forcing.input_name not in self._input_names:
            raise RuntimeError(
                f"latent_forcing 要喂 {forcing.input_name!r}，但它不在图的输入里"
                f"（{sorted(self._input_names)}）—— 图没被真正改写，集合会退化成逐位相同的"
                "确定性预报。请删掉那份缓存的 .onnx 让它重建。")
        if not targets:
            raise RuntimeError(
                "潜空间强迫的 metadata 里没有任何目标张量（spw.ar1.targets 为空），"
                "说明图是在注入之前就被写出来的，请删掉重新构建。")
        self._latent_n_targets = len(targets)
        log.info("潜空间强迫自检：输入 %s，目标 %d 个，sigma=%s tau=%s 天 rank=%s",
                 forcing.input_name, len(targets), meta.get("spw.ar1.sigma"),
                 meta.get("spw.ar1.tau_days"), meta.get("spw.ar1.rank"))

    def _check_gpu_state_frames(self):
        """gpu_state=True 的前提是「图输出帧数 == 输入帧数」，加载时就查。

        GPU 常驻分支（base.run 的 forward_gpu）绕过子类的 forward override，
        也绕过 numpy 分支里的帧数校验（base.py 的 state.shape[1] != in_frames）。
        对「图只出 1 帧、靠 forward override 手工补成 in_frames 帧」的模型
        （如 iwc_fgvp_gdn2），开 GPU 常驻会把 1 帧当完整 state 回填给需要
        in_frames 帧的图，且 forward_gpu 里的 _spare buffer 复用还假设输入输出
        同形。两者都不会报错，只会算错。所以在加载期直接拦下。
        """
        if not self.gpu_state:
            return
        expected = getattr(self, "history_steps", None)
        if not expected or expected <= 1:
            return
        shape = self._out_shape
        # 只在能确定「图输出 5D 且帧维为 1」时才拦——这是 forward override 补齐
        # 帧数的确凿特征（iwc_fgvp_gdn2）。其余情况（动态轴是字符串、维数不是 5、
        # 帧维是别的值）一律放过：本函数的目的是拦已知的静默错误，不是猜测未知
        # 图的布局，误伤一个能跑的模型比漏拦更糟。
        if not shape or len(shape) != 5 or shape[1] != 1:
            return
        raise ValueError(
            f"{type(self).__name__} 设了 gpu_state=True，但图输出只有 1 帧，"
            f"而输入需要 history_steps={expected} 帧。"
            "该模型靠 forward() override 手工补齐帧数，而 GPU 常驻分支走的是 "
            "forward_gpu()，不经过 override（补齐被跳过），也不做帧数校验；"
            "forward_gpu 的 _spare buffer 复用还假设输入输出同形。开启会静默"
            "算错。请设 gpu_state=False，或为该模型实现 forward_gpu() 并在其中"
            "补齐帧数。"
        )

    def _latent_driver(self, batch, step):
        """取本步要喂的 xi（潜空间 AR(1) 的驱动状态）。

        键 = (成员号, batch, 进程 rank)：成员号让 51 个成员拿到互不相同的随机流（cli 的集合
        编排在每次 load 之前设 self.member_index），batch 让同一步里的多条轨迹各拿一份，
        rank 见下面的注释。step == 0 时重抽 xi_0 —— 每条轨迹（每批起报）都必须从新的噪声
        相位开始，否则新起报会继承上一条轨迹的相位。重抽不重置随机流，所以「重抽 = 新轨迹」，
        而整个任务的抽样序列仍可复现。

        为什么不覆盖 reset_runtime_state：那是「换 session 之后状态还在不在」的契约
        （cli 的 _guard_segment_model 靠它判断分段 rollout 能不能用），而 xi 的递推本来就在
        驱动侧、与 session 无关。用 step 判新轨迹，语义等价且不牵动那条契约。
        """
        forcing = getattr(self, "latent_forcing", None)
        if not forcing or self._latent_n_targets <= 0:
            return None
        from xmetai_inference.perturbation.spec import _rank_offset

        batch = int(batch)
        # 进程 rank 必须计进熵：只开 latent_forcing（没有 m1a 权重）时，集合走的是
        # 「一个 session + 一批轨迹」那条路，成员身份 = 批里的位置、member_index 恒为 0。
        # 多卡按成员切时每个 rank 跑一部分成员，不把 rank 计进熵的话各 rank 的 xi 序列
        # **逐位相同** —— member_000 与 member_001 拿到同一份噪声，静默地变成重复成员。
        rank = int(_rank_offset())
        key = (int(getattr(self, "member_index", 0)), batch, rank)
        if self._ar1 is None or self._ar1_key != key:
            from xmetai_inference.perturbation import Ar1Driver

            self._ar1 = Ar1Driver(forcing, self._latent_n_targets,
                                  member_index=key[0], slot=rank, batch=batch,
                                  hour_interval=getattr(self, "hour_interval", None))
            self._ar1_key = key
        elif int(step) == 0:
            self._ar1.reset()
        return self._ar1

    def forward(self, x, step, valid_time):
        feeds = {"input": _as_numpy_onnx(x)}
        if valid_time is not None:
            for k, v in _scalar_feeds(valid_time, step).items():
                if k in self._input_names:
                    feeds[k] = v
        driver = self._latent_driver(int(x.shape[0]), step)
        if driver is not None:
            feeds[self.latent_forcing.input_name] = driver.value(int(x.shape[0]))
        # 输出完整滑动窗口 (1, in_frames, C, H, W)：输入 [t-1,t] → 输出 [t,t+1]，
        # 末帧才是本步预报。基类 run 会取 state[:, -1] 当预报、用完整 state
        # 回填（state = result），与官方 inference.py 的 `new_input = model.run(...)`
        # 一致。这里原样返回完整输出，不再只切末帧。
        result = self.session.run([self._pred_name], feeds)[0]
        if driver is not None:
            # 递推放在 run 之后：本步喂进去的是 xi_t，下一步该用 xi_{t+1}。
            driver.advance()
        return result

    # ---- GPU 常驻（IOBinding）：state 全程待在 GPU，只在落盘时搬回 CPU ----
    # 相比 forward（每步 numpy H2D + D2H），forward_gpu 用 OrtValue 绑定输入/输出到
    # 显存，省掉每步的 H2D/D2H 与 ORT 反复分配显存的开销。仅当模型前后处理/回填都
    # 是恒等（如 FuXi-Ens）时启用，见 gpu_state。
    def to_gpu(self, x):
        return ort.OrtValue.ortvalue_from_numpy(
            _as_numpy_onnx(x), "cuda", self.device_id)

    def forward_gpu(self, state, step, valid_time):
        # 输出 buffer 复用：上一步 run 用完的输入 state 已在 _spare 里，直接拿它当输出。
        # 输入 state 与本步输出是不同 OrtValue（spare 是上一拍回收的另一个 buffer），
        # run 读输入、写输出不会重叠。shape 与 _out_shape 一致（模型输入输出同形）。
        out = self._spare if self._spare is not None else \
            ort.OrtValue.ortvalue_from_shape_and_type(
                self._out_shape, np.float32, "cuda", self.device_id)
        self._spare = None
        binding = self.session.io_binding()
        binding.bind_ortvalue_input("input", state)
        if valid_time is not None:
            for k, v in _scalar_feeds(valid_time, step).items():
                if k in self._input_names:
                    binding.bind_cpu_input(k, v)
        binding.bind_ortvalue_output(self._pred_name, out)
        self.session.run_with_iobinding(binding)
        # 输入 state 已被 run 读完，回收为下一个输出 buffer。
        self._spare = state
        return out

    def release(self):
        """丢掉当前 session（m1a 逐成员加载时，成员之间调用）。

        m1a 每个成员是一份独立权重，逐成员 load 会不断新建 session；不显式放手的话
        权重副本会一直挂到函数返回（51 份 x ~3 GB）。置 None 后 ORT 引用计数归零、
        显存随之释放；下一次 load() 会重建 session 并重新填好 _pred_name 等派生属性。

        常驻池（config 的 resident_members）下的语义：额度为 1 时连池一起清空 —— 就
        是上面那段原行为，逐位不变；额度 > 1 时池子留着（那正是这个开关的意义），要连
        池一起放手用 release_all()。
        """
        self.session = None
        self._spare = None
        self._ar1 = None
        if self._pool is not None and self._pool.budget == 1:
            self._pool.purge()

    def release_all(self):
        """把当前 session 与整个常驻池一起放开（进程收尾，或想立刻把显存还回去）。"""
        self.session = None
        self._spare = None
        self._ar1 = None
        if self._pool is not None:
            self._pool.purge()
        gc.collect()

    def pool_stats(self):
        """常驻池计数（budget / size / builds / hits / evictions）；日志与自检用。"""
        if self._pool is None:
            return {"budget": normalize_budget(self.resident_members), "size": 0,
                    "builds": 0, "hits": 0, "evictions": 0}
        return self._pool.stats()

    def _acquire_session(self, path):
        """按**解析后的路径**取 session：池里命中就复用，否则新建并入池。

        键是路径而不是成员号，理由见 backends/session_pool.py 的模块注释（扰动/潜空间
        强迫把指纹烘进了文件名，所以同一个路径 = 同一份权重、同一张图）。
        """
        if self._pool is None:
            self._pool = SessionPool(self.resident_members, logger=log)
        session, hit = self._pool.acquire(path, lambda: self._make_session(path))
        self._pool_hit = hit
        if hit:
            log.info("session 命中共驻池（驻留 %d 份，额度 %d）：%s",
                     len(self._pool), self._pool.budget, os.path.basename(path))
        return session

    def to_numpy(self, state):
        return state.numpy()

    def describe(self):
        return f"providers={self.session.get_providers()}"
