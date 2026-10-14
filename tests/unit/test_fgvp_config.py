# -*- coding: utf-8 -*-
"""``configs/fgvp.py`` 的单元测试：配置与模型契约之间的一致性。

这个文件测什么、不测什么
------------------------
config 里写的是一串**声明**（哪个模型类、什么通道、什么单位换算、几张卡、什么时段），
这些声明只在别处被消费时才生效 —— 写错了大多不报错，只是静默不生效：单位换算的键名
拼错一个，那个变量就悄悄不换算；`vars` 里变量名拼错，落盘时就少一个变量或直接没有；
`cuda_devices` 比 `gpus` 短，要到多卡启动时才 IndexError。

所以这里测的是**声明之间的一致性**：config 说的和模型类的契约对不对得上、自身各处
对不对得上。**它证明不了这些值是对的** —— 值本身对不对（比如单位换算系数）要靠
`tests/integration/test_fgvp_realdata.py` 跟标准答案对拍。这里的作用是**变更探测器**：
谁的编辑动了这些约定，测试就红。

不测的：需要真实数据 / ONNX / .so 的东西（那是集成测试的活）。
"""
import re
from datetime import datetime

import pytest

from xmetai_inference.configs.base import InferConfig, load_config, parse_times
from xmetai_inference.models import FUXI_ENS_CHANNELS, get_model_class

#: 模型通道名 → 基础变量名。规则抄自 processed_multi_zarr._raw_channel_name：
#: 只有「字母+数字」结尾的才拆（z50 → z），带尾字母的（t2m/u10m）和纯字母的
#: （ssr/msl/tp）原样保留。
_BASE_NAME_RE = re.compile(r"([A-Za-z]+)(\d+)")

#: 本 config 声明落盘的 11 个变量。
EXPECTED_VARS = {
    "z500", "q700", "t700", "t850", "u850", "v850",
    "u10m", "v10m", "t2m", "msl", "tp",
}


def _base_names(channels):
    """通道名集合 → 基础变量名集合（去层号）。"""
    out = set()
    for name in channels:
        match = _BASE_NAME_RE.fullmatch(name)
        out.add(match.group(1) if match else name)
    return out


@pytest.fixture(scope="module")
def cfg():
    """按生产路径加载配置（`load_config` 而非直接 import）。

    走 `load_config` 才顺带验证了 `processor_specs()` 没有把 model_processing 和旧的
    pre/recurrent/output_processors 同时配上（那会直接抛 ValueError），以及 cfg 确实是
    InferConfig、三个 Processor 都声明全了。
    """
    return load_config("fgvp")


@pytest.fixture(scope="module")
def model_cls(cfg):
    """config **自己声明的**模型类。

    刻意从 `cfg.model_class` 解析而不是写死名字：下面那些通道/变量一致性用例要验的是
    「这份 config 和它声明的模型对不对得上」。若写死，别人把 model_class 改成集合档
    之后，那些用例还在拿旧模型校验，绿得毫无意义 —— 而"改成了别的模型"这件事由
    `test_model_class_is_the_gdn2_backbone` 显式抓。名字拼错则在解析这一层就炸。
    """
    return get_model_class(cfg.model_class)


# ---------------------------------------------------------------------------
# 身份与模型契约
# ---------------------------------------------------------------------------

def test_loads_as_inferconfig_named_fgvp(cfg):
    assert isinstance(cfg, InferConfig)
    assert cfg.name == "fgvp"


def test_model_class_is_the_gdn2_backbone(cfg, model_cls):
    """config 的 model_class 必须解析到 GatedDeltaNet 骨干那一支。

    搞混的代价不小：同目录还有个 `iwc_fgvp_mamba_ens_onnx`（51 成员集合档），
    两者通道布局相同，写错名字不会立刻报错，但推理语义完全变了。
    """
    assert cfg.model_class == "iwc_fgvp_gdn2_onnx"
    assert model_cls.__name__ == "IwcFgvpGdn2Model"
    assert model_cls.windowed_output is False


def test_hour_interval_matches_model(cfg, model_cls):
    """起报间隔 / 时间步长必须和模型契约一致，否则时间轴整体错位。"""
    assert cfg.hour_interval == 6
    assert model_cls.hour_interval == 6


def test_members_is_one_for_deterministic_model(cfg, model_cls):
    """确定性模型必须 members=1。

    这个模型图只出一帧、没有扰动通路，members>1 只会把同一份结果存 N 遍。集合档是
    **另一个模型类**（iwc_fgvp_mamba_ens_onnx），不是靠这里改数字。
    """
    assert model_cls.forecast_type == "deterministic"
    assert model_cls.members == 1
    assert cfg.members == 1


def test_gpu_state_is_off_by_default(cfg, monkeypatch):
    """GPU 常驻默认必须关 —— 它和这个模型的窗口形状不相容。

    `gpu_state=True` 的前提是「输入输出同形」（要靠复用 spare buffer 把 state 留在
    显存里），而本模型输入 2 帧、输出 1 帧（`windowed_output=False`），前提不成立。
    `OnnxInferModel.load` 有启动期断言拦这个组合，但那是**跑到加载模型才炸**；这里
    在配置层面先钉住默认值，让它在收集阶段就红。

    这里必须重新 load 一次（gpu_state 是模块 exec 时读环境变量算出来的，改环境不会
    影响已经建好的对象）。要 load 的是**`cfg` fixture 那份文件**、不是硬编码的
    "fgvp" —— 否则这个用例验的就永远是包内那份原始配置，而不是被测对象。
    """
    monkeypatch.delenv("XMETAI_GPU_STATE", raising=False)
    fresh = load_config(cfg._source_path)
    assert fresh.gpu_state is False, "默认若为真，这个模型跑不起来"


def test_env_override_only_accepts_literal_one(cfg, monkeypatch):
    """`XMETAI_GPU_STATE` 只有字面 "1" 才算开。

    把这个行为写成测试是因为它是个**容易踩的开关**：写 `true` / `yes` / `on` 都当
    没设，静默走默认值（关）。下游想要「开了」却得到「没开」时不会有任何提示。
    """
    monkeypatch.setenv("XMETAI_GPU_STATE", "1")
    assert load_config(cfg._source_path).gpu_state is True

    for falsy in ("true", "yes", "on", "0"):
        monkeypatch.setenv("XMETAI_GPU_STATE", falsy)
        assert load_config(cfg._source_path).gpu_state is False, (
            f"{falsy!r} 不该被当成开"
        )


# ---------------------------------------------------------------------------
# 通道 / 变量的一致性
# ---------------------------------------------------------------------------

def test_declared_vars_are_real_model_output_channels(cfg, model_cls):
    """`vars` 里每个名字都必须是模型真的会输出的通道。

    拼错一个（比如 u850 写成 u0850）不会报错，只是那个变量不落盘 —— 而且往往是
    跑完整年的推理之后才发现。
    """
    declared = {v.strip() for v in cfg.vars.split(",") if v.strip()}
    assert declared == EXPECTED_VARS, "落盘变量清单变了，确认是有意的再改这里"

    available = set(model_cls.output_channels)
    missing = declared - available
    assert not missing, f"vars 里有模型不输出的通道：{sorted(missing)}"


def test_unit_convert_keys_are_real_base_variables(cfg, model_cls):
    """单位换算的键名必须能对上一个真实的基础变量。

    换算表的键是**基础变量名**（`q` 而不是 `q700`）—— 上层变量的层号在读取时才拼上。
    所以判据是把通道名去层号后取集合，再比。键名拼错（ssrd 写成 ssdr）就是静默不
    换算：那个通道以错误单位进模型，误差不会报错，只会让预报变差。
    """
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    scales = processors["unit_convert"]["scales"]

    assert set(scales) == {"q", "ssr", "ssrd", "fdir", "ttr", "tp"}, (
        "换算变量清单变了，确认是有意的"
    )

    unknown = set(scales) - _base_names(model_cls.input_channels)
    assert not unknown, f"换算表里有对不上任何输入通道的键：{sorted(unknown)}"


def test_unit_convert_scales_are_the_agreed_conventions(cfg):
    """钉住换算系数的具体取值。

    这些值是实测对齐过的约定（上游 store 单位 → 模型期望单位），改动会静默改变所有
    预报结果，所以在这里显式锁住。注意这条只能防「无意改动」，防不了「约定本身是错的」
    —— 后者得靠集成测试跟标准答案对拍。
    """
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    scales = processors["unit_convert"]["scales"]

    assert scales["q"] == 1000.0, "q：store kg/kg → 模型 g/kg"
    assert scales["tp"] == 6000.0, "tp：累积 m → 模型 mm"
    for field in ("ssr", "ssrd", "fdir", "ttr"):
        assert scales[field] == pytest.approx(6.0 / 3600.0), f"{field}：辐射场换算"


def test_fill_missing_targets_real_channel_and_is_strict(cfg, model_cls):
    """缺测填充的两条不变量。

    1. 规则里的变量名得是真通道（`sst` 拼错就是静默不填，NaN 一路带进模型）；
    2. `unconfigured` 必须是 "error" —— 这条是安全属性：没配规则的通道一旦缺测就
       报错，而不是悄悄填个默认值。缺测处理最容易出的问题就是「以为填了其实没填」，
       宁可炸。
    """
    processors = {p["name"]: p for p in cfg.dataset["processors"]}
    fill = processors["fill_missing"]

    assert set(fill["rules"]) == {"sst"}, "填充规则的目标变了，确认是有意的"
    unknown = set(fill["rules"]) - _base_names(model_cls.input_channels)
    assert not unknown, f"填充规则里有对不上任何输入通道的变量：{sorted(unknown)}"

    assert fill["rules"]["sst"]["method"] == "constant"
    assert fill["rules"]["sst"]["value"] == pytest.approx(287.066)
    assert fill["unconfigured"] == "error", (
        "未配置的通道缺测必须报错；改成别的值是放宽标准，要显式确认"
    )


def test_model_processing_is_empty(cfg):
    """模型侧三个 Processor 必须都空 —— 归一化烘焙在图里。

    图输入就是物理量，图内自带归一化/反归一化/累积通道清零。这里若挂上任何
    input/output Processor，就会与图内的处理**叠加**（比如反归一化做两遍），
    而数值不会报错，只是全错。
    """
    input_specs, recurrent_specs, output_specs = cfg.processor_specs()
    assert list(input_specs) == []
    assert list(recurrent_specs) == []
    assert list(output_specs) == []


# ---------------------------------------------------------------------------
# 运行参数自洽
# ---------------------------------------------------------------------------

def test_cuda_devices_cover_every_rank(cfg):
    """`cuda_devices` 至少要有 `gpus` 个。

    `cli._run_infer` 多卡分支按 `cuda_devices[rank] for rank in range(gpus)` 取卡，
    短一个就是启动时 IndexError —— 而且只在多卡时才炸，单卡测不出来。
    """
    devices = [d for d in str(cfg.cuda_devices).split(",") if d.strip()]
    assert cfg.gpus >= 1
    assert len(devices) >= cfg.gpus, (
        f"gpus={cfg.gpus} 但只给了 {len(devices)} 个卡号：{cfg.cuda_devices!r}"
    )


def test_times_expand_to_four_daily_inits(cfg):
    """`times` 展开成 4 个每日 00Z 起报。

    这里有个容易看漏的地方：`hour_interval=6`（模型的时间步长）会让**不带 `:N`
    后缀**的区间按 6 小时展开 —— 那不是想要的（要每天一次）。所以 `times`
    结尾那个 `:24` 是必需的，不是冗余。这条测试就是钉住它。
    """
    inits = parse_times(cfg.times, cfg.hour_interval)
    assert inits == [
        datetime(2025, 1, 2), datetime(2025, 1, 3),
        datetime(2025, 1, 4), datetime(2025, 1, 5),
    ], f"起报列表变了：{inits}"


def test_steps_and_interval_span_the_expected_lead_time(cfg):
    """`steps × hour_interval` = 预报跨度，应当是 15 天。

    单独一条是因为这两个数一起决定输出目录里有多少个 `NNN.nc`，改一个忘一个很常见
    （比如想让跨度更长，只把 steps 调大，但 hour_interval 也顺手改了）。
    """
    assert cfg.hour_interval == 6
    assert cfg.steps == 60
    assert cfg.steps * cfg.hour_interval == 15 * 24, "预报跨度应为 15 天"


def test_dynamic_stores_are_the_expected_trio(cfg):
    """三个切片 store 的名单。

    用集合比而不是列表：通道是**按名字**选的，store 的先后顺序不影响结果。这里关心
    的是"是不是这三个"，不是"什么顺序"。
    """
    paths = cfg.dataset["paths"]
    assert cfg.dataset["type"] == "processed_multi_zarr"
    assert len(paths) == 3

    names = {p.replace("\\", "/").rsplit("/", 1)[-1] for p in paths}
    assert names == {
        "era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
        "era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
        "era5_cldrad_2025.01-2026.07.c8.p25.h6.zarr",
    }, f"数据源名单变了：{sorted(names)}"

    # 三个都在同一个根下 —— 换 ERA5_STORE_ROOT 时不会有一个留在旧路径
    roots = {p.replace("\\", "/").rsplit("/", 1)[0] for p in paths}
    assert len(roots) == 1, f"三个 store 不在同一个根下：{sorted(roots)}"
