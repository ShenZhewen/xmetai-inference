# -*- coding: utf-8 -*-
"""推理运行时：基于合成图的接线自检。

本包只保留推理相关代码：

  * selftest.py   合成图上的图改写 / SPA 数学断言
  * e2e.py        真实框架路径下的端到端接线预检
  * __main__.py   分析与测试入口（probe / self-test / e2e / check / spread / verdict）

图手术与造产物（SPW / SPA）在 `xmetai_inference.weights`。造产物没有
命令行：改模块末尾 `if __name__ == "__main__"` 里的字面量后直接运行

    python -m xmetai_inference.weights.spw     # groups / members
    python -m xmetai_inference.weights.spa     # latent build / audit

离线分析（probe / check / spread / verdict / a(t) 审计 / 逐尺度谱）在
`xmetai_inference.analysis`；分析与测试命令仍由本包统一分发：

    python -m xmetai_inference.perturbation <command> ...
"""


def __getattr__(name):
    """惰性入口：支持 from xmetai_inference.perturbation import run_e2e。

    e2e 会 import runner（torch）与 models，放在模块顶层会让「只想跑合成图自检」
    的调用方也背上这些依赖，所以这里按需解析。
    """
    if name == "run_e2e":
        from .e2e import run_e2e

        return run_e2e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
