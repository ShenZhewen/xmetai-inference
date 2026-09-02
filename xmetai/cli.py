"""Top-level ``xmetai`` command."""

from __future__ import annotations

import argparse
import sys


def _root_parser():
    parser = argparse.ArgumentParser(
        prog="xmetai",
        description="天气模型推理与评测工具",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["infer", "eval-single", "eval-ens"],
        help="infer=推理，eval-single=确定性评测，eval-ens=集合评测",
    )
    return parser


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _root_parser().print_help()
        return 0

    command, command_args = args[0], args[1:]
    if command == "infer":
        from xmetai.inference import main as command_main
    elif command == "eval-single":
        from xmetai.util.eval_single_util import main as command_main
    elif command == "eval-ens":
        from xmetai.util.eval_ens_util import main as command_main
    else:
        _root_parser().error(
            f"未知命令 {command!r}；可选 infer、eval-single、eval-ens")
    return command_main(command_args)
