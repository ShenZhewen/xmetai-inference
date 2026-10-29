# -*- coding: utf-8 -*-
"""启动期校验 ``cli._check_model_batch`` 的三条静态规则。

图是不是动态 batch 轴要 load 读图才知道（``backends/onnx.py`` 设
``batch_axis_dynamic``），不在这一层判；这里只钉住与图无关、配错必然跑不通的组合。
"""
import pytest

from xmetai_inference.cli import _check_model_batch


def test_none_defaults_to_one():
    assert _check_model_batch(None, gpu_state=False, has_recurrent=False) == 1


def test_valid_value_passes_through():
    assert _check_model_batch(6, gpu_state=False, has_recurrent=False) == 6


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_below_one(bad):
    with pytest.raises(SystemExit, match=">= 1"):
        _check_model_batch(bad, gpu_state=False, has_recurrent=False)


def test_rejects_gpu_state_with_recurrent():
    with pytest.raises(SystemExit, match="recurrent"):
        _check_model_batch(1, gpu_state=True, has_recurrent=True)


def test_rejects_batch_gt_one_with_gpu_state():
    with pytest.raises(SystemExit, match="互斥"):
        _check_model_batch(2, gpu_state=True, has_recurrent=False)
