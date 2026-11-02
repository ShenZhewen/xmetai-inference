# -*- coding: utf-8 -*-
"""Static startup checks for the three rules enforced by ``cli._check_model_batch``.

Whether the graph has a dynamic batch axis can only be determined by loading it
(``backends/onnx.py`` sets ``batch_axis_dynamic``), so it is not validated here;
this module pins only the graph-independent combinations that cannot run when
misconfigured.
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
