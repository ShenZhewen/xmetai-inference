# -*- coding: utf-8 -*-
"""Output-variable mapping and the .npy fallback buffer in ``xmetai_inference/cli.py``.

These pin the two silent-failure cases: an unknown ``--vars`` name must fail instead
of dropping the field, and the fallback buffer must contain exactly the channels that
the NetCDF path would have written.
"""
import numpy as np
import pytest

from xmetai_inference.cli import _fallback_buffer, _resolve_output_indices

CHANNELS = ["z500", "t2m", "tp"]


def test_not_requested_saves_every_channel():
    assert _resolve_output_indices(CHANNELS, None) == [0, 1, 2]


def test_names_are_case_insensitive_and_stripped():
    assert _resolve_output_indices(CHANNELS, [" Z500 ", "tp"]) == [0, 2]


def test_unknown_name_raises_with_suggestion():
    with pytest.raises(SystemExit, match="z500"):
        _resolve_output_indices(CHANNELS, ["z_500"])


def test_fallback_buffer_keeps_only_saved_channels():
    buf = np.zeros((3, 2, 2), dtype=np.float32)
    buf[2] = 1.0
    out = _fallback_buffer(buf, [2])
    assert out.shape == (1, 2, 2)
    assert np.all(out == 1.0)


def test_fallback_buffer_is_unchanged_when_all_channels_saved():
    buf = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    out = _fallback_buffer(buf, [0, 1, 2])
    assert out.shape == buf.shape
    assert np.array_equal(out, buf)
