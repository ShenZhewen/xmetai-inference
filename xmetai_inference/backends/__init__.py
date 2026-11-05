# -*- coding: utf-8 -*-
"""Inference backend package facade: re-exports BaseInferModel only.

Do not re-export the onnx/pt2 engines here: onnx.py imports onnxruntime at module
level, so exposing it through the facade would drag ONNX Runtime into any import of
a backends symbol in PT2-only environments.
"""
from .base import BaseInferModel

__all__ = ["BaseInferModel"]
