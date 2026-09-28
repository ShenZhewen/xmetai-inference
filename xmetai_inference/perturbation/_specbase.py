# -*- coding: utf-8 -*-
"""两个 spec 类共用的样板：配置解析、源身份、指纹。

`WeightPerturbSpec`（spec.py）与 `LatentForcingSpec`（latent.py）在这三件事上原本是逐字
重复的。指纹决定落盘目录名（`spw-members-<指纹>/`、`<stem>.ar1-<指纹>.onnx`），**必须
逐字节稳定**，所以这里只搬同一段代码：按类不同的部分都走钩子（`config_error` 的文案、
`fingerprint_payload` 的附加键），不动序列化结果。
"""
from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from typing import Any, Mapping


class SpecBase:
    """spec 类的公共样板。子类至少要提供 `budgets()`。"""

    #: `from_config` 的类型报错里用的名字。**不要加类型注解**——加了它就变成 dataclass 字段。
    config_error = "spec"

    @classmethod
    def from_config(cls, value: Any):
        """配置字段 -> 规格；None/False 表示不启用。"""
        if value is None or value is False:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        if value is True:
            return cls()
        raise TypeError(
            f"{cls.config_error} 应为 dict / {cls.__name__} / None，"
            f"实际 {type(value).__name__}")

    def budgets(self) -> dict:
        """参与指纹的字段。子类必须实现。"""
        raise NotImplementedError

    def source_identity(self, src_path: str) -> dict:
        """源模型身份：文件换了，成员集 / 改写图就必须失效。"""
        stat = os.stat(src_path)
        return {
            "source": osp.basename(src_path),
            "size": int(stat.st_size),
            "mtime": int(stat.st_mtime),
        }

    def fingerprint_payload(self, src_path: str) -> dict:
        """进指纹的完整载荷。子类可以加键（例如成员集的落盘格式版本）。"""
        return {"spec": self.budgets(), "source_identity": self.source_identity(src_path)}

    def fingerprint(self, src_path: str) -> str:
        payload = self.fingerprint_payload(src_path)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:10]

