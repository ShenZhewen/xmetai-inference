# -*- coding: utf-8 -*-
"""常驻 session 池：按「解析后的模型路径」做 LRU 驻留，省掉反复建 session。

为什么需要它：m1a 的集合是「一个成员一份冻结权重 = 一个 onnx session」。按成员优先
认领时同一个 worker 会在整轮里反复回到同一批成员；按起报优先认领时更极端 —— 每个任务
都换成员。而 ORT 建一次 session 的代价不只是读一遍权重（latent_mixer 下 1.49 GB），
还有一遍 cuDNN 预热（cudnn_conv_algo_search="EXHAUSTIVE" 时每层卷积都要搜一遍 kernel）。
所以「显存够大就别放手」是有真实收益的，这正是 config 里 resident_members 的含义。

额度语义（config 的 resident_members）：
  1     缺省 = 一次只驻留 1 份（进来第 2 份就把第 1 份还给 ORT），这是小显存的口径；
  N > 1 最多驻留 N 份，第 N+1 份进来时把**最久没用过**的那份换出；
  -1    不限（全常驻）。

池子只存不透明对象（真身是 ``onnxruntime.InferenceSession``），本模块**不 import
onnxruntime** —— 于是它能在没有 ORT 的环境里单独自检（见 backends/selftest.py），
也符合 backends 包的约定（门面不拖 ONNX Runtime，见 backends/__init__.py）。

为什么键是「解析后的路径」而不是成员号：扰动 / 潜空间强迫的每一份变体都把指纹烘进了
文件名（见 perturbation/），所以「同一个路径」就等于「同一份权重、同一张图」。用成员号
当键反而会错：m3 / latent 是一个 session 服务全部成员（member_index 会变、路径不变），
按成员号索引会把同一个 session 存成 51 份。
"""
from __future__ import annotations

import collections
import logging
import os

log = logging.getLogger(__name__)


def normalize_budget(value):
    """resident_members -> 池子额度：0 / None / 非法值都当 1（缺省口径），负数 = 不限。"""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 1
    if value < 0:
        return -1
    return max(1, value)


class SessionPool:
    """路径 -> session 的 LRU 池。额度 1 时退化成「用一个换一个」，与改造前等价。"""

    def __init__(self, budget=1, logger=None):
        self.budget = normalize_budget(budget)
        self.log = logger or log
        self.builds = 0
        self.hits = 0
        self.evictions = 0
        self._items = collections.OrderedDict()

    def acquire(self, key, build):
        """取 session：命中则复用（返回 hit=True），否则调用 ``build()`` 新建并入池。

        ``build`` 是惰性回调（池子自己不知道 session 怎么造），于是本模块与
        onnxruntime 完全解耦。
        """
        key = os.path.abspath(key)
        cached = self._items.pop(key, None)
        if cached is not None:
            self._items[key] = cached          # 重新入尾 = 这条刚被用过（LRU 更新）
            self.hits += 1
            return cached, True
        session = build()
        self.builds += 1
        self._items[key] = session
        self._evict()
        return session, False

    def _evict(self):
        """按额度从 LRU 一端换出；额度 -1 = 不换。"""
        if self.budget < 0:
            return
        dropped = []
        while len(self._items) > self.budget:
            dropped.append(self._items.popitem(last=False)[0])
        if dropped:
            self.evictions += len(dropped)
            self.log.info("常驻 session 池已满（额度 %d），换出 %d 份：%s",
                          self.budget, len(dropped),
                          "、".join(os.path.basename(item) for item in dropped))

    def purge(self):
        """清空池子（驻留的 session 全部放开，显存随之归还）。返回清掉的份数。"""
        count = len(self._items)
        self._items.clear()
        return count

    def __contains__(self, key):
        return os.path.abspath(key) in self._items

    def __len__(self):
        return len(self._items)

    def keys(self):
        """驻留中的路径，按 LRU 排序（最久没用过的在前）。"""
        return list(self._items)

    def stats(self):
        return {
            "budget": self.budget,
            "size": len(self._items),
            "builds": self.builds,
            "hits": self.hits,
            "evictions": self.evictions,
        }


__all__ = ["SessionPool", "normalize_budget"]
