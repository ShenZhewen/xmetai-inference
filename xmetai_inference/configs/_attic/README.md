# _attic —— 2026-10-13 清理归档

**只归档、没删**（`mv _attic/<文件> ..` 即可复原）。

这四档都是一次性探针，而且**没有任何一条保留下来的 Job yaml 引用它们**
（它们只被 `tmp/_attic/szw-fgvp-ens-ar1-arms.yaml` / `-b.yaml` 引用，那两个 yaml 也归档了）：

| 档 | 干什么的 |
|---|---|
| `fgvp_ens_ar1.py` | 臂 C 在 **gdn2 旧锚点**上（当前族用的是 20261008 那份导出） |
| `fgvp_ens_ar1hi.py` | σ=20 探针，判"这根线通电吗"；判决已写进 §4.1 第 10–11 轮 |
| `fgvp_ens_ar1state_s04c4.py` | 4 卡桥，量卡数效应；已结算 |
| `fgvp_ens_ar1state_s50a.py` | σ=0.5 哨兵，已结算 |

顶层保留的阶梯档（`s00/s02/s04/s08` `t005/t1/tinf` `l1/l4/l8/l16/l24`）
和位置对照（`head_s04/prior_s04`）都被至少一个在用的 yaml 引用着，**没动**。
