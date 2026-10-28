# -*- coding: utf-8 -*-
"""公共工具层：data/ 负责 Dataset 读取与 DataLoader batch，utils/ 按处理阶段分两侧：

  * ``dataset_processors.py``：**数据侧**，跑在 DataLoader worker，原始 store → 模型输入；
  * ``model_processors.py``：**模型侧**，跑在主进程，自回归回填与输出反变换；
  * ``logging_util.py``：日志配置。
"""
