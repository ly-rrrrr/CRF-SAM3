# ReVOS 训练布局

`train/`、`valid/` 各包含 `JPEGImages/`、`meta_expressions.json`、`mask_dict.json` 和 `mask_dict_foreground.json`。源服务器通过链接组织原始 ReVOS 视频；本地仅保留抽样帧。

构建：`Sa2VA-main/tools/prepare_revos_mevis_layout.py`。`manifest.json` 记录来源与构建统计，仅在本地保留。此目录与 `revos/` 共享数据，不重复计入数据集数量。

规模、采样范围与复现要求见 [数据集说明](../../../README.md)。上述代码路径相对仓库根目录；原始数据文件不进入 Git。
