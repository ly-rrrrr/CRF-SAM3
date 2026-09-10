# ReVOS

来源目录包括 `LV-VIS/`、`MOSE/`、`OVIS/`、`TAO/`、`UVO/`。根目录保存 `meta_expressions_train_.json`、`meta_expressions_valid_.json`、`mask_dict.json` 和 `mask_dict_foreground.json`。

评测：`Sa2VA-main/tools/eval/eval_revos.py`。统一训练布局见相邻的 `revos_prepared/`；它不构成另一个数据集。

规模、采样范围与复现要求见 [数据集说明](../../../README.md)。上述代码路径相对仓库根目录；原始数据文件不进入 Git。
