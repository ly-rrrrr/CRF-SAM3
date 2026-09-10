# MeViS

`train/`、`valid_u/` 包含 `JPEGImages/`、`meta_expressions.json` 和 `mask_dict.json`；本地 `valid/` 包含帧与表达式元数据，不含该掩码字典。

推理：`Sa2VA-main/run_crf_sam3_dataset.py`；评测：`Sa2VA-main/eval_mapp_sam3.py`。

规模、采样范围与复现要求见 [数据集说明](../../../README.md)。上述代码路径相对仓库根目录；原始数据文件不进入 Git。
