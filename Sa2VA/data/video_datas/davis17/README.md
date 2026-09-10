# DAVIS17

`train/`、`valid/` 包含 `JPEGImages/`、`Annotations/` 和 `mask_dict.pkl`；表达式位于 `meta_expressions/<split>/meta_expressions.json`。

推理：`Sa2VA-main/run_crf_sam3_davis_dataset.py`；评测：`Sa2VA-main/tools/eval/eval_davis_png.py`。

规模、采样范围与复现要求见 [数据集说明](../../../README.md)。上述代码路径相对仓库根目录；原始数据文件不进入 Git。
