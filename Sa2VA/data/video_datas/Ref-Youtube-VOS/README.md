# Ref-Youtube-VOS

`train/` 包含 `JPEGImages/`、`Annotations/`、`mask_dict.pkl`、`meta.json` 和 `meta_expressions.json`；`valid/` 包含帧、PNG 标注和 `meta_expressions_challenge.json`。

推理：`Sa2VA-main/run_crf_sam3_refytros_dataset.py`；评测：`Sa2VA-main/tools/eval/eval_refytros_png.py`。

规模、采样范围与复现要求见 [数据集说明](../../../README.md)。上述代码路径相对仓库根目录；原始数据文件不进入 Git。
