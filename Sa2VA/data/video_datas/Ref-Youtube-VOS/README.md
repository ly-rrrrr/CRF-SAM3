# Ref-Youtube-VOS Layout

Expected root:

```text
Sa2VA/data/video_datas/Ref-Youtube-VOS/
```

Expected split layout:

```text
Ref-Youtube-VOS/
|-- train/
|   |-- JPEGImages/
|   |-- Annotations/
|   |-- mask_dict.pkl
|   `-- meta_expressions.json
`-- valid/
    |-- JPEGImages/
    |-- Annotations/
    `-- meta_expressions_challenge.json
```

Role in this project:

- Used for Ref-Youtube-VOS / Ref-YTROS-style training and evaluation.
- The main runner is `Sa2VA-main/run_crf_sam3_refytros_dataset.py`.
- The evaluator is `Sa2VA-main/tools/eval/eval_refytros_png.py`.
- This dataset is central to the recent category-extraction prompt study because failure-subset candidate recall improved substantially after prompt refinement.

Local transfer package note:

- `sample_inventory.json` reports 66 sampled train expressions and 66 sampled eval expressions for Ref-Youtube-VOS.
- Raw frames, annotations, and serialized mask dictionaries are intentionally not tracked in Git.
