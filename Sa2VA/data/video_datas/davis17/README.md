# DAVIS17 Layout

Expected root:

```text
Sa2VA/data/video_datas/davis17/
```

Expected split layout:

```text
davis17/
|-- meta_expressions/
|   |-- train/
|   |   `-- meta_expressions.json
|   `-- valid/
|       `-- meta_expressions.json
|-- train/
|   |-- JPEGImages/
|   |-- Annotations/
|   `-- mask_dict.pkl
`-- valid/
    |-- JPEGImages/
    |-- Annotations/
    `-- mask_dict.pkl
```

Role in this project:

- Used as a DAVIS-style RVOS benchmark split.
- The main DAVIS17 runner is `Sa2VA-main/run_crf_sam3_davis_dataset.py`.
- The DAVIS17 evaluator is `Sa2VA-main/tools/eval/eval_davis_png.py`.

Local transfer package note:

- `sample_inventory.json` reports 67 sampled train expressions and 67 sampled eval expressions for DAVIS17.
- Raw frames, annotations, and serialized mask dictionaries are intentionally not tracked in Git.
