# MeViS Layout

Expected root:

```text
Sa2VA/data/video_datas/mevis/
```

Expected split layout:

```text
mevis/
|-- train/
|   |-- JPEGImages/
|   |-- mask_dict.json
|   `-- meta_expressions.json
`-- valid_u/
    |-- JPEGImages/
    |-- mask_dict.json
    `-- meta_expressions.json
```

Role in this project:

- Used for RVOS training and MeViS validation.
- The main MeViS runner is `Sa2VA-main/run_crf_sam3_dataset.py`.
- The MeViS evaluator is `Sa2VA-main/eval_mapp_sam3.py`.

Local transfer package note:

- `sample_inventory.json` reports 67 sampled train expressions and 67 sampled eval expressions for MeViS.
- Raw frames and masks are intentionally not tracked in Git.
