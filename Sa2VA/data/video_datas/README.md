# Video Dataset Layout

This directory records the expected dataset organization for the CRF-SAM3 RVOS experiments.

Tracked in Git:

- this README;
- dataset-specific README files;
- optional `.gitkeep` placeholders.

Not tracked in Git:

- JPEG frames;
- annotation PNG masks;
- `mask_dict.json` / `mask_dict.pkl`;
- full metadata files that belong to the raw dataset packages;
- generated predictions.

The sampled transfer package contained:

| Dataset | Train Samples | Eval Samples |
|---|---:|---:|
| MeViS | 67 | 67 |
| DAVIS17 | 67 | 67 |
| Ref-Youtube-VOS | 66 | 66 |

The exact sampled expressions are listed in `../../../sample_inventory.json`.

Expected top-level layout:

```text
video_datas/
|-- mevis/
|   |-- train/
|   `-- valid_u/
|-- davis17/
|   |-- meta_expressions/
|   |-- train/
|   `-- valid/
`-- Ref-Youtube-VOS/
    |-- train/
    `-- valid/
```
