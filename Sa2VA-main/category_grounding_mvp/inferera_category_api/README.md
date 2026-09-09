# Inferera ReVOS Category API

This directory is an offline annotation utility for ReVOS training supervision.
It sends the complete referring expression and up to three temporally ordered
original RGB frames in chronological order, each with the GT-derived target
bbox overlaid in red. The API returns only:

```json
{"category": "..."}
```

The bbox depends on GT masks. Do not use its outputs in valid/test inference or
reported segmentation evaluation.

## Setup

Fill in `.env` locally:

```bash
INFERERA_API_KEY=your_key_here
```

The script loads this file itself; it never prints the key.

## Small audit run

```bash
nohup bash category_grounding_mvp/inferera_category_api/run_revos_train_qwen37plus.sh \
  --max-samples 50 \
  > category_grounding_mvp/inferera_category_api/logs/revos_train_qwen37plus_50.log 2>&1 &
```

By default, only the final JSON is saved under `results/`; debug images are written only when `--debug-dir` is explicitly supplied. To resume a stopped run, invoke the same command:
completed sample IDs in the JSON result are skipped.
