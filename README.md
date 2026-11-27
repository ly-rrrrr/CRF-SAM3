# CRF-SAM3 Research Mirror

A compact research repository for referring video object segmentation (RVOS), combining category grounding, SAM3 candidate recall, and candidate-aware reranking.

```text
referring expression + video frames
  -> category grounding
  -> SAM3 candidate tube recall
  -> candidate-aware visual scoring
  -> mask selection and evaluation
```

## Main Method

- **Category grounding:** learn category queries from referring expressions and supporting frames with Qwen3-VL.
- **Candidate recall:** use SAM3-DMS to recall object tubes; the ReVOS configuration recalls up to 24 candidates and retains up to 7.
- **Candidate reranking:** highlight each candidate in sampled frames and score its compatibility with the expression using a Yes/No discriminator.
- **Offline training:** train with cached candidates and IoU supervision, using listwise ranking, alignment, and hard-negative objectives as configured.

## Repository Layout

| Path | Contents |
|---|---|
| `Sa2VA-main/category_grounding_mvp/` | Category grounding, data construction, and inference |
| `Sa2VA-main/crf_sam3_attn/` | Candidate reranker, training, and cache metadata |
| `Sa2VA-main/SAM3-DMS/` | Candidate recall implementation |
| `Sa2VA-main/configs/` | Dataset and inference configurations |
| `Sa2VA-main/tools/eval/` | Evaluation tools |
| `Sa2VA/data/video_datas/` | Dataset layout documentation |
| `experiments/` | 2026-11 new mainline implementation: Ground v2, heatmap point seeds, unified `crf_sam3_v2` pipeline (standalone) |

## Training and Inference

Run from `Sa2VA-main/` after configuring the data paths, base models, learned weights, and Python environment.

Three-dataset reranker training:

```bash
python -u crf_sam3_attn/train.py --config crf_sam3_attn/config_joint_train_train.yaml
```

ReVOS reranker training:

```bash
bash crf_sam3_attn/run_revos_train_warmstart_3domain_api_categories_recall24_keep7_20260805.sh
```

The ReVOS inference configuration is `configs/crf_sam3_config_revos_valid_revos_query_ranker_20260806.yaml`; evaluation is implemented in `tools/eval/eval_revos.py`.

## Data Policy

See [dataset documentation](Sa2VA/README.md) for splits, annotation layouts, metadata counts, and local sample coverage.

Git retains implementation code, configurations, dataset layout documentation, and sampled cache metadata. Experiment results, evaluation outputs, and paper materials are kept outside the repository. Raw media, mask dictionaries, checkpoints, tokenizer payloads, training logs, generated training sets, and full prediction payloads are excluded.

Datasets include MeViS, DAVIS17, Ref-Youtube-VOS, and ReVOS; `revos_prepared` is a derived ReVOS training layout. `sample_inventory.json` describes local samples, which do not replace full evaluation datasets. Full reproduction requires the original datasets and model weights.
