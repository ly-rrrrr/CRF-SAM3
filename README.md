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
| `Sa2VA-main/crf_sam3_*/` | Evaluation summaries and selected supporting records |
| `Sa2VA/data/video_datas/` | Dataset layout documentation |
| `论文代理/` | Paper figures and analysis |

See [PAPER_EVIDENCE_MAP.md](PAPER_EVIDENCE_MAP.md) for source and result entry points.

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

## Recorded Results

| Dataset / split | J | F | J&F | Evaluation file under `Sa2VA-main/` |
|---|---:|---:|---:|---|
| MeViS valid_u | 68.83 | 74.07 | 71.45 | `crf_sam3_mevis_valid_u_joint_train_poswin_mergedprompt/evaluation_results.json` |
| DAVIS17 valid | 74.06 | 78.73 | 76.39 | `crf_sam3_davis17_valid_query_grounding_semantic_expand_topk5_final_20260703/evaluation_results_davis_png.json` |
| Ref-Youtube-VOS valid (overlay) | 65.23 | 67.72 | 66.47 | `crf_sam3_refytros_valid_query_grounding_semantic_expand_topk5_failures_only_20260702/overlay_evaluation_results_refytros_png.json` |
| ReVOS valid | 57.20 | 62.28 | 59.74 | `crf_sam3_revos_valid_revos_query_ranker_20260806/revos_valid_official.json` |

The first three rows use expression-level metrics. The Ref-Youtube-VOS overlay combines 666 base expressions with 168 patched expressions. ReVOS uses the official overall aggregation; its audit covers 5,822 expressions with zero invalid predictions. Each result is tied to its own run configuration and evaluation protocol.

## Data Policy

Git retains source code, configurations, dataset layout documentation, sampled cache metadata, evaluation summaries, and paper materials. Raw media, mask dictionaries, checkpoints, tokenizer payloads, training logs, generated training sets, and full prediction payloads are excluded.

Datasets include MeViS, DAVIS17, Ref-Youtube-VOS, and ReVOS; `revos_prepared` is a derived ReVOS training layout. `sample_inventory.json` describes local samples, which do not replace full evaluation datasets. Full reproduction requires the original datasets and model weights.
