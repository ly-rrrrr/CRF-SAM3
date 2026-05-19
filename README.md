# CRF-SAM3 Research Mirror

This repository is a compact research mirror for an RVOS (Referring Video Object Segmentation) project centered on the `crf_sam3_attn` line. The goal is not to train a new segmenter from scratch. The project studies the full chain:

```text
referring expression
  -> category / target noun extraction
  -> SAM3 text prompt
  -> candidate tube recall
  -> offline cache with IoU supervision
  -> candidate reranking and training
  -> top-k mask selection and evaluation
```

The current main line is `Sa2VA-main/crf_sam3_attn`.

## Research Question

Modern RVOS systems often fail before the final mask prediction step: the language model may extract the wrong object noun, produce a weak prompt for SAM-style recall, or create a candidate set that never contains the true referent. This project treats RVOS as a two-stage problem:

1. **Candidate recall quality**: extract the right category noun from the referring expression so SAM3 receives a text prompt close to the real referent.
2. **Candidate reranking**: given a set of SAM3 candidate object tubes, train a model to score and select the candidate or top-k candidates that best match the expression.

The recent key finding is that improving only the category-extraction prompt can substantially improve candidate recall on the Ref-Youtube-VOS failure subset. In other words, the upstream prompt is not a cosmetic detail: it changes the upper bound of downstream reranking.

## Core Idea

The project moves from direct generative localization toward **concept recall plus semantic reranking**.

Traditional RVOS pipelines often ask a vision-language model to directly identify or localize the referred object. This makes a single language or grounding mistake expensive. CRF-SAM3 instead uses a category noun as a high-recall query for SAM3, gathers candidate object tubes, and then asks a candidate-aware discriminator to rank them.

The philosophy is:

- recall all plausible objects first;
- preserve temporal object tubes rather than isolated frames;
- judge each highlighted candidate as the subject of the expression;
- optimize the ranking within the candidate set, not independent yes/no labels only;
- keep candidate-generation artifacts in an offline cache so training is stable and reproducible.

## Main Method

### 1. Concept Extraction

The system extracts:

- `category`: the primary SAM3 recall query;
- `backup_categories`: fallback query nouns;
- `num_targets`: estimated number of referred instances;
- `masked_expression`: expression with the head noun replaced by `[MASK]` in teacher-style prompts.

Important prompt files:

- `Sa2VA-main/configs/category_num_targets_inference_merged.txt`
- `Sa2VA-main/configs/category_extract_teacher_inference.txt`
- `Sa2VA-main/tools/teacher_data_pipeline/teacher_extract_prompt.txt`

The current prompt design emphasizes the head noun of the referred noun phrase, while allowing visually grounded backup categories only as fallback queries.

### 2. SAM3 Candidate Recall

The extracted category is used as the SAM3 text prompt. SAM3 produces object masks over the video, and masks are grouped into candidate tubes by object id.

The cache stores, for each `(video_id, exp_id)`:

- expression text;
- category and SAM3 prompt text;
- candidate object ids;
- per-candidate frame indices;
- `iou_map` against ground truth;
- summary statistics such as `max_j`, `sum_j`, and `mean_j`;
- candidate mask files in `masks/obj_*` directories.

Only cache metadata is tracked in this repository. Large mask images are intentionally ignored.

### 3. Candidate Reranking

The reranker is implemented in `Sa2VA-main/crf_sam3_attn/model.py`.

Each candidate is visualized by overlaying its mask on sampled frames. The model receives the highlighted candidate and an anchor-style Yes/No prompt:

```text
Treat the masked / highlighted object as the anchor subject of the referring expression.
If the expression becomes true when grounded on this object in the frame sequence, answer Yes; otherwise answer No.
```

The main inference score is:

```text
s_i = P(Yes | expression, candidate_i)
```

Training uses listwise ranking over all candidates for the same expression:

```text
L_cls = logsumexp(s_j / tau, j in C) - logsumexp(s_j / tau, j in P)
```

where `C` is the candidate set and `P` is the positive candidate set determined from IoU.

The current training objective combines:

- listwise ranking loss;
- mask-aware soft alignment loss;
- hard-negative rank penalty over all violating negative candidates, instead of only top-k hardest negatives.

### 4. Offline Training

The offline cache decouples expensive SAM3 recall from Qwen3-VL candidate training. The main training command is:

```bash
cd Sa2VA-main
python -u crf_sam3_attn/train.py --config crf_sam3_attn/config_joint_train_train.yaml
```

The joint training config mixes MeViS, DAVIS17, and Ref-Youtube-VOS with dataset-specific IoU thresholds, negative policies, and repeat factors:

- `Sa2VA-main/crf_sam3_attn/config_joint_train_train.yaml`

## Repository Layout

```text
.
|-- README.md
|-- README_transfer_mirror.txt
|-- sample_inventory.json
|-- transfer_mirror_manifest.json
|-- Sa2VA/
|   `-- data/video_datas/
|       |-- mevis/
|       |-- davis17/
|       `-- Ref-Youtube-VOS/
`-- Sa2VA-main/
    |-- crf_sam3_attn/
    |   |-- architecture.md
    |   |-- METHOD.md
    |   |-- config_joint_train_train.yaml
    |   |-- dataset.py
    |   |-- model.py
    |   |-- train.py
    |   |-- train_data.py
    |   `-- offline_cache_*/.../meta.json
    |-- configs/
    |-- tools/eval/
    |-- tools/teacher_data_pipeline/
    |-- crf_sam3_demo.py
    |-- run_crf_sam3_dataset.py
    |-- run_crf_sam3_davis_dataset.py
    |-- run_crf_sam3_refytros_dataset.py
    `-- eval_mapp_sam3.py
```

## Data Policy

The original dataset and generated mask files are too large for normal Git hosting. This repository therefore keeps:

- original directory names and dataset layout documentation;
- sampled inventory metadata;
- offline cache metadata (`meta.json`);
- evaluation summaries and score JSON files.

It does not track:

- raw video frames;
- ground-truth mask images;
- predicted mask PNG files;
- large serialized mask dictionaries;
- model checkpoints or merged weights.

For exact sampled expressions in the local transfer package, see:

- `sample_inventory.json`
- `transfer_mirror_manifest.json`
- `README_transfer_mirror.txt`

The transfer mirror contains sampled content only, not full datasets.

## Evaluation Entry Points

### MeViS

Prediction:

```bash
cd Sa2VA-main
python run_crf_sam3_dataset.py --config configs/crf_sam3_config.yaml
```

Evaluation:

```bash
python eval_mapp_sam3.py \
  --pred_dir crf_sam3_mevis_valid_u_joint_train_poswin_mergedprompt \
  --meta_file /path/to/mevis/valid_u/meta_expressions.json \
  --mask_dict /path/to/mevis/valid_u/mask_dict.json \
  --output evaluation_results.json
```

### DAVIS17

Prediction:

```bash
cd Sa2VA-main
python run_crf_sam3_davis_dataset.py --config configs/crf_sam3_config.yaml
```

Evaluation:

```bash
python tools/eval/eval_davis_png.py \
  --pred_root crf_sam3_davis17_valid_joint_train_poswin_mergedprompt \
  --output crf_sam3_davis17_valid_joint_train_poswin_mergedprompt/evaluation_results_davis_png.json
```

### Ref-Youtube-VOS

Prediction:

```bash
cd Sa2VA-main
python run_crf_sam3_refytros_dataset.py --config configs/crf_sam3_config.yaml
```

Evaluation:

```bash
python tools/eval/eval_refytros_png.py \
  --pred_root crf_sam3_refytros_valid_joint_train_poswin_mergedprompt \
  --output crf_sam3_refytros_valid_joint_train_poswin_mergedprompt/evaluation_results_refytros_png.json
```

## Current Recorded Results

The checked-in result JSON files record the following expression-level metrics:

| Dataset | J | F | J&F | Result File |
|---|---:|---:|---:|---|
| MeViS | 68.83 | 74.07 | 71.45 | `Sa2VA-main/crf_sam3_mevis_valid_u_joint_train_poswin_mergedprompt/evaluation_results.json` |
| DAVIS17 | 72.00 | 76.16 | 74.08 | `Sa2VA-main/crf_sam3_davis17_valid_joint_train_poswin_mergedprompt/evaluation_results_davis_png.json` |
| Ref-Youtube-VOS | 63.38 | 66.04 | 64.71 | `Sa2VA-main/crf_sam3_refytros_valid_joint_train_poswin_mergedprompt/evaluation_results_refytros_png.json` |

These files should be read together with `scores.json`, `frame_coverage.json`, and `expression_details/` under each result directory.

## Important Code Files

- `Sa2VA-main/crf_sam3_attn/architecture.md`: current implementation-aligned method overview.
- `Sa2VA-main/crf_sam3_attn/METHOD.md`: paper-style method description.
- `Sa2VA-main/crf_sam3_attn/dataset.py`: sample construction, offline cache loading, IoU supervision.
- `Sa2VA-main/crf_sam3_attn/model.py`: Qwen3-VL reranker, LoRA, listwise loss, alignment loss.
- `Sa2VA-main/crf_sam3_attn/train.py`: training loop and checkpoint export.
- `Sa2VA-main/crf_sam3_demo.py`: concept extraction, SAM3 recall, visual tube construction, candidate scoring.
- `Sa2VA-main/tools/teacher_data_pipeline/`: teacher-style category extraction pipeline.

## Notes for Paper Writing

This repository supports a paper direction centered on a diagnosis of modern RVOS systems:

- many errors originate in the conversion from expression to recall prompt;
- candidate set quality is the upper bound of downstream ranking;
- SAM-style segmenters are strong recall engines when given a good noun query;
- VLMs are more reliable as candidate judges when the candidate is explicitly highlighted and treated as the sentence anchor;
- offline candidate caches make candidate recall, ranking supervision, and ablation analysis separable.

The intended narrative is a bridge between prompt-driven foundation segmentation and trainable candidate reranking for referring video object segmentation.
