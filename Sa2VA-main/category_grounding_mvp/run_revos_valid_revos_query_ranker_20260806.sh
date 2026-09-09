#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/mnt/csip-500/share/kongjiawei/Sa2VA-main
cd "${WORKSPACE}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
REVOS_ROOT=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos
QUERY_MODULE=category_grounding_mvp/outputs/qwen3vl_revos_query_grounding_api_bf16_stage1_20260805/checkpoint-900
CATEGORY_CACHE=category_grounding_mvp/outputs/revos_valid_revos_query_grounding_categories_ckpt900_20260806.json
FRAME_CACHE=category_grounding_mvp/outputs/revos_valid_query_grounding_frame_cache_20260806
CONFIG=configs/crf_sam3_config_revos_valid_revos_query_ranker_20260806.yaml
OUT_DIR=crf_sam3_revos_valid_revos_query_ranker_20260806
EVAL_WORKERS="${EVAL_WORKERS:-8}"

mkdir -p logs category_grounding_mvp/outputs "${FRAME_CACHE}"

test -x "${PYTHON}"
test -f "${QUERY_MODULE}/query_grounding_modules.pt"
test -f crf_sam3_attn_outputs_revos_warmstart_3domain_api_categories_recall24_keep7_20260805/final/adapter_model.safetensors
test -f "${REVOS_ROOT}/meta_expressions_valid_.json"
test -f "${REVOS_ROOT}/mask_dict.json"
test -f "${REVOS_ROOT}/mask_dict_foreground.json"

if "${PYTHON}" -c 'import json, sys; records=json.load(open(sys.argv[1])).get("records", []); assert len(records) == 5822 and all(record.get("pred_category") for record in records)' "${CATEGORY_CACHE}" 2>/dev/null; then
  printf '[stage 1/3] category cache already has 5,822 records; skip query inference\n'
else
  printf '[stage 1/3] ReVOS-trained query grounding -> ReVOS valid category cache\n'
  "${PYTHON}" -u category_grounding_mvp/predict_query_grounding_revos_meta.py \
    --model-path Qwen3-VL-4B-Instruct \
    --query-module-dir "${QUERY_MODULE}" \
    --data-root "${REVOS_ROOT}" \
    --expression-file meta_expressions_valid_.json \
    --output-json "${CATEGORY_CACHE}" \
    --frame-cache-root "${FRAME_CACHE}" \
    --source-training-dataset 'ReVOS train' \
    --frames-per-sample 3 \
    --max-image-side 336 \
    --max-new-tokens 32 \
    --bf16 \
    --device-map '' \
    --save-every 25 \
    --resume
fi

printf '[stage 2/3] ReVOS valid: ReVOS category query + ReVOS-warm-start scorer + DMS recall24/keep7\n'
"${PYTHON}" -u run_crf_sam3_revos_dataset.py \
  --config "${CONFIG}" \
  --data_root "${REVOS_ROOT}" \
  --expression_file meta_expressions_valid_.json \
  --output_dir "${OUT_DIR}" \
  --skip_existing

printf '[stage 3/3] Complete prediction audit and official ReVOS evaluation\n'
"${PYTHON}" -u tools/eval/eval_revos_partial_parallel.py \
  "${OUT_DIR}/results.json" \
  --exp-path "${REVOS_ROOT}/meta_expressions_valid_.json" \
  --mask-path "${REVOS_ROOT}/mask_dict.json" \
  --output-json "${OUT_DIR}/revos_valid_jf_audit.json" \
  --output-csv "${OUT_DIR}/revos_valid_jf_audit.csv" \
  --workers "${EVAL_WORKERS}"

"${PYTHON}" -u tools/eval/eval_revos.py \
  "${OUT_DIR}/results.json" \
  --exp_path "${REVOS_ROOT}/meta_expressions_valid_.json" \
  --mask_path "${REVOS_ROOT}/mask_dict.json" \
  --foreground_mask_path "${REVOS_ROOT}/mask_dict_foreground.json" \
  --save_json_name revos_valid_official.json \
  --save_csv_name revos_valid_official.csv \
  --workers "${EVAL_WORKERS}"

printf '[done] %s/revos_valid_official.json\n' "${OUT_DIR}"
