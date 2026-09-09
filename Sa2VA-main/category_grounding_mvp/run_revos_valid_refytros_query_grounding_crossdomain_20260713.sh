#!/usr/bin/env bash
set -euo pipefail

cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
REVOS_ROOT=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos
QUERY_MODULE=category_grounding_mvp/outputs/qwen3vl_refytvos_query_grounding_no_obj_bf16_stage1_20260630/checkpoint-1453
CATEGORY_CACHE=category_grounding_mvp/outputs/revos_valid_refytros_query_grounding_categories_20260713.json
CONFIG=configs/crf_sam3_config_query_grounding_revos_valid_crossdomain_20260713.yaml
OUT_DIR=crf_sam3_revos_valid_refytros_query_grounding_crossdomain_20260713

printf '[stage 1/3] RefYTROS-trained query grounding -> ReVOS valid category cache\n'
"${PYTHON}" -u category_grounding_mvp/predict_query_grounding_revos_meta.py \
  --model-path /mnt/csip-500/share/kongjiawei/Sa2VA-main/Qwen3-VL-4B-Instruct \
  --query-module-dir "${QUERY_MODULE}" \
  --data-root "${REVOS_ROOT}" \
  --expression-file meta_expressions_valid_.json \
  --output-json "${CATEGORY_CACHE}" \
  --frames-per-sample 3 \
  --max-image-side 336 \
  --max-new-tokens 32 \
  --bf16 \
  --device-map '' \
  --save-every 25 \
  --resume

printf '[stage 2/3] ReVOS valid inference: cross-domain category + final ranker + DMS recall24/keep7\n'
"${PYTHON}" -u run_crf_sam3_revos_dataset.py \
  --config "${CONFIG}" \
  --data_root "${REVOS_ROOT}" \
  --expression_file meta_expressions_valid_.json \
  --output_dir "${OUT_DIR}" \
  --skip_existing

printf '[stage 3/3] Official local ReVOS J/F evaluation\n'
"${PYTHON}" -u tools/eval/eval_revos.py \
  "${OUT_DIR}/results.json" \
  --exp_path "${REVOS_ROOT}/meta_expressions_valid_.json" \
  --mask_path "${REVOS_ROOT}/mask_dict.json" \
  --foreground_mask_path "${REVOS_ROOT}/mask_dict_foreground.json" \
  --save_json_name revos_valid.json \
  --save_csv_name revos_valid.csv

printf '[done] %s/revos_valid.json\n' "${OUT_DIR}"
