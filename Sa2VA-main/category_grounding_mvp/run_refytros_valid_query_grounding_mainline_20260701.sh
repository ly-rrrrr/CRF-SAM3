#!/usr/bin/env bash
set -euo pipefail
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
export CUDA_VISIBLE_DEVICES=3
PYTHON=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
CACHE_JSON=category_grounding_mvp/outputs/refytros_valid_query_grounding_categories_20260701.json
OUT_DIR=/mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_refytros_valid_query_grounding_mainline_20260701
EVAL_JSON=${OUT_DIR}/evaluation_results_refytros_png.json

echo "[stage 1/3] query-grounding category cache"
${PYTHON} category_grounding_mvp/predict_query_grounding_refytros_meta.py \
  --model-path /mnt/csip-500/share/kongjiawei/Sa2VA-main/Qwen3-VL-4B-Instruct \
  --query-module-dir /mnt/csip-500/share/kongjiawei/Sa2VA-main/category_grounding_mvp/outputs/qwen3vl_refytvos_query_grounding_no_obj_bf16_stage1_20260630 \
  --data-root /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS \
  --split valid \
  --meta-file meta_expressions_challenge.json \
  --output-json ${CACHE_JSON} \
  --frames-per-sample 3 \
  --max-new-tokens 64 \
  --bf16 \
  --device-map auto

echo "[stage 2/3] CRF-SAM3 RefYTROS valid prediction with query-grounding category cache"
${PYTHON} run_crf_sam3_refytros_dataset.py \
  --config configs/crf_sam3_config_query_grounding_refytros_valid_20260701.yaml \
  --backend mllm \
  --base_mllm_path /mnt/csip-500/share/kongjiawei/Sa2VA-main/Qwen3-VL-4B-Instruct \
  --mllm_adapter_path /mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_attn_outputs_joint_train_poswin_dms_recall24_keep7_filter_failures_20260623/checkpoint_39000 \
  --sam3_checkpoint /mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt \
  --data_root /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS \
  --split valid \
  --meta_file meta_expressions_challenge.json \
  --output_dir ${OUT_DIR} \
  --num_tube_frames 8 \
  --skip_existing

echo "[stage 3/3] RefYTROS PNG J/F evaluation"
${PYTHON} tools/eval/eval_refytros_png.py \
  --pred_root ${OUT_DIR} \
  --data_root /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS \
  --split valid \
  --meta_file meta_expressions_challenge.json \
  --output_json ${EVAL_JSON}

echo "[done] evaluation json: ${EVAL_JSON}"
