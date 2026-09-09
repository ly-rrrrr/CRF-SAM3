#!/usr/bin/env bash
set -euo pipefail

cd /mnt/csip-500/share/kongjiawei/Sa2VA-main

PYTHON=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
OUT_DIR=category_grounding_mvp/data/revos_siglip2_openimages_hierarchy_bestview_20260703

cat <<EOF
[start] ReVOS mask-crop pseudo category retrieval with OpenImages hierarchy, best visible GT-mask frame
[model] siglip2-base-patch16-224
[vocab] category_grounding_mvp/data/openimages_vocab/processed_20260703/categories.jsonl
[retrieval_text] display_prompt_text
[output_prompt] sam3_prompt_text
[crop] bbox crop with original background
[frame_selection] best_quality
[output] ${OUT_DIR}
EOF

CUDA_VISIBLE_DEVICES=3 "${PYTHON}" -u category_grounding_mvp/build_revos_mask_crop_pseudo_category.py \
  --model-path siglip2-base-patch16-224 \
  --meta-expressions /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos/meta_expressions_train_.json \
  --mask-dict /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos/mask_dict.json \
  --image-root /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos \
  --vocab-jsonl category_grounding_mvp/data/openimages_vocab/processed_20260703/categories.jsonl \
  --vocab-text-field display_prompt_text \
  --output-dir "${OUT_DIR}" \
  --frames-per-sample 1 \
  --frame-selection best_quality \
  --min-area 16 \
  --padding-ratio 0.15 \
  --background original \
  --top-k 10 \
  --text-batch-size 256 \
  --device cuda \
  --dtype float16 \
  --resume

echo "[done] ReVOS OpenImages hierarchy best-view pseudo category retrieval"
