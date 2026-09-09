#!/usr/bin/env bash
set -euo pipefail

cd /mnt/csip-500/share/kongjiawei/Sa2VA-main

PYTHON=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
CONFIG=configs/crf_sam3_config_query_grounding_davis17_valid_semantic_expand_topk5_final_20260703.yaml
DATA_ROOT=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/davis17
OUTPUT_DIR=crf_sam3_davis17_valid_query_grounding_semantic_expand_topk5_final_20260703
META_PATH=${DATA_ROOT}/meta_expressions/valid/meta_expressions.json
MASK_DICT=${DATA_ROOT}/valid/mask_dict.pkl
EVAL_JSON=${OUTPUT_DIR}/evaluation_results_davis_png.json

cat <<EOF
[start] DAVIS17 valid latest scheme with final adapter
[config] ${CONFIG}
[data_root] ${DATA_ROOT}
[output] ${OUTPUT_DIR}
[eval_json] ${EVAL_JSON}
EOF

echo "[run] DAVIS17 valid prediction with final adapter + DMS recall24/keep7 + semantic expansion top_k=5"
CUDA_VISIBLE_DEVICES=3 "${PYTHON}" -u run_crf_sam3_davis_dataset.py \
  --config "${CONFIG}" \
  --data_root "${DATA_ROOT}" \
  --split valid \
  --output_dir "${OUTPUT_DIR}" \
  --skip_existing

echo "[eval] DAVIS17 valid PNG evaluation"
"${PYTHON}" -u tools/eval/eval_davis_png.py \
  --pred_root "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --mask_dict_path "${MASK_DICT}" \
  --output_json "${EVAL_JSON}"

echo "[summary]"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path

path = Path("crf_sam3_davis17_valid_query_grounding_semantic_expand_topk5_final_20260703/evaluation_results_davis_png.json")
if path.exists():
    data = json.load(open(path, "r", encoding="utf-8"))
    print(json.dumps({
        "overall_expression_level": data.get("overall_expression_level"),
        "overall_video_level": data.get("overall_video_level"),
        "coverage": data.get("coverage"),
    }, ensure_ascii=False, indent=2))
else:
    print(f"missing evaluation json: {path}")
PY

echo "[done] DAVIS17 valid latest scheme with final adapter"
