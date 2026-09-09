#!/usr/bin/env bash
set -euo pipefail

cd /mnt/csip-500/share/kongjiawei/Sa2VA-main

PYTHON=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
BASE_OUT=crf_sam3_refytros_valid_query_grounding_mainline_20260701
PATCH_ONLY_OUT=crf_sam3_refytros_valid_query_grounding_semantic_expand_topk5_failures_only_20260702
CONFIG=configs/crf_sam3_config_query_grounding_refytros_valid_semantic_expand_topk5_failpatch_20260702.yaml
FAIL_META=/mnt/csip-500/share/kongjiawei/Sa2VA-main/category_grounding_mvp/outputs/refytros_valid_mainline_failures_jf_lt_0p5_meta_20260702.json
DATA_ROOT=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS
OVERLAY_JSON=${PATCH_ONLY_OUT}/overlay_evaluation_results_refytros_png.json

mkdir -p "${PATCH_ONLY_OUT}"

cat <<EOF
[start] RefYTROS top-k=5 failure-only patch
[base] ${BASE_OUT}
[patch_only] ${PATCH_ONLY_OUT}
[config] ${CONFIG}
[fail_meta] ${FAIL_META}
EOF

rm -f "${PATCH_ONLY_OUT}/scores.json" \
      "${PATCH_ONLY_OUT}/frame_coverage.json" \
      "${PATCH_ONLY_OUT}/run_summary.json" \
      "${PATCH_ONLY_OUT}/config.yaml" \
      "${OVERLAY_JSON}"

echo "[run] rerun failed expressions with semantic expansion top_k=5"
echo "[run] python=${PYTHON}"
echo "[run] start_ts=$(date '+%F %T')"
set +e
CUDA_VISIBLE_DEVICES=3 "${PYTHON}" -u run_crf_sam3_refytros_dataset.py \
  --config "${CONFIG}" \
  --data_root "${DATA_ROOT}" \
  --split valid \
  --meta_file "${FAIL_META}" \
  --output_dir "${PATCH_ONLY_OUT}"
run_status=$?
set -e
echo "[run] exit_status=${run_status}"
echo "[run] end_ts=$(date '+%F %T')"
if [[ ${run_status} -ne 0 ]]; then
  exit ${run_status}
fi

echo "[eval] overlay evaluation: patch failures over base full result"
"${PYTHON}" category_grounding_mvp/eval_refytros_overlay_png.py \
  --base-root "${BASE_OUT}" \
  --patch-root "${PATCH_ONLY_OUT}" \
  --data-root "${DATA_ROOT}" \
  --split valid \
  --meta-file meta_expressions_challenge.json \
  --output-json "${OVERLAY_JSON}"

echo "[summary]"
"${PYTHON}" - <<'INNERPY'
import json
base = json.load(open("crf_sam3_refytros_valid_query_grounding_mainline_20260701/evaluation_results_refytros_png.json"))
patched = json.load(open("crf_sam3_refytros_valid_query_grounding_semantic_expand_topk5_failures_only_20260702/overlay_evaluation_results_refytros_png.json"))
print(json.dumps({
    "base_expression_level": base["overall_expression_level"],
    "patched_overlay_expression_level": patched["overall_expression_level"],
    "overlay": patched.get("overlay", {}),
    "delta_J": round(patched["overall_expression_level"]["J"] - base["overall_expression_level"]["J"], 4),
    "delta_F": round(patched["overall_expression_level"]["F"] - base["overall_expression_level"]["F"], 4),
    "delta_J&F": round(patched["overall_expression_level"]["J&F"] - base["overall_expression_level"]["J&F"], 4),
}, ensure_ascii=False, indent=2))
INNERPY

echo "[done] RefYTROS top-k=5 failure-only patch"
