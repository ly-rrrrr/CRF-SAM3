#!/usr/bin/env bash
set -euo pipefail
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
export CUDA_VISIBLE_DEVICES=0
PY=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
DATA_ROOT=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS
SPLIT=valid
META=meta_expressions_challenge.json
BASE_CFG=configs/crf_sam3_config_query_grounding_refytros_valid_20260701.yaml
PROTO_CFG=configs/crf_sam3_config_query_grounding_refytros_valid_lvis_proto_alpha02_20260702.yaml
BASE_OUT=crf_sam3_refytros_valid_sampletest_baseline_20260702
PROTO_OUT=crf_sam3_refytros_valid_sampletest_lvis_proto_alpha02_20260702
SAMPLES=("4f6662e4e0 0" "03fe6115d4 4" "03fe6115d4 2")

run_one() {
  local cfg=$1
  local out=$2
  local vid=$3
  local exp=$4
  echo "[run] cfg=${cfg} sample=${vid}/${exp} out=${out}"
  ${PY} run_crf_sam3_refytros_dataset.py \
    --config "${cfg}" \
    --data_root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --meta_file "${META}" \
    --video_id "${vid}" \
    --exp_id "${exp}" \
    --output_dir "${out}"
  echo "[eval] out=${out} sample=${vid}/${exp}"
  ${PY} tools/eval/eval_refytros_png.py \
    --pred_root "${out}" \
    --data_root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --meta_file "${META}" \
    --video_id "${vid}" \
    --exp_id "${exp}" \
    --output_json "${out}/eval_${vid}_${exp}.json"
}

for item in "${SAMPLES[@]}"; do
  read -r vid exp <<< "${item}"
  run_one "${BASE_CFG}" "${BASE_OUT}" "${vid}" "${exp}"
  run_one "${PROTO_CFG}" "${PROTO_OUT}" "${vid}" "${exp}"
done

echo "[done] sample tests finished"
