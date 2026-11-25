#!/usr/bin/env bash
set -euo pipefail
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_v2
source ../ground_v2/scripts/env.sh
export CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1
P=$REPO/crf_sam3_refytros_valid_query_grounding_mainline_20260701
Y=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/valid
OUT=runs/noisy_channel
trap 'rc=$?; echo "$rc" > "$OUT/pipeline.exitcode"' EXIT
JF1=$P/evaluation_results_refytros_png.json
[ ! -f runs/refytvos_v1words/words/evaluation_results_refytros_png.json ] || JF1=runs/refytvos_v1words/words/evaluation_results_refytros_png.json
args=(--model-path "$MODEL" --gckpt ../ground_v2/runs/gv2_revos --data-root "$Y/JPEGImages" --meta "$Y/meta_expressions_challenge.json" --v1-scores "$P/scores.json" --v2-scores runs/refytvos_gv2revos/scores.jsonl --jf-v1 "$JF1" --jf-v2 runs/refytvos_gv2revos/words/evaluation_results_refytros_png.json --out "$OUT")
echo smoke > "$OUT/pipeline.stage"
$PY tools/noisy_channel.py "${args[@]}" --limit 5
$PY runs/noisy_channel/check_complete.py 5
echo full > "$OUT/pipeline.stage"
$PY tools/noisy_channel.py "${args[@]}"
$PY runs/noisy_channel/check_complete.py 241
echo complete > "$OUT/pipeline.stage"
