#!/usr/bin/env bash
set -euo pipefail
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_v2
export GCKPT=../ground_v2/runs/gv2_revos OUT=runs/refytvos_gv2revos GPU=2
trap 'rc=$?; echo "$rc" > "$OUT/pipeline.exitcode"; date -Is; echo "[pipeline] exit=$rc"' EXIT
printf '%s
' smoke > "$OUT/pipeline.stage"
date -Is
echo '[pipeline] smoke: 3 videos (full heatmap export first)'
LIMIT=3 bash run_v2.sh refytvos
printf '%s
' full > "$OUT/pipeline.stage"
echo '[pipeline] smoke passed; continuing full inference and evaluation'
unset LIMIT
bash run_v2.sh refytvos
printf '%s
' complete > "$OUT/pipeline.stage"
