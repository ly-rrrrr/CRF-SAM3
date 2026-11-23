#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
# Fail before launching any model if CUDA is unavailable; never silently run A on CPU.
for gpu in 2 3; do
 CUDA_VISIBLE_DEVICES="$gpu" /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python -c 'import torch; torch.cuda.init(); print(torch.cuda.get_device_name(0))' || exit 1
done
(CUDA_VISIBLE_DEVICES=2 bash scripts/04_eval.sh v1 > logs/eval_v1_val.log 2>&1; echo $? > logs/eval_v1_val.exitcode) &
a=$!
(CUDA_VISIBLE_DEVICES=3 bash scripts/04_eval.sh runs/gv2_revos > logs/eval_v2_val.log 2>&1; status=$?; echo "$status" > logs/eval_v2_val.exitcode
 if [ "$status" -eq 0 ]; then
 CUDA_VISIBLE_DEVICES=3 SUFFIX=_nosummary bash scripts/04_eval.sh runs/gv2_revos --drop-summary > logs/eval_v2_val_nosummary.log 2>&1
 echo $? > logs/eval_v2_val_nosummary.exitcode
 fi) &
b=$!
wait "$a"
wait "$b"
