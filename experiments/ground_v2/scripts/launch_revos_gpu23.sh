#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2,3
source scripts/env.sh
mkdir -p runs/gv2_revos logs
set +e
"$PY" -u -m torch.distributed.run --nproc_per_node=2 --master_port=29623 -m gv2.train \
  --model-path "$MODEL" --train-jsonl "$DATA/train.jsonl" --val-jsonl "$DATA/val.jsonl" \
  --max-val-samples 200 --output-dir runs/gv2_revos \
  --num-train-epochs 2 --feature-layer 12 --bf16 --gradient-checkpointing \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 --warmup-ratio 0.03 --max-grad-norm 1.0 \
  --logging-steps 10 --eval-steps 500 --save-steps 500 \
  > runs/gv2_revos.log 2>&1
status=$?
printf '%s\n' "$status" > logs/gv2_revos_gpu23.exitcode
exit "$status"
