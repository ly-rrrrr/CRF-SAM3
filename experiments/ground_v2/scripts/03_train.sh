#!/usr/bin/env bash
# 完整训练（消融表 E 行）。其他行：在命令末尾追加 README 表里的参数，并改 RUN
set -euo pipefail; cd "$(dirname "$0")/.."; source scripts/env.sh
OUT=${OUT:-$RUN}; mkdir -p "$(dirname "$OUT")"
$PY -u -m gv2.train --model-path $MODEL --train-jsonl $DATA/train.jsonl --val-jsonl $DATA/val.jsonl \
  --max-val-samples 200 --output-dir $OUT --num-train-epochs 2 --feature-layer 12 \
  --bf16 --gradient-checkpointing "$@" 2>&1 | tee -a $OUT.log
