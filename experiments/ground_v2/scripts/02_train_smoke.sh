#!/usr/bin/env bash
# 冒烟：20 步、50 条，确认显存和 loss 正常下降。OOM 时见 README「显存不够」
set -euo pipefail; cd "$(dirname "$0")/.."; source scripts/env.sh
mkdir -p runs; : > runs/gv2_smoke.log
$PY -u -m gv2.train --model-path $MODEL --train-jsonl $DATA/train.jsonl --output-dir runs/gv2_smoke \
  --max-train-samples 50 --max-steps 20 --logging-steps 1 --save-steps 1000 --bf16 --gradient-checkpointing "$@" \
  2>&1 | tee -a runs/gv2_smoke.log
$PY -m gv2.plot_log runs/gv2_smoke.log     # 曲线 -> runs/gv2_smoke.png
