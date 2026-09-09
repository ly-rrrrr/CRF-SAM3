#!/usr/bin/env bash
set -euo pipefail

cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python

exec "$PY" -u category_grounding_mvp/train_query_grounding_qwen3vl.py \
  --model-path /mnt/csip-500/share/kongjiawei/Sa2VA-main/Qwen3-VL-4B-Instruct \
  --train-jsonl category_grounding_mvp/data/revos_query_grounding_api_20260717/train.jsonl \
  --val-jsonl category_grounding_mvp/data/revos_query_grounding_api_20260717/val.jsonl \
  --output-dir category_grounding_mvp/outputs/qwen3vl_revos_query_grounding_api_bf16_stage1_20260805 \
  --num-queries 8 \
  --query-dim 1024 \
  --conditioner-layers 2 \
  --decoder-layers 2 \
  --num-heads 8 \
  --mask-dim 256 \
  --mask-logit-scale 10.0 \
  --modulator-bottleneck 512 \
  --lambda-mask 1.0 \
  --lambda-cat 0.5 \
  --mask-positive-threshold 0.05 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --warmup-steps 60 \
  --logging-steps 10 \
  --eval-steps 300 \
  --save-steps 300 \
  --bf16 \
  --gradient-checkpointing \
  --device-map auto
