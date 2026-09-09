#!/usr/bin/env bash
set -euo pipefail

cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${PYTHON_BIN:-/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python}"
CONFIG="crf_sam3_attn/config_revos_train_warmstart_3domain_api_categories_recall24_keep7_20260805.yaml"
LOG="crf_sam3_attn/logs/revos_train_warmstart_3domain_api_categories_recall24_keep7_gpu0_20260805.log"

mkdir -p "$(dirname "$LOG")"
echo "[start] $(date '+%F %T') config=$CONFIG cuda=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
"$PYTHON_BIN" -u crf_sam3_attn/train_resume.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
