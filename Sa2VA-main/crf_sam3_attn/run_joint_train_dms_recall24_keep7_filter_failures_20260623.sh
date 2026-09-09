#!/usr/bin/env bash
set -euo pipefail

cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python}"
CONFIG="crf_sam3_attn/config_joint_train_train_dms_recall24_keep7_filter_failures_20260623.yaml"
LOG="crf_sam3_attn/logs/joint_train_dms_recall24_keep7_filter_failures_20260623.log"

mkdir -p "$(dirname "$LOG")"
echo "[start] $(date '+%F %T') config=$CONFIG cuda=$CUDA_VISIBLE_DEVICES" | tee "$LOG"
"$PYTHON_BIN" crf_sam3_attn/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
