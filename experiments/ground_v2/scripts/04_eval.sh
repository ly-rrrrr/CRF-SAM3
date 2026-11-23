#!/usr/bin/env bash
# 消融表的前四列：在 18 个留出视频（val.jsonl，1041 条）上测
# 用法：bash scripts/04_eval.sh v1            -> A 行（现有 Ground，原分辨率=它的训练设置）
#       bash scripts/04_eval.sh runs/gv2_revos -> v2；加 --drop-summary 得 E' 行
set -euo pipefail; cd "$(dirname "$0")/.."; source scripts/env.sh
WHO=$1; shift
if [ "$WHO" = v1 ]; then
  $PY -u -m gv2.evaluate --model-path $MODEL --jsonl $DATA/val.jsonl --v1-repo $REPO --v1-query-module-dir $V1_CKPT \
    --max-image-side 0 --bf16 --device-map '' --out runs/eval_v1_val.json "$@"
else
  $PY -u -m gv2.evaluate --model-path $MODEL --jsonl $DATA/val.jsonl --ckpt $WHO --bf16 --device-map '' \
    --out $WHO/eval_val${SUFFIX:-}.json "$@"
fi
