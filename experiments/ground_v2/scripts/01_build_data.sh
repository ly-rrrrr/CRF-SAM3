#!/usr/bin/env bash
# 在 v1 的 train/val jsonl 上补别类负样本、同类忽略区、目标缺席帧。看 summary.json 的 records_with_neg_ratio
set -euo pipefail; cd "$(dirname "$0")/.."; source scripts/env.sh
for split in train val; do
  $PY -m gv2.build_data --train-jsonl $V1_DATA/$split.jsonl \
    --meta-expressions $REVOS_ROOT/meta_expressions_train_.json --mask-dict $REVOS_ROOT/mask_dict.json \
    --image-root $REVOS_ROOT --out $DATA/$split.jsonl
  mv $DATA/summary.json $DATA/summary_$split.json
done
