#!/usr/bin/env bash
# ReVOS valid 全量推理：类别词缓存（给 CRFSAM3Pipeline）+ 每条样本的热力图 npz（给 p1a 补召回）
set -euo pipefail; cd "$(dirname "$0")/.."; source scripts/env.sh
CKPT=${1:-$RUN}
$PY -u -m gv2.predict --model-path $MODEL --ckpt $CKPT --data-root $REVOS_ROOT \
  --expression-file meta_expressions_valid_.json --frames-per-sample 3 \
  --output-json $CKPT/revos_valid_categories.json --heat-dir $CKPT/heat_revos_valid \
  --bf16 --device-map '' --save-every 25 --resume
