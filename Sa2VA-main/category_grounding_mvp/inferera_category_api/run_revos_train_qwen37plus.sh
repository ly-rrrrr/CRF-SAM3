#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${ROOT}/../.." && pwd)
PYTHON=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -u "${ROOT}/generate_revos_bbox_categories.py" \
  --data-root /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos \
  --expression-file meta_expressions_train_.json \
  --model qwen3.7-plus \
  "$@"
