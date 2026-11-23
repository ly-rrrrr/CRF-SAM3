#!/usr/bin/env bash
# CPU 测试：不需要权重和 GPU，约 1 分钟
set -euo pipefail; cd "$(dirname "$0")/.."; source scripts/env.sh
CUDA_VISIBLE_DEVICES= $PY tests/test_gv2.py
