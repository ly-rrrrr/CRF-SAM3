#!/usr/bin/env bash
# 画训练曲线 + 打印健康检查。训练进行中也能随时跑。
#   bash scripts/plot.sh                       # 默认读 runs/gv2_revos.log -> runs/gv2_revos.png
#   bash scripts/plot.sh runs/gv2_rowB.log     # 其他 run
#   bash scripts/plot.sh runs/gv2_revos.log runs/gv2_rowB.log --labels E B --out runs/E_vs_B.png
set -euo pipefail; cd "$(dirname "$0")/.."; source scripts/env.sh
if [ $# -eq 0 ]; then set -- "$RUN.log"; fi
$PY -m gv2.plot_log "$@"
