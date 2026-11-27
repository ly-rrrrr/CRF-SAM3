#!/usr/bin/env bash
# Explicit server defaults; override any value in the environment for another machine.
REPO=${REPO:-/mnt/csip-500/share/kongjiawei/Sa2VA-main}
PY=${PY:-/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python}
MODEL=${MODEL:-$REPO/Qwen3-VL-4B-Instruct}
REVOS_ROOT=${REVOS_ROOT:-/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos}
RUN=${RUN:-runs/gv2_revos}
DATA=${DATA:-data/revos_v2}
