# 按服务器实际路径修改这一个文件即可，其余脚本都 source 它
PY=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python
REPO=/mnt/csip-500/share/kongjiawei/Sa2VA-main
MODEL=$REPO/Qwen3-VL-4B-Instruct
REVOS_ROOT=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos
V1_DATA=$REPO/category_grounding_mvp/data/revos_query_grounding_api_20260717     # v1 的 train.jsonl / val.jsonl
V1_CKPT=$REPO/category_grounding_mvp/outputs/qwen3vl_revos_query_grounding_api_bf16_stage1_20260805/checkpoint-900
DATA=data/revos_v2            # v2 数据输出目录（相对 ground_v2/）
RUN=runs/gv2_revos            # v2 训练输出目录
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
