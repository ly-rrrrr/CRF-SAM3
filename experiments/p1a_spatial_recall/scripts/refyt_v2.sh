#!/usr/bin/env bash
# Ref-YouTube-VOS valid：论文主线 + 热力图点种子（两行一次跑完，掩码都按帧位置写，已避开帧错位）
# 放法同 revos_v2.sh：p1a_spatial_recall/ 与 ground_v2/ 在同一个上级目录，在 p1a_spatial_recall/ 下运行。
#   GPU=2 bash scripts/refyt_v2.sh heat      # 约 20–30 分钟：用 ReVOS 训的 Ground v2 给 834 条导出热力图（只用 Qwen）
#   GPU=2 LIMIT=3 bash scripts/refyt_v2.sh run   # 试跑前 3 个视频（几分钟）
#   GPU=2 bash scripts/refyt_v2.sh run       # 约 1.5–2 小时：主线推理 + 点种子 + Verify（Qwen + SAM3 同卡）
#   bash scripts/refyt_v2.sh eval            # CPU：两行各跑一次 eval_refytros_png.py，写 runs/refyt_v2/summary.md
#   GPU=2 bash scripts/refyt_v2.sh all       # heat → run → eval
# heat 和 run 都能断点续跑
set -uo pipefail
cd "$(dirname "$0")/.."
HERE=$(pwd)
GV2=${GV2:-../ground_v2}
GV2=$(cd "$GV2" && pwd)
source "$GV2/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
REFYT=${REFYT:-/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS}
CONFIG=${CONFIG:-$REPO/configs/crf_sam3_config_query_grounding_refytros_valid_20260701.yaml}
ADAPTER=${ADAPTER:-$REPO/crf_sam3_attn_outputs_joint_train_poswin_dms_recall24_keep7_filter_failures_20260623/checkpoint_39000}
SAM3_CKPT=${SAM3_CKPT:-$REPO/../sam3-main/sam3-main/checkpoints/sam3/sam3.pt}
GCKPT=${GCKPT:-$GV2/$RUN}                                 # Ground v2 权重（ReVOS 训练）
OUT=${OUT:-runs/refyt_v2}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HEAT=$OUT/heat
FIXREF=${FIXREF:-$HERE/../refytvos_frame_fix/runs/mainline_framefix/evaluation_results_refytros_png.json}

step_heat() {
  (cd "$GV2" && $PY -u -m gv2.predict --model-path "$MODEL" --ckpt "$GCKPT" \
      --data-root "$REFYT/valid/JPEGImages" --expression-file "$REFYT/valid/meta_expressions_challenge.json" \
      --frames-per-sample 3 --output-json "$HEAT/refyt_valid_v2_categories.json" --heat-dir "$HEAT/npz" \
      --bf16 --device-map '' --save-every 25 --resume) 2>&1 | tee -a "$OUT/heat.log" | grep --line-buffered -v "^\s*$" | tail -n 3
  local n; n=$(find "$HEAT/npz" -name "*.npz" 2>/dev/null | wc -l)
  echo "热力图：$n / 834"
  [ "$n" -ge 834 ] || { echo "热力图没导完，重跑 heat 会接着跑；仍报错就把 $OUT/heat.log 最后 30 行发给我"; return 1; }
}

step_run() {
  $PY -u -m p1a.refyt_point --repo "$REPO" --config "$CONFIG" --adapter "$ADAPTER" \
      --base-mllm "$MODEL" --sam3-checkpoint "$SAM3_CKPT" --num-tube-frames 8 \
      --data-root "$REFYT" --heat-dir "$HEAT/npz" --out "$OUT" --gt ${LIMIT:+--limit $LIMIT} 2>&1 \
    | tee -a "$OUT/run.log" | grep --line-buffered -E '^\[(refyt|error|[0-9]+/[0-9]+)\]|Error|Traceback'
  return "${PIPESTATUS[0]}"
}

step_eval() {
  for row in base point; do
    echo "== 评测 $row"
    (cd "$REPO" && $PY -u tools/eval/eval_refytros_png.py --pred_root "$OUT/$row" --data_root "$REFYT" --split valid \
        --meta_file meta_expressions_challenge.json --output_json "$OUT/$row/evaluation_results_refytros_png.json") | tail -3
  done
  REFS=(--ref "论文主线（原写法，帧错位）=$REPO/crf_sam3_refytros_valid_query_grounding_mainline_20260701/evaluation_results_refytros_png.json")
  [ -f "$FIXREF" ] && REFS+=(--ref "论文主线（修复帧错位，含 backup 词）=$FIXREF")
  $PY -m p1a.refyt_point summary --out "$OUT" "${REFS[@]}"
  echo "完成。把 $OUT/summary.md 发给我"
}

case "${1:-}" in
  heat) step_heat ;;
  run)  step_run && echo "推理完成，接着跑：bash scripts/refyt_v2.sh eval" ;;
  eval) step_eval ;;
  all)  step_heat && step_run && step_eval ;;
  *) echo "用法: bash scripts/refyt_v2.sh heat|run|eval|all"; exit 1 ;;
esac
