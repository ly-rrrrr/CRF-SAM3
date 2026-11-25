#!/usr/bin/env bash
# Ref-YouTube-VOS 帧错位检查与修复重跑。放在服务器任意位置，在本目录下运行：
#   bash scripts/run.sh check            # CPU，约 2 分钟：统计受影响帧，并用论文主线的已有预测证明/量化错位
#   GPU=2 bash scripts/run.sh rerun      # GPU，约 1–1.5 小时：主线配置原样重跑，只把掩码按帧位置写出，然后评测
#   bash scripts/run.sh compare          # 打印修复前后对比（rerun 结束时会自动打印）
set -uo pipefail
cd "$(dirname "$0")/.."
HERE=$(pwd)
PY=${PY:-/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python}
REPO=${REPO:-/mnt/csip-500/share/kongjiawei/Sa2VA-main}
REFYT=${REFYT:-/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS}
REVOS_ROOT=${REVOS_ROOT:-/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos}
MAINLINE=${MAINLINE:-$REPO/crf_sam3_refytros_valid_query_grounding_mainline_20260701}
OUT=${OUT:-$HERE/runs}
FIXED=$OUT/mainline_framefix
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

compare() {
  $PY - "$MAINLINE/evaluation_results_refytros_png.json" "$FIXED/evaluation_results_refytros_png.json" <<'EOF' | tee "$OUT/compare.md"
import json, sys
rows = [("论文主线（原写法）", sys.argv[1]), ("主线 + 按帧位置写掩码", sys.argv[2])]
print("| Ref-YTVOS valid | J | F | J&F（表达式级） | J&F（视频级） |\n|---|---|---|---|---|")
for name, p in rows:
    try:
        d = json.load(open(p)); e, v = d["overall_expression_level"], d["overall_video_level"]
        print(f"| {name} | {e['J']:.2f} | {e['F']:.2f} | **{e['J&F']:.2f}** | {v['J&F']:.2f} |")
    except FileNotFoundError:
        print(f"| {name} | 未找到 {p} | | | |")
EOF
}

case "${1:-}" in
  check)
    PRED=(--pred-root "$MAINLINE")
    [ -d "$REPO/crf_sam3_refytros_valid_joint_train_poswin_mergedprompt" ] && PRED+=(--pred-root "$REPO/crf_sam3_refytros_valid_joint_train_poswin_mergedprompt")
    EXTRA=()
    [ -f "$REVOS_ROOT/meta_expressions_valid_.json" ] && EXTRA+=(--revos-meta "$REVOS_ROOT/meta_expressions_valid_.json")
    $PY check_frame_index.py --refytvos-root "$REFYT" --split valid "${PRED[@]}" "${EXTRA[@]}" --out "$OUT/check"
    echo "完成。把 $OUT/check/report.md 发给我" ;;
  rerun)
    mkdir -p "$FIXED"
    (cd "$REPO" && $PY -u "$HERE/run_fixed.py" refytros \
        --config configs/crf_sam3_config_query_grounding_refytros_valid_20260701.yaml \
        --backend mllm \
        --base_mllm_path "$REPO/Qwen3-VL-4B-Instruct" \
        --mllm_adapter_path "$REPO/crf_sam3_attn_outputs_joint_train_poswin_dms_recall24_keep7_filter_failures_20260623/checkpoint_39000" \
        --sam3_checkpoint /mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt \
        --data_root "$REFYT" --split valid --meta_file meta_expressions_challenge.json \
        --output_dir "$FIXED" --num_tube_frames 8 --skip_existing ${LIMIT:+--max_videos $LIMIT}) 2>&1 \
      | tee -a "$OUT/rerun.log" | grep --line-buffered -E "run_fixed|Videos:|Error|Traceback|error"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "重跑报错，把 $OUT/rerun.log 最后 50 行发给我"; exit 1; }
    (cd "$REPO" && $PY -u tools/eval/eval_refytros_png.py --pred_root "$FIXED" --data_root "$REFYT" --split valid \
        --meta_file meta_expressions_challenge.json --output_json "$FIXED/evaluation_results_refytros_png.json") \
      | tail -5
    compare
    echo "完成。把上面的表（$OUT/compare.md）发给我" ;;
  compare) compare ;;
  *) echo "用法: bash scripts/run.sh check|rerun|compare"; exit 1 ;;
esac
