#!/usr/bin/env bash
# 最终 J&F：用论文原版 Verify 给三种候选池打分，再跑官方 eval_revos.py
#   旧词池（复现论文，作对照） / v2 词池 / v2 词池 + 热力图点种子
#
# 放法和 revos_v2.sh 一样：p1a_spatial_recall/ 与 ground_v2/ 在同一个上级目录，在 p1a_spatial_recall/ 下运行。
#   GPU=2 bash scripts/verify_v2.sh check     # 第 0 步（约 5 分钟）：只用旧词池和 v2 词池打 20 条，并和论文那次的 scores.json 逐候选对分数
#   bash scripts/verify_v2.sh fidelity        # 只重做上面的比对（CPU，几秒），不重跑模型
#   GPU=2 bash scripts/verify_v2.sh points    # 第 1 步（约 3–4 小时）：全部 5,822 条跑点种子并保存轨迹（只用 SAM3）
#   GPU=2 bash scripts/verify_v2.sh predict   # 第 2 步（约 4–6 小时）：Verify 打分，三行结果一次出（只用 Qwen3-VL）
#   bash scripts/verify_v2.sh eval            # 第 3 步（CPU，约 20 分钟）：官方评测三次，写 runs/verify_v2/compare.md
#   GPU=2 bash scripts/verify_v2.sh all       # 第 1–3 步连着跑（适合放一晚上）
# 第 1、2 步都能断点续跑：中断后重跑同一命令，会跳过已完成的样本
set -uo pipefail
cd "$(dirname "$0")/.."
HERE=$(pwd)
GV2=${GV2:-../ground_v2}
GV2=$(cd "$GV2" && pwd)
source "$GV2/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
SAM3_CKPT=${SAM3_CKPT:-$REPO/../sam3-main/sam3-main/checkpoints/sam3/sam3.pt}
POOL=${POOL:-$GV2/runs/pool_revos}
HEAT=${HEAT:-$GV2/$RUN/heat_revos_valid}
CATS=${CATS:-$GV2/$RUN/revos_valid_categories.json}
CONFIG=${CONFIG:-$REPO/configs/crf_sam3_config_revos_valid_revos_query_ranker_20260806.yaml}
PAPER=${PAPER:-$REPO/crf_sam3_revos_valid_revos_query_ranker_20260806}   # 论文那次的输出目录（有 scores.json 和 revos_valid_official.json）
SUB=${SUB:-runs/revos_v2}             # 样本清单（和 revos_v2.sh 共用）
PTS=${PTS:-runs/revos_v2/point_all}   # 点种子轨迹
OUT=${OUT:-runs/verify_v2}
META=$REVOS_ROOT/meta_expressions_valid_.json
ROWS=${ROWS:-old,v2,v2_point}
mkdir -p "$OUT" "$SUB"
OUT=$(cd "$OUT" && pwd)
STEP=${1:-}
noise() { grep --line-buffered -v "propagate_in_video"; }

need() { for f in "$@"; do [ -e "$f" ] || { echo "找不到 $f"; exit 1; }; done; }

step_points() {
  need "$POOL/per_expression.csv" "$POOL/pools" "$HEAT" "$CATS" "$SAM3_CKPT"
  [ -s "$SUB/samples_all.json" ] || $PY -m p1a.revos_subsets --per-expression "$POOL/per_expression.csv" --out "$SUB" || exit 1
  GTKW="{\"mask_dict\": \"$REVOS_ROOT/mask_dict.json\", \"meta_expressions\": \"$META\"}"
  $PY -u -m p1a.run_p1a --samples "$SUB/samples_all.json" --video-root "$REVOS_ROOT" \
      --heat-dir "$HEAT" --pool-dir "$POOL/pools" --category-json "$CATS" \
      --gt revos --gt-kw "$GTKW" --dms-repo "$REPO/SAM3-DMS" --sam3-checkpoint "$SAM3_CKPT" \
      --modes point --budget 2 --max-new 3 --offload-video-to-cpu --save-tubes \
      --out "$PTS" 2>&1 | noise | tee -a "$PTS.log"
  local n; n=$(wc -l < "$PTS/results.jsonl" 2>/dev/null || echo 0)
  echo "点种子：已完成 $n / 5822 条"
  if [ "$n" -lt 5822 ]; then
    echo "还有样本没完成（多半是个别视频报错）。先重跑一次：GPU=$CUDA_VISIBLE_DEVICES bash scripts/verify_v2.sh points"
    echo "重跑后仍不满 5822，把 $PTS.log 里的 [error] 行发给我"; return 1
  fi
}

step_predict() {  # $1 = 输出目录, 其余参数透传
  local out=$1; shift
  need "$CONFIG" "$POOL/per_expression.csv" "$POOL/pools" "$META"
  mkdir -p "$out"
  $PY -u -m p1a.verify_pool predict --repo "$REPO" --config "$CONFIG" --revos-root "$REVOS_ROOT" --meta "$META" \
      --per-expression "$POOL/per_expression.csv" --pool-dir "$POOL/pools" --point-dir "$PTS" \
      --out "$out" "$@" 2>&1 | tee -a "$out/predict.log" | grep --line-buffered -E '^\[(verify|error|[0-9]+/[0-9]+)\]|Error|Traceback'   # 完整输出在 predict.log
  return "${PIPESTATUS[0]}"
}

step_eval() {
  need "$OUT/scores.jsonl"
  $PY -m p1a.verify_pool merge --meta "$META" --rows "$ROWS" --out "$OUT"
  [ $? -eq 2 ] && echo "注意：有样本没跑完，按空掩码计分（见 $OUT/<行>/missing.txt）。最好先重跑 predict 补齐。"
  for row in ${ROWS//,/ }; do
    echo "== 官方评测 $row"
    (cd "$REPO" && $PY -u tools/eval/eval_revos.py "$OUT/$row/results.json" \
        --exp_path "$META" --mask_path "$REVOS_ROOT/mask_dict.json" \
        --foreground_mask_path "$REVOS_ROOT/mask_dict_foreground.json" \
        --save_json_name revos_valid_official.json --save_csv_name revos_valid_official.csv \
        --workers "${EVAL_WORKERS:-8}") || { echo "评测 $row 失败"; return 1; }
  done
  $PY -m p1a.verify_pool compare --meta "$META" --rows "$ROWS" --out "$OUT" --paper-json "$PAPER/revos_valid_official.json"
  echo "完成。结果表在 $OUT/compare.md，把它发给我"
}

case "$STEP" in
  check)
    rm -rf "$OUT/check"
    step_predict "$OUT/check" --rows old,v2 --limit "${LIMIT:-20}" || { echo "试跑报错，把 $OUT/check/predict.log 最后 50 行发给我"; exit 1; }
    if [ -f "$PAPER/scores.json" ]; then
      $PY -m p1a.verify_pool fidelity --meta "$META" --out "$OUT/check" --paper-scores "$PAPER/scores.json"
      rc=$?
      [ $rc -eq 0 ] && echo "自检通过：GPU=$CUDA_VISIBLE_DEVICES bash scripts/verify_v2.sh all" || echo "自检没通过，把上面几行 [fidelity] 发给我"
      exit $rc
    else
      echo "没找到论文那次的 $PAPER/scores.json，跳过逐候选比对（可用 PAPER=<目录> 指定）。试跑本身正常。"
    fi ;;
  fidelity)  # 只重做比对（CPU，几秒）：用已有的 check 结果
    need "$OUT/check/scores.jsonl" "$PAPER/scores.json"
    $PY -m p1a.verify_pool fidelity --meta "$META" --out "$OUT/check" --paper-scores "$PAPER/scores.json" ;;
  points)  step_points ;;
  predict) step_predict "$OUT" --rows "$ROWS" && echo "打分完成，接着跑：bash scripts/verify_v2.sh eval" ;;
  eval)    step_eval ;;
  all)     step_points && step_predict "$OUT" --rows "$ROWS" && step_eval ;;
  *) echo "用法: bash scripts/verify_v2.sh check|fidelity|points|predict|eval|all"; exit 1 ;;
esac
