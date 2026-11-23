#!/usr/bin/env bash
# 热力图辅助扩展 SAM3 候选池：ReVOS valid，用 v2 热力图 + 第 4 步 v2 词的候选池
#
# 把 p1a_spatial_recall/ 放在 ground_v2/ 旁边（同一个上级目录），在 p1a_spatial_recall/ 下运行：
#   GPU=2 bash scripts/revos_v2.sh check            # 第 0 步（必须）：1 分钟自检 SAM3 的检测、点/框跟踪接口
#   GPU=2 LIMIT=10 bash scripts/revos_v2.sh zero    # 试跑 10 条
#   GPU=2 bash scripts/revos_v2.sh zero             # v2 词 0 候选的样本（约 420 条，先跑这个）
#   GPU=2 bash scripts/revos_v2.sh ok               # 300 条本来可用的样本：看误触发会多塞多少候选
#   GPU=2 bash scripts/revos_v2.sh fail             # 有候选但都不可用的样本（约 1,250 条，最慢，可最后跑）
#   bash scripts/revos_v2.sh summary                # 汇总成 runs/revos_v2/summary.md（CPU，随时可跑）
# 每个子集都能断点续跑：重跑同一命令会跳过 results.jsonl 里已有的样本
set -uo pipefail
cd "$(dirname "$0")/.."
GV2=${GV2:-../ground_v2}
source "$GV2/scripts/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
SAM3_CKPT=${SAM3_CKPT:-$REPO/../sam3-main/sam3-main/checkpoints/sam3/sam3.pt}
POOL=${POOL:-$GV2/runs/pool_revos}
HEAT=${HEAT:-$GV2/$RUN/heat_revos_valid}
CATS=${CATS:-$GV2/$RUN/revos_valid_categories.json}
OUT=${OUT:-runs/revos_v2}
mkdir -p "$OUT"
STEP=${1:-}

if [ "$STEP" = check ]; then
  $PY -m p1a.check_sam3_api --dms-repo "$REPO/SAM3-DMS" --sam3-checkpoint "$SAM3_CKPT" \
      --video "$REVOS_ROOT/MOSE/train/08746283" --text cat 2>&1 | grep --line-buffered -v "propagate_in_video" | tee "$OUT/check_sam3_api.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then echo "自检通过，可以试跑：GPU=$CUDA_VISIBLE_DEVICES LIMIT=10 bash scripts/revos_v2.sh zero"
  else echo "自检未通过（退出码 $rc），不要继续。把 $OUT/check_sam3_api.log 发给我"; fi
  exit "$rc"
fi

if [ "$STEP" = summary ]; then
  $PY -m p1a.revos_summary --per-expression "$POOL/per_expression.csv" --runs "$OUT" --out "$OUT/summary.md"
  exit 0
fi

case "$STEP" in zero|fail|ok) ;; *) echo "用法: bash scripts/revos_v2.sh check|zero|ok|fail|summary"; exit 1 ;; esac
for f in "$POOL/per_expression.csv" "$POOL/pools" "$HEAT" "$CATS" "$SAM3_CKPT"; do
  [ -e "$f" ] || { echo "找不到 $f"; exit 1; }
done
[ -s "$OUT/samples_$STEP.json" ] || $PY -m p1a.revos_subsets --per-expression "$POOL/per_expression.csv" --out "$OUT"
MODES=${MODES:-rescue,point,oracle_gt}
[ "$STEP" = ok ] && MODES=${MODES_OK:-rescue,point}
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
GTKW="{\"mask_dict\": \"$REVOS_ROOT/mask_dict.json\", \"meta_expressions\": \"$REVOS_ROOT/meta_expressions_valid_.json\"}"
$PY -u -m p1a.run_p1a --samples "$OUT/samples_$STEP.json" --video-root "$REVOS_ROOT" \
    --heat-dir "$HEAT" --pool-dir "$POOL/pools" --category-json "$CATS" \
    --gt revos --gt-kw "$GTKW" --dms-repo "$REPO/SAM3-DMS" --sam3-checkpoint "$SAM3_CKPT" \
    --modes "$MODES" --det-thresh 0.2 --budget 2 --max-new 3 --offload-video-to-cpu \
    ${LIM[@]+"${LIM[@]}"} --out "$OUT/run_$STEP" 2>&1 | grep --line-buffered -v "propagate_in_video" | tee -a "$OUT/run_$STEP.log"
$PY -m p1a.revos_summary --per-expression "$POOL/per_expression.csv" --runs "$OUT" --out "$OUT/summary.md" > /dev/null
echo "完成 $STEP。汇总在 $OUT/summary.md；把它和 $OUT/run_$STEP.log 最后 30 行发给我"
