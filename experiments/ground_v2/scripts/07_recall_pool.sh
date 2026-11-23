#!/usr/bin/env bash
# 第 4 步：用旧 Ground 词和 v2 词各跑一遍 SAM3 召回（论文协议 recall24 -> keep7），统计候选池能不能变好
#
# 用法（在 ground_v2/ 目录下）：
#   GPUS="2" LIMIT=5 bash scripts/07_recall_pool.sh      # 先试 5 个 (视频, 词)，确认能跑通
#   GPUS="2 3" bash scripts/07_recall_pool.sh            # 正式跑：每张卡一个进程分片，约 1,650 次召回
#   bash scripts/07_recall_pool.sh score                 # 只重新打分（召回都跑完后，CPU 即可）
#
# 断点续跑：已经跑完的 (视频, 词) 存在 $OUT/pools/ 里，重跑同一命令会跳过
# 结果：$OUT/report.md（对比表）、per_expression.csv（逐条）、pools/（候选 mask，后面重跑 Verify 用）
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
OUT=${OUT:-runs/pool_revos}
SAM3_CKPT=${SAM3_CKPT:-$REPO/../sam3-main/sam3-main/checkpoints/sam3/sam3.pt}
OLD_CSV=${OLD_CSV:-$REPO/paper_assets/revos_qualitative_20261008/all_cases_diagnostics.csv}
NEW_JSON=${NEW_JSON:-$RUN/revos_valid_categories.json}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
mkdir -p "$OUT/logs"
for f in "$SAM3_CKPT" "$OLD_CSV" "$NEW_JSON" "$REVOS_ROOT/mask_dict.json"; do
  [ -e "$f" ] || { echo "找不到 $f（可用环境变量 SAM3_CKPT / OLD_CSV / NEW_JSON 指定）"; exit 1; }
done
COMMON=(--revos-root "$REVOS_ROOT" --old-csv "$OLD_CSV" --new-json "$NEW_JSON" --out "$OUT")

if [ "${1:-}" != "score" ]; then
  read -ra G <<< "$GPUS"; N=${#G[@]}
  LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
  pids=()
  for i in "${!G[@]}"; do
    echo "启动分片 $i/$N 在 GPU ${G[$i]}，日志 $OUT/logs/recall_$i.log"
    CUDA_VISIBLE_DEVICES=${G[$i]} $PY -u -m gv2.recall_pool recall "${COMMON[@]}" --repo "$REPO" --sam3-ckpt "$SAM3_CKPT" \
      --shard "$i" --num-shards "$N" ${LIM[@]+"${LIM[@]}"} > "$OUT/logs/recall_$i.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "分片 $i 失败，看 $OUT/logs/recall_$i.log 最后几行："; tail -20 "$OUT/logs/recall_$i.log"; fail=1; }
  done
  grep -h "ERROR" "$OUT"/logs/recall_*.log | tail -5
  [ $fail -eq 0 ] || exit 1
fi

$PY -u -m gv2.recall_pool score "${COMMON[@]}" 2>&1 | tee "$OUT/logs/score.log"
echo "完成。把 $OUT/report.md、$OUT/report.json、$OUT/per_expression.csv 发给我（pools/ 先留在服务器上）"
