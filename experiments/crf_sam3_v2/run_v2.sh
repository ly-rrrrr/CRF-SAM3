#!/usr/bin/env bash
# CRF-SAM3 新主线，一个入口：Ground v2（按数据集训练）→ 词 + 热力图 → SAM3 召回 → 热力图点种子 → Verify
# → 按帧位置写掩码 → 官方评测。所有数据集用同一份代码（v2/mainline.py）。
#
# 放法：crf_sam3_v2/、ground_v2/、p1a_spatial_recall/ 放在同一个上级目录（就是你现在的「论文」目录结构），
# 在 crf_sam3_v2/ 下运行：
#   bash run_v2.sh test                      # CPU 自检，约 1 分钟
#   bash run_v2.sh inventory                 # 只读盘点（数据量、缓存、权重、磁盘），只用 CPU，结果在 runs/inventory.md
#   GPU=2 LIMIT=3 bash run_v2.sh refytvos    # 试跑：只跑前 3 个视频（Ground v2 没训好时会先训练，见下）
#   GPU=2 bash run_v2.sh refytvos            # 全流程：训练 Ref-YTVOS 的 Ground v2（GPU=2,3 时两卡训练）→ 热力图（约 30 分钟）
#                                            #          → 主线推理（约 2 小时）→ 评测，结果在 runs/refytvos/summary.md
#   GPU=2 bash run_v2.sh revos               # 全流程：ReVOS 的 Ground v2 已训好，热力图已有就复用 → 主线推理（10 小时以上）→ 评测
#   GPU=2 bash run_v2.sh refytvos ground     # 也可以只跑一步：ground | heat | run | eval
# 可用环境变量改的东西：
#   GCKPT=<Ground v2 目录>   用哪份 Ground v2（有 gv2_config.json 就不训练），例如 GCKPT=../ground_v2/runs/gv2_revos
#   OUT=<输出目录>           默认 runs/<数据集>；换一份 Ground 时记得换 OUT，免得和别的结果混在一起
#   CONFIG=<主线 yaml>       Verify 权重、SAM3-DMS、recall/keep 个数都在 yaml 里
#   RUN_ARGS="..."           透传给 v2/mainline.py，例如 RUN_ARGS="--keep-backup" 打开 backup 词
#   WORD_K=2|3               Ground 选词时取前 K 个词，用反向打分（noisy-channel）重排；不设或 0 = 只用贪心词（原做法）
#                            重排结果单独存成 ground/categories_k<K>_<打分方式>.json
#   WORD_SCORE=yesno|rev     重排怎么打分：yesno 直接问「是不是指代对象」（默认），rev 反向打分
#   GPU / LIMIT / SAM3_CKPT / REFYT / EVAL_WORKERS
# 每一步都能断点续跑：中断后重跑同一条命令即可（训练除外：训练中断会从头开始）。
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
GV2=$(cd "${GV2:-../ground_v2}" && pwd)
P1A=$(cd "${P1A:-../p1a_spatial_recall}" && pwd)
source "$GV2/scripts/env.sh"                 # PY REPO MODEL REVOS_ROOT RUN ...
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$HERE:$P1A${PYTHONPATH:+:$PYTHONPATH}"
REFYT=${REFYT:-/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS}
SAM3_CKPT=${SAM3_CKPT:-/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt}

DS=${1:-}; STEP=${2:-all}
case "$DS" in
  refytvos)
    CONFIG=${CONFIG:-$REPO/configs/crf_sam3_config_query_grounding_refytros_valid_20260701.yaml}
    GCKPT=${GCKPT:-$GV2/runs/gv2_refytvos}
    GDATA=$GV2/data/refytvos_v2
    VAL_ROOT=$REFYT/valid/JPEGImages
    VAL_META=$REFYT/valid/meta_expressions_challenge.json
    DATA_ROOT=$REFYT
    REFS=(--ref "论文主线（原写法，帧错位）=$REPO/crf_sam3_refytros_valid_query_grounding_mainline_20260701/evaluation_results_refytros_png.json")
    FIX=$HERE/../refytvos_frame_fix/runs/mainline_framefix/evaluation_results_refytros_png.json
    [ -f "$FIX" ] && REFS+=(--ref "论文主线（只修帧错位）=$FIX") ;;
  revos)
    CONFIG=${CONFIG:-$REPO/configs/crf_sam3_config_revos_valid_revos_query_ranker_20260806.yaml}
    GCKPT=${GCKPT:-$GV2/$RUN}
    GDATA=$GV2/$DATA
    VAL_ROOT=$REVOS_ROOT
    VAL_META=$REVOS_ROOT/meta_expressions_valid_.json
    DATA_ROOT=$REVOS_ROOT
    REFS=(--ref "论文结果=$REPO/crf_sam3_revos_valid_revos_query_ranker_20260806/revos_valid_official.json")
    VP=$P1A/runs/verify_v2/v2_point/revos_valid_official.json
    [ -f "$VP" ] && REFS+=(--ref "上一轮拼接实验（旧词池 ∪ v2 词池 + 点种子）=$VP") ;;
  test)
    $PY "$HERE/tests/test_mainline.py" && $PY "$HERE/tests/test_inventory.py" && $PY "$HERE/tests/test_ground_rerank.py" \
      && (cd "$GV2" && CUDA_VISIBLE_DEVICES= $PY tests/test_build_refytvos.py)
    exit $? ;;
  inventory)                                 # 只读盘点，只用 CPU，结果在 runs/inventory.md
    CUDA_VISIBLE_DEVICES= $PY "$HERE/tools/inventory.py" --repo "$REPO" --revos-root "$REVOS_ROOT" \
      --gv2 "$GV2" --p1a "$P1A" --out "$HERE/runs/inventory.md"
    exit $? ;;
  *) echo "用法: bash run_v2.sh refytvos|revos [all|ground|heat|run|eval]   或   bash run_v2.sh test|inventory"; exit 1 ;;
esac
OUT=${OUT:-$HERE/runs/$DS}
abs() { case "$1" in /*) echo "$1" ;; *) echo "$HERE/$1" ;; esac; }   # 相对路径一律按 crf_sam3_v2/ 解释
OUT=$(abs "$OUT"); GCKPT=$(abs "$GCKPT"); CONFIG=$(abs "$CONFIG")
mkdir -p "$OUT"
GOUT=$OUT/ground                            # Ground v2 在 valid 上的词和热力图
NEXP=$($PY -c "import json,sys; v=json.load(open(sys.argv[1]))['videos']; print(sum(len(x['expressions']) for x in v.values()))" "$VAL_META")
need() { for f in "$@"; do [ -e "$f" ] || { echo "找不到 $f"; exit 1; }; done; }

# ---------------------------------------------------------------- 1. Ground v2（按数据集训练）
step_ground() {
  if [ -f "$GCKPT/gv2_config.json" ]; then echo "[ground] 已有训练好的 Ground v2：$GCKPT，跳过训练"; return 0; fi
  if [ "$DS" = revos ]; then
    echo "[ground] 训练 ReVOS 的 Ground v2（和之前 ground_v2/scripts/03_train.sh 相同）"
    [ -s "$GDATA/train.jsonl" ] || (cd "$GV2" && bash scripts/01_build_data.sh) || return 1
  else
    local V1=$REPO/category_grounding_mvp/data/refytvos_query_grounding
    if [ ! -s "$V1/train.jsonl" ]; then
      echo "[ground] 先生成 v1 的 Ref-YTVOS 训练清单（仓库自带脚本，CPU，几分钟）"
      (cd "$REPO" && $PY category_grounding_mvp/build_refytvos_query_grounding.py \
          --meta-expressions "$REFYT/train/meta_expressions.json" --meta-objects "$REFYT/train/meta.json" \
          --image-root "$REFYT/train/JPEGImages" --mask-dict "$REFYT/train/mask_dict.pkl" --output-dir "$V1") || return 1
    fi
    if [ ! -s "$GDATA/val.jsonl" ]; then
      echo "[ground] 补别类负样本 / 同类忽略区 / 目标缺席帧（CPU，约 10–20 分钟）"
      for split in train val; do
        (cd "$GV2" && $PY -m gv2.build_refytvos --v1-jsonl "$V1/$split.jsonl" --ytvos-train "$REFYT/train" \
            --out "$GDATA/$split.jsonl") || return 1
        mv "$GDATA/summary.json" "$GDATA/summary_$split.json"
      done
    fi
    local iou; iou=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['target_png_iou_mean'])" "$GDATA/summary_train.json")
    echo "[ground] v1 目标掩码与 Annotations PNG 的平均 IoU：$iou（应接近 1）"
    $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.9 else 1)" "$iou" || {
      echo "目标掩码和 PNG 对不上，先别训练。把 $GDATA/summary_train.json 发给我"; return 1; }
  fi
  # 和 ReVOS 那次（launch_revos_gpu23.sh）同一套超参；等效 batch 固定为 8：单卡累积 8 步，两卡各累积 4 步
  local ngpu acc launch=()
  ngpu=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  acc=$((8 / ngpu))
  [ "$ngpu" -gt 1 ] && launch=(-m torch.distributed.run --nproc_per_node="$ngpu" --master_port="${PORT:-29624}")
  echo "[ground] 训练：$ngpu 张卡，每卡累积 $acc 步（另开终端 cd $GV2 && $PY -m gv2.plot_log $GCKPT.log 看曲线）"
  mkdir -p "$(dirname "$GCKPT")"
  (cd "$GV2" && $PY -u ${launch[@]+"${launch[@]}"} -m gv2.train --model-path "$MODEL" \
      --train-jsonl "$GDATA/train.jsonl" --val-jsonl "$GDATA/val.jsonl" --max-val-samples 200 --output-dir "$GCKPT" \
      --num-train-epochs 2 --feature-layer 12 --bf16 --gradient-checkpointing \
      --per-device-train-batch-size 1 --gradient-accumulation-steps "$acc" --learning-rate 1e-4 --warmup-ratio 0.03 \
      --max-grad-norm 1.0 --logging-steps 10 --eval-steps 500 --save-steps 500) 2>&1 \
    | tee -a "$GCKPT.log" | grep --line-buffered -E "^\[gv2\]|Error|Traceback|'loss'"
  [ -f "$GCKPT/gv2_config.json" ] || { echo "训练没有正常结束，把 $GCKPT.log 最后 50 行发给我"; return 1; }
  echo "[ground] 在留出视频上评测 Ground v2（词正确率、热力图落在目标上的比例）"
  (cd "$GV2" && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} $PY -u -m gv2.evaluate --model-path "$MODEL" --jsonl "$GDATA/val.jsonl" --ckpt "$GCKPT" \
      --bf16 --device-map '' --out "$GCKPT/eval_val.json") 2>&1 | tail -n 15
}

# ---------------------------------------------------------------- 2. valid 上的词和热力图
CATS=$GOUT/categories.json; HEAT=$GOUT/npz
if [ "$DS" = revos ] && [ -z "${FRESH_HEAT:-}" ] && [ "$GCKPT" = "$GV2/$RUN" ] && [ -f "$GCKPT/revos_valid_categories.json" ]; then
  CATS=$GCKPT/revos_valid_categories.json; HEAT=$GCKPT/heat_revos_valid      # 之前 05_predict_revos_valid.sh 已导出，直接复用
fi
step_heat() {
  need "$GCKPT/gv2_config.json"
  local n; n=$(find "$HEAT" -name "*.npz" 2>/dev/null | wc -l)
  if [ "$n" -ge "$NEXP" ] && [ -f "$CATS" ]; then echo "[heat] 已有 $n 个热力图：$HEAT，跳过"; return 0; fi
  (cd "$GV2" && $PY -u -m gv2.predict --model-path "$MODEL" --ckpt "$GCKPT" --data-root "$VAL_ROOT" \
      --expression-file "$VAL_META" --frames-per-sample 3 --output-json "$CATS" --heat-dir "$HEAT" \
      --bf16 --device-map '' --save-every 25 --resume) 2>&1 | tee -a "$OUT/heat.log" | grep --line-buffered -E "^\[[0-9]+/[0-9]+\]|Error|Traceback" | awk 'NR%50==1 {print; fflush()}'
  n=$(find "$HEAT" -name "*.npz" 2>/dev/null | wc -l)
  echo "[heat] 热力图 $n / $NEXP"
  [ "$n" -ge "$NEXP" ] || { echo "热力图没导完，重跑会接着跑；仍报错就把 $OUT/heat.log 最后 30 行发给我"; return 1; }
}

# ---------------------------------------------------------------- 2b. 选词重排（WORD_K>0 时）
RUN_CATS=$CATS
WORD_K=${WORD_K:-0}
[ "$WORD_K" -gt 0 ] && RUN_CATS=$GOUT/categories_k${WORD_K}_${WORD_SCORE:-yesno}.json
step_rerank() {
  [ "$WORD_K" -gt 0 ] || return 0
  need "$CATS" "$GCKPT/gv2_config.json"
  local n; n=$($PY -c "import json,sys; print(sum('candidates' in r for r in json.load(open(sys.argv[1]))['records']))" "$RUN_CATS" 2>/dev/null || echo 0)
  if [ "$n" -ge "$NEXP" ]; then echo "[rerank] 已有 K=$WORD_K 的重排结果（$n 条），跳过"; return 0; fi
  echo "[rerank] Ground 取前 $WORD_K 个词，反向打分重排（单卡，约 20–40 分钟）"
  (cd "$GV2" && CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} PYTHONPATH="$GV2:$HERE${PYTHONPATH:+:$PYTHONPATH}" \
      $PY -u -m v2.ground_rerank --model-path "$MODEL" --ckpt "$GCKPT" --categories "$CATS" --out "$RUN_CATS" \
      --k "$WORD_K" --score "${WORD_SCORE:-yesno}" ${LIMIT:+--limit $((LIMIT * 20))}) 2>&1 | tee -a "$OUT/rerank.log" \
    | grep --line-buffered -E "^\[rerank|Error|Traceback"
  [ -f "$RUN_CATS" ] || { echo "重排没有输出，把 $OUT/rerank.log 最后 30 行发给我"; return 1; }
}

# ---------------------------------------------------------------- 3. 主线推理
step_run() {
  need "$CONFIG" "$RUN_CATS" "$HEAT" "$SAM3_CKPT"
  $PY -u -m v2.mainline run --dataset "$DS" --repo "$REPO" --config "$CONFIG" --data-root "$DATA_ROOT" \
      --categories "$RUN_CATS" --heat-dir "$HEAT" --base-mllm "$MODEL" --sam3-checkpoint "$SAM3_CKPT" \
      --out "$OUT" --gt ${LIMIT:+--limit $LIMIT --allow-missing-words} ${RUN_ARGS:-} 2>&1 \
    | tee -a "$OUT/run.log" | grep --line-buffered -E '^\[(v2|error|[0-9]+/[0-9]+)\]|Error|Traceback'
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || echo "有样本报错（见 $OUT/run.log 里的 [error]）。先重跑同一条命令补齐；仍报错就把那几行发给我"
  return "$rc"
}

# ---------------------------------------------------------------- 4. 评测
step_eval() {
  need "$OUT/scores.jsonl"
  $PY -m v2.mainline export --dataset "$DS" --data-root "$DATA_ROOT" --out "$OUT"
  [ $? -eq 2 ] && echo "注意：有样本没跑完，按空掩码计分（见 $OUT/<words|final>/missing.txt）"
  for row in words final; do
    echo "== 评测 $row"
    mkdir -p "$OUT/$row"
    if [ "$DS" = revos ]; then
      (cd "$REPO" && $PY -u tools/eval/eval_revos.py "$OUT/$row/results.json" --exp_path "$VAL_META" \
          --mask_path "$REVOS_ROOT/mask_dict.json" --foreground_mask_path "$REVOS_ROOT/mask_dict_foreground.json" \
          --save_json_name revos_valid_official.json --save_csv_name revos_valid_official.csv \
          --workers "${EVAL_WORKERS:-8}") > "$OUT/$row/eval.log" 2>&1
    else
      (cd "$REPO" && $PY -u tools/eval/eval_refytros_png.py --pred_root "$OUT/$row" --data_root "$REFYT" --split valid \
          --meta_file meta_expressions_challenge.json --output_json "$OUT/$row/evaluation_results_refytros_png.json") > "$OUT/$row/eval.log" 2>&1
    fi
    local rc=$?
    tail -n 3 "$OUT/$row/eval.log"
    [ "$rc" -eq 0 ] || { echo "评测 $row 失败，把 $OUT/$row/eval.log 最后 30 行发给我"; return 1; }
  done
  $PY -m v2.mainline summary --dataset "$DS" --out "$OUT" "${REFS[@]}"
  echo "完成。把 $OUT/summary.md 发给我"
}

case "$STEP" in
  ground) step_ground ;;
  heat)   step_heat ;;
  rerank) step_rerank ;;
  run)    step_run ;;
  eval)   step_eval ;;
  all)    step_ground && step_heat && step_rerank && step_run && { [ -n "${LIMIT:-}" ] && echo "试跑完成（LIMIT=$LIMIT），去掉 LIMIT 跑全量" || step_eval; } ;;
  *) echo "用法: bash run_v2.sh $DS all|ground|heat|rerank|run|eval"; exit 1 ;;
esac
