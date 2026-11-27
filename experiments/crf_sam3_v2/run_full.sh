#!/usr/bin/env bash
# CRF-SAM3 server adaptation. See SERVER_README.md; Ground training restarts, not resumes.
# 只读 video_datas/ 和 $REPO，不改它们；所有产物都在 crf_sam3_v2/runs/ 下。规则见 docs/training_set_spec.md。
#
#   bash run_full.sh test            CPU 自检（假数据，约 2 分钟）
#   bash run_full.sh splits          1  划分：v1 的 val 视频原样保留 + 其余视频 10%（seed 42）→ runs/data/splits.json（CPU）
#   bash run_full.sh ground-data     2a Ground 训练数据：ReVOS + Ref-YTVOS，全量和两折 → runs/data/ground/（CPU）
#   bash run_full.sh ground-train    2b 训练三份 Ground v2：full、cf0、cf1（双卡，一份接一份）→ runs/ground/<名字>
#                                       只训一份：bash run_full.sh ground-train full
#   bash run_full.sh ground-infer    3  每个视频用没见过它的那份 Ground 出词和热力图（训练集 + holdout + 四个验证集，两卡并行）
#   bash run_full.sh cache           4  四个数据集的 Verify 训练缓存（和推理同一套候选池，两卡分片）→ runs/verify/cache/
#   bash run_full.sh verify-train    5  Verify 四数据集联合训练（双卡）→ runs/verify/ckpt/final
#   bash run_full.sh eval            6  四个验证集推理 + 官方评测 → runs/eval/<数据集>/summary.md
#   bash run_full.sh all             1 → 6 依次跑
# 环境变量：GPUS（默认 2,3）、GROUND_EPOCHS（默认 1，Sa2VA 也是 1 个 epoch）、ONLY（只跑这些数据集，逗号分隔）、
#           LIMIT（缓存每个分片只跑前 N 个视频）、GROUND_MAX_STEPS / VERIFY_MAX_STEPS（训练只跑 N 步），试跑用；
#           VD、REFYT、SAM3_CKPT、PORT。
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
GV2=$(cd "${GV2:-../ground_v2}" && pwd)
P1A=$(cd "${P1A:-../p1a_spatial_recall}" && pwd)
source "$HERE/scripts/server_env.sh"                 # PY REPO MODEL REVOS_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$HERE:$P1A${PYTHONPATH:+:$PYTHONPATH}"
VD=${VD:-/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas}
REFYT=${REFYT:-$VD/Ref-Youtube-VOS}
SAM3_CKPT=${SAM3_CKPT:-/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt}
GPUS=${GPUS:-2,3}
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NGPU=${#GPU_LIST[@]}
[ "$NGPU" -le 2 ] || { echo "Use one or two GPUs"; exit 1; }
DATASETS=${ONLY:-mevis,davis17,refytvos,revos}
R=${RUN_ROOT:-$HERE/runs}
R=$(realpath -m "$R")
if [ -n "${GROUND_MAX_STEPS:-}${VERIFY_MAX_STEPS:-}${LIMIT:-}" ]; then
  case "${1:-}" in all|eval) echo "Do not run all/eval with smoke limits"; exit 1 ;; esac
fi
SPLITS=$R/data/splits.json
GDATA=$R/data/ground
LOGS=$R/logs
mkdir -p "$LOGS"
CUDA_VISIBLE_DEVICES= "$PY" "$HERE/tools/record_run.py" "$R/provenance" "$0" "$@"
need() { for f in "$@"; do [ -e "$f" ] || { echo "找不到 $f（先跑前一阶段）"; exit 1; }; done; }
cpu() { CUDA_VISIBLE_DEVICES= "$PY" "$@"; }

# ---------------------------------------------------------------- 0. 自检
stage_test() {
  local rc=0
  for t in test_server_guards test_stage_data test_pipeline_glue test_verify_cache test_mainline test_audit_trainsets test_ground_rerank test_inventory; do
    echo "== $t"; cpu "$HERE/tests/$t.py" > "$LOGS/$t.log" 2>&1 && tail -n 1 "$LOGS/$t.log" || { tail -n 30 "$LOGS/$t.log"; rc=1; }
  done
  return $rc
}

# ---------------------------------------------------------------- 1. 划分
stage_splits() {
  cpu "$HERE/tools/make_splits.py" --video-datas "$VD" --repo "$REPO" --out "$SPLITS"
}

# ---------------------------------------------------------------- 2a. Ground 数据
stage_ground_data() {
  need "$SPLITS"
  if [ -f "$GDATA/COMPLETE" ]; then echo "[ground-data] 已有 $GDATA，跳过（要重建先删掉它）"; return 0; fi
  cpu "$HERE/v2/build_ground.py" --video-datas "$VD" --repo "$REPO" --gv2 "$GV2" --splits "$SPLITS" --out "$GDATA" \
    2>&1 | tee "$LOGS/ground_data.log" || return 1
  return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------- 2b. Ground 训练（ReVOS v2 的配方：等效 batch 8、lr 1e-4、两卡各累积 4 步）
stage_ground_train() {
  local which=${1:-full,cf0,cf1} name
  for name in ${which//,/ }; do
    local out=$R/ground/$name data=$GDATA/$name/train.jsonl
    need "$GDATA/COMPLETE" "$data" "$GDATA/val.jsonl"
    [ -z "${GROUND_MAX_STEPS:-}" ] || out=$R/smoke/ground/$name
    if [ -d "$out" ] && [ ! -f "$out/COMPLETE" ]; then
      echo "Incomplete Ground directory $out; use a fresh RUN_ROOT or archive it before restarting (Ground resume unsupported)"; return 1
    fi
    if [ -f "$out/COMPLETE" ]; then echo "[ground-train] $name 已训好：$out"; continue; fi
    local acc=$((8 / NGPU)) launch=()
    [ "$NGPU" -gt 1 ] && launch=(-m torch.distributed.run --nproc_per_node="$NGPU" --master_port="${PORT:-29631}")
    echo "[ground-train] $name：$(wc -l < "$data") 条，$NGPU 张卡（$GPUS），每卡累积 $acc 步，${GROUND_EPOCHS:-1} 个 epoch；日志 $out.log"
    mkdir -p "$(dirname "$out")"
    (cd "$GV2" && CUDA_VISIBLE_DEVICES=$GPUS "$PY" -u ${launch[@]+"${launch[@]}"} -m gv2.train --model-path "$MODEL" \
        --train-jsonl "$data" --val-jsonl "$GDATA/val.jsonl" --max-val-samples 200 --output-dir "$out" \
        --num-train-epochs "${GROUND_EPOCHS:-1}" --feature-layer 12 --bf16 --gradient-checkpointing \
        --per-device-train-batch-size 1 --gradient-accumulation-steps "$acc" --learning-rate 1e-4 --warmup-ratio 0.03 \
        --max-grad-norm 1.0 --logging-steps 10 --eval-steps 500 --save-steps 500 \
        ${GROUND_MAX_STEPS:+--max-steps $GROUND_MAX_STEPS}) >> "$out.log" 2>&1 || return 1
    [ -f "$out/gv2_config.json" ] || { echo "[ground-train] $name 没有正常结束，看 $out.log 最后 50 行"; return 1; }
    (cd "$GV2" && CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" -u -m gv2.evaluate --model-path "$MODEL" --jsonl "$GDATA/val.jsonl" \
        --ckpt "$out" --bf16 --device-map '' --out "$out/eval_val.json") 2>&1 | tail -n 8 || return 1
    touch "$out/COMPLETE"
  done
}

# ---------------------------------------------------------------- 3. Ground 推理（交叉拟合）
stage_ground_infer() {
  need "$SPLITS" "$GDATA/members.json"
  local plan=$R/ground_infer/jobs.txt
  mkdir -p "$R/ground_infer"
  cpu "$HERE/tools/ground_infer_plan.py" plan --video-datas "$VD" --refyt "$REFYT" --revos-root "$REVOS_ROOT" \
    --splits "$SPLITS" --members "$GDATA/members.json" --out "$R/ground_infer" --parts "$NGPU" --datasets "$DATASETS" > "$plan" || return 1
  local g pids=()
  for g in $(seq 0 $((NGPU - 1))); do
    (
      while read -r set model part root meta outj heat n; do
        [ "$part" = "$g" ] || continue
        case ",$DATASETS," in *",${set%_valid},"*) ;; *) continue ;; esac
        need "$R/ground/$model/COMPLETE"
        echo "[ground-infer] GPU ${GPU_LIST[$g]}: $set/$model 分片 $part（$n 条）"
        (cd "$GV2" && CUDA_VISIBLE_DEVICES=${GPU_LIST[$g]} "$PY" -u -m gv2.predict --model-path "$MODEL" --ckpt "$R/ground/$model" \
            --data-root "$root" --expression-file "$meta" --frames-per-sample 3 --output-json "$outj" --heat-dir "$heat" \
            --bf16 --device-map '' --save-every 50 --resume) >> "$LOGS/ground_infer_gpu$g.log" 2>&1 || { echo "[ground-infer] failed: $set/$model"; exit 1; }
      done < "$plan"
    ) &
    pids+=($!)
  done
  local wait_rc=0
  for p in "${pids[@]}"; do wait "$p" || wait_rc=1; done
  [ "$wait_rc" -eq 0 ] || return 1
  cpu "$HERE/tools/ground_infer_plan.py" merge --out "$R/ground_infer" --datasets "$DATASETS"
}

# ---------------------------------------------------------------- 4. Verify 训练缓存
stage_cache() {
  need "$SPLITS" "$GDATA/members.json"
  local yaml=$R/verify/mainline_cache.yaml d part g rc=0
  cpu "$HERE/tools/make_mainline_yaml.py" --repo "$REPO" --out "$yaml" || return 1
  for d in ${DATASETS//,/ }; do
    for part in train holdout; do
      local pids=()
      for g in $(seq 0 $((NGPU - 1))); do
        CUDA_VISIBLE_DEVICES=${GPU_LIST[$g]} "$PY" -u -m v2.verify_cache --dataset "$d" --part "$part" --video-datas "$VD" \
          --splits "$SPLITS" --members "$GDATA/members.json" --words "$R/ground_infer/$d" --out "$R/verify/cache/$d" \
          --repo "$REPO" --config "$yaml" --base-mllm "$MODEL" --sam3-checkpoint "$SAM3_CKPT" --shard "$g/$NGPU" \
          ${LIMIT:+--limit $LIMIT} >> "$LOGS/cache_${d}_${part}_$g.log" 2>&1 &
        pids+=($!)
      done
      for p in "${pids[@]}"; do wait "$p" || rc=1; done
      "$PY" - "$R/verify/cache/$d" "$part" <<'EOF'
import json, sys, collections, pathlib
d, part = pathlib.Path(sys.argv[1]), sys.argv[2]
c = collections.Counter(json.loads(l)["status"] for p in d.glob(f"done_{part}_*.jsonl") for l in p.read_text().splitlines() if l.strip())
print(f"[cache] {d.name}/{part}: {dict(c)}")
EOF
    done
  done
  [ "$rc" -eq 0 ] || echo "[cache] 有分片报错（见 $LOGS/cache_*.log 里的 [error]），重跑 bash run_full.sh cache 会只补没做完的"
  return $rc
}

# ---------------------------------------------------------------- 5. Verify 联合训练
stage_verify_train() {
  if [ -n "${VERIFY_MAX_STEPS:-}" ]; then
    # smoke test: may run on a partial (LIMIT) cache so trainer problems show up before the full cache is built;
    # the trainer only reads expressions that already have meta.json, and writes to runs/smoke/verify
    echo "[verify-train] smoke: skipping the cache completeness check (partial cache allowed)"
  else
    for d in ${DATASETS//,/ }; do
      cpu "$HERE/tools/check_cache_complete.py" --root "$R/verify/cache/$d" --splits "$SPLITS" --dataset "$d" --part train --video-datas "$VD" || return 1
    done
  fi
  local cfg=$R/verify/train.yaml vout=$R/verify/ckpt
  if [ -n "${VERIFY_MAX_STEPS:-}" ]; then cfg=$R/smoke/verify/train.yaml; vout=$R/smoke/verify/ckpt; fi
  if [ -f "$vout/COMPLETE" ]; then echo "Verify complete: $vout"; return 0; fi
  if [ -d "$vout/final" ] && [ ! -f "$vout/COMPLETE" ]; then
    echo "Uncommitted final checkpoint: $vout; inspect before resuming"; return 1
  fi
  cpu "$HERE/tools/make_verify_config.py" --repo "$REPO" --cache "$R/verify/cache" --datasets "$DATASETS" --out "$cfg" \
    --output-dir "$vout" ${VERIFY_MAX_STEPS:+--max-steps $VERIFY_MAX_STEPS} || return 1
  local launch=()
  [ "$NGPU" -gt 1 ] && launch=(-m torch.distributed.run --nproc_per_node="$NGPU" --master_port="${PORT:-29632}")
  echo "[verify-train] $NGPU 张卡（$GPUS）；日志 $LOGS/verify_train.log；中断后重跑同一命令会从最近的 checkpoint 接着训"
  CUDA_VISIBLE_DEVICES=$GPUS "$PY" -u ${launch[@]+"${launch[@]}"} -m v2.train_verify_ddp --repo "$REPO" --config "$cfg" \
    --auto-resume >> "$LOGS/verify_train.log" 2>&1 || return 1
  [ -d "$vout/final" ] || { echo "[verify-train] 没有正常结束，看 $LOGS/verify_train.log 最后 50 行"; return 1; }
  touch "$vout/COMPLETE"
}

# ---------------------------------------------------------------- 6. 四个验证集推理 + 评测
eval_one() {  # $1 dataset  $2 gpu
  local d=$1 g=$2 out=$R/eval/$1 root
  case "$d" in
    mevis) root=$VD/mevis ;; davis17) root=$VD/davis17 ;; refytvos) root=$REFYT ;; revos) root=$REVOS_ROOT ;;
  esac
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$g "$PY" -u -m v2.mainline run --dataset "$d" --repo "$REPO" --config "$R/verify/mainline_eval.yaml" \
    --data-root "$root" --categories "$R/ground_infer/${d}_valid/full/categories.json" \
    --heat-dir "$R/ground_infer/${d}_valid/full/npz" --base-mllm "$MODEL" --sam3-checkpoint "$SAM3_CKPT" \
    --out "$out" --gt >> "$out/run.log" 2>&1 || { echo "[eval] $d 推理有样本报错，见 $out/run.log；重跑会补齐"; return 1; }
  "$PY" -m v2.mainline export --dataset "$d" --data-root "$root" --out "$out" || return 1
  local row rc=0
  for row in words final; do
    case "$d" in
      revos) (cd "$REPO" && "$PY" -u tools/eval/eval_revos.py "$out/$row/results.json" --exp_path "$REVOS_ROOT/meta_expressions_valid_.json" \
               --mask_path "$REVOS_ROOT/mask_dict.json" --foreground_mask_path "$REVOS_ROOT/mask_dict_foreground.json" \
               --save_json_name revos_valid_official.json --save_csv_name revos_valid_official.csv --workers 8) ;;
      mevis) (cd "$REPO" && PYTHONPATH="$REPO" "$PY" -u tools/eval/eval_mevis.py "$out/$row/results.json" \
               --mevis_exp_path "$VD/mevis/valid_u/meta_expressions.json" --mevis_mask_path "$VD/mevis/valid_u/mask_dict.json" \
               --save_name mevis_valu.json) ;;
      davis17) (cd "$REPO" && "$PY" -u tools/eval/eval_davis_png.py --pred_root "$out/$row" \
               --meta_path "$VD/davis17/meta_expressions/valid/meta_expressions.json" \
               --mask_dict_path "$VD/davis17/valid/mask_dict.pkl" --output_json "$out/$row/evaluation_results_davis_png.json") ;;
      refytvos) (cd "$REPO" && "$PY" -u tools/eval/eval_refytros_png.py --pred_root "$out/$row" --data_root "$REFYT" --split valid \
               --meta_file meta_expressions_challenge.json --output_json "$out/$row/evaluation_results_refytros_png.json") ;;
    esac > "$out/$row/eval.log" 2>&1 || { echo "[eval] $d/$row 评测失败，见 $out/$row/eval.log"; rc=1; }
  done
  "$PY" -m v2.mainline summary --dataset "$d" --out "$out" > /dev/null
  echo "[eval] $d 完成：$out/summary.md"
  return $rc
}
stage_eval() {
  need "$R/verify/ckpt/COMPLETE" "$R/verify/ckpt/final"
  cpu "$HERE/tools/make_mainline_yaml.py" --repo "$REPO" --adapter "$R/verify/ckpt/final" --out "$R/verify/mainline_eval.yaml" || return 1
  if [ "$NGPU" -eq 1 ]; then
    for d in ${DATASETS//,/ }; do eval_one "$d" "${GPU_LIST[0]}" || return 1; done
    return 0
  fi
  # ReVOS 最大（5,822 条），单独占一张卡；其余三个依次在另一张卡上
  local a=${GPU_LIST[0]} b=${GPU_LIST[$((NGPU - 1))]} pids=() d
  (for d in mevis davis17 refytvos; do case ",$DATASETS," in *",$d,"*) eval_one "$d" "$a" || exit 1 ;; esac; done) & pids+=($!)
  (case ",$DATASETS," in *",revos,"*) eval_one revos "$b" ;; esac) & pids+=($!)
  local wait_rc=0
  for p in "${pids[@]}"; do wait "$p" || wait_rc=1; done
  [ "$wait_rc" -eq 0 ] || return 1
  cat "$R"/eval/*/summary.md
}

case "${1:-}" in
  test) stage_test ;;
  splits) stage_splits ;;
  ground-data) stage_ground_data ;;
  ground-train) stage_ground_train "${2:-full,cf0,cf1}" ;;
  ground-infer) stage_ground_infer ;;
  cache) stage_cache ;;
  verify-train) stage_verify_train ;;
  eval) stage_eval ;;
  all) stage_splits && stage_ground_data && stage_ground_train && stage_ground_infer && stage_cache && stage_verify_train && stage_eval ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
