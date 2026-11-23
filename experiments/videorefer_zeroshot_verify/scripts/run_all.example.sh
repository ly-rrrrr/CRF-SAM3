#!/usr/bin/env bash
# Full pipeline example. Usage: bash scripts/run_all.example.sh scripts/paths.sh
set -euo pipefail
source "$1"
PKG="$(cd "$(dirname "$0")/.." && pwd)"
QUANT=${QUANT:-none}        # none | 8bit | 4bit

# Step 1: frozen SAM3 replay (CRF-SAM3 env)
$CRF_PY "$PKG/step1_dump_candidates.py" --crf_root "$CRF_ROOT" --sam3_checkpoint "$SAM3_CKPT" \
  --revos_root "$REVOS" --historical_scores "$HIST/scores.json" --historical_results "$HIST/results.json" \
  --out_dir "$WORK"

# Step 2: VideoRefer zero-shot scoring (videorefer env)
run_vr () {  # $1 label, rest = extra args
  local label=$1; shift
  $VR_PY "$PKG/step2_score_videorefer.py" --manifest "$WORK/manifest.jsonl" --image_root "$REVOS" \
    --videorefer_repo "$VR_REPO" --model_path "$VR_CKPT" --vision_tower "$SIGLIP" --quant "$QUANT" \
    --out "$WORK/vr_${label}.jsonl" "$@"
}
run_vr yesno_p0_m8  --mode yesno  --prompt_id p0 --mask_frames 8
run_vr yesno_p0_m4  --mode yesno  --prompt_id p0 --mask_frames 4
run_vr choice_c0_m8 --mode choice --prompt_id c0 --mask_frames 8

# Step 3: evaluation (CPU)
$CRF_PY "$PKG/step3_evaluate.py" --manifest "$WORK/manifest.jsonl" --revos_root "$REVOS" \
  --official_csv "$HIST/revos_valid_official.csv" \
  --vr yesno_p0_m8="$WORK/vr_yesno_p0_m8.jsonl" \
  --vr yesno_p0_m4="$WORK/vr_yesno_p0_m4.jsonl" \
  --vr choice_c0_m8="$WORK/vr_choice_c0_m8.jsonl" \
  --out_dir "$WORK/eval" --workers 16 \
  --export_results --historical_results "$HIST/results.json"
echo "report: $WORK/eval/report.md"
