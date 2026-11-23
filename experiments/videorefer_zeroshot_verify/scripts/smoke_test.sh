#!/usr/bin/env bash
# CPU-only end-to-end check of the bookkeeping (no SAM3, no VideoRefer weights).
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python}
T=${1:-/tmp/vr_smoke}
rm -rf "$T"; mkdir -p "$T"
$PY "$PKG/tools/make_fake_revos.py" --out "$T/revos"
$PY "$PKG/step1_dump_candidates.py" --revos_root "$T/revos" \
    --historical_scores "$T/revos/hist/scores.json" --historical_results "$T/revos/hist/results.json" \
    --out_dir "$T/work" --mock_recall --mock_gt_mask_dict "$T/revos/mask_dict.json"
for MODE in yesno choice; do
  $PY "$PKG/step2_score_videorefer.py" --manifest "$T/work/manifest.jsonl" --image_root "$T/revos" \
      --mode $MODE --mask_frames 4 --out "$T/work/vr_${MODE}.jsonl" --dry_run
done
$PY "$PKG/step3_evaluate.py" --manifest "$T/work/manifest.jsonl" --revos_root "$T/revos" \
    --official_csv "$T/revos/hist/revos_valid_official.csv" \
    --vr yesno_p0="$T/work/vr_yesno.jsonl" --vr choice_c0="$T/work/vr_choice.jsonl" \
    --out_dir "$T/eval" --workers 2 --export_results --historical_results "$T/revos/hist/results.json"
echo "SMOKE TEST OK -> $T/eval/report.md"
