#!/usr/bin/env bash
# Usage (inside ring_flow/):
#   bash scripts/run_all.sh check   # no training, ~1 min: read camera motion / local motion from ring derivatives
#   bash scripts/run_all.sh render  # representation check: rings + splats vs uniform grid, ~1 min
#   GPU=0 bash scripts/run_all.sh train   # calculus engine on the synthetic world (renders data first, then trains)
#   FRAMES_DIR=/path/to/frames bash scripts/run_all.sh real   # training-free readout on your own video frames
#   GPU=0 bash scripts/run_all.sh v1        # v1: learned encoder/derivative/integrator on synthetic video + probes
#   GPU=0 FRAMES_DIR=/path/to/frames bash scripts/run_all.sh v1real   # v1 trained on your own unlabeled video
#   GPU=0 bash scripts/run_all.sh v2        # v2: equation of motion on the rings; trained at 15 fps only, tested at 60/30/15/7.5 fps (also trains v1 at 15 fps for comparison)
#   GPU=0 bash scripts/run_all.sh v23       # v2.3 (best so far): camera solved from the equation, network learns local flow + source
#   GPU=0 bash scripts/run_all.sh v21       # v2.1: same, plus a whole-frame camera term (4 numbers) and costlier local flow / source
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
case "${1:-}" in
  check)  python -m rf.check_motion --out runs/check_motion --n 20 --frames 16 ;;
  render) python -m rf.demo_render --out runs/demo_render --n 32 ;;
  train)  CUDA_VISIBLE_DEVICES=$GPU python -m rf.train --out runs/ringflow_v0 --cache data \
            --n_train ${N_TRAIN:-3000} --steps ${STEPS:-20000} --workers ${WORKERS:-8} 2>&1 | tee runs/ringflow_v0.log ;;
  real)   python -m rf.check_motion --out runs/check_motion_real --n 2 --frames 8 --frames_dir "${FRAMES_DIR:?set FRAMES_DIR}" --real_dt ${REAL_DT:-0.0333333} ;;
  v1)     mkdir -p runs; CUDA_VISIBLE_DEVICES=$GPU python -m rf.v1_train --out runs/v1 --cache data \
            --n_train ${N_TRAIN:-3000} --steps ${STEPS:-30000} --workers ${WORKERS:-8} 2>&1 | tee runs/v1.log ;;
  v1real) mkdir -p runs; CUDA_VISIBLE_DEVICES=$GPU python -m rf.v1_train --out runs/v1_real --data real \
            --frames_dir "${FRAMES_DIR:?set FRAMES_DIR}" --real_dt ${REAL_DT:-0.0333333} --steps ${STEPS:-30000} 2>&1 | tee runs/v1_real.log ;;
  v2)     mkdir -p runs
          CUDA_VISIBLE_DEVICES=$GPU python -m rf.v1_train --out runs/v1_15fps --cache data --strides 2 \
            --n_train ${N_TRAIN:-3000} --steps ${STEPS_V1:-30000} --workers ${WORKERS:-8} 2>&1 | tee runs/v1_15fps.log
          CUDA_VISIBLE_DEVICES=$GPU python -m rf.v2_train --out runs/v2 --cache data --v1_ckpt runs/v1_15fps/ringworld.pt \
            --n_train ${N_TRAIN:-3000} --steps ${STEPS:-20000} --workers ${WORKERS:-8} 2>&1 | tee runs/v2.log ;;
  v21)    mkdir -p runs
          [ -f runs/v1_15fps/ringworld.pt ] || CUDA_VISIBLE_DEVICES=$GPU python -m rf.v1_train --out runs/v1_15fps --cache data --strides 2 \
            --n_train ${N_TRAIN:-3000} --steps ${STEPS_V1:-30000} --workers ${WORKERS:-8} 2>&1 | tee runs/v1_15fps.log
          CUDA_VISIBLE_DEVICES=$GPU python -m rf.v2_train --out runs/v21 --cache data --v1_ckpt runs/v1_15fps/ringworld.pt \
            --global_cam --lam_loc ${LAM_LOC:-0.05} --lam_q ${LAM_Q:-0.02} --n_frames ${N_FRAMES:-7} \
            --n_train ${N_TRAIN:-3000} --steps ${STEPS:-20000} --workers ${WORKERS:-8} 2>&1 | tee runs/v21.log ;;
  v23)    mkdir -p runs
          [ -f runs/v1_15fps/ringworld.pt ] || CUDA_VISIBLE_DEVICES=$GPU python -m rf.v1_train --out runs/v1_15fps --cache data --strides 2 \
            --n_train ${N_TRAIN:-3000} --steps ${STEPS_V1:-30000} --workers ${WORKERS:-8} 2>&1 | tee runs/v1_15fps.log
          CUDA_VISIBLE_DEVICES=$GPU python -m rf.v2_train --out runs/v23 --cache data --v1_ckpt runs/v1_15fps/ringworld.pt \
            --cam_lsq --lam_res 1 --lam_loc ${LAM_LOC:-0.01} --lam_q ${LAM_Q:-0.02} --n_frames ${N_FRAMES:-7} \
            --n_train ${N_TRAIN:-3000} --steps ${STEPS:-20000} --workers ${WORKERS:-8} 2>&1 | tee runs/v23.log ;;
  *) echo "usage: $0 {check|render|train|real|v1|v1real|v2|v21|v23}"; exit 1 ;;
esac
