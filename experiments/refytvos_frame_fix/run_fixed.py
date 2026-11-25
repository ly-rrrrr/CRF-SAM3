#!/usr/bin/env python3
"""Run an UNCHANGED CRF-SAM3 dataset runner with the mask writer keyed by frame position only.

Everything else (config, Ground words, SAM3 recall, Verify) is the repo's own code. The only change:
the pipeline's masks_dict is keyed by SAM3 frame position (sorted JPEG order), so frame i of the sorted
meta frame list takes masks_dict[i]. The repo's writer tried int(frame_name) first, which on
Ref-YouTube-VOS (frames 00000, 00005, ...) writes the mask of a later frame into frames 00005..

  cd $REPO && python /path/run_fixed.py refytros <the usual run_crf_sam3_refytros_dataset.py args>
  cd $REPO && python /path/run_fixed.py revos    <the usual run_crf_sam3_revos_dataset.py args>
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def save_masks_by_position(masks_dict: Dict[int, Dict[int, np.ndarray]], output_dir: str, video_id: str, exp_id: str,
                           expected_frames: Optional[List[str]] = None, frame_hw: Optional[Tuple[int, int]] = None):
    import run_crf_sam3_dataset as base
    from PIL import Image

    if expected_frames is None:  # the runners always pass it; keep the original path otherwise
        return _ORIG_SAVE(masks_dict, output_dir, video_id, exp_id, expected_frames, frame_hw)
    exp_dir = os.path.join(output_dir, "Annotations", video_id, str(exp_id))
    os.makedirs(exp_dir, exist_ok=True)
    frames = [str(x) for x in expected_frames]
    if frame_hw is not None:
        h, w = int(frame_hw[0]), int(frame_hw[1])
    else:
        first = next((m for fm in masks_dict.values() for m in fm.values()), None)
        if first is None:
            raise RuntimeError(f"Cannot infer mask size for {video_id}/{exp_id}")
        h, w = first.shape[:2]
    missing = 0
    for pos, name in enumerate(frames):
        fm = masks_dict.get(pos, {})
        if not fm:
            missing += 1
        Image.fromarray(base._make_instance_mask(fm, h, w)).save(os.path.join(exp_dir, f"{name}.png"))
    return {"saved_frames": len(frames), "expected_frames": len(frames), "missing_pred_filled": missing, "out_dir": exp_dir}


def build_revos_rles_by_position(masks_dict, frame_names, h, w):
    import run_crf_sam3_revos_dataset as rv

    return [rv._encode_mask_to_rle(rv._compose_binary_mask(masks_dict.get(i, {}), h, w)) for i in range(len(frame_names))]


_ORIG_SAVE = None


def main():
    global _ORIG_SAVE
    if len(sys.argv) < 2 or sys.argv[1] not in ("refytros", "revos"):
        raise SystemExit("usage: python run_fixed.py refytros|revos <runner args>")
    which = sys.argv.pop(1)
    sys.path.insert(0, os.getcwd())
    import run_crf_sam3_dataset as base

    _ORIG_SAVE = base.save_masks
    if which == "refytros":
        import run_crf_sam3_refytros_dataset as runner

        runner.save_masks = save_masks_by_position
    else:
        import run_crf_sam3_revos_dataset as runner

        runner._build_revos_rles = build_revos_rles_by_position
        runner.save_masks = save_masks_by_position
    print(f"[run_fixed] {runner.__file__}: masks written by frame position", flush=True)
    sys.argv[0] = runner.__file__
    runner.main()


if __name__ == "__main__":
    main()
