#!/usr/bin/env python3
"""
Evaluate DAVIS17 PNG predictions directly (no results.json conversion required).

Expected prediction layout:
  <pred_root>/Annotations/<video_id>/<exp_id>/<frame>.png
or legacy:
  <pred_root>/<video_id>/<exp_id>/<frame>.png

Ground truth:
  meta_expressions.json + mask_dict.pkl
"""

import os
import json
import pickle
import argparse
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METRICS_DIR = os.path.join(ROOT, "third_parts", "revos", "utils")
if METRICS_DIR not in sys.path:
    sys.path.append(METRICS_DIR)
from metircs import db_eval_iou, db_eval_boundary


def load_meta_with_anno(meta_path: str) -> Dict:
    with open(meta_path, "r") as f:
        videos = json.load(f)["videos"]

    # Align with tools/eval/eval_davis.py anno assignment logic.
    anno_count = 0
    for vid in sorted(videos.keys()):
        exprs = videos[vid]["expressions"]
        # Must match tools/eval/eval_davis.py exactly (lexicographic sort)
        exp_ids = sorted(list(exprs.keys()))
        for exp_id in exp_ids:
            exprs[exp_id]["anno_id"] = [anno_count]
            anno_count += 1
    return videos


def load_mask_dict(mask_dict_path: str):
    if mask_dict_path.endswith(".pkl"):
        with open(mask_dict_path, "rb") as f:
            return pickle.load(f)
    if mask_dict_path.endswith(".json"):
        with open(mask_dict_path, "r") as f:
            return json.load(f)
    raise ValueError(f"Unsupported mask_dict format: {mask_dict_path}")


def _resolve_pred_dir(pred_root: str, video_id: str, exp_id: str) -> str:
    cands = [
        os.path.join(pred_root, "Annotations", video_id, exp_id),
        os.path.join(pred_root, video_id, exp_id),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    # default to submit-style path even if not created yet
    return cands[0]


def _infer_gt_hw(anno_ids: List[int], mask_dict: Dict, num_frames: int) -> Tuple[int, int]:
    for frame_idx in range(num_frames):
        for anno_id in anno_ids:
            rle = mask_dict[str(anno_id)][frame_idx]
            if rle:
                m = cocomask.decode(rle)
                return int(m.shape[0]), int(m.shape[1])
    raise RuntimeError("No non-empty GT mask found to infer size")


def _load_pred_binary(pred_file: str, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if not os.path.exists(pred_file):
        return np.zeros((h, w), dtype=np.uint8)

    pred = np.array(Image.open(pred_file))
    pred = (pred > 0).astype(np.uint8)

    if pred.shape != (h, w):
        pred_img = Image.fromarray((pred * 255).astype(np.uint8))
        pred_img = pred_img.resize((w, h), Image.NEAREST)
        pred = (np.array(pred_img) > 0).astype(np.uint8)

    return pred


def evaluate_single_expression(
    video_id: str,
    exp_id: str,
    vid_info: Dict,
    mask_dict: Dict,
    pred_root: str,
):
    frames = [str(x) for x in vid_info["frames"]]
    num_frames = len(frames)
    anno_ids = vid_info["expressions"][exp_id]["anno_id"]

    h, w = _infer_gt_hw(anno_ids, mask_dict, num_frames)

    gt_masks = np.zeros((num_frames, h, w), dtype=np.uint8)
    pred_masks = np.zeros((num_frames, h, w), dtype=np.uint8)

    pred_dir = _resolve_pred_dir(pred_root, video_id, exp_id)
    missing_pred_frames = 0

    for frame_idx, frame_name in enumerate(frames):
        # GT
        for anno_id in anno_ids:
            rle = mask_dict[str(anno_id)][frame_idx]
            if rle:
                gt_masks[frame_idx] += cocomask.decode(rle).astype(np.uint8)

        # Prediction
        pred_file = os.path.join(pred_dir, f"{frame_name}.png")
        if not os.path.exists(pred_file):
            missing_pred_frames += 1
        pred_masks[frame_idx] = _load_pred_binary(pred_file, (h, w))

    j = float(db_eval_iou(gt_masks, pred_masks).mean())
    f = float(db_eval_boundary(gt_masks, pred_masks).mean())
    jf = float((j + f) / 2.0)

    return {
        "J": j,
        "F": f,
        "J&F": jf,
        "num_frames": num_frames,
        "missing_pred_frames": missing_pred_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DAVIS17 PNG predictions directly")
    parser.add_argument("--pred_root", type=str, required=True, help="Prediction root directory")
    parser.add_argument(
        "--meta_path",
        type=str,
        default="/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/davis17/meta_expressions/valid/meta_expressions.json",
        help="Path to DAVIS meta_expressions.json",
    )
    parser.add_argument(
        "--mask_dict_path",
        type=str,
        default="/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/davis17/valid/mask_dict.pkl",
        help="Path to DAVIS mask_dict.pkl/json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output json path (default: <pred_root>/evaluation_results_davis_png.json)",
    )
    parser.add_argument("--video_id", type=str, default=None, help="Only evaluate one video")
    parser.add_argument("--exp_id", type=str, default=None, help="Only evaluate one expression (requires --video_id)")
    args = parser.parse_args()

    videos = load_meta_with_anno(args.meta_path)
    mask_dict = load_mask_dict(args.mask_dict_path)

    if args.output_json is None:
        args.output_json = os.path.join(args.pred_root, "evaluation_results_davis_png.json")

    # evaluation scope
    if args.video_id is not None:
        if args.video_id not in videos:
            raise KeyError(f"video_id not in meta: {args.video_id}")
        eval_videos = {args.video_id: videos[args.video_id]}
    else:
        eval_videos = videos

    per_expression = {}
    per_video = {}

    total_expected_frames = 0
    total_missing_pred_frames = 0

    for video_id in sorted(eval_videos.keys()):
        vid_info = eval_videos[video_id]
        exprs = vid_info["expressions"]

        if args.video_id is not None and args.exp_id is not None:
            exp_ids = [str(args.exp_id)]
        else:
            exp_ids = sorted(exprs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

        video_expr_results = []

        for exp_id in exp_ids:
            if exp_id not in exprs:
                continue

            result = evaluate_single_expression(
                video_id=video_id,
                exp_id=exp_id,
                vid_info=vid_info,
                mask_dict=mask_dict,
                pred_root=args.pred_root,
            )

            key = f"{video_id}_{exp_id}"
            per_expression[key] = {
                "video_id": video_id,
                "exp_id": exp_id,
                **result,
            }
            video_expr_results.append(result)

            total_expected_frames += int(result["num_frames"])
            total_missing_pred_frames += int(result["missing_pred_frames"])

        if video_expr_results:
            vj = float(np.mean([x["J"] for x in video_expr_results]))
            vf = float(np.mean([x["F"] for x in video_expr_results]))
            per_video[video_id] = {
                "J": vj,
                "F": vf,
                "J&F": float((vj + vf) / 2.0),
                "num_expressions": len(video_expr_results),
            }

    if not per_expression:
        raise RuntimeError("No expressions evaluated. Check pred_root/meta/split consistency.")

    expr_J = float(np.mean([x["J"] for x in per_expression.values()]))
    expr_F = float(np.mean([x["F"] for x in per_expression.values()]))
    video_J = float(np.mean([x["J"] for x in per_video.values()])) if per_video else 0.0
    video_F = float(np.mean([x["F"] for x in per_video.values()])) if per_video else 0.0

    out = {
        "overall_expression_level": {
            "J": round(expr_J * 100, 2),
            "F": round(expr_F * 100, 2),
            "J&F": round((expr_J + expr_F) * 50, 2),
            "num_expressions": len(per_expression),
        },
        "overall_video_level": {
            "J": round(video_J * 100, 2),
            "F": round(video_F * 100, 2),
            "J&F": round((video_J + video_F) * 50, 2),
            "num_videos": len(per_video),
        },
        "coverage": {
            "total_expected_frames": int(total_expected_frames),
            "total_missing_pred_frames": int(total_missing_pred_frames),
            "missing_ratio": round(total_missing_pred_frames / max(total_expected_frames, 1), 6),
        },
        "per_video": per_video,
        "per_expression": per_expression,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 50)
    print(f"Evaluated expressions: {len(per_expression)}")
    print(f"Evaluated videos: {len(per_video)}")
    print(f"Expression-level: J={out['overall_expression_level']['J']:.2f} "
          f"F={out['overall_expression_level']['F']:.2f} "
          f"J&F={out['overall_expression_level']['J&F']:.2f}")
    print(f"Video-level:      J={out['overall_video_level']['J']:.2f} "
          f"F={out['overall_video_level']['F']:.2f} "
          f"J&F={out['overall_video_level']['J&F']:.2f}")
    print(f"Missing predicted frames: {total_missing_pred_frames}/{total_expected_frames}")
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
