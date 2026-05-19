#!/usr/bin/env python3
"""
Evaluate Ref-Youtube-ROS PNG predictions directly.

Prediction layout (supported):
  1) <pred_root>/Annotations/<video_id>/<exp_id>/<frame>.png
  2) <pred_root>/<video_id>/<exp_id>/<frame>.png

Ground-truth layout:
  <data_root>/<split>/Annotations/<video_id>/<exp_id>/<frame>.png
  <data_root>/<split>/meta_expressions_challenge.json

Metrics:
  J (IoU), F (boundary), J&F
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METRICS_DIR = os.path.join(ROOT, "third_parts", "revos", "utils")
if METRICS_DIR not in sys.path:
    sys.path.append(METRICS_DIR)
from metircs import db_eval_iou, db_eval_boundary


def _resolve_pred_dir(pred_root: str, video_id: str, exp_id: str) -> str:
    cands = [
        os.path.join(pred_root, "Annotations", video_id, exp_id),
        os.path.join(pred_root, video_id, exp_id),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return cands[0]


def _resolve_gt_dir(data_root: str, split: str, video_id: str, exp_id: str) -> str:
    return os.path.join(data_root, split, "Annotations", video_id, exp_id)


def _infer_hw(
    data_root: str,
    split: str,
    video_id: str,
    exp_id: str,
    frames: List[str],
) -> Tuple[int, int]:
    gt_dir = _resolve_gt_dir(data_root, split, video_id, exp_id)
    for fr in frames:
        gt_path = os.path.join(gt_dir, f"{fr}.png")
        if os.path.exists(gt_path):
            m = np.array(Image.open(gt_path))
            return int(m.shape[0]), int(m.shape[1])

    # Fallback: infer from RGB frame.
    jpeg_dir = os.path.join(data_root, split, "JPEGImages", video_id)
    for fr in frames:
        jpg_path = os.path.join(jpeg_dir, f"{fr}.jpg")
        if os.path.exists(jpg_path):
            m = np.array(Image.open(jpg_path).convert("L"))
            return int(m.shape[0]), int(m.shape[1])
    raise RuntimeError(f"Cannot infer size for {video_id}/{exp_id}")


def _load_binary_mask(mask_path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if not os.path.exists(mask_path):
        return np.zeros((h, w), dtype=np.uint8)

    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8)

    if arr.shape != (h, w):
        arr_img = Image.fromarray((arr * 255).astype(np.uint8))
        arr_img = arr_img.resize((w, h), Image.NEAREST)
        arr = (np.array(arr_img) > 0).astype(np.uint8)
    return arr


def evaluate_single_expression(
    video_id: str,
    exp_id: str,
    vid_info: Dict,
    pred_root: str,
    data_root: str,
    split: str,
) -> Dict:
    frames = [str(x) for x in vid_info["frames"]]
    num_frames = len(frames)
    gt_dir = _resolve_gt_dir(data_root, split, video_id, exp_id)
    pred_dir = _resolve_pred_dir(pred_root, video_id, exp_id)
    pred_dir_exists = os.path.isdir(pred_dir)

    h, w = _infer_hw(data_root, split, video_id, exp_id, frames)
    gt_masks = np.zeros((num_frames, h, w), dtype=np.uint8)
    pred_masks = np.zeros((num_frames, h, w), dtype=np.uint8)

    missing_gt_frames = 0
    missing_pred_frames = 0

    for i, fr in enumerate(frames):
        gt_path = os.path.join(gt_dir, f"{fr}.png")
        pred_path = os.path.join(pred_dir, f"{fr}.png")

        if not os.path.exists(gt_path):
            missing_gt_frames += 1
        if not os.path.exists(pred_path):
            missing_pred_frames += 1

        gt_masks[i] = _load_binary_mask(gt_path, (h, w))
        pred_masks[i] = _load_binary_mask(pred_path, (h, w))

    j_arr = db_eval_iou(gt_masks, pred_masks)
    f_arr = db_eval_boundary(gt_masks, pred_masks)

    j = float(np.nanmean(j_arr))
    f = float(np.nanmean(f_arr))
    if np.isnan(j):
        j = 0.0
    if np.isnan(f):
        f = 0.0

    return {
        "J": j,
        "F": f,
        "J&F": (j + f) / 2.0,
        "num_frames": num_frames,
        "missing_gt_frames": missing_gt_frames,
        "missing_pred_frames": missing_pred_frames,
        "pred_dir_exists": pred_dir_exists,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ref-Youtube-ROS PNG predictions")
    parser.add_argument("--pred_root", type=str, required=True, help="Prediction root")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/ref-youtube-ros",
        help="Dataset root for ref-youtube-ros",
    )
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test", "train"])
    parser.add_argument(
        "--meta_file",
        type=str,
        default="meta_expressions_challenge.json",
        help="Metadata file under <data_root>/<split>",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path (default: <pred_root>/evaluation_results_refytros_png.json)",
    )
    parser.add_argument("--video_id", type=str, default=None, help="Only evaluate one video id")
    parser.add_argument(
        "--exp_id",
        type=str,
        default=None,
        help="Only evaluate one expression id (requires --video_id)",
    )
    args = parser.parse_args()

    meta_path = os.path.join(args.data_root, args.split, args.meta_file)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, "r") as f:
        videos = json.load(f)["videos"]

    if args.output_json is None:
        args.output_json = os.path.join(args.pred_root, "evaluation_results_refytros_png.json")

    if args.video_id is not None:
        if args.video_id not in videos:
            raise KeyError(f"video_id not found in meta: {args.video_id}")
        eval_videos = {args.video_id: videos[args.video_id]}
    else:
        eval_videos = videos

    per_expression: Dict[str, Dict] = {}
    per_video: Dict[str, Dict] = {}

    total_expected_frames = 0
    total_missing_gt_frames = 0
    total_missing_pred_frames = 0
    total_missing_pred_dirs = 0

    for video_id in sorted(eval_videos.keys()):
        vid_info = eval_videos[video_id]
        exprs = vid_info.get("expressions", {})

        if args.video_id is not None and args.exp_id is not None:
            exp_ids = [str(args.exp_id)]
        else:
            exp_ids = sorted(exprs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

        v_results = []
        for exp_id in exp_ids:
            if exp_id not in exprs:
                continue

            one = evaluate_single_expression(
                video_id=video_id,
                exp_id=exp_id,
                vid_info=vid_info,
                pred_root=args.pred_root,
                data_root=args.data_root,
                split=args.split,
            )

            key = f"{video_id}_{exp_id}"
            per_expression[key] = {
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": exprs[exp_id].get("exp", ""),
                "obj_id": exprs[exp_id].get("obj_id", None),
                **one,
            }
            v_results.append(one)

            total_expected_frames += int(one["num_frames"])
            total_missing_gt_frames += int(one["missing_gt_frames"])
            total_missing_pred_frames += int(one["missing_pred_frames"])
            if not one["pred_dir_exists"]:
                total_missing_pred_dirs += 1

        if v_results:
            vj = float(np.mean([x["J"] for x in v_results]))
            vf = float(np.mean([x["F"] for x in v_results]))
            per_video[video_id] = {
                "J": vj,
                "F": vf,
                "J&F": (vj + vf) / 2.0,
                "num_expressions": len(v_results),
            }

    if not per_expression:
        raise RuntimeError("No expression was evaluated. Please check pred_root/meta/split.")

    expr_j = float(np.mean([x["J"] for x in per_expression.values()]))
    expr_f = float(np.mean([x["F"] for x in per_expression.values()]))
    video_j = float(np.mean([x["J"] for x in per_video.values()])) if per_video else 0.0
    video_f = float(np.mean([x["F"] for x in per_video.values()])) if per_video else 0.0

    out = {
        "overall_expression_level": {
            "J": round(expr_j * 100, 2),
            "F": round(expr_f * 100, 2),
            "J&F": round((expr_j + expr_f) * 50, 2),
            "num_expressions": len(per_expression),
        },
        "overall_video_level": {
            "J": round(video_j * 100, 2),
            "F": round(video_f * 100, 2),
            "J&F": round((video_j + video_f) * 50, 2),
            "num_videos": len(per_video),
        },
        "coverage": {
            "total_expected_frames": int(total_expected_frames),
            "total_missing_gt_frames": int(total_missing_gt_frames),
            "total_missing_pred_frames": int(total_missing_pred_frames),
            "total_missing_pred_dirs": int(total_missing_pred_dirs),
            "missing_pred_ratio": round(total_missing_pred_frames / max(total_expected_frames, 1), 6),
        },
        "per_video": per_video,
        "per_expression": per_expression,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 52)
    print("Ref-Youtube-ROS Evaluation Done")
    print("=" * 52)
    print(f"Evaluated {len(per_expression)} expressions across {len(per_video)} videos")
    print(f"Expression-level  J: {out['overall_expression_level']['J']:.2f}")
    print(f"Expression-level  F: {out['overall_expression_level']['F']:.2f}")
    print(f"Expression-level J&F: {out['overall_expression_level']['J&F']:.2f}")
    print(f"Video-level       J: {out['overall_video_level']['J']:.2f}")
    print(f"Video-level       F: {out['overall_video_level']['F']:.2f}")
    print(f"Video-level      J&F: {out['overall_video_level']['J&F']:.2f}")
    print(f"Missing pred frames: {out['coverage']['total_missing_pred_frames']} / {out['coverage']['total_expected_frames']}")
    print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()

