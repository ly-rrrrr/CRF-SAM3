#!/usr/bin/env python3
"""Evaluate Ref-Youtube-ROS PNG predictions with patch-over-base roots.

For each expression, the evaluator reads prediction masks from --patch-root if
that expression directory exists; otherwise it falls back to --base-root. This
lets a small failure-only rerun be evaluated as a full validation result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_DIR = os.path.join(ROOT, "third_parts", "revos", "utils")
if METRICS_DIR not in sys.path:
    sys.path.append(METRICS_DIR)
from metircs import db_eval_boundary, db_eval_iou


def _resolve_expr_dir(root: str, video_id: str, exp_id: str) -> str:
    cands = [
        os.path.join(root, "Annotations", video_id, exp_id),
        os.path.join(root, video_id, exp_id),
    ]
    for path in cands:
        if os.path.isdir(path):
            return path
    return cands[0]


def _resolve_pred_dir(base_root: str, patch_root: str, video_id: str, exp_id: str) -> tuple[str, str]:
    patch_dir = _resolve_expr_dir(patch_root, video_id, exp_id)
    if os.path.isdir(patch_dir):
        return patch_dir, "patch"
    return _resolve_expr_dir(base_root, video_id, exp_id), "base"


def _resolve_gt_dir(data_root: str, split: str, video_id: str, exp_id: str) -> str:
    return os.path.join(data_root, split, "Annotations", video_id, exp_id)


def _infer_hw(data_root: str, split: str, video_id: str, exp_id: str, frames: List[str]) -> Tuple[int, int]:
    gt_dir = _resolve_gt_dir(data_root, split, video_id, exp_id)
    for frame in frames:
        path = os.path.join(gt_dir, f"{frame}.png")
        if os.path.exists(path):
            mask = np.array(Image.open(path))
            return int(mask.shape[0]), int(mask.shape[1])
    jpeg_dir = os.path.join(data_root, split, "JPEGImages", video_id)
    for frame in frames:
        path = os.path.join(jpeg_dir, f"{frame}.jpg")
        if os.path.exists(path):
            image = np.array(Image.open(path).convert("L"))
            return int(image.shape[0]), int(image.shape[1])
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


def evaluate_one(
    video_id: str,
    exp_id: str,
    vid_info: Dict,
    base_root: str,
    patch_root: str,
    data_root: str,
    split: str,
) -> Dict:
    frames = [str(x) for x in vid_info["frames"]]
    h, w = _infer_hw(data_root, split, video_id, exp_id, frames)
    pred_dir, pred_source = _resolve_pred_dir(base_root, patch_root, video_id, exp_id)
    gt_dir = _resolve_gt_dir(data_root, split, video_id, exp_id)

    gt_masks = np.zeros((len(frames), h, w), dtype=np.uint8)
    pred_masks = np.zeros((len(frames), h, w), dtype=np.uint8)
    missing_gt_frames = 0
    missing_pred_frames = 0

    for i, frame in enumerate(frames):
        gt_path = os.path.join(gt_dir, f"{frame}.png")
        pred_path = os.path.join(pred_dir, f"{frame}.png")
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
        "num_frames": len(frames),
        "missing_gt_frames": missing_gt_frames,
        "missing_pred_frames": missing_pred_frames,
        "pred_dir_exists": os.path.isdir(pred_dir),
        "pred_source": pred_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="valid", choices=["valid", "test", "train"])
    parser.add_argument("--meta-file", default="meta_expressions_challenge.json")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    meta_path = os.path.join(args.data_root, args.split, args.meta_file)
    with open(meta_path, "r", encoding="utf-8") as f:
        videos = json.load(f)["videos"]

    per_expression: Dict[str, Dict] = {}
    per_video: Dict[str, Dict] = {}
    patch_count = 0
    base_count = 0
    total_expected_frames = 0
    total_missing_gt_frames = 0
    total_missing_pred_frames = 0
    total_missing_pred_dirs = 0

    for video_id in sorted(videos.keys()):
        vid_info = videos[video_id]
        exprs = vid_info.get("expressions", {})
        v_results = []
        for exp_id in sorted(exprs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            one = evaluate_one(video_id, str(exp_id), vid_info, args.base_root, args.patch_root, args.data_root, args.split)
            if one["pred_source"] == "patch":
                patch_count += 1
            else:
                base_count += 1
            expr_info = exprs[exp_id]
            one.update(
                {
                    "video_id": video_id,
                    "exp_id": str(exp_id),
                    "expression": expr_info.get("exp", ""),
                    "obj_id": expr_info.get("obj_id"),
                }
            )
            key = f"{video_id}_{exp_id}"
            per_expression[key] = one
            v_results.append(one)
            total_expected_frames += int(one["num_frames"])
            total_missing_gt_frames += int(one["missing_gt_frames"])
            total_missing_pred_frames += int(one["missing_pred_frames"])
            if not one["pred_dir_exists"]:
                total_missing_pred_dirs += 1
        if v_results:
            per_video[video_id] = {
                "J": float(np.mean([x["J"] for x in v_results])),
                "F": float(np.mean([x["F"] for x in v_results])),
                "J&F": float(np.mean([x["J&F"] for x in v_results])),
                "num_expressions": len(v_results),
            }

    expr_values = list(per_expression.values())
    video_values = list(per_video.values())
    result = {
        "overall_expression_level": {
            "J": round(float(np.mean([x["J"] for x in expr_values])) * 100.0, 2),
            "F": round(float(np.mean([x["F"] for x in expr_values])) * 100.0, 2),
            "J&F": round(float(np.mean([x["J&F"] for x in expr_values])) * 100.0, 2),
            "num_expressions": len(expr_values),
        },
        "overall_video_level": {
            "J": round(float(np.mean([x["J"] for x in video_values])) * 100.0, 2),
            "F": round(float(np.mean([x["F"] for x in video_values])) * 100.0, 2),
            "J&F": round(float(np.mean([x["J&F"] for x in video_values])) * 100.0, 2),
            "num_videos": len(video_values),
        },
        "overlay": {
            "base_root": args.base_root,
            "patch_root": args.patch_root,
            "patch_expressions": patch_count,
            "base_expressions": base_count,
        },
        "coverage": {
            "total_expected_frames": total_expected_frames,
            "total_missing_gt_frames": total_missing_gt_frames,
            "total_missing_pred_frames": total_missing_pred_frames,
            "total_missing_pred_dirs": total_missing_pred_dirs,
            "missing_pred_ratio": total_missing_pred_frames / max(total_expected_frames, 1),
        },
        "per_video": per_video,
        "per_expression": per_expression,
    }
    output_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["overall_expression_level"], ensure_ascii=False, indent=2))
    print(json.dumps(result["overlay"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
