#!/usr/bin/env python3
"""Evaluate only complete ReVOS predictions from a live results snapshot.

This follows the official ReVOS per-expression J/F definitions, but does not
assume that every expression has already been predicted.  It is intended for
an in-progress run and never changes the prediction JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from pycocotools import mask as cocomask

from third_parts.revos.utils.metircs import db_eval_boundary, db_eval_iou


_VIDEOS: Dict[str, Any] = {}
_PREDICTIONS: Dict[str, Any] = {}
_MASKS: Dict[str, Any] = {}


def read_json_retry(path: Path, retries: int = 20) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(min(0.25 * (attempt + 1), 2.0))
    raise RuntimeError(f"Could not read a stable JSON snapshot: {path}: {last_error}")


def decode_mask(rle: Any, height: int, width: int) -> np.ndarray:
    if not rle:
        return np.zeros((height, width), dtype=bool)
    decoded = cocomask.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def expression_is_complete(video: Dict[str, Any], prediction: Dict[str, Any]) -> bool:
    frames = video.get("frames", [])
    masks = prediction.get("prediction_masks")
    return isinstance(masks, list) and len(masks) == len(frames)


def evaluate_expression(item: Tuple[str, str]) -> Dict[str, Any]:
    video_id, exp_id = item
    video = _VIDEOS[video_id]
    prediction = _PREDICTIONS[video_id][exp_id]
    height = int(video["height"])
    width = int(video["width"])
    frame_count = len(video["frames"])

    gt_masks = np.zeros((frame_count, height, width), dtype=bool)
    pred_masks = np.zeros((frame_count, height, width), dtype=bool)
    anno_ids = video["expressions"][exp_id].get("anno_id", [])

    for frame_index in range(frame_count):
        for anno_id in anno_ids:
            anno_frames = _MASKS.get(str(anno_id), [])
            if frame_index < len(anno_frames):
                gt_masks[frame_index] |= decode_mask(anno_frames[frame_index], height, width)
        pred_masks[frame_index] = decode_mask(
            prediction["prediction_masks"][frame_index], height, width
        )

    j = float(db_eval_iou(gt_masks, pred_masks).mean())
    f = float(db_eval_boundary(gt_masks, pred_masks).mean())
    return {
        "video_name": video_id,
        "exp_id": exp_id,
        "exp": video["expressions"][exp_id].get("exp", ""),
        "type_id": video["expressions"][exp_id].get("type_id"),
        "J": j,
        "F": f,
        "JF": (j + f) / 2.0,
        "frame_count": frame_count,
    }


def init_worker() -> None:
    # The parent loads the read-only objects before fork; avoid pickling the
    # 294 MB mask dictionary into every worker.
    return None


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = {
        "referring": [row for row in rows if row.get("type_id") == 0],
        "reason": [row for row in rows if row.get("type_id") == 1],
    }

    def mean_group(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group_rows:
            return {"count": 0, "J": None, "F": None, "JF": None}
        j = float(np.mean([row["J"] for row in group_rows]))
        f = float(np.mean([row["F"] for row in group_rows]))
        return {"count": len(group_rows), "J": j, "F": f, "JF": (j + f) / 2.0}

    group_results = {name: mean_group(group_rows) for name, group_rows in groups.items()}
    available = [result for result in group_results.values() if result["count"] > 0]
    official_partial = {
        "count": len(rows),
        "J": float(np.mean([result["J"] for result in available])) if available else None,
        "F": float(np.mean([result["F"] for result in available])) if available else None,
        "JF": float(np.mean([result["JF"] for result in available])) if available else None,
        "aggregation": "equal mean of available referring/reason groups; official overall protocol when both groups are complete",
    }
    micro = mean_group(rows)
    return {
        "num_completed_expressions": len(rows),
        "groups": group_results,
        "official_partial_overall": official_partial,
        "micro_expression_average": micro,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--exp-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_payload = read_json_retry(Path(args.pred_path))
    exp_payload = read_json_retry(Path(args.exp_path))
    masks = read_json_retry(Path(args.mask_path))
    videos = exp_payload["videos"]

    predictions = {
        video_id: exp_results
        for video_id, exp_results in pred_payload.items()
        if video_id in videos and isinstance(exp_results, dict)
    }
    global _VIDEOS, _PREDICTIONS, _MASKS
    _VIDEOS = videos
    _PREDICTIONS = predictions
    _MASKS = masks
    expected_items = sum(len(video.get("expressions", {})) for video in videos.values())
    complete_items = [
        (video_id, str(exp_id))
        for video_id, exp_results in predictions.items()
        for exp_id, prediction in exp_results.items()
        if str(exp_id) in videos[video_id].get("expressions", {})
        and expression_is_complete(videos[video_id], prediction)
    ]

    if args.workers <= 1:
        rows = [evaluate_expression(item) for item in complete_items]
    else:
        # Fork shares the read-only JSON objects through copy-on-write on Linux.
        context = mp.get_context("fork")
        with context.Pool(
            processes=args.workers,
            initializer=init_worker,
        ) as pool:
            rows = list(pool.imap(evaluate_expression, complete_items, chunksize=1))

    rows.sort(key=lambda row: (row["video_name"], str(row["exp_id"])))
    summary = aggregate(rows)
    summary.update(
        {
            "prediction_path": str(Path(args.pred_path).resolve()),
            "expression_path": str(Path(args.exp_path).resolve()),
            "mask_path": str(Path(args.mask_path).resolve()),
            "complete_prediction_count": len(complete_items),
            "expected_expression_count": expected_items,
            "incomplete_prediction_count": expected_items - len(complete_items),
            "metric_source": "third_parts.revos.utils.metircs.db_eval_iou/db_eval_boundary",
        }
    )
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_name", "exp_id", "type_id", "J", "F", "JF", "frame_count", "exp"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_json={output_json}")
    print(f"saved_csv={output_csv}")


if __name__ == "__main__":
    main()
