#!/usr/bin/env python3
"""Evaluate only complete predictions in a running ReVOS results snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as cocomask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from third_parts.revos.utils.metircs import db_eval_boundary, db_eval_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=Path)
    parser.add_argument("--exp-path", type=Path, required=True)
    parser.add_argument("--mask-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--read-retries", type=int, default=10)
    return parser.parse_args()


def load_snapshot(path: Path, retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1.0)
    raise RuntimeError(f"Unable to read a consistent snapshot from {path}: {last_error}")


def decode_gt(video: dict[str, Any], anno_ids: list[Any], masks: dict[str, Any]) -> np.ndarray:
    frame_count = len(video["frames"])
    result = np.zeros((frame_count, int(video["height"]), int(video["width"])), dtype=bool)
    for anno_id in anno_ids:
        anno_frames = masks.get(str(anno_id), [])
        for frame_index, rle in enumerate(anno_frames[:frame_count]):
            if not rle:
                continue
            decoded = cocomask.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2)
            result[frame_index] |= decoded.astype(bool)
    return result


def evaluate_masks(gt_masks: np.ndarray, prediction_rles: list[Any]) -> tuple[float, float]:
    j_values: list[float] = []
    f_values: list[float] = []
    for frame_index, rle in enumerate(prediction_rles):
        prediction = cocomask.decode(rle)
        if prediction.ndim == 3:
            prediction = prediction.any(axis=2)
        prediction = prediction.astype(bool)
        gt = gt_masks[frame_index]
        if prediction.shape != gt.shape:
            raise ValueError(f"mask shape mismatch: prediction={prediction.shape}, gt={gt.shape}")
        j_values.append(float(db_eval_iou(gt, prediction)))
        f_values.append(float(db_eval_boundary(gt, prediction)))
    return float(np.mean(j_values)), float(np.mean(f_values))


WORKER_META: dict[str, Any] = {}
WORKER_MASKS: dict[str, Any] = {}


def init_worker(meta: dict[str, Any], masks: dict[str, Any]) -> None:
    global WORKER_META, WORKER_MASKS
    WORKER_META = meta
    WORKER_MASKS = masks


def evaluate_video(job: tuple[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    video_id, predictions = job
    video = WORKER_META.get(video_id)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    if video is None or not isinstance(predictions, dict):
        return rows, [{"video_id": video_id, "reason": "video_missing_from_metadata"}]

    frame_count = len(video["frames"])
    gt_cache: dict[tuple[str, ...], np.ndarray] = {}
    for exp_id, prediction_record in predictions.items():
        expression = video.get("expressions", {}).get(str(exp_id))
        if expression is None:
            invalid.append({"video_id": video_id, "exp_id": str(exp_id), "reason": "expression_missing"})
            continue
        prediction_rles = prediction_record.get("prediction_masks", []) if isinstance(prediction_record, dict) else []
        if not isinstance(prediction_rles, list) or len(prediction_rles) != frame_count:
            invalid.append(
                {
                    "video_id": video_id,
                    "exp_id": str(exp_id),
                    "reason": "incomplete_prediction_frames",
                    "expected": frame_count,
                    "actual": len(prediction_rles) if isinstance(prediction_rles, list) else None,
                }
            )
            continue
        anno_key = tuple(str(value) for value in expression.get("anno_id", []))
        if anno_key not in gt_cache:
            gt_cache[anno_key] = decode_gt(video, list(anno_key), WORKER_MASKS)
        try:
            j_value, f_value = evaluate_masks(gt_cache[anno_key], prediction_rles)
        except Exception as error:
            invalid.append({"video_id": video_id, "exp_id": str(exp_id), "reason": "metric_error", "error": str(error)})
            continue
        rows.append(
            {
                "video_id": video_id,
                "exp_id": str(exp_id),
                "expression": expression.get("exp", ""),
                "type_id": int(expression.get("type_id", 2)),
                "J_raw": j_value,
                "F_raw": f_value,
                "J": round(100.0 * j_value, 4),
                "F": round(100.0 * f_value, 4),
                "JF": round(50.0 * (j_value + f_value), 4),
            }
        )
    return rows, invalid


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def pct(value: float) -> float | None:
    return None if math.isnan(value) else round(100.0 * value, 4)


def group_metrics(rows: list[dict[str, Any]], type_id: int) -> tuple[dict[str, Any], float, float]:
    selected = [row for row in rows if row["type_id"] == type_id]
    j_value = mean([row["J_raw"] for row in selected])
    f_value = mean([row["F_raw"] for row in selected])
    return {
        "count": len(selected),
        "J": pct(j_value),
        "F": pct(f_value),
        "JF": pct((j_value + f_value) / 2.0),
    }, j_value, f_value


def main() -> None:
    args = parse_args()
    meta = load_snapshot(args.exp_path, args.read_retries)["videos"]
    predictions = load_snapshot(args.pred_path, args.read_retries)
    masks = load_snapshot(args.mask_path, args.read_retries)

    total_by_type: dict[int, int] = {}
    for video in meta.values():
        for expression in video.get("expressions", {}).values():
            type_id = int(expression.get("type_id", 2))
            total_by_type[type_id] = total_by_type.get(type_id, 0) + 1

    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    jobs = list(predictions.items())
    worker_count = max(1, min(args.workers, len(jobs)))
    context = mp.get_context("fork")
    with context.Pool(worker_count, initializer=init_worker, initargs=(meta, masks)) as pool:
        for video_rows, video_invalid in pool.imap_unordered(evaluate_video, jobs):
            rows.extend(video_rows)
            invalid.extend(video_invalid)
            print(f"[partial-eval] evaluated={len(rows)} videos_done_progress", flush=True)

    referring, referring_j, referring_f = group_metrics(rows, 0)
    reasoning, reasoning_j, reasoning_f = group_metrics(rows, 1)
    overall_j = (referring_j + reasoning_j) / 2.0
    overall_f = (referring_f + reasoning_f) / 2.0
    official_overall = {
        "count": referring["count"] + reasoning["count"],
        "J": pct(overall_j),
        "F": pct(overall_f),
        "JF": pct((overall_j + overall_f) / 2.0),
    }

    scored = [row for row in rows if row["type_id"] in (0, 1)]
    micro_j = mean([row["J_raw"] for row in scored])
    micro_f = mean([row["F_raw"] for row in scored])
    completed_by_type: dict[int, int] = {}
    for row in rows:
        completed_by_type[row["type_id"]] = completed_by_type.get(row["type_id"], 0) + 1

    total_expressions = sum(total_by_type.values())
    payload = {
        "scope": "partial snapshot; complete predictions only",
        "warning": "The running job processes a non-random dataset prefix, so this is not the final full-validation metric.",
        "prediction_path": str(args.pred_path),
        "total_expressions": total_expressions,
        "evaluated_expressions": len(rows),
        "invalid_predictions": len(invalid),
        "coverage_percent": round(100.0 * len(rows) / max(total_expressions, 1), 4),
        "coverage_by_type": {
            str(type_id): {
                "completed": completed_by_type.get(type_id, 0),
                "total": count,
                "percent": round(100.0 * completed_by_type.get(type_id, 0) / max(count, 1), 4),
            }
            for type_id, count in sorted(total_by_type.items())
        },
        "metrics": {
            "referring": referring,
            "reasoning": reasoning,
            "overall_type_balanced_official": official_overall,
            "completed_sample_weighted_audit": {
                "count": len(scored),
                "J": pct(micro_j),
                "F": pct(micro_f),
                "JF": pct((micro_j + micro_f) / 2.0),
            },
        },
        "invalid_examples": invalid[:100],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["video_id", "exp_id", "expression", "type_id", "J", "F", "JF"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fieldnames})
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
