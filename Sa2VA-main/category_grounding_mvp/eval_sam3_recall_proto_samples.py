#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crf_sam3_demo import build_sam3_recall_backend


def frame_sort_key(name: str):
    stem = Path(name).stem
    return int(stem) if stem.isdigit() else stem


def load_category_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    data = json.load(path.open("r", encoding="utf-8"))
    records = data.get("records", data if isinstance(data, list) else [])
    out = {}
    for rec in records:
        out[(str(rec.get("video_id")), str(rec.get("exp_id")))] = rec
    return out


def load_gt_mask(data_root: Path, split: str, video_id: str, exp_id: str, frame: str, shape: tuple[int, int]) -> np.ndarray:
    p = data_root / split / "Annotations" / video_id / exp_id / f"{frame}.png"
    if not p.exists():
        return np.zeros(shape, dtype=np.uint8)
    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def infer_hw(jpeg_dir: Path, frame: str) -> tuple[int, int]:
    img = cv2.imread(str(jpeg_dir / f"{frame}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read frame: {jpeg_dir / (frame + '.jpg')}")
    return int(img.shape[0]), int(img.shape[1])


def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter / union)


def ranges(indices: list[int]) -> str:
    if not indices:
        return ""
    out = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        out.append(f"{start:05d}" if start == prev else f"{start:05d}-{prev:05d}")
        start = prev = idx
    out.append(f"{start:05d}" if start == prev else f"{start:05d}-{prev:05d}")
    return ", ".join(out)


def evaluate_tracks(
    tracks: dict[int, dict[int, np.ndarray]],
    data_root: Path,
    split: str,
    video_id: str,
    exp_id: str,
    frame_names: list[str],
    save_dir: Path,
) -> dict[str, Any]:
    h, w = infer_hw(data_root / split / "JPEGImages" / video_id, frame_names[0])
    gt_by_idx = {
        idx: load_gt_mask(data_root, split, video_id, exp_id, frame, (h, w))
        for idx, frame in enumerate(frame_names)
    }
    gt_present = [idx for idx, m in gt_by_idx.items() if m.any()]
    gt_empty = [idx for idx, m in gt_by_idx.items() if not m.any()]
    obj_ids = sorted({obj for per_frame in tracks.values() for obj in per_frame.keys()})
    candidates = []
    save_dir.mkdir(parents=True, exist_ok=True)
    for obj_id in obj_ids:
        obj_dir = save_dir / f"obj_{obj_id}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        js_all = []
        js_present = []
        empty_fp = 0
        pred_present = []
        for idx, frame in enumerate(frame_names):
            pred = tracks.get(idx, {}).get(obj_id)
            if pred is None:
                pred = np.zeros((h, w), dtype=np.uint8)
            pred = (pred > 0).astype(np.uint8)
            gt = gt_by_idx[idx]
            if pred.any():
                pred_present.append(idx)
                cv2.imwrite(str(obj_dir / f"{frame}.png"), pred * 255)
            score = jaccard(pred, gt)
            js_all.append(score)
            if gt.any():
                js_present.append(score)
            elif pred.any():
                empty_fp += 1
        candidates.append(
            {
                "obj_id": obj_id,
                "j_all": float(np.mean(js_all)) if js_all else 0.0,
                "j_present": float(np.mean(js_present)) if js_present else 0.0,
                "empty_fp_rate": float(empty_fp / len(gt_empty)) if gt_empty else 0.0,
                "pred_frames": len(pred_present),
                "pred_range": ranges(pred_present),
            }
        )
    candidates.sort(key=lambda x: (x["j_all"], x["j_present"]), reverse=True)
    return {
        "num_candidates": len(candidates),
        "gt_present_frames": len(gt_present),
        "gt_range": ranges(gt_present),
        "best_j_all": candidates[0]["j_all"] if candidates else 0.0,
        "best_j_present": candidates[0]["j_present"] if candidates else 0.0,
        "best_obj_id": candidates[0]["obj_id"] if candidates else None,
        "candidates": candidates,
    }


def recall_tracks(recall, video_path: Path, category: str, candidate_top_k: int) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, float]]]:
    tracks: dict[int, dict[int, np.ndarray]] = {}
    scores: dict[int, dict[int, float]] = {}
    for frame_idx, frame_masks, frame_scores in recall.recall_all_instances_stream(
        video_path=str(video_path),
        category=category,
        candidate_top_k=candidate_top_k,
    ):
        tracks[int(frame_idx)] = frame_masks
        scores[int(frame_idx)] = frame_scores
    return tracks, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/crf_sam3_config_query_grounding_refytros_valid_20260701.yaml")
    ap.add_argument("--proto-config", default="configs/crf_sam3_config_query_grounding_refytros_valid_lvis_proto_alpha02_20260702.yaml")
    ap.add_argument("--data-root", default="/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--meta-file", default="meta_expressions_challenge.json")
    ap.add_argument("--category-cache", default="category_grounding_mvp/outputs/refytros_valid_query_grounding_categories_20260701.json")
    ap.add_argument("--output-dir", default="category_grounding_mvp/outputs/sam3_lvis_proto_sample_recall_20260702")
    ap.add_argument("--sample", action="append", required=True, help="video_id/exp_id")
    ap.add_argument("--candidate-top-k", type=int, default=7)
    ap.add_argument("--override-category", default=None, help="Use this category prompt for all samples.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    meta = json.load((data_root / args.split / args.meta_file).open("r", encoding="utf-8"))["videos"]
    category_cache = load_category_cache(Path(args.category_cache))
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "baseline": yaml.safe_load(open(args.base_config, "r", encoding="utf-8")),
        "lvis_proto_alpha02": yaml.safe_load(open(args.proto_config, "r", encoding="utf-8")),
    }
    summary = []
    for method, cfg in configs.items():
        print(f"[load] method={method}", flush=True)
        recall = build_sam3_recall_backend(cfg["model"]["sam3_checkpoint"], cfg["sam3_recall"])
        for sample in args.sample:
            video_id, exp_id = sample.split("/", 1)
            exp = meta[video_id]["expressions"][str(exp_id)]
            rec = category_cache.get((video_id, str(exp_id)), {})
            category = args.override_category or rec.get("pred_category") or rec.get("category")
            if not category:
                raise RuntimeError(f"No category for {sample}")
            frame_names = sorted(
                [p.stem for p in (data_root / args.split / "JPEGImages" / video_id).glob("*.jpg")],
                key=frame_sort_key,
            )
            print(f"[run] {method} {sample} category={category!r} frames={len(frame_names)}", flush=True)
            tracks, scores = recall_tracks(
                recall,
                data_root / args.split / "JPEGImages" / video_id,
                category,
                args.candidate_top_k,
            )
            item_dir = output_dir / method / video_id / str(exp_id)
            metrics = evaluate_tracks(tracks, data_root, args.split, video_id, str(exp_id), frame_names, item_dir / "candidates")
            item = {
                "method": method,
                "video_id": video_id,
                "exp_id": str(exp_id),
                "expression": exp.get("exp", ""),
                "obj_id": exp.get("obj_id"),
                "category": category,
                **metrics,
            }
            json.dump(item, (item_dir / "metrics.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
            summary.append(item)
            print(
                f"[metrics] {method} {sample} cand={item['num_candidates']} "
                f"bestJ_all={item['best_j_all']:.4f} bestJ_present={item['best_j_present']:.4f} "
                f"bestObj={item['best_obj_id']} gt={item['gt_range']}",
                flush=True,
            )
        del recall
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    json.dump({"records": summary}, (output_dir / "summary.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
