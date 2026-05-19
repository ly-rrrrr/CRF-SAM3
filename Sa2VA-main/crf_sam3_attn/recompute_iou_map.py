#!/usr/bin/env python3
"""Recompute offline-cache iou_map without rerunning SAM3/Qwen.

This script reads cached candidate masks from:
  offline_cache_root/<split>/<video_id>/<exp_id>/masks/obj_*/<frame>.png
and recomputes `iou_map` in meta.json using GT masks.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import yaml


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from crf_sam3_attn.dataset import CRFSAM3AttnDataset, summarize_iou_map  # noqa: E402


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    p1 = os.path.abspath(os.path.join(base_dir, path))
    if os.path.exists(p1):
        return p1
    p2 = os.path.abspath(os.path.join(_REPO_ROOT, path))
    return p2


def _parse_obj_id(dirname: str):
    if not dirname.startswith("obj_"):
        return None
    tail = dirname[4:]
    if not tail.isdigit():
        return None
    return int(tail)


def _load_cached_all_masks(exp_dir: str) -> Dict[int, Dict[int, np.ndarray]]:
    all_masks: Dict[int, Dict[int, np.ndarray]] = {}
    mask_root = os.path.join(exp_dir, "masks")
    if not os.path.isdir(mask_root):
        return all_masks

    for obj_dir in sorted(glob.glob(os.path.join(mask_root, "obj_*"))):
        obj_name = os.path.basename(obj_dir)
        obj_id = _parse_obj_id(obj_name)
        if obj_id is None:
            continue
        for mask_path in sorted(glob.glob(os.path.join(obj_dir, "*.png"))):
            stem = os.path.splitext(os.path.basename(mask_path))[0]
            if not stem.isdigit():
                continue
            frame_idx = int(stem)
            mask = np.array(Image.open(mask_path))
            all_masks.setdefault(frame_idx, {})[obj_id] = (mask > 0).astype(np.uint8)
    return all_masks


def _iter_meta_paths(split_root: str, video_id: str = None, exp_id: int = None):
    if video_id is not None and exp_id is not None:
        p = os.path.join(split_root, video_id, str(exp_id), "meta.json")
        if os.path.isfile(p):
            yield p
        return
    if video_id is not None:
        for p in sorted(glob.glob(os.path.join(split_root, video_id, "*", "meta.json"))):
            yield p
        return
    for p in sorted(glob.glob(os.path.join(split_root, "*", "*", "meta.json"))):
        yield p


def _update_iou_map(dataset: CRFSAM3AttnDataset, meta_path: str, dry_run: bool = False) -> Tuple[bool, str]:
    exp_dir = os.path.dirname(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    video_id = meta.get("video_id")
    exp_id = int(meta.get("exp_id"))

    all_masks = _load_cached_all_masks(exp_dir)
    if not all_masks:
        return False, "no_cached_masks"

    gt_masks = dataset._load_expression_gt_masks(video_id, exp_id)
    if not gt_masks:
        return False, "no_gt_masks"

    iou_map = dataset._compute_iou_map(all_masks, gt_masks)
    if not iou_map:
        return False, "empty_iou_map"

    new_iou_map = {str(int(k)): float(v) for k, v in sorted(iou_map.items(), key=lambda kv: int(kv[0]))}
    iou_summary = summarize_iou_map(new_iou_map)
    old_iou_map = meta.get("iou_map", {})
    old_summary = {
        "max_j": float(meta.get("max_j", 0.0)),
        "sum_j": float(meta.get("sum_j", 0.0)),
        "mean_j": float(meta.get("mean_j", 0.0)),
        "num_candidates": int(meta.get("num_candidates", 0)),
    }
    changed = old_iou_map != new_iou_map or old_summary != iou_summary
    if changed:
        meta["iou_map"] = new_iou_map
        meta["max_j"] = iou_summary["max_j"]
        meta["sum_j"] = iou_summary["sum_j"]
        meta["mean_j"] = iou_summary["mean_j"]
        meta["num_candidates"] = iou_summary["num_candidates"]
        if not dry_run:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
    return changed, "ok"


def main():
    parser = argparse.ArgumentParser(description="Recompute iou_map for offline cache.")
    parser.add_argument("--config", required=True, help="Training config yaml, e.g. crf_sam3_attn/config_refytros_train_train.yaml")
    parser.add_argument("--video_id", default=None, help="Optional single video id")
    parser.add_argument("--exp_id", type=int, default=None, help="Optional single exp id (must be used with --video_id)")
    parser.add_argument("--cache_root", default=None, help="Override offline cache root")
    parser.add_argument("--data_root", default=None, help="Override base_cfg dataset.data_root")
    parser.add_argument("--gt_masks_root", default=None, help="Override base_cfg dataset.gt_masks_root")
    parser.add_argument("--mask_dict_path", default=None, help="Override train_cfg data.mask_dict_path")
    parser.add_argument("--gt_source", default=None, help="Override GT source: auto, gt_masks_root, or mask_dict")
    parser.add_argument("--dry_run", action="store_true", help="Compute and report only, do not write meta.json")
    parser.add_argument("--report_every", type=int, default=200, help="Progress print interval")
    args = parser.parse_args()

    if args.exp_id is not None and args.video_id is None:
        raise ValueError("--exp_id requires --video_id")

    cfg_path = os.path.abspath(args.config)
    cfg_dir = os.path.dirname(cfg_path)

    train_cfg = _load_yaml(cfg_path)
    base_cfg_path = _resolve_path(train_cfg["base_config"], cfg_dir)
    base_cfg = _load_yaml(base_cfg_path)
    data_cfg = dict(train_cfg.get("data", {}))

    if args.data_root:
        base_cfg.setdefault("dataset", {})["data_root"] = args.data_root
    if args.gt_masks_root:
        base_cfg.setdefault("dataset", {})["gt_masks_root"] = args.gt_masks_root
    if args.mask_dict_path:
        data_cfg["mask_dict_path"] = args.mask_dict_path
    if args.gt_source:
        data_cfg["gt_source"] = args.gt_source

    # Offline mode to avoid loading heavy models.
    data_cfg["offline_mode"] = True
    data_cfg["offline_prepare"] = False
    data_cfg["offline_strict"] = False

    dataset = CRFSAM3AttnDataset(base_cfg, data_cfg)
    split = str(data_cfg.get("split") or base_cfg["dataset"]["split"])

    cache_root = args.cache_root or data_cfg.get("offline_cache_root", "crf_sam3_attn/offline_cache")
    if not os.path.isabs(cache_root):
        cache_root = os.path.join(_REPO_ROOT, cache_root)
    split_root = os.path.join(os.path.abspath(cache_root), split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Offline split root not found: {split_root}")

    total = 0
    updated = 0
    skipped = 0
    reason_count = {}
    for meta_path in _iter_meta_paths(split_root, args.video_id, args.exp_id):
        total += 1
        changed, reason = _update_iou_map(dataset, meta_path, dry_run=args.dry_run)
        if reason != "ok":
            skipped += 1
            reason_count[reason] = reason_count.get(reason, 0) + 1
        elif changed:
            updated += 1
        if args.report_every > 0 and total % args.report_every == 0:
            print(f"[progress] processed={total} updated={updated} skipped={skipped}")

    print("=== Recompute iou_map done ===")
    print(f"split_root: {split_root}")
    print(f"processed: {total}")
    print(f"updated:   {updated}")
    print(f"skipped:   {skipped}")
    if reason_count:
        print(f"skip_reasons: {reason_count}")
    print(f"gt_source: {dataset.gt_source}")
    print(f"dry_run:   {args.dry_run}")


if __name__ == "__main__":
    main()
