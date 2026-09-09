#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as cocomask
except Exception:
    cocomask = None


def resolve_meta_path(data_root: Path, split: str, explicit: Optional[Path]) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        data_root / split / "meta_expressions.json",
        data_root / "meta_expressions" / split / "meta_expressions.json",
        data_root / "meta_expressions.json",
    ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("meta_expressions.json not found. Tried: " + ", ".join(str(p) for p in candidates))


def resolve_mask_dict_path(data_root: Path, split: str, explicit: Optional[Path]) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend([
        data_root / split / "mask_dict.pkl",
        data_root / split / "mask_dict.json",
        data_root / "mask_dict.pkl",
        data_root / "mask_dict.json",
    ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("mask_dict not found. Tried: " + ", ".join(str(p) for p in candidates))


def load_meta(meta_path: Path) -> Dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mask_dict(mask_dict_path: Path) -> Dict[Any, Any]:
    if mask_dict_path.suffix == ".pkl":
        with mask_dict_path.open("rb") as f:
            return pickle.load(f)
    with mask_dict_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_total_expressions(meta: Dict[str, Any]) -> int:
    return sum(len(v.get("expressions", {})) for v in meta.get("videos", {}).values())


def infer_mask_dict_mode(meta: Dict[str, Any], mask_dict: Dict[Any, Any]) -> str:
    if not isinstance(mask_dict, dict) or not mask_dict:
        return "obj_or_anno"
    int_keys = []
    for key in mask_dict.keys():
        try:
            int_keys.append(int(key))
        except (TypeError, ValueError):
            return "obj_or_anno"
    total_expr = count_total_expressions(meta)
    if total_expr > 0 and len(int_keys) == total_expr and min(int_keys) == 0 and max(int_keys) == total_expr - 1:
        return "expression_index"
    return "obj_or_anno"


def build_expression_index(meta: Dict[str, Any], sorted_order: bool) -> Dict[Tuple[str, int], int]:
    videos = meta.get("videos", {})
    index_map: Dict[Tuple[str, int], int] = {}
    idx = 0
    video_items = videos.items()
    if sorted_order:
        video_items = [(vid, videos[vid]) for vid in sorted(videos.keys())]
    for video_id, video_meta in video_items:
        exprs = video_meta.get("expressions", {})
        exp_ids = exprs.keys()
        if sorted_order:
            exp_ids = sorted(exp_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))
        for exp_id in exp_ids:
            index_map[(str(video_id), int(exp_id))] = idx
            idx += 1
    return index_map


def compare_index_orders(meta: Dict[str, Any]) -> Dict[str, Any]:
    insertion = build_expression_index(meta, sorted_order=False)
    sorted_map = build_expression_index(meta, sorted_order=True)
    mismatches = []
    for key, insertion_idx in insertion.items():
        sorted_idx = sorted_map.get(key)
        if sorted_idx != insertion_idx:
            mismatches.append({
                "video_id": key[0],
                "exp_id": key[1],
                "insertion_idx": insertion_idx,
                "sorted_idx": sorted_idx,
            })
            if len(mismatches) >= 10:
                break
    return {
        "num_total": len(insertion),
        "num_mismatches": sum(1 for key, v in insertion.items() if sorted_map.get(key) != v),
        "first_mismatches": mismatches,
        "insertion_index": insertion,
        "sorted_index": sorted_map,
    }


def decode_mask_rle(mask_rle: Any) -> Optional[np.ndarray]:
    if cocomask is None or not mask_rle:
        return None
    if isinstance(mask_rle, (list, tuple)):
        merged = None
        for item in mask_rle:
            decoded = decode_mask_rle(item)
            if decoded is None:
                continue
            if merged is None:
                merged = decoded.astype(np.uint8)
            else:
                merged = np.logical_or(merged > 0, decoded > 0).astype(np.uint8)
        return merged
    if isinstance(mask_rle, dict):
        counts = mask_rle.get("counts")
        if isinstance(counts, str):
            mask_rle = dict(mask_rle)
            mask_rle["counts"] = counts.encode("utf-8")
    decoded = cocomask.decode(mask_rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2).astype(np.uint8)
    return decoded.astype(np.uint8)


def load_expression_masks(
    meta: Dict[str, Any],
    mask_dict: Dict[Any, Any],
    mode: str,
    video_id: str,
    exp_id: int,
    order_report: Dict[str, Any],
) -> Dict[str, Any]:
    videos = meta.get("videos", {})
    vid_info = videos.get(video_id)
    if vid_info is None:
        raise KeyError(f"video_id not found: {video_id}")
    exp_info = vid_info.get("expressions", {}).get(str(exp_id))
    if exp_info is None:
        raise KeyError(f"exp_id not found under {video_id}: {exp_id}")

    num_frames = len(vid_info.get("frames", []))
    result = {
        "video_id": video_id,
        "exp_id": int(exp_id),
        "expression": exp_info.get("exp", ""),
        "mode": mode,
        "num_video_frames": num_frames,
        "obj_id": exp_info.get("obj_id", []),
        "anno_id": exp_info.get("anno_id", []),
    }

    if mode == "expression_index":
        insertion_idx = order_report["insertion_index"].get((video_id, int(exp_id)))
        sorted_idx = order_report["sorted_index"].get((video_id, int(exp_id)))
        result["insertion_index"] = insertion_idx
        result["sorted_index"] = sorted_idx
        seq = mask_dict.get(str(insertion_idx))
        if seq is None:
            seq = mask_dict.get(insertion_idx)
        if not isinstance(seq, list):
            raise KeyError(f"mask_dict sequence not found for expression index {insertion_idx}")
        gt_masks: Dict[int, np.ndarray] = {}
        upper = len(seq) if num_frames <= 0 else min(num_frames, len(seq))
        for frame_idx in range(upper):
            decoded = decode_mask_rle(seq[frame_idx])
            if decoded is not None:
                gt_masks[frame_idx] = decoded
        result["sequence_len"] = len(seq)
        result["decoded_num_frames"] = len(gt_masks)
        result["frame_indices"] = sorted(gt_masks.keys())
        return {"meta": result, "masks": gt_masks}

    target_ids = exp_info.get("anno_id", []) or exp_info.get("obj_id", []) or []
    if isinstance(target_ids, (int, str)):
        target_ids = [target_ids]
    target_ids = [int(x) for x in target_ids]
    seqs = []
    max_len = 0
    for target_id in target_ids:
        seq = mask_dict.get(str(target_id))
        if seq is None:
            seq = mask_dict.get(target_id)
        if isinstance(seq, list):
            seqs.append((target_id, seq))
            max_len = max(max_len, len(seq))
    gt_masks: Dict[int, np.ndarray] = {}
    for frame_idx in range(max_len):
        merged = None
        for _, seq in seqs:
            if frame_idx >= len(seq):
                continue
            decoded = decode_mask_rle(seq[frame_idx])
            if decoded is None:
                continue
            if merged is None:
                merged = decoded.astype(np.uint8)
            else:
                merged = np.logical_or(merged > 0, decoded > 0).astype(np.uint8)
        if merged is not None:
            gt_masks[frame_idx] = merged
    result["target_ids"] = target_ids
    result["sequence_lengths"] = {str(target_id): len(seq) for target_id, seq in seqs}
    result["decoded_num_frames"] = len(gt_masks)
    result["frame_indices"] = sorted(gt_masks.keys())
    return {"meta": result, "masks": gt_masks}


def export_masks(export_dir: Path, masks: Dict[int, np.ndarray]) -> List[str]:
    export_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for frame_idx, mask in sorted(masks.items()):
        out_path = export_dir / f"{frame_idx:05d}.png"
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(out_path)
        written.append(str(out_path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify mask_dict indexing and export decoded expression masks.")
    parser.add_argument("--data-root", required=True, help="Dataset root, e.g. .../davis17 or .../mevis")
    parser.add_argument("--split", required=True, help="Dataset split, e.g. train/valid/valid_u")
    parser.add_argument("--video-id", default=None, help="Optional target video_id")
    parser.add_argument("--exp-id", type=int, default=None, help="Optional target exp_id")
    parser.add_argument("--meta-path", default=None, help="Optional explicit meta_expressions.json path")
    parser.add_argument("--mask-dict-path", default=None, help="Optional explicit mask_dict path")
    parser.add_argument("--export-dir", default=None, help="Optional export directory for decoded binary masks")
    parser.add_argument("--report-json", default=None, help="Optional output json path for the verification report")
    args = parser.parse_args()

    if (args.video_id is None) != (args.exp_id is None):
        raise ValueError("--video-id and --exp-id must be provided together")
    if cocomask is None:
        raise RuntimeError("pycocotools is required for decoding mask_dict")

    data_root = Path(args.data_root).resolve()
    meta_path = resolve_meta_path(data_root, args.split, Path(args.meta_path).resolve() if args.meta_path else None)
    mask_dict_path = resolve_mask_dict_path(data_root, args.split, Path(args.mask_dict_path).resolve() if args.mask_dict_path else None)

    meta = load_meta(meta_path)
    mask_dict = load_mask_dict(mask_dict_path)
    mode = infer_mask_dict_mode(meta, mask_dict)
    order_report = compare_index_orders(meta)

    report: Dict[str, Any] = {
        "data_root": str(data_root),
        "split": args.split,
        "meta_path": str(meta_path),
        "mask_dict_path": str(mask_dict_path),
        "mask_dict_entries": len(mask_dict),
        "total_expressions": count_total_expressions(meta),
        "mode": mode,
        "index_order_check": {
            "applicable": mode == "expression_index",
            "num_total": order_report["num_total"],
            "num_mismatches": order_report["num_mismatches"] if mode == "expression_index" else 0,
            "first_mismatches": order_report["first_mismatches"] if mode == "expression_index" else [],
            "note": "order check only matters for expression_index mask_dict" if mode != "expression_index" else "",
        },
    }

    if args.video_id is not None:
        payload = load_expression_masks(meta, mask_dict, mode, str(args.video_id), int(args.exp_id), order_report)
        report["target_expression"] = payload["meta"]
        if args.export_dir:
            written = export_masks(Path(args.export_dir).resolve(), payload["masks"])
            report["exported_mask_dir"] = str(Path(args.export_dir).resolve())
            report["exported_mask_count"] = len(written)
            report["exported_mask_preview"] = written[:10]

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
