#!/usr/bin/env python3
"""Build mask-supervised ReVOS query-grounding data from frozen API categories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from pycocotools import mask as mask_utils


def normalize_rle(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(x) for x in rle.get("size", [])], "counts": counts}


def merge_frame_rles(mask_dict: dict[str, Any], anno_ids: list[int], frame_idx: int) -> dict[str, Any] | None:
    rles = []
    for anno_id in anno_ids:
        sequence = mask_dict.get(str(anno_id))
        if not isinstance(sequence, list) or frame_idx < 0 or frame_idx >= len(sequence):
            continue
        rle = sequence[frame_idx]
        if isinstance(rle, dict) and rle.get("counts"):
            rles.append(rle)
    if not rles:
        return None
    merged = rles[0] if len(rles) == 1 else mask_utils.merge(rles)
    if int(mask_utils.area(merged)) <= 0:
        return None
    return normalize_rle(merged)


def image_path(image_root: Path, video_id: str, frame: str) -> Path | None:
    for extension in (".jpg", ".jpeg", ".png"):
        path = image_root / video_id / f"{frame}{extension}"
        if path.exists():
            return path
    return None


def split_videos(records: list[dict[str, Any]], val_ratio: float, seed: int) -> set[str]:
    videos = sorted({str(record["video_id"]) for record in records})
    rng = random.Random(seed)
    rng.shuffle(videos)
    count = max(1, round(len(videos) * val_ratio)) if videos else 0
    return set(videos[:count])


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category-cache", required=True)
    parser.add_argument("--meta-expressions", required=True)
    parser.add_argument("--mask-dict", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_payload = json.loads(Path(args.category_cache).read_text(encoding="utf-8"))
    category_records = cache_payload.get("records", [])
    meta = json.loads(Path(args.meta_expressions).read_text(encoding="utf-8")).get("videos", {})
    mask_dict = json.loads(Path(args.mask_dict).read_text(encoding="utf-8"))
    image_root = Path(args.image_root)

    records = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for category_record in category_records:
        video_id = str(category_record["video_id"])
        exp_id = int(category_record["exp_id"])
        video_meta = meta.get(video_id)
        if not isinstance(video_meta, dict):
            skip("missing_video_meta")
            continue
        expression_meta = video_meta.get("expressions", {}).get(str(exp_id))
        if not isinstance(expression_meta, dict):
            skip("missing_expression_meta")
            continue
        frames = [str(frame) for frame in video_meta.get("frames", [])]
        anno_ids = category_record.get("anno_id") or expression_meta.get("anno_id", [])
        if isinstance(anno_ids, (int, str)):
            anno_ids = [anno_ids]
        anno_ids = [int(value) for value in anno_ids]
        if not anno_ids:
            skip("missing_anno_id")
            continue

        selected = []
        for frame_idx in category_record.get("frame_indices", []):
            frame_idx = int(frame_idx)
            if frame_idx not in selected:
                selected.append(frame_idx)
        frame_names = []
        frame_paths = []
        mask_rles = []
        mask_areas = []
        mask_frame_indices = []
        for frame_idx in selected:
            if frame_idx < 0 or frame_idx >= len(frames):
                continue
            rle = merge_frame_rles(mask_dict, anno_ids, frame_idx)
            path = image_path(image_root, video_id, frames[frame_idx])
            if rle is None or path is None:
                continue
            frame_names.append(frames[frame_idx])
            frame_paths.append(str(path))
            mask_rles.append(rle)
            mask_areas.append(int(mask_utils.area(rle)))
            mask_frame_indices.append(frame_idx)
        if not mask_rles:
            skip("no_valid_supervision_frame")
            continue

        expression = str(expression_meta.get("exp", category_record.get("expression", "")))
        records.append(
            {
                "sample_id": f"revos:{video_id}:{exp_id}",
                "video_id": video_id,
                "exp_id": exp_id,
                "obj_id": ",".join(str(value) for value in anno_ids),
                "expression": expression,
                "category": str(category_record["category"]),
                "raw_category": str(category_record.get("raw_category", category_record["category"])),
                "frames": frame_names,
                "frame_paths": frame_paths,
                "mask_rles": mask_rles,
                "mask_frame_indices": mask_frame_indices,
                "mask_areas": mask_areas,
                "dataset": "revos",
            }
        )

    val_videos = split_videos(records, args.val_ratio, args.seed)
    train = [record for record in records if record["video_id"] not in val_videos]
    val = [record for record in records if record["video_id"] in val_videos]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    summary = {
        "category_cache": str(Path(args.category_cache).resolve()),
        "records": len(records),
        "train_records": len(train),
        "val_records": len(val),
        "train_videos": len({record["video_id"] for record in train}),
        "val_videos": len(val_videos),
        "skipped": skipped,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
