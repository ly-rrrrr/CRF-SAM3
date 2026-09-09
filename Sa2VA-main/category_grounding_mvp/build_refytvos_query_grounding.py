#!/usr/bin/env python3
"""Build Ref-Youtube-VOS data for mask-supervised target-query grounding.

Each record contains original sampled frames, aligned target masks, the referring
expression, and the target category. GT masks are stored only as supervision
metadata; this builder does not create highlight or crop evidence images.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pycocotools import mask as mask_utils

DEFAULT_META_EXPRESSIONS = (
    "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/"
    "Ref-Youtube-VOS/train/meta_expressions.json"
)
DEFAULT_META_OBJECTS = (
    "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/"
    "Ref-Youtube-VOS/train/meta.json"
)
DEFAULT_IMAGE_ROOT = (
    "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/"
    "Ref-Youtube-VOS/train/JPEGImages"
)
DEFAULT_MASK_DICT = (
    "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/"
    "Ref-Youtube-VOS/train/mask_dict.pkl"
)


@dataclass(frozen=True)
class QueryGroundingSample:
    sample_id: str
    global_index: int
    video_id: str
    exp_id: int
    obj_id: str
    expression: str
    category: str
    frames: List[str]
    frame_paths: List[str]
    mask_rles: List[Dict[str, Any]]
    mask_frame_indices: List[int]
    mask_areas: List[int]


def frame_image_path(image_root: Path, video_id: str, frame_name: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        path = image_root / video_id / f"{frame_name}{ext}"
        if path.exists():
            return path
    return image_root / video_id / f"{frame_name}.jpg"


def mask_area(rle: Dict[str, Any] | None) -> int:
    if not rle:
        return 0
    try:
        return int(mask_utils.area(rle))
    except Exception:
        decoded = mask_utils.decode(rle)
        return int(decoded.sum())


def normalize_rle(rle: Dict[str, Any]) -> Dict[str, Any]:
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(rle.get("size", [])), "counts": counts}


def sample_evenly(items: Sequence[int], k: int) -> List[int]:
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in idxs]


def split_by_video(samples: Sequence[QueryGroundingSample], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    videos = sorted({s.video_id for s in samples})
    if len(videos) <= 1:
        return set(videos), set()
    rng = random.Random(seed)
    rng.shuffle(videos)
    n_val = max(1, int(round(len(videos) * val_ratio))) if videos else 0
    return set(videos[n_val:]), set(videos[:n_val])


def build_samples(
    meta_expressions: Path,
    meta_objects: Path,
    image_root: Path,
    mask_dict: Dict[str, Any],
    frames_per_sample: int,
    min_area: int,
    max_samples: int,
) -> tuple[List[QueryGroundingSample], Dict[str, int]]:
    expr_meta = json.loads(meta_expressions.read_text(encoding="utf-8"))
    obj_meta = json.loads(meta_objects.read_text(encoding="utf-8"))
    obj_videos = obj_meta.get("videos", {})
    samples: List[QueryGroundingSample] = []
    stats = {
        "missing_mask": 0,
        "missing_fields": 0,
        "no_visible_mask": 0,
        "missing_image": 0,
    }
    global_index = 0

    for video_id, video_meta in expr_meta.get("videos", {}).items():
        frames = list(video_meta.get("frames", []))
        obj_map = obj_videos.get(video_id, {}).get("objects", {})
        for exp_id, exp_data in video_meta.get("expressions", {}).items():
            if max_samples and len(samples) >= max_samples:
                return samples, stats

            mask_seq = mask_dict.get(str(global_index))
            obj_id = str(exp_data.get("obj_id", ""))
            expression = str(exp_data.get("exp", "")).strip()
            category = str(obj_map.get(obj_id, {}).get("category", "")).strip()
            if not mask_seq:
                stats["missing_mask"] += 1
                global_index += 1
                continue
            if not frames or not obj_id or not expression or not category:
                stats["missing_fields"] += 1
                global_index += 1
                continue

            n = min(len(frames), len(mask_seq))
            visible = [i for i in range(n) if mask_area(mask_seq[i]) >= min_area]
            selected = sample_evenly(visible, frames_per_sample)
            if not selected:
                stats["no_visible_mask"] += 1
                global_index += 1
                continue

            frame_names: List[str] = []
            frame_paths: List[str] = []
            mask_rles: List[Dict[str, Any]] = []
            areas: List[int] = []
            ok = True
            for idx in selected:
                frame_name = str(frames[idx])
                image_path = frame_image_path(image_root, str(video_id), frame_name)
                if not image_path.exists():
                    ok = False
                    break
                rle = normalize_rle(mask_seq[idx])
                frame_names.append(frame_name)
                frame_paths.append(str(image_path))
                mask_rles.append(rle)
                areas.append(mask_area(mask_seq[idx]))
            if not ok:
                stats["missing_image"] += 1
                global_index += 1
                continue

            sample_id = f"refytvos:{video_id}:{exp_id}:{global_index}"
            samples.append(
                QueryGroundingSample(
                    sample_id=sample_id,
                    global_index=global_index,
                    video_id=str(video_id),
                    exp_id=int(exp_id),
                    obj_id=obj_id,
                    expression=expression,
                    category=category,
                    frames=frame_names,
                    frame_paths=frame_paths,
                    mask_rles=mask_rles,
                    mask_frame_indices=[int(i) for i in selected],
                    mask_areas=[int(a) for a in areas],
                )
            )
            global_index += 1

    return samples, stats


def write_jsonl(path: Path, samples: Sequence[QueryGroundingSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-expressions", default=DEFAULT_META_EXPRESSIONS)
    parser.add_argument("--meta-objects", default=DEFAULT_META_OBJECTS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-dict", default=DEFAULT_MASK_DICT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-sample", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.mask_dict, "rb") as f:
        mask_dict = pickle.load(f)

    samples, skip_stats = build_samples(
        meta_expressions=Path(args.meta_expressions),
        meta_objects=Path(args.meta_objects),
        image_root=Path(args.image_root),
        mask_dict=mask_dict,
        frames_per_sample=args.frames_per_sample,
        min_area=args.min_area,
        max_samples=args.max_samples,
    )
    train_videos, val_videos = split_by_video(samples, args.val_ratio, args.seed)
    train = [s for s in samples if s.video_id in train_videos]
    val = [s for s in samples if s.video_id in val_videos]
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)

    summary = {
        "mode": "mask_supervised_query_grounding",
        "base_samples": len(samples),
        "train_records": len(train),
        "val_records": len(val),
        "train_videos": len(train_videos),
        "val_videos": len(val_videos),
        "video_overlap": len(train_videos & val_videos),
        "frames_per_sample": args.frames_per_sample,
        "min_area": args.min_area,
        "skip_stats": skip_stats,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
