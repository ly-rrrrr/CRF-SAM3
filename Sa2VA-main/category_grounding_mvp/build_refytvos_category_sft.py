#!/usr/bin/env python3
"""Build category-only multimodal SFT data from Ref-Youtube-VOS train split."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


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


@dataclass(frozen=True)
class RefCategorySample:
    video_id: str
    exp_id: int
    obj_id: str
    expression: str
    category: str
    frame_paths: List[str]
    view_id: str


def list_frames(video_dir: Path) -> List[Path]:
    frames: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        frames.extend(video_dir.glob(ext))
    return sorted(frames)


def sample_evenly(frames: Sequence[Path], k: int) -> List[Path]:
    if k <= 0:
        raise ValueError("--frames-per-sample must be positive")
    if not frames:
        return []
    if len(frames) <= k:
        return list(frames)
    idxs = [round(i * (len(frames) - 1) / (k - 1)) for i in range(k)]
    out: List[int] = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    for idx in range(len(frames)):
        if len(out) >= k:
            break
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return [frames[i] for i in sorted(out[:k])]


def sample_random_view(frames: Sequence[Path], k: int, rng: random.Random) -> List[Path]:
    if not frames:
        return []
    if len(frames) <= k:
        return list(frames)
    return sorted(rng.sample(list(frames), k))


def load_base_samples(
    meta_expressions: Path,
    meta_objects: Path,
    image_root: Path,
    frames_per_sample: int,
) -> List[RefCategorySample]:
    expr_meta = json.loads(meta_expressions.read_text(encoding="utf-8"))
    obj_meta = json.loads(meta_objects.read_text(encoding="utf-8"))
    obj_videos = obj_meta.get("videos", {})
    samples: List[RefCategorySample] = []

    for video_id, video_meta in expr_meta.get("videos", {}).items():
        frame_paths = sample_evenly(list_frames(image_root / video_id), frames_per_sample)
        if not frame_paths:
            continue
        obj_map = obj_videos.get(video_id, {}).get("objects", {})
        for exp_id, exp_data in video_meta.get("expressions", {}).items():
            obj_id = str(exp_data.get("obj_id", ""))
            expression = str(exp_data.get("exp", "")).strip()
            category = str(obj_map.get(obj_id, {}).get("category", "")).strip()
            if not obj_id or not expression or not category:
                continue
            samples.append(
                RefCategorySample(
                    video_id=str(video_id),
                    exp_id=int(exp_id),
                    obj_id=obj_id,
                    expression=expression,
                    category=category,
                    frame_paths=[str(p) for p in frame_paths],
                    view_id="even",
                )
            )
    return samples


def add_random_views(
    samples: Sequence[RefCategorySample],
    image_root: Path,
    frames_per_sample: int,
    copies: int,
    seed: int,
) -> List[RefCategorySample]:
    if copies <= 0:
        return []
    rng = random.Random(seed)
    frames_by_video: Dict[str, List[Path]] = {}
    out: List[RefCategorySample] = []
    for sample in samples:
        frames = frames_by_video.get(sample.video_id)
        if frames is None:
            frames = list_frames(image_root / sample.video_id)
            frames_by_video[sample.video_id] = frames
        for view_idx in range(copies):
            view_frames = sample_random_view(frames, frames_per_sample, rng)
            if not view_frames:
                continue
            out.append(
                RefCategorySample(
                    video_id=sample.video_id,
                    exp_id=sample.exp_id,
                    obj_id=sample.obj_id,
                    expression=sample.expression,
                    category=sample.category,
                    frame_paths=[str(p) for p in view_frames],
                    view_id=f"random_{view_idx}",
                )
            )
    return out


def split_by_video(
    samples: Sequence[RefCategorySample],
    val_ratio: float,
    seed: int,
) -> tuple[List[str], List[str]]:
    video_ids = sorted({s.video_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    n_val = max(1, int(round(len(video_ids) * val_ratio))) if video_ids else 0
    val_videos = sorted(video_ids[:n_val])
    train_videos = sorted(video_ids[n_val:])
    return train_videos, val_videos


def to_record(sample: RefCategorySample) -> Dict[str, Any]:
    rec = asdict(sample)
    rec["target_text"] = json.dumps({"category": sample.category}, ensure_ascii=False)
    return rec


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-expressions", default=DEFAULT_META_EXPRESSIONS)
    parser.add_argument("--meta-objects", default=DEFAULT_META_OBJECTS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-sample", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--random-view-copies",
        type=int,
        default=0,
        help="Extra random frame views per training sample.",
    )
    args = parser.parse_args()

    meta_expressions = Path(args.meta_expressions)
    meta_objects = Path(args.meta_objects)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    base_samples = load_base_samples(
        meta_expressions=meta_expressions,
        meta_objects=meta_objects,
        image_root=image_root,
        frames_per_sample=args.frames_per_sample,
    )
    if args.max_samples and args.max_samples > 0:
        base_samples = base_samples[: args.max_samples]

    train_videos, val_videos = split_by_video(base_samples, args.val_ratio, args.seed)
    train_video_set = set(train_videos)
    val_video_set = set(val_videos)

    train_samples = [s for s in base_samples if s.video_id in train_video_set]
    val_samples = [s for s in base_samples if s.video_id in val_video_set]
    train_aug = add_random_views(
        train_samples,
        image_root=image_root,
        frames_per_sample=args.frames_per_sample,
        copies=args.random_view_copies,
        seed=args.seed + 1009,
    )
    train_all = list(train_samples) + train_aug

    n_train = write_jsonl(output_dir / "train.jsonl", (to_record(s) for s in train_all))
    n_val = write_jsonl(output_dir / "val.jsonl", (to_record(s) for s in val_samples))

    categories: Dict[str, int] = {}
    for sample in base_samples:
        categories[sample.category] = categories.get(sample.category, 0) + 1
    summary = {
        "meta_expressions": str(meta_expressions),
        "meta_objects": str(meta_objects),
        "image_root": str(image_root),
        "frames_per_sample": args.frames_per_sample,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "base_samples": len(base_samples),
        "train_records": n_train,
        "val_records": n_val,
        "train_videos": len(train_videos),
        "val_videos": len(val_videos),
        "random_view_copies": args.random_view_copies,
        "num_categories": len(categories),
        "top_categories": sorted(categories.items(), key=lambda x: (-x[1], x[0]))[:30],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
