#!/usr/bin/env python3
"""Build mask-guided category-grounding SFT data from Ref-Youtube-VOS.

This dataset implements privileged visual evidence:
- no_mask view: expression + original frames -> category
- mask_evidence view: expression + original frames + training-only target highlights/crops -> category

The two views share the same pair_id and target category. Inference can later use
only the no_mask view while training benefits from mask evidence.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
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
class MaskGuidedSample:
    global_index: int
    video_id: str
    exp_id: int
    obj_id: str
    expression: str
    category: str
    frames: List[str]
    mask_frame_indices: List[int]
    mask_areas: List[int]
    original_frame_paths: List[str]
    highlight_frame_paths: List[str]
    crop_frame_paths: List[str]
    pair_id: str


def frame_image_path(image_root: Path, video_id: str, frame_name: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        path = image_root / video_id / f"{frame_name}{ext}"
        if path.exists():
            return path
    return image_root / video_id / f"{frame_name}.jpg"


def decode_rle(rle: Dict[str, Any] | None) -> np.ndarray:
    if not rle:
        return np.zeros((0, 0), dtype=bool)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def mask_area(rle: Dict[str, Any] | None) -> int:
    if not rle:
        return 0
    try:
        return int(mask_utils.area(rle))
    except Exception:
        return int(decode_rle(rle).sum())


def sample_evenly(items: Sequence[int], k: int) -> List[int]:
    if not items:
        return []
    if len(items) <= k:
        return list(items)
    idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in idxs]


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    pad = int(round(max(bw, bh) * pad_ratio))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    )


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = mask.shape[:2]
    target_w, target_h = size
    if width == target_w and height == target_h:
        return mask
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.asarray(img) > 0


def save_evidence_images(
    image_path: Path,
    mask: np.ndarray,
    out_prefix: Path,
    alpha: float,
    crop_pad_ratio: float,
) -> tuple[Path, Path] | None:
    image = Image.open(image_path).convert("RGB")
    mask = resize_mask(mask, image.size)
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None

    arr = np.asarray(image).astype(np.float32)
    overlay = arr.copy()
    red = np.array([255, 48, 48], dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * red
    # Slightly dim background while keeping image distribution RGB-natural.
    overlay[~mask] = 0.78 * overlay[~mask]
    highlight = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")

    x0, y0, x1, y1 = expand_bbox(bbox, image.width, image.height, crop_pad_ratio)
    crop = image.crop((x0, y0, x1, y1))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    highlight_path = out_prefix.with_suffix(".highlight.jpg")
    crop_path = out_prefix.with_suffix(".crop.jpg")
    highlight.save(highlight_path, quality=92)
    crop.save(crop_path, quality=92)
    return highlight_path, crop_path


def split_by_video(samples: Sequence[MaskGuidedSample], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    videos = sorted({s.video_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(videos)
    n_val = max(1, int(round(len(videos) * val_ratio))) if videos else 0
    val = set(videos[:n_val])
    train = set(videos[n_val:])
    return train, val


def iter_base_samples(
    meta_expressions: Path,
    meta_objects: Path,
    image_root: Path,
    mask_dict: Dict[str, Any],
    frames_per_sample: int,
    evidence_dir: Path,
    alpha: float,
    crop_pad_ratio: float,
    min_area: int,
    max_samples: int,
) -> List[MaskGuidedSample]:
    expr_meta = json.loads(meta_expressions.read_text(encoding="utf-8"))
    obj_meta = json.loads(meta_objects.read_text(encoding="utf-8"))
    obj_videos = obj_meta.get("videos", {})
    samples: List[MaskGuidedSample] = []
    global_index = 0

    for video_id, video_meta in expr_meta.get("videos", {}).items():
        frames = list(video_meta.get("frames", []))
        obj_map = obj_videos.get(video_id, {}).get("objects", {})
        for exp_id, exp_data in video_meta.get("expressions", {}).items():
            if max_samples and len(samples) >= max_samples:
                return samples
            mask_seq = mask_dict.get(str(global_index))
            obj_id = str(exp_data.get("obj_id", ""))
            expression = str(exp_data.get("exp", "")).strip()
            category = str(obj_map.get(obj_id, {}).get("category", "")).strip()
            if not mask_seq or not frames or not obj_id or not expression or not category:
                global_index += 1
                continue
            n = min(len(frames), len(mask_seq))
            nonempty = [i for i in range(n) if mask_area(mask_seq[i]) >= min_area]
            selected = sample_evenly(nonempty, frames_per_sample)
            selected_areas = [mask_area(mask_seq[i]) for i in selected]
            if not selected:
                global_index += 1
                continue

            original_paths: List[str] = []
            highlight_paths: List[str] = []
            crop_paths: List[str] = []
            selected_frames: List[str] = []
            ok = True
            for idx in selected:
                frame_name = str(frames[idx])
                image_path = frame_image_path(image_root, video_id, frame_name)
                if not image_path.exists():
                    ok = False
                    break
                mask = decode_rle(mask_seq[idx])
                safe_video = video_id.replace("/", "__")
                out_prefix = evidence_dir / safe_video / str(exp_id) / f"{frame_name}_{global_index}"
                evidence = save_evidence_images(
                    image_path=image_path,
                    mask=mask,
                    out_prefix=out_prefix,
                    alpha=alpha,
                    crop_pad_ratio=crop_pad_ratio,
                )
                if evidence is None:
                    ok = False
                    break
                hi, crop = evidence
                selected_frames.append(frame_name)
                original_paths.append(str(image_path))
                highlight_paths.append(str(hi))
                crop_paths.append(str(crop))
            if ok:
                pair_id = f"refytvos:{video_id}:{exp_id}:{global_index}"
                samples.append(
                    MaskGuidedSample(
                        global_index=global_index,
                        video_id=str(video_id),
                        exp_id=int(exp_id),
                        obj_id=obj_id,
                        expression=expression,
                        category=category,
                        frames=selected_frames,
                        mask_frame_indices=[int(i) for i in selected],
                        mask_areas=[int(a) for a in selected_areas],
                        original_frame_paths=original_paths,
                        highlight_frame_paths=highlight_paths,
                        crop_frame_paths=crop_paths,
                        pair_id=pair_id,
                    )
                )
            global_index += 1
    return samples


def to_records(sample: MaskGuidedSample, include_mask_evidence: bool = True) -> List[Dict[str, Any]]:
    base = asdict(sample)
    target_text = json.dumps({"category": sample.category}, ensure_ascii=False)
    no_mask = dict(base)
    no_mask.update(
        {
            "input_mode": "no_mask",
            "frame_paths": sample.original_frame_paths,
            "target_text": target_text,
        }
    )
    records = [no_mask]
    if include_mask_evidence:
        mask_view = dict(base)
        mask_view.update(
            {
                "input_mode": "mask_evidence",
                "frame_paths": sample.original_frame_paths,
                "target_text": target_text,
            }
        )
        records.append(mask_view)
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-expressions", default=DEFAULT_META_EXPRESSIONS)
    parser.add_argument("--meta-objects", default=DEFAULT_META_OBJECTS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask-dict", default=DEFAULT_MASK_DICT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-sample", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--crop-pad-ratio", type=float, default=0.35)
    parser.add_argument("--no-mask-evidence-records", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    evidence_dir = output_dir / "evidence_images"
    mask_dict = pickle.load(open(args.mask_dict, "rb"))
    samples = iter_base_samples(
        meta_expressions=Path(args.meta_expressions),
        meta_objects=Path(args.meta_objects),
        image_root=Path(args.image_root),
        mask_dict=mask_dict,
        frames_per_sample=args.frames_per_sample,
        evidence_dir=evidence_dir,
        alpha=args.alpha,
        crop_pad_ratio=args.crop_pad_ratio,
        min_area=args.min_area,
        max_samples=args.max_samples,
    )
    train_videos, val_videos = split_by_video(samples, args.val_ratio, args.seed)
    include_mask_evidence = not args.no_mask_evidence_records

    train_samples = [s for s in samples if s.video_id in train_videos]
    val_samples = [s for s in samples if s.video_id in val_videos]
    n_train = write_jsonl(
        output_dir / "train.jsonl",
        (record for s in train_samples for record in to_records(s, include_mask_evidence)),
    )
    n_val = write_jsonl(
        output_dir / "val.jsonl",
        (record for s in val_samples for record in to_records(s, include_mask_evidence)),
    )
    summary = {
        "dataset": "Ref-Youtube-VOS mask-guided category grounding",
        "base_samples": len(samples),
        "train_base_samples": len(train_samples),
        "val_base_samples": len(val_samples),
        "train_records": n_train,
        "val_records": n_val,
        "train_videos": len(train_videos),
        "val_videos": len(val_videos),
        "video_overlap": len(train_videos & val_videos),
        "frames_per_sample": args.frames_per_sample,
        "include_mask_evidence": include_mask_evidence,
        "evidence_dir": str(evidence_dir),
        "min_area": args.min_area,
        "alpha": args.alpha,
        "crop_pad_ratio": args.crop_pad_ratio,
        "frame_sampling": "select frames from mask_dict where mask area >= min_area, then sample evenly over those non-empty mask frames",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
