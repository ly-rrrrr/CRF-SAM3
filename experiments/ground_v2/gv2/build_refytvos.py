#!/usr/bin/env python3
"""Ground-v2 training data for Ref-YouTube-VOS (the ReVOS counterpart is gv2.build_data).

Input: the v1 jsonl written by the repo's category_grounding_mvp/build_refytvos_query_grounding.py
(video_id, exp_id, obj_id, category, frames, frame_paths, mask_rles, mask_frame_indices; the target
masks come from train/mask_dict.pkl), plus train/meta.json (category of EVERY annotated object) and
train/Annotations/<video>/<frame>.png (palette PNG, pixel value = object id).

For every record and every selected frame it adds the same fields as gv2.build_data:
  neg_rles : union of the OTHER objects of the video whose category group differs from the target's
  ign_rles : union of the other objects of the same category group (left out of the loss)
  absent_frames : frames where the target object is not in its PNG (max --max-absent)
Unlike ReVOS, every object has an official category here (meta.json), so no voting is needed.

Also checks that the v1 target masks (mask_dict.pkl, indexed by global expression order) agree with the
PNG of obj_id on the selected frames: summary["target_png_iou_mean"] should be ~1.0. If it is low, the
v1 jsonl and the PNGs disagree about which object is the target; stop and look before training.

  python -m gv2.build_refytvos --v1-jsonl .../refytvos_query_grounding/train.jsonl \
      --ytvos-train .../Ref-Youtube-VOS/train --out data/refytvos_v2/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .common import CategoryNormalizer, decode_rle, normalize_rle, read_jsonl, sample_frame_indices


def encode(mask: np.ndarray) -> Optional[Dict]:
    from pycocotools import mask as mask_utils

    if mask is None or not mask.any():
        return None
    return normalize_rle(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))


class VideoPNGs:
    """Palette PNGs of one video, read once each."""

    def __init__(self, ann_root: Path, vid: str) -> None:
        self.dir = ann_root / vid
        self.cache: Dict[str, Optional[np.ndarray]] = {}

    def __call__(self, frame: str) -> Optional[np.ndarray]:
        if frame not in self.cache:
            p = self.dir / f"{frame}.png"
            if not p.is_file():
                self.cache[frame] = None
            else:
                with Image.open(p) as im:
                    if im.mode not in ("P", "L"):
                        raise SystemExit(f"{p}: mode {im.mode}, expected a palette PNG whose pixel value is the object id")
                    self.cache[frame] = np.array(im)
        return self.cache[frame]


def union(ids_png: Optional[np.ndarray], ids: List[int]) -> Optional[Dict]:
    if ids_png is None or not ids:
        return None
    return encode(np.isin(ids_png, ids))


def find_image(root: Path, vid: str, frame: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png"):
        p = root / vid / f"{frame}{ext}"
        if p.exists():
            return str(p)
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1-jsonl", required=True)
    ap.add_argument("--ytvos-train", required=True, help="Ref-Youtube-VOS/train (meta.json, meta_expressions.json, Annotations, JPEGImages)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synonyms", default="", help='optional JSON {"group": ["word", ...]} merged into the defaults')
    ap.add_argument("--max-absent", type=int, default=4)
    a = ap.parse_args(argv)

    root = Path(a.ytvos_train)
    norm = CategoryNormalizer.from_file(a.synonyms)
    objects = json.loads((root / "meta.json").read_text(encoding="utf-8"))["videos"]
    frames_of = {v: [str(f) for f in d.get("frames", [])]
                 for v, d in json.loads((root / "meta_expressions.json").read_text(encoding="utf-8"))["videos"].items()}
    ann, img = root / "Annotations", root / "JPEGImages"
    records = read_jsonl(a.v1_jsonl)
    by_vid: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_vid[str(r["video_id"])].append(i)

    stats, ious = Counter(), []
    out: List[Optional[Dict]] = [None] * len(records)
    for vid, idxs in by_vid.items():
        pngs = VideoPNGs(ann, vid)
        cats = {int(o): norm(d.get("category")) for o, d in objects.get(vid, {}).get("objects", {}).items()}
        frames = frames_of.get(vid, [])
        for i in idxs:
            r = dict(records[i])
            tgt = int(r["obj_id"])
            group = norm(r.get("category")) or cats.get(tgt)
            others = [o for o in cats if o != tgt]
            neg = [o for o in others if cats[o] != group]
            ign = [o for o in others if cats[o] == group]
            sel = [int(f) for f in r.get("mask_frame_indices") or []]
            if not frames or len(sel) != len(r["frame_paths"]):
                stats["skipped_no_frame_indices"] += 1
                out[i] = r
                continue
            neg_rles, ign_rles = [], []
            for k, f in enumerate(sel):
                ids_png = pngs(frames[f])
                if ids_png is None:
                    stats["selected_frames_without_png"] += 1
                neg_rles.append(union(ids_png, neg))
                ign_rles.append(union(ids_png, ign))
                v1 = decode_rle(r["mask_rles"][k])
                if ids_png is not None and v1 is not None and v1.shape == ids_png.shape:
                    p = ids_png == tgt
                    ious.append(float((v1 & p).sum()) / max(float((v1 | p).sum()), 1.0))
            absent_pool = []
            for f in range(len(frames)):
                if f in sel:
                    continue
                ids_png = pngs(frames[f])
                if ids_png is not None and not (ids_png == tgt).any():
                    absent_pool.append(f)
            absent = []
            for k in sample_frame_indices(len(absent_pool), a.max_absent):
                f = absent_pool[k]
                path = find_image(img, vid, frames[f])
                if path:
                    ids_png = pngs(frames[f])
                    absent.append({"frame_index": f, "frame_path": path,
                                   "neg_rle": union(ids_png, neg), "ign_rle": union(ids_png, ign)})
            r.update(neg_rles=neg_rles, ign_rles=ign_rles, absent_frames=absent, target_group=group)
            stats["records"] += 1
            stats["records_with_neg"] += any(x is not None for x in neg_rles)
            stats["records_with_ign"] += any(x is not None for x in ign_rles)
            stats["records_with_absent"] += bool(absent)
            stats["objects_in_video_total"] += len(cats)
            out[i] = r

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = dict(stats)
    n = max(1, stats["records"])
    for k in ("records_with_neg", "records_with_ign", "records_with_absent"):
        s[k + "_ratio"] = round(stats[k] / n, 4)
    s["mean_objects_per_video"] = round(stats.pop("objects_in_video_total", 0) / n, 2)
    s.pop("objects_in_video_total", None)
    s["target_png_iou_mean"] = round(float(np.mean(ious)), 4) if ious else None
    s["target_png_iou_below_0.9"] = int(sum(x < 0.9 for x in ious))
    s["target_png_frames_checked"] = len(ious)
    (p.parent / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
