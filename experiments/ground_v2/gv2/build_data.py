#!/usr/bin/env python3
"""Add Ground-v2 supervision fields to an existing ReVOS query-grounding jsonl.

Input: the v1 jsonl written by build_revos_query_grounding_from_category_cache.py
(video_id, exp_id, obj_id, category, frame_paths, mask_rles, mask_frame_indices),
plus ReVOS meta_expressions and mask_dict.

For every record and every selected frame it adds
  neg_rles : union of OTHER annotated objects in the video whose category group differs
             from the target's  (hard negatives, weight --neg-weight in training)
  ign_rles : union of other objects with the SAME category group (left out of the loss,
             so the heatmap stays category-level)
and a list ``absent_frames`` (frames where the target is not visible) for the collator.

An object's category = majority of the (normalised) categories of the single-object
expressions that refer to it in the same jsonl. Objects with no such expression are
"unknown": by default treated as plain background (weight 1, same as v1).

Prints and writes summary.json with coverage numbers -- check ``records_with_neg``
before training: if it is near 0, change 2b has no effect on your data.

  python -m gv2.build_data --train-jsonl .../train.jsonl \
      --meta-expressions .../meta_expressions_train_.json --mask-dict .../mask_dict.json \
      --image-root .../revos --out data/revos_v2/train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import CategoryNormalizer, merge_rles, read_jsonl, sample_frame_indices


def parse_ids(value: Any) -> List[int]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).split(",") if str(v).strip()]


def frame_rle(mask_dict: Dict[str, Any], ids: List[int], f: int) -> Optional[Dict[str, Any]]:
    rles = []
    for i in ids:
        seq = mask_dict.get(str(i))
        if isinstance(seq, list) and 0 <= f < len(seq) and isinstance(seq[f], dict) and seq[f].get("counts"):
            rles.append(seq[f])
    return merge_rles(rles)


def find_image(root: Path, vid: str, frame: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png"):
        p = root / vid / f"{frame}{ext}"
        if p.exists():
            return str(p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--meta-expressions", required=True)
    ap.add_argument("--mask-dict", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synonyms", default="", help='optional JSON {"group": ["word", ...]} merged into the defaults')
    ap.add_argument("--unknown-policy", choices=["background", "ignore"], default="background")
    ap.add_argument("--max-absent", type=int, default=4, help="absent-frame candidates kept per record")
    args = ap.parse_args()

    norm = CategoryNormalizer.from_file(args.synonyms)
    records = read_jsonl(args.train_jsonl)
    meta = json.loads(Path(args.meta_expressions).read_text(encoding="utf-8"))["videos"]
    mask_dict = json.loads(Path(args.mask_dict).read_text(encoding="utf-8"))
    root = Path(args.image_root)

    votes: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        ids = parse_ids(r.get("obj_id"))
        if len(ids) == 1 and r.get("category"):
            votes[str(r["video_id"])][ids[0]][norm(r["category"])] += 1
    obj_cat = {v: {i: c.most_common(1)[0][0] for i, c in objs.items()} for v, objs in votes.items()}

    stats = Counter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for r in records:
            vid = str(r["video_id"])
            vmeta = meta.get(vid, {})
            frames = [str(x) for x in vmeta.get("frames", [])]
            tgt_ids = parse_ids(r.get("obj_id"))
            all_ids = sorted({i for e in vmeta.get("expressions", {}).values() for i in parse_ids(e.get("anno_id"))})
            tgt_group = norm(r.get("category"))
            neg_ids, ign_ids = [], []
            for i in all_ids:
                if i in tgt_ids:
                    continue
                g = obj_cat.get(vid, {}).get(i)
                if g is None:
                    stats["unknown_objects"] += 1
                    if args.unknown_policy == "ignore":
                        ign_ids.append(i)
                elif g == tgt_group:
                    ign_ids.append(i)
                else:
                    neg_ids.append(i)
            sel = [int(f) for f in r.get("mask_frame_indices") or []]
            if len(sel) != len(r["frame_paths"]):
                stats["skipped_no_frame_indices"] += 1
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            r = dict(r)
            r["neg_rles"] = [frame_rle(mask_dict, neg_ids, f) for f in sel]
            r["ign_rles"] = [frame_rle(mask_dict, ign_ids, f) for f in sel]
            absent_pool = [f for f in range(len(frames)) if f not in sel and frame_rle(mask_dict, tgt_ids, f) is None]
            absent = []
            for k in sample_frame_indices(len(absent_pool), args.max_absent):
                f = absent_pool[k]
                path = find_image(root, vid, frames[f])
                if path:
                    absent.append({"frame_index": f, "frame_path": path,
                                   "neg_rle": frame_rle(mask_dict, neg_ids, f), "ign_rle": frame_rle(mask_dict, ign_ids, f)})
            r["absent_frames"] = absent
            r["target_group"] = tgt_group
            stats["records"] += 1
            stats["records_with_neg"] += any(x is not None for x in r["neg_rles"])
            stats["records_with_ign"] += any(x is not None for x in r["ign_rles"])
            stats["records_with_absent"] += bool(absent)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = dict(stats)
    n = max(1, stats["records"])
    for k in ("records_with_neg", "records_with_ign", "records_with_absent"):
        summary[k + "_ratio"] = round(stats[k] / n, 4)
    summary["objects_with_category"] = sum(len(v) for v in obj_cat.values())
    (out_path.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
