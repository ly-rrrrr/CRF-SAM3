#!/usr/bin/env python3
"""CPU-only: WHY is the best cached candidate below J=0.5?  (no SAM3, no Qwen)

Reads the existing offline candidate caches + GT and assigns every expression
whose best candidate has J* < --thr to exactly one failure type, judged on the
best candidate b (highest video J) unless stated otherwise:

  fragmented      union of the best two candidates reaches J >= thr
                  (the target is split across tubes / IDs)
  wrong_instance  b never overlaps GT (IoU < 0.05 on every GT frame) — SAM3
                  recalled other objects only
  temporal_gap    on frames where b exists its IoU is good (median >= 0.5) but
                  b covers < 60 % of GT-visible frames — tube starts late, is
                  lost after occlusion, or is suppressed for a while
  id_drift        b has a segment with IoU >= 0.5 and later/earlier frames where
                  it exists with IoU < 0.1 — the track jumped to a distractor
  merged          b is present but over-covers: GT mostly inside b
                  (recall >= 0.7) while precision < 0.5 — several objects in one mask
  partial         b is present but under-covers: precision >= 0.7, recall < 0.5
                  — part of the object only
  low_quality     anything else with overlap

It also records per sample: J*, number of candidates, GT size (fraction of the
frame), GT-visible frame count, b's temporal coverage, median co-present IoU,
pixel precision / recall, so other cut-offs can be applied later.

Example
  python -m p1a.failure_taxonomy --cache-root .../offline_cache_davis17_dms_recall24_keep7_20260620 \
     --split train --video-root .../davis17/train/JPEGImages \
     --gt palette --gt-kw '{"annotation_root": ".../davis17/train/Annotations", "meta_expressions": ".../meta_expressions/train/meta_expressions.json"}' \
     --out runs/taxonomy_davis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from .common import (Tube, _resize_bool, build_gt_loader, iter_cache_samples, list_frames,
                     load_cache_candidates, load_sample_list, video_j)


def per_frame(tube: Tube, gt: Tube):
    """Yield (t, iou, inter, pred_area, gt_area) on GT frames where GT is non-empty."""
    for t, g in gt.items():
        if not g.any():
            continue
        p = tube.get(t)
        if p is None or not p.any():
            yield t, None, 0, 0, int(g.sum())
            continue
        p = _resize_bool(p, g.shape)
        inter = int(np.logical_and(p, g).sum())
        union = int(np.logical_or(p, g).sum())
        yield t, inter / union, inter, int(p.sum()), int(g.sum())


def union_tube(a: Tube, b: Tube) -> Tube:
    out = dict(a)
    for t, m in b.items():
        out[t] = m if t not in out else (out[t] | _resize_bool(m, out[t].shape))
    return out


def classify(cands: Dict[int, Tube], gt: Tube, thr: float) -> Dict:
    js = {k: video_j(t, gt) for k, t in cands.items()}
    vis = [t for t, g in gt.items() if g.any()]
    frame_area = None
    for g in gt.values():
        frame_area = g.size
        break
    gt_frac = float(np.mean([gt[t].sum() / frame_area for t in vis])) if vis and frame_area else 0.0
    rec = {"num_candidates": len(cands), "gt_visible_frames": len(vis), "gt_area_frac": gt_frac}
    if not cands:
        rec.update(j_star=0.0, type="no_candidate")
        return rec
    order = sorted(js, key=lambda k: -js[k])
    b = cands[order[0]]
    rec["j_star"] = js[order[0]]
    rows = list(per_frame(b, gt))
    present = [r for r in rows if r[1] is not None]
    ious = [r[1] for r in present]
    inter = sum(r[2] for r in present)
    parea = sum(r[3] for r in present)
    garea = sum(r[4] for r in present)
    rec["temporal_cov"] = len(present) / max(len(rows), 1)
    rec["median_iou_present"] = float(np.median(ious)) if ious else 0.0
    rec["precision"] = inter / parea if parea else 0.0
    rec["recall_present"] = inter / garea if garea else 0.0
    j_pair = 0.0
    if len(order) >= 2:
        j_pair = video_j(union_tube(cands[order[0]], cands[order[1]]), gt)
    rec["j_union_top2"] = j_pair
    if rec["j_star"] >= thr:
        rec["type"] = "ok"
    elif j_pair >= thr:
        rec["type"] = "fragmented"
    elif not ious or max(ious) < 0.05:
        rec["type"] = "wrong_instance"
    elif rec["median_iou_present"] >= 0.5 and rec["temporal_cov"] < 0.6:
        rec["type"] = "temporal_gap"
    elif max(ious) >= 0.5 and sum(1 for x in ious if x < 0.1) >= max(2, 0.2 * len(ious)):
        rec["type"] = "id_drift"
    elif rec["recall_present"] >= 0.7 and rec["precision"] < 0.5:
        rec["type"] = "merged"
    elif rec["precision"] >= 0.7 and rec["recall_present"] < 0.5:
        rec["type"] = "partial"
    else:
        rec["type"] = "low_quality"
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--gt", default="palette", choices=["palette", "png_dir", "revos"])
    ap.add_argument("--gt-kw", default="{}")
    ap.add_argument("--samples", default="", help="optional list; default = every meta.json in the cache")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--all", action="store_true", help="also classify J*>=thr samples (type=ok)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gt_loader = build_gt_loader(a.gt, **json.loads(a.gt_kw))
    if a.samples:
        keys = [(str(s["video_id"]), str(s["exp_id"])) for s in load_sample_list(a.samples)]
    else:
        keys = list(iter_cache_samples(a.cache_root, a.split))
    if a.limit:
        keys = keys[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    frames_cache: Dict[str, List[str]] = {}
    for n, (vid, eid) in enumerate(keys, 1):
        if vid not in frames_cache:
            frames_cache[vid] = list_frames(Path(a.video_root) / vid)
        gt = gt_loader(vid, eid, frames_cache[vid])
        if not gt:
            continue
        cands = load_cache_candidates(a.cache_root, a.split, vid, eid)
        rec = classify(cands, gt, a.thr)
        if rec["type"] == "ok" and not a.all:
            continue
        rec.update(video_id=vid, exp_id=eid)
        rows.append(rec)
        if n % 200 == 0:
            print(f"[{n}/{len(keys)}] failures so far: {len(rows)}", flush=True)

    fields = sorted({k for r in rows for k in r})
    with open(out / "taxonomy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    fails = [r for r in rows if r["type"] != "ok"]
    cnt = Counter(r["type"] for r in fails)
    summ = {
        "scanned": len(keys),
        "failures(J*<thr)": len(fails),
        "types": {k: {"n": v, "pct_of_failures": round(100 * v / max(len(fails), 1), 1)} for k, v in cnt.most_common()},
        "median_gt_area_frac_failures": float(np.median([r["gt_area_frac"] for r in fails])) if fails else None,
    }
    if a.all:
        ok = [r for r in rows if r["type"] == "ok"]
        summ["median_gt_area_frac_ok"] = float(np.median([r["gt_area_frac"] for r in ok])) if ok else None
    (out / "taxonomy_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
