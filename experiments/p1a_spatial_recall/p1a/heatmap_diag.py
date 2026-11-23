#!/usr/bin/env python3
"""CPU-only diagnostic: can the response map act as a recall gate?

For each sample and each Ground frame it reports, after inference (GT is used
only for scoring):
  * gt_blob_cov  - share of GT tokens inside the best blob   (is the target region found?)
  * gt_blob_pur  - share of that blob's tokens that are GT    (instance-level or category-level?)
  * top_is_gt    - the highest-mass blob overlaps GT (>= 30 % of GT tokens)
  * n_blobs
  * uncovered    - with --cache-root: the GT blob is NOT explained by the cached
                   SAM3 candidates (coverage < thr), i.e. P1-A would trigger on it
  * slot_corr    - mean pairwise correlation among active slots (slot collapse)

Two input modes:
  --diagnostic-dir  the 2026-11-12 package (case_*.npz with gt_token_mask)
  --heat-dir        output of heatmap_dump + a GT loader (+ optional cache)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .seeds import aggregate, blob_coverage, find_blobs, mask_to_token_grid, split_frames


def slot_collapse(responses: np.ndarray) -> float:
    act = (responses > 0.5).mean(1) > 0.005
    r = responses[act]
    if len(r) < 2:
        return float("nan")
    c = np.corrcoef(r.astype(np.float64))
    iu = np.triu_indices(len(r), 1)
    return float(np.nanmean(c[iu]))


def frame_stats(fm: np.ndarray, gt_tok: np.ndarray, args, cand_masks=None) -> Dict:
    blobs = find_blobs(fm, 0, 0, args.abs_thr, args.rel_thr, args.min_tokens)
    row = {"n_blobs": len(blobs), "gt_tokens": int(gt_tok.sum())}
    if not gt_tok.any():
        row.update(gt_blob_cov=np.nan, gt_blob_pur=np.nan, top_is_gt=np.nan, uncovered=np.nan)
        return row
    best = None
    for b in blobs:
        cov = (b.token_mask & gt_tok).sum() / gt_tok.sum()
        if best is None or cov > best[0]:
            best = (cov, (b.token_mask & gt_tok).sum() / b.size, b)
    if best is None:
        row.update(gt_blob_cov=0.0, gt_blob_pur=0.0, top_is_gt=0.0, uncovered=np.nan)
        return row
    top = blobs[0]
    row["gt_blob_cov"] = float(best[0])
    row["gt_blob_pur"] = float(best[1])
    row["top_is_gt"] = float((top.token_mask & gt_tok).sum() / gt_tok.sum() >= 0.3)
    if cand_masks is not None:
        row["uncovered"] = float(blob_coverage(best[2], fm, cand_masks) < args.coverage_thr)
    else:
        row["uncovered"] = np.nan
    return row


def run_diagnostic_dir(args) -> List[Dict]:
    rows = []
    meta = json.loads((Path(args.diagnostic_dir) / "results.json").read_text(encoding="utf-8"))
    for i, case in enumerate(meta["cases"]):
        z = np.load(Path(args.diagnostic_dir) / f"case_{i:02d}.npz")
        R, G, grid = z["responses"], z["gt_token_mask"].astype(bool), z["image_grid_thw"]
        fmaps = split_frames(aggregate(R, args.agg), grid)
        gmaps = split_frames(G, grid)
        sc = slot_collapse(R)
        for f, (fm, gm) in enumerate(zip(fmaps, gmaps)):
            row = {"sample": case["sample"]["sample_id"], "frame": f, "slot_corr": sc}
            row.update(frame_stats(fm, gm, args))
            rows.append(row)
    return rows


def run_heat_dir(args) -> List[Dict]:
    from .common import build_gt_loader, list_frames, load_cache_candidates

    gt_kw = json.loads(args.gt_kw)
    gt_loader = build_gt_loader(args.gt, **gt_kw)
    rows = []
    for npz in sorted(Path(args.heat_dir).rglob("*.npz")):
        vid, eid = npz.parent.name, npz.stem
        z = np.load(npz)
        R = z["responses"].astype(np.float32)
        grid, merge, fidx = z["image_grid_thw"], int(z["merge"]), z["frame_indices"].tolist()
        stems = list_frames(Path(args.video_root) / vid)
        gt = gt_loader(vid, eid, stems)
        cands = load_cache_candidates(args.cache_root, args.split, vid, eid) if args.cache_root else None
        fmaps = split_frames(aggregate(R, args.agg), grid, merge)
        sc = slot_collapse(R)
        for f, (fm, t) in enumerate(zip(fmaps, fidx)):
            g = gt.get(t)
            if g is None:
                continue
            gt_tok = mask_to_token_grid(g, *fm.shape) >= args.gt_token_thr
            cm = None if cands is None else [tube[t] for tube in cands.values() if t in tube]
            row = {"sample": f"{vid}/{eid}", "frame": t, "slot_corr": sc}
            row.update(frame_stats(fm, gt_tok, args, cm))
            rows.append(row)
    return rows


def summarize(rows: List[Dict]) -> Dict:
    def m(k):
        v = np.array([r[k] for r in rows if r.get(k) is not None and not np.isnan(r[k])], dtype=float)
        return (float(v.mean()) if v.size else float("nan")), int(v.size)

    out = {"frames": len(rows), "samples": len({r["sample"] for r in rows})}
    for k in ("gt_blob_cov", "gt_blob_pur", "top_is_gt", "n_blobs", "uncovered", "slot_corr"):
        out[k], out[k + "_n"] = m(k)
    v = np.array([r["gt_blob_cov"] for r in rows if not np.isnan(r.get("gt_blob_cov", np.nan))])
    out["frac_frames_gt_found(cov>=0.5)"] = float((v >= 0.5).mean()) if v.size else float("nan")
    p = np.array([r["gt_blob_pur"] for r in rows if not np.isnan(r.get("gt_blob_pur", np.nan))])
    out["frac_frames_instance_level(pur>=0.8)"] = float((p >= 0.8).mean()) if p.size else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagnostic-dir", default="")
    ap.add_argument("--heat-dir", default="")
    ap.add_argument("--video-root", default="")
    ap.add_argument("--gt", default="palette", choices=["palette", "png_dir", "revos"])
    ap.add_argument("--gt-kw", default="{}", help='JSON, e.g. {"annotation_root": "...", "meta_expressions": "..."}')
    ap.add_argument("--cache-root", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--agg", default="max")
    ap.add_argument("--abs-thr", type=float, default=0.5)
    ap.add_argument("--rel-thr", type=float, default=0.6)
    ap.add_argument("--min-tokens", type=int, default=2)
    ap.add_argument("--coverage-thr", type=float, default=0.5)
    ap.add_argument("--gt-token-thr", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = run_diagnostic_dir(args) if args.diagnostic_dir else run_heat_dir(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"diag_rows_{args.agg}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows)
    (out / f"diag_summary_{args.agg}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
