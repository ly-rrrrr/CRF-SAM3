#!/usr/bin/env python3
"""Fill the ablation table on a held-out jsonl that has GT masks (e.g. val.jsonl from gv2.build_data).

Columns (per model):
  word_acc            normalised predicted category == normalised label category
  word_acc_multinoun  same, on expressions that mention >= 2 known category words ("a man ... a monkey")
  mass_in_gt          share of heatmap mass on GT tokens (frames where the target is visible)
  mass_on_other_cat   share of heatmap mass on annotated other-category objects (needs neg_rles). NOTE: in ReVOS the
                      "other" objects often CONTAIN the target (person vs. their pants), so this counts GT mass too
  mass_on_other_excl  same, but on other-category tokens that are NOT GT tokens (the leak we actually care about)
  gt_area / other_excl_area   token share of GT / of other-not-GT: what a perfectly flat heatmap would score on
                      mass_in_gt / mass_on_other_excl (so mass / area = how much more concentrated than flat)
  peak_in_gt          argmax token of each frame lies on GT

Backends
  v2:  --ckpt <gv2 output dir>
  v1:  --v1-repo <Sa2VA-main> --v1-query-module-dir <checkpoint-900>   (row A of the table)

  python -m gv2.evaluate --model-path .../Qwen3-VL-4B-Instruct --jsonl data/revos_v2/val.jsonl \
      --ckpt runs/gv2_revos --bf16 --out runs/gv2_revos/eval_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch

from .common import CategoryNormalizer, decode_rle, read_jsonl, token_fraction
from .data import resize_cached


def frame_token_masks(rec: Dict[str, Any], grids: np.ndarray, merge: int, thr: float):
    gts, negs = [], []
    neg_rles = rec.get("neg_rles") or [None] * len(rec["frame_paths"])
    for rle, nrle, g in zip(rec["mask_rles"], neg_rles, grids.tolist()):
        _, gh, gw = [int(x) for x in g]
        gts.append(token_fraction(decode_rle(rle), gh, gw, merge) >= thr)
        negs.append(token_fraction(decode_rle(nrle), gh, gw, merge) >= thr)
    return gts, negs


def heat_metrics(heat: np.ndarray, grids: np.ndarray, merge: int, gts, negs) -> Dict[str, float]:
    sizes = [int(t * h * w) // (merge * merge) for t, h, w in grids.tolist()]
    out = {"mass_in_gt": [], "mass_on_other_cat": [], "peak_in_gt": [], "mass_on_other_excl": [], "gt_area": [],
           "other_excl_area": []}
    off = 0
    for n, gt, ng in zip(sizes, gts, negs):
        h = np.clip(heat[off: off + n].astype(np.float64), 0, None)
        off += n
        if not gt.any() or h.sum() <= 0:
            continue
        out["mass_in_gt"].append(float((h * gt).sum() / h.sum()))
        out["peak_in_gt"].append(float(gt[int(h.argmax())]))
        out["gt_area"].append(float(gt.mean()))
        if ng.any():
            out["mass_on_other_cat"].append(float((h * ng).sum() / h.sum()))
            ex = ng & ~gt
            if ex.any():
                out["mass_on_other_excl"].append(float((h * ex).sum() / h.sum()))
                out["other_excl_area"].append(float(ex.mean()))
    return {k: (float(np.mean(v)) if v else None) for k, v in out.items()}


class V2Backend:
    def __init__(self, a):
        from .model import base_model_of, load_for_inference

        self.processor, self.model = load_for_inference(a.model_path, a.ckpt, torch.bfloat16 if a.bf16 else torch.float32,
                                                        a.device_map, a.device, a.drop_summary)
        self.merge = int(base_model_of(self.model.qwen).model.visual.spatial_merge_size)
        self.a = a

    def __call__(self, rec):
        from .predict import predict_one

        pred, raw, heat, grid = predict_one(self.processor, self.model, rec, 32, self.a.max_image_side, self.a.frame_cache)
        return pred, heat.astype(np.float32), grid


class V1Backend:
    """Existing Ground (8 queries, MAX over queries), via the repository code."""

    def __init__(self, a):
        sys.path.insert(0, str(Path(a.v1_repo) / "category_grounding_mvp"))
        from eval_refytvos_prompt_vs_query_grounding import build_query_messages
        from predict_query_grounding_refytros_meta import load_query_model, predict_one

        self.build_query_messages, self.predict_one = build_query_messages, predict_one
        self.processor, self.qg = load_query_model(SimpleNamespace(
            model_path=a.model_path, query_module_dir=a.v1_query_module_dir, bf16=a.bf16,
            device_map=a.device_map, device=a.device))
        self.merge = int(self.qg.qwen.model.visual.spatial_merge_size)
        self.cap = {}
        self.qg.mask_head.register_forward_hook(lambda m, i, o: self.cap.__setitem__("l", o.detach().float().cpu()))
        self.a = a

    def __call__(self, rec):
        r = dict(rec)
        r["frame_paths"] = [resize_cached(p, self.a.max_image_side, self.a.frame_cache) for p in rec["frame_paths"]]
        pred, _, _ = self.predict_one(self.processor, self.qg, r, 64)
        heat = torch.sigmoid(self.cap.pop("l")[0]).max(0).values.numpy()
        grid = self.processor.apply_chat_template(self.build_query_messages(r), tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors="pt")["image_grid_thw"].numpy()
        return pred, heat, grid


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--v1-repo", default="")
    ap.add_argument("--v1-query-module-dir", default="")
    ap.add_argument("--max-image-side", type=int, default=0, help="v1: use its TRAINING side (e.g. 448) for a fair row A")
    ap.add_argument("--frame-cache", default="runs/_frame_cache")
    ap.add_argument("--synonyms", default="")
    ap.add_argument("--positive-threshold", type=float, default=0.05)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--drop-summary", action="store_true", help="ablation row E': run without the summary tokens")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device-map", default="auto", choices=["", "auto"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if bool(a.ckpt) == bool(a.v1_query_module_dir):
        ap.error("give exactly one of --ckpt (v2) or --v1-query-module-dir (v1)")

    norm = CategoryNormalizer.from_file(a.synonyms)
    recs = read_jsonl(a.jsonl)
    if a.max_samples > 0:
        recs = recs[: a.max_samples]
    vocab = {norm(r["category"]) for r in recs if r.get("category")}
    backend = V2Backend(a) if a.ckpt else V1Backend(a)

    rows: List[Dict[str, Any]] = []
    for n, rec in enumerate(recs, 1):
        pred, heat, grid = backend(rec)
        gts, negs = frame_token_masks(rec, grid, backend.merge, a.positive_threshold)
        words = {norm(w) for w in rec["expression"].lower().replace(",", " ").split()} & vocab
        row = {"sample_id": rec.get("sample_id"), "expression": rec["expression"], "category": rec.get("category"),
               "pred": pred, "word_ok": norm(pred) == norm(rec.get("category")), "multinoun": len(words) >= 2,
               **heat_metrics(heat, grid, backend.merge, gts, negs)}
        rows.append(row)
        print(f"[{n}/{len(recs)}] {row['sample_id']} gt={row['category']} pred={pred} ok={row['word_ok']} "
              f"mass_in_gt={row['mass_in_gt']}", flush=True)

    def mean(key, subset=None):
        vals = [r[key] for r in rows if (subset is None or subset(r)) and r[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {"n": len(rows), "word_acc": mean("word_ok"), "n_multinoun": sum(r["multinoun"] for r in rows),
               "word_acc_multinoun": mean("word_ok", lambda r: r["multinoun"]), "mass_in_gt": mean("mass_in_gt"),
               "mass_on_other_cat": mean("mass_on_other_cat"), "peak_in_gt": mean("peak_in_gt"),
               "mass_on_other_excl": mean("mass_on_other_excl"), "gt_area": mean("gt_area"),
               "other_excl_area": mean("other_excl_area"),
               "backend": "v2" if a.ckpt else "v1", "ckpt": a.ckpt or a.v1_query_module_dir,
               "max_image_side": a.max_image_side}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
