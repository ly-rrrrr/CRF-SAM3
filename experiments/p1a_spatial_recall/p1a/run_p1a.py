#!/usr/bin/env python3
"""Stage 2 (GPU, SAM3 only): evidence-gated recall completion and oracle J*.

Pipeline per sample (text recall is never replaced, only supplemented):
  0. candidates C = cached text-recall tubes (or --recompute-text)
  1. blobs = connected components of the Ground MAX response on the M Ground frames;
     a blob is *uncovered evidence* if < coverage_thr of its mass lies on C
  2. for each uncovered blob (top --budget by mass) run the requested modes:
       rescue    : SAM3 detections on that frame with score >= --det-thresh
                   (default thresholds hide everything < 0.7); keep detections
                   inside the blob and not already in C; seed the tracker with
                   the detection box -> new tube
       point     : tracker positive click at the blob's local peaks -> new tube
       exemplar  : text + blob box as SAM3 exemplar on that frame (ablation)
       text_cfg  : (not gated) rerun text recall with --sam3-overrides; reports
                   both union J* and replace J* (info.j_star_replace)
       text_word : (not gated) rerun text recall with the word from
                   --alt-word-json (e.g. dataset category / coarser word); a
                   diagnostic for "is the word the problem", not a method
       repropagate: (not gated, deployable) for each existing candidate, take its
                   largest-mask frame as anchor, re-seed the SAM3 tracker there
                   (box + interior point) and propagate both ways; adds the
                   re-propagated tube and the gap-filled tube (original where
                   present, re-propagated elsewhere). Targets temporal gaps /
                   tracks killed by the video heuristics.
       det_probe : (not gated, adds no tube) single-frame SAM3 detection with
                   --probe-thresh on the probe frames; records the best GT IoU
                   of any detection, its score and rank, for the main word and
                   (if given) the alt word. Tells whether the detector sees the
                   target at all and whether the 0.5/0.7 thresholds hide it.
  3. oracle_gt   : tracker box from GT on the probe frame with the largest GT
                   (upper bound for any single-seed method; not gated)
  Probe frames = the Ground frames of the heatmap, or, without --heat-dir,
  --probe-frames frames sampled like Ground. Gated modes need --heat-dir.
  New tubes duplicating C (tube IoU > --dup-iou) are dropped. J* is recomputed
  as max over C ∪ new tubes.

Output: <out>/results.jsonl (one line per sample) and optional tube RLEs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import contextlib

from .common import (build_gt_loader, list_frames, load_cache_candidates, load_cache_meta, load_pool_candidates,
                     load_sample_list, sample_frame_indices, tube_iou, tube_to_rle, video_j)
from .seeds import mask_to_token_grid, select_seeds, split_frames, aggregate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--heat-dir", default="", help="needed only for rescue/point/exemplar")
    p.add_argument("--cache-root", default="")
    p.add_argument("--pool-dir", default="",
                   help="ground_v2 runs/pool_revos/pools: read the text-recall candidates of (video, --category-json word)")
    p.add_argument("--split", default="train")
    p.add_argument("--recompute-text", action="store_true", help="run text recall instead of reading the cache")
    p.add_argument("--category-json", default="", help="Ground prediction JSON (records with video_id, exp_id, pred_category)")
    p.add_argument("--gt", default="palette", choices=["palette", "png_dir", "revos"])
    p.add_argument("--gt-kw", default="{}")
    p.add_argument("--dms-repo", required=True)
    p.add_argument("--sam3-checkpoint", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--offload-video-to-cpu", action="store_true")
    p.add_argument("--modes", default="rescue,point,oracle_gt")
    p.add_argument("--agg", default="max")
    p.add_argument("--abs-thr", type=float, default=0.5)
    p.add_argument("--rel-thr", type=float, default=0.6)
    p.add_argument("--coverage-thr", type=float, default=0.5)
    p.add_argument("--budget", type=int, default=2, help="uncovered blobs used per sample")
    p.add_argument("--max-new", type=int, default=3, help="max new tubes per mode per sample")
    p.add_argument("--det-thresh", type=float, default=0.2)
    p.add_argument("--det-in-blob", type=float, default=0.5, help="min share of a detection's token area inside the blob")
    p.add_argument("--dup-iou", type=float, default=0.7)
    p.add_argument("--no-gate", action="store_true", help="use all blobs, not only uncovered ones")
    p.add_argument("--recall-top-k", type=int, default=24)
    p.add_argument("--keep-top-k", type=int, default=7)
    p.add_argument("--sam3-overrides", default="{}",
                   help='JSON for mode text_cfg, e.g. {"new_det_thresh": 0.5, "hotstart_delay": 0}')
    p.add_argument("--alt-word-json", default="",
                   help="for text_word/det_probe: JSON {\"vid/exp\": word} or records with video_id, exp_id, word")
    p.add_argument("--repro-top", type=int, default=7, help="repropagate: candidates re-seeded per sample")
    p.add_argument("--probe-thresh", type=float, default=0.05)
    p.add_argument("--probe-frames", type=int, default=3, help="frames used when there is no heatmap")
    p.add_argument("--save-tubes", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_categories(path: str) -> Dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = payload.get("records", payload) if isinstance(payload, dict) else payload
    return {f"{r['video_id']}/{r['exp_id']}": r.get("pred_category") or r.get("category", "") for r in recs}


def load_alt_words(path: str) -> Dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" not in payload:
        return {str(k): str(v) for k, v in payload.items()}
    recs = payload.get("records", payload) if isinstance(payload, dict) else payload
    return {f"{r['video_id']}/{r['exp_id']}": r.get("word") or r.get("category", "") for r in recs}


def probe(sess, t: int, word: str, g: np.ndarray, thresh: float) -> Dict:
    dets = sorted(sess.detect_on_frame(t, word, thresh), key=lambda d: -d["score"])
    best, best_i = 0.0, -1
    for i, d in enumerate(dets):
        j = _miou(g, d["mask"])
        if j > best:
            best, best_i = j, i
    return {"frame": t, "word": word, "n_dets": len(dets), "best_iou": best,
            "best_score": dets[best_i]["score"] if best_i >= 0 else None,
            "best_rank": best_i + 1 if best_i >= 0 else None,
            "top_score": dets[0]["score"] if dets else None}


GATED = {"rescue", "point", "exemplar"}


def is_dup(tube, pool: List[Dict[int, np.ndarray]], thr: float) -> bool:
    return any(tube_iou(tube, t) > thr for t in pool)


def det_inside_blob(det_mask: np.ndarray, blob, ht: int, wt: int) -> float:
    g = mask_to_token_grid(det_mask, ht, wt)
    return float((g * blob.token_mask).sum() / max(g.sum(), 1e-6))


def interior_point(mask: np.ndarray) -> List[float]:
    from scipy import ndimage as nd

    d = nd.distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return [(x + 0.5) / mask.shape[1], (y + 0.5) / mask.shape[0]]


class _SessionReuse:
    """One SAM3 session per video: consecutive samples of the same video share the decoded frames."""

    def __init__(self, factory, predictor, offload: bool):
        self.factory, self.p, self.offload = factory, predictor, offload
        self.video, self.sess = None, None

    def get(self, video_dir: str):
        if self.video != video_dir:
            self.close()
            self.sess = self.factory(self.p, video_dir, self.offload)
            self.sess.__enter__()
            self.video = video_dir
        return contextlib.nullcontext(self.sess)

    def close(self):
        if self.sess is not None:
            try:
                self.sess.__exit__(None, None, None)
            finally:
                self.video, self.sess = None, None


def main():
    a = parse_args()
    from .sam3_ops import SAM3Session, build_predictor

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
    done = set()
    res_path = out / "results.jsonl"
    if res_path.is_file():
        done = {json.loads(l)["key"] for l in res_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    samples = load_sample_list(a.samples)
    samples.sort(key=lambda s: str(s["video_id"]))  # group by video so the SAM3 session is reused
    if a.limit:
        samples = samples[: a.limit]
    gt_loader = build_gt_loader(a.gt, **json.loads(a.gt_kw))
    cats = load_categories(a.category_json)
    alt = load_alt_words(a.alt_word_json)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    if GATED & set(modes) and not a.heat_dir:
        raise SystemExit(f"modes {sorted(GATED & set(modes))} need --heat-dir")
    predictor = build_predictor(a.dms_repo, a.sam3_checkpoint, a.gpu)
    sessions = _SessionReuse(SAM3Session, predictor, a.offload_video_to_cpu)
    n_err = 0
    for n, s in enumerate(samples, 1):
        vid, eid = str(s["video_id"]), str(s["exp_id"])
        key = f"{vid}/{eid}"
        if key in done:
            continue
        try:
            _run_one(a, n, len(samples), s, vid, eid, key, cats, alt, modes, gt_loader, sessions, out, res_path)
        except Exception as e:  # not written to results.jsonl, so a re-run retries it
            n_err += 1
            print(f"[error] {key}: {type(e).__name__}: {e}", flush=True)
            sessions.close()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    sessions.close()
    if n_err:
        print(f"{n_err} samples failed; re-run the same command to retry them")


def _run_one(a, n, total, s, vid, eid, key, cats, alt, modes, gt_loader, sessions, out, res_path):
    if True:
        heat = Path(a.heat_dir) / vid / f"{eid}.npz" if a.heat_dir else None
        if heat is not None and not heat.is_file():
            if GATED & set(modes):
                print(f"[skip] no heatmap for {key}")
                return
            heat = None
        t0 = time.time()
        meta = load_cache_meta(a.cache_root, a.split, vid, eid) if a.cache_root else {}
        category = cats.get(key) or s.get("sam3_prompt_text") or meta.get("sam3_prompt_text") or s.get("category") or meta.get("category", "")
        video_dir = Path(a.video_root) / vid
        stems = list_frames(video_dir)
        gt = gt_loader(vid, eid, stems)
        if heat is not None:
            z = np.load(heat)
            R = z["responses"].astype(np.float32)
            grid, merge, fidx = z["image_grid_thw"], int(z["merge"]), [int(x) for x in z["frame_indices"]]
            expr = str(z["expression"]) if "expression" in z else s.get("expression", "")
        else:
            R = None
            fidx = sample_frame_indices(len(stems), a.probe_frames)
            expr = s.get("expression", "")
        alt_word = alt.get(key, "")
        rec = {"key": key, "category": category, "alt_word": alt_word, "expression": expr}
        if a.pool_dir:
            pooled = load_pool_candidates(a.pool_dir, vid, category)
            if pooled is None:
                print(f"[skip] no recall pool for {key} '{category}'")
                return
        def seeds_of(cands):
            base = list(cands.values())
            j_old = max([video_j(t, gt) for t in base], default=0.0)
            rec.update(num_candidates=len(base), j_old=j_old)
            blobs, shape_of = [], {}
            if R is not None:
                sel = select_seeds(R, grid, fidx, cands, merge=merge, agg=a.agg, abs_thr=a.abs_thr,
                                   rel_thr=a.rel_thr, coverage_thr=a.coverage_thr, budget=a.budget)
                blobs = sel["blobs"] if a.no_gate else sel["uncovered"]
                blobs = sorted(blobs, key=lambda b: -b.mass)[: a.budget]
                rec["n_blobs"] = len(sel["blobs"])
                rec["n_uncovered"] = len(sel["uncovered"])
                rec["triggered"] = bool(sel["triggered"])
                rec["seeds"] = [b.to_json() for b in blobs]
                fmaps = split_frames(aggregate(R, a.agg), grid, merge)
                shape_of = {t: fm.shape for t, fm in zip(fidx, fmaps)}
            return base, j_old, blobs, shape_of

        pre = None
        if a.pool_dir and R is not None and set(modes) <= GATED:
            # gated modes only: decide on CPU first; with no seed there is nothing for SAM3 to do
            pre = seeds_of(pooled)
            if not pre[2]:
                for mode in modes:
                    rec[mode] = {"added": 0, "j_new_tubes": [], "j_star": pre[1], "info": []}
                rec["sec"] = round(time.time() - t0, 1)
                with open(res_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{n}/{total}] {key} '{category}' J*old={pre[1]:.3f} no seed, SAM3 skipped", flush=True)
                return
        with sessions.get(str(video_dir)) as sess:
            sess.warm_text = category or None
            if a.pool_dir:
                cands = pooled
            elif a.recompute_text or not a.cache_root:
                cands = sess.text_recall(category, a.recall_top_k, a.keep_top_k)
            else:
                cands = load_cache_candidates(a.cache_root, a.split, vid, eid)
            base, j_old, blobs, shape_of = pre if pre is not None else seeds_of(cands)

            saved = {}
            for mode in modes:
                new: List[Dict[int, np.ndarray]] = []
                info: List[Dict] = []
                if mode == "rescue":
                    det_cache: Dict[int, list] = {}
                    for b in blobs:
                        t = b.frame_idx
                        if t not in det_cache:
                            det_cache[t] = sess.detect_on_frame(t, category, a.det_thresh)
                        ht, wt = shape_of[t]
                        existing = [tube[t] for tube in base + new if t in tube]
                        for d in sorted(det_cache[t], key=lambda d: -d["score"]):
                            if len(new) >= a.max_new:
                                break
                            if det_inside_blob(d["mask"], b, ht, wt) < a.det_in_blob:
                                continue
                            if any(_miou(d["mask"], e) > 0.5 for e in existing):
                                continue
                            tube = sess.track_seed(t, points=[interior_point(d["mask"])], labels=[1], box_xyxy=d["box"])
                            if tube and not is_dup(tube, base + new, a.dup_iou):
                                new.append(tube)
                                existing.append(d["mask"])
                                info.append({"frame": t, "det_score": d["score"], "box": d["box"]})
                elif mode == "point":
                    for b in blobs:
                        t = b.frame_idx
                        for pt in b.peaks:
                            if len(new) >= a.max_new:
                                break
                            if _on_any(pt, [tube[t] for tube in base + new if t in tube]):
                                continue
                            tube = sess.track_seed(t, points=[pt], labels=[1])
                            if tube and not is_dup(tube, base + new, a.dup_iou):
                                new.append(tube)
                                info.append({"frame": t, "point": pt})
                elif mode == "exemplar":
                    for b in blobs:
                        tubes = sess.exemplar_recall(b.frame_idx, b.box, category, a.recall_top_k)
                        for tube in tubes.values():
                            if len(new) >= a.max_new:
                                break
                            if tube and not is_dup(tube, base + new, a.dup_iou):
                                new.append(tube)
                                info.append({"frame": b.frame_idx, "box": b.box})
                elif mode == "text_cfg":
                    # rerun the SAME word with relaxed SAM3 video heuristics (no Ground map involved)
                    tubes = sess.text_recall(category, a.recall_top_k, a.keep_top_k, json.loads(a.sam3_overrides))
                    alone = [video_j(t, gt) for t in tubes.values()]
                    for tube in tubes.values():
                        if tube and not is_dup(tube, base + new, a.dup_iou):
                            new.append(tube)
                    info.append({"j_star_replace": max(alone, default=0.0), "num_tubes": len(tubes)})
                elif mode == "text_word":
                    if alt_word:
                        tubes = sess.text_recall(alt_word, a.recall_top_k, a.keep_top_k)
                        alone = [video_j(t, gt) for t in tubes.values()]
                        for tube in tubes.values():
                            if tube and not is_dup(tube, base + new, a.dup_iou):
                                new.append(tube)
                        info.append({"word": alt_word, "j_star_replace": max(alone, default=0.0), "num_tubes": len(tubes)})
                elif mode == "repropagate":
                    order = sorted(base, key=lambda tb: -len(tb))[: a.repro_top]
                    for k, tb in enumerate(order):
                        t = max(tb, key=lambda f: int(tb[f].sum()))
                        m = tb[t]
                        ys, xs = np.where(m)
                        h, w = m.shape
                        box = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
                        rp = sess.track_seed(t, points=[interior_point(m)], labels=[1], box_xyxy=box, obj_id=1000 + k)
                        if not rp:
                            continue
                        filled = dict(rp)
                        filled.update(tb)
                        for kind, tube in (("filled", filled), ("repro", rp)):
                            if not is_dup(tube, base + new, a.dup_iou):
                                new.append(tube)
                                info.append({"cand": k, "anchor": t, "kind": kind,
                                             "frames_before": len(tb), "frames_after": len(tube)})
                elif mode == "det_probe":
                    for t in fidx:
                        if t not in gt or not gt[t].any():
                            continue
                        for w in [category] + ([alt_word] if alt_word and alt_word != category else []):
                            info.append(probe(sess, t, w, gt[t], a.probe_thresh))
                elif mode == "oracle_gt":
                    areas = [(int(gt[t].sum()) if t in gt else 0, t) for t in fidx]
                    area, t = max(areas)
                    if area > 0:
                        ys, xs = np.where(gt[t])
                        h, w = gt[t].shape
                        box = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
                        tube = sess.track_seed(t, points=[interior_point(gt[t])], labels=[1], box_xyxy=box)
                        if tube:
                            new.append(tube)
                            info.append({"frame": t, "box": box})
                else:
                    raise ValueError(mode)
                js = [video_j(t, gt) for t in new]
                rec[mode] = {
                    "added": len(new),
                    "j_new_tubes": js,
                    "j_star": max([j_old] + js),
                    "info": info,
                }
                if a.save_tubes and new:
                    saved[mode] = [tube_to_rle(t) for t in new]
        rec["sec"] = round(time.time() - t0, 1)
        with open(res_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if saved:
            p = out / "tubes" / vid / f"{eid}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(saved), encoding="utf-8")
        msg = " ".join(f"{m}={rec[m]['j_star']:.3f}(+{rec[m]['added']})" for m in modes)
        print(f"[{n}/{total}] {key} '{category}' J*old={j_old:.3f} uncovered={rec.get('n_uncovered', '-')} {msg} {rec['sec']}s", flush=True)


def _miou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        import cv2

        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _on_any(pt, masks) -> bool:
    for m in masks:
        h, w = m.shape
        x, y = min(int(pt[0] * w), w - 1), min(int(pt[1] * h), h - 1)
        if m[y, x]:
            return True
    return False


if __name__ == "__main__":
    main()
