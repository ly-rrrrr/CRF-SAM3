#!/usr/bin/env python
"""Step 1 (GPU, CRF-SAM3 environment): export the real candidate mask tubes.

Why this step exists
--------------------
The historical ReVOS-valid run (crf_sam3_revos_valid_revos_query_ranker_20260806)
was executed with save_visual_tubes=false. It kept the Qwen3-VL verifier scores of
every candidate (scores.json) and the final prediction (results.json), but NOT the
masks of the unselected candidates. To feed the same candidate sets to VideoRefer
we re-run the *frozen* SAM3-DMS recall with the recorded recall word
(used_category_query) and the same recall24 -> keep7 budget, exactly like
paper_assets/revos_qualitative_20261008/scripts/replay_recall.py did for the
qualitative cases (where the candidate ids matched and the selected candidate was
pixel-identical to the archived prediction).

Nothing in Ground / SAM3 / Verify is modified; only DMSConceptRecall from the
existing repo is imported. Each replayed candidate set is audited against the
historical record:
  * ids_match       : replayed kept ids == historical all_candidates
  * selected_exact  : union of the historically selected ids, replayed, equals the
                      archived final mask on every frame (needs --historical_results)

Outputs (under --out_dir):
  candidates/<video>/<category>.json.gz   candidate tubes (RLE)
  pair_status[.shardK].jsonl              one line per (video, category)
  manifest.jsonl                          one line per expression (built at the end,
                                          or with --manifest_only)
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.io_utils import (  # noqa: E402
    append_jsonl, cand_relpath, encode_rle, exp_key, pair_key, read_json_gz,
    read_jsonl, shard_list, write_json_gz,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crf_root", required=False, default=None,
                   help="CRF-SAM3 code dir that contains crf_sam3_demo.py (i.e. .../Sa2VA-main)")
    p.add_argument("--dms_repo", default=None, help="SAM3-DMS repo (default: <crf_root>/SAM3-DMS)")
    p.add_argument("--sam3_checkpoint", default=None, help="sam3.pt")
    p.add_argument("--revos_root", required=True,
                   help="ReVOS root containing meta_expressions_valid_.json and the video folders")
    p.add_argument("--expression_file", default="meta_expressions_valid_.json")
    p.add_argument("--image_root", default=None, help="video frame root (default: revos_root)")
    p.add_argument("--historical_scores", required=True,
                   help="historical scores.json of the Qwen3-VL verifier run")
    p.add_argument("--historical_results", default=None,
                   help="historical results.json (RLE predictions). Optional, enables the pixel audit")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--recall_top_k", type=int, default=24)
    p.add_argument("--keep_top_k", type=int, default=7)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--memory_fraction", type=float, default=0.0,
                   help="optional torch.cuda.set_per_process_memory_fraction (0 = off)")
    p.add_argument("--include_uncontested", action="store_true",
                   help="also replay expressions whose choice is forced (#cand <= #selected)")
    p.add_argument("--max_pairs", type=int, default=None, help="pilot: only the first N (video, category) pairs")
    p.add_argument("--video_ids", default=None, help="comma separated video ids to restrict to")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--manifest_only", action="store_true", help="only (re)build manifest.jsonl from pair_status files")
    p.add_argument("--mock_recall", action="store_true", help=argparse.SUPPRESS)  # smoke test only
    p.add_argument("--mock_gt_mask_dict", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Expression bookkeeping
# ----------------------------------------------------------------------------

def load_inputs(args):
    meta = json.load(open(os.path.join(args.revos_root, args.expression_file)
                          if not os.path.isabs(args.expression_file) else args.expression_file))["videos"]
    hist = json.load(open(args.historical_scores))
    return meta, hist


def is_contested(rec) -> bool:
    return len(rec.get("all_candidates", [])) > len(rec.get("selected_candidates", []))


def collect_pairs(args, meta, hist):
    """Group historical expressions by (video_id, recall category)."""
    pairs = defaultdict(list)
    for rec in hist:
        if not rec.get("all_candidates"):
            continue
        if not args.include_uncontested and not is_contested(rec):
            continue
        vid = rec["video_id"]
        if vid not in meta:
            continue
        cat = rec.get("used_category_query") or rec.get("category")
        pairs[pair_key(vid, cat)].append(rec)
    keys = sorted(pairs)
    if args.video_ids:
        allow = {v.strip() for v in args.video_ids.split(",") if v.strip()}
        keys = [k for k in keys if k.split("||")[0] in allow]
    if args.max_pairs is not None:
        keys = keys[: args.max_pairs]
    keys = shard_list(keys, args.num_shards, args.shard_id)
    return keys, pairs


# ----------------------------------------------------------------------------
# Recall backends
# ----------------------------------------------------------------------------

class FrozenDMSRecall:
    """Thin wrapper around crf_sam3_demo.DMSConceptRecall (unchanged)."""

    def __init__(self, args):
        import torch
        crf_root = os.path.abspath(args.crf_root)
        sys.path.insert(0, crf_root)
        os.chdir(crf_root)  # the repo uses some relative paths internally
        from crf_sam3_demo import DMSConceptRecall
        if args.memory_fraction and args.memory_fraction > 0:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
        dms_repo = args.dms_repo or os.path.join(crf_root, "SAM3-DMS")
        # Same flags as configs/crf_sam3_config_revos_valid_revos_query_ranker_20260806.yaml
        self.model = DMSConceptRecall(
            dms_repo, args.sam3_checkpoint, gpus_to_use=[args.gpu],
            async_loading_frames=True, video_loader_type="cv2",
            offload_video_to_cpu=True, offload_output_to_cpu_for_eval=True,
            trim_past_non_cond_mem_for_eval=False,
        )
        self.torch = torch

    def stream(self, video_dir, category, recall_top_k, keep_top_k):
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t, masks, scores in self.model.recall_all_instances_stream(
                video_dir, category, candidate_top_k=recall_top_k, candidate_keep_top_k=keep_top_k
            ):
                yield t, masks, scores


class MockRecall:
    """Smoke-test backend: candidates = GT objects of the video + shifted copies."""

    def __init__(self, args):
        from common.io_utils import decode_rle
        self.decode = decode_rle
        self.mask_dict = json.load(open(args.mock_gt_mask_dict))
        self.meta = None

    def stream(self, video_dir, category, recall_top_k, keep_top_k):
        vid = self._vid
        v = self.meta[vid]
        anno_ids = sorted({a for e in v["expressions"].values() for a in e["anno_id"]})
        for t in range(len(v["frames"])):
            masks, scores = {}, {}
            for i, a in enumerate(anno_ids):
                rle = self.mask_dict[str(a)][t]
                if rle:
                    m = self.decode(rle)
                    if m.sum() > 0:
                        masks[i] = m
                        scores[i] = 0.9 - 0.1 * i
            # one distractor: shifted copy of the first object
            if 0 in masks:
                masks[len(anno_ids)] = np.roll(masks[0], 7, axis=1)
                scores[len(anno_ids)] = 0.5
            yield t, masks, scores


# ----------------------------------------------------------------------------
# Main replay loop
# ----------------------------------------------------------------------------

def replay_pair(backend, args, meta, vid, category, image_root):
    v = meta[vid]
    n_frames, h, w = len(v["frames"]), int(v["height"]), int(v["width"])
    video_dir = os.path.join(image_root, vid)
    rles = defaultdict(dict)   # obj -> {t: rle}
    sums = defaultdict(float)
    t0 = time.time()
    for t, masks, scores in backend.stream(video_dir, category, args.recall_top_k, args.keep_top_k):
        for k, m in masks.items():
            k = int(k)
            m = np.asarray(m)
            if m.ndim == 3:
                m = m[0]
            m = (m > 0).astype(np.uint8)
            if m.shape != (h, w):
                raise RuntimeError(f"mask shape {m.shape} != meta {(h, w)} for {vid}")
            if m.any():
                rles[k][str(int(t))] = encode_rle(m)
            sums[k] += float(scores[k])
    # native SAM3 score-sum keep-top-k, same tie-break as the pipeline
    keep = sorted(sums, key=lambda k: (-sums[k], k))[: args.keep_top_k]
    keep = sorted(keep)
    cand = {
        "video_id": vid,
        "category": category,
        "frames": [str(x) for x in v["frames"]],
        "height": h,
        "width": w,
        "candidate_ids": keep,
        "native_score_sums": {str(k): sums[k] for k in keep},
        "all_score_sums": {str(k): sums[k] for k in sorted(sums)},
        "masks": {str(k): rles.get(k, {}) for k in keep},
        "config": {"recall_top_k": args.recall_top_k, "keep_top_k": args.keep_top_k,
                   "backend": "mock" if args.mock_recall else "SAM3-DMS frozen replay",
                   "offload_video_to_cpu": True, "offload_output_to_cpu_for_eval": True},
        "elapsed_seconds": time.time() - t0,
    }
    return cand


def audit_pair(cand, recs, hist_results):
    from pycocotools import mask as cm
    out = {}
    h, w = cand["height"], cand["width"]
    n = len(cand["frames"])
    zero = None
    for rec in recs:
        key = exp_key(rec["video_id"], rec["exp_id"])
        hist_ids = sorted(int(x) for x in rec["all_candidates"])
        a = {"ids_match": hist_ids == sorted(cand["candidate_ids"]),
             "replay_ids": cand["candidate_ids"], "hist_ids": hist_ids}
        if hist_results is not None:
            pred = hist_results.get(rec["video_id"], {}).get(str(rec["exp_id"]))
            if pred is None or len(pred.get("prediction_masks", [])) != n:
                a["selected_exact_frames"] = None
            else:
                sel = [str(int(x)) for x in rec["selected_candidates"]]
                exact = 0
                for t in range(n):
                    parts = [cand["masks"].get(s, {}).get(str(t)) for s in sel]
                    parts = [dict(p, counts=p["counts"].encode()) for p in parts if p is not None]
                    if parts:
                        mine = cm.merge(parts, intersect=False) if len(parts) > 1 else parts[0]
                    else:
                        if zero is None:
                            zero = cm.encode(np.asfortranarray(np.zeros((h, w), dtype=np.uint8)))
                        mine = zero
                    ref = pred["prediction_masks"][t]
                    ref_counts = ref["counts"].encode() if isinstance(ref["counts"], str) else ref["counts"]
                    if mine["counts"] == ref_counts:
                        exact += 1
                    else:  # canonical RLEs can still differ in trailing form; compare pixels
                        mm = cm.decode(mine).astype(bool)
                        rr = cm.decode(dict(ref, counts=ref_counts)).astype(bool)
                        exact += int(np.array_equal(mm, rr))
                a["selected_exact_frames"] = exact
            a["num_frames"] = n
        out[key] = a
    return out


def build_manifest(args, meta, hist):
    status = {}
    for path in sorted(glob.glob(os.path.join(args.out_dir, "pair_status*.jsonl"))):
        for row in read_jsonl(path):
            if row.get("status") == "ok" or row["pair"] not in status:
                status[row["pair"]] = row
    n_written = 0
    out_path = os.path.join(args.out_dir, "manifest.jsonl")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in hist:
            vid, eid = rec["video_id"], str(rec["exp_id"])
            if vid not in meta or eid not in meta[vid]["expressions"]:
                continue
            e = meta[vid]["expressions"][eid]
            cat = rec.get("used_category_query") or rec.get("category")
            pk = pair_key(vid, cat)
            st = status.get(pk)
            audit = (st or {}).get("audit", {}).get(exp_key(vid, eid), {})
            row = {
                "key": exp_key(vid, eid),
                "video_id": vid,
                "exp_id": eid,
                "expression": e.get("exp", rec.get("expression", "")),
                "type_id": int(e.get("type_id", -1)),
                "anno_id": e.get("anno_id", []),
                "category": cat,
                "all_candidates": [int(x) for x in rec.get("all_candidates", [])],
                "selected_candidates": [int(x) for x in rec.get("selected_candidates", [])],
                "k": len(rec.get("selected_candidates", [])),
                "qwen_scores": {str(k): float(v.get("score", v.get("prob_yes", 0.0)))
                                for k, v in (rec.get("scores") or {}).items()},
                "contested": is_contested(rec),
                "cand_file": (st or {}).get("cand_file") if (st or {}).get("status") == "ok" else None,
                "replay_status": (st or {}).get("status", "not_run"),
                "ids_match": audit.get("ids_match"),
                "selected_exact_frames": audit.get("selected_exact_frames"),
                "num_frames": len(meta[vid]["frames"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
    os.replace(tmp, out_path)
    rows = read_jsonl(out_path)
    con = [r for r in rows if r["contested"]]
    ok = [r for r in con if r["cand_file"]]
    match = [r for r in ok if r["ids_match"]]
    exact = [r for r in match if r.get("selected_exact_frames") is not None
             and r["selected_exact_frames"] == r["num_frames"]]
    audited = [r for r in match if r.get("selected_exact_frames") is not None]
    summary = {
        "expressions": len(rows),
        "contested": len(con),
        "contested_replayed": len(ok),
        "contested_ids_match": len(match),
        "contested_selected_pixel_exact": len(exact),
        "contested_pixel_audited": len(audited),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "replay_audit_summary.json"), "w"), indent=2)
    print("[manifest]", json.dumps(summary))
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args.out_dir = os.path.abspath(args.out_dir)
    meta, hist = load_inputs(args)
    if args.manifest_only:
        build_manifest(args, meta, hist)
        return

    image_root = os.path.abspath(args.image_root or args.revos_root)
    keys, pairs = collect_pairs(args, meta, hist)
    status_path = os.path.join(
        args.out_dir, "pair_status.jsonl" if args.num_shards <= 1 else f"pair_status.shard{args.shard_id}.jsonl")
    done = {r["pair"] for r in read_jsonl(status_path) if r.get("status") == "ok"}
    todo = [k for k in keys if k not in done]
    print(f"[step1] pairs total={len(keys)} done={len(keys) - len(todo)} todo={len(todo)}")

    hist_results = None
    if args.historical_results:
        print("[step1] loading historical results.json for the pixel audit ...")
        hist_results = json.load(open(args.historical_results))

    if args.mock_recall:
        backend = MockRecall(args)
        backend.meta = meta
    else:
        if not (args.crf_root and args.sam3_checkpoint):
            raise SystemExit("--crf_root and --sam3_checkpoint are required (unless --manifest_only)")
        backend = FrozenDMSRecall(args)

    try:
        from tqdm import tqdm
        it = tqdm(todo, desc="replay")
    except Exception:
        it = todo
    for pk in it:
        vid, cat = pk.split("||", 1)
        recs = pairs[pk]
        cat_raw = recs[0].get("used_category_query") or recs[0].get("category")
        rel = cand_relpath(vid, cat)
        row = {"pair": pk, "video_id": vid, "category": cat_raw, "cand_file": rel}
        try:
            if args.mock_recall:
                backend._vid = vid
            cand = replay_pair(backend, args, meta, vid, cat_raw, image_root)
            write_json_gz(os.path.join(args.out_dir, "candidates", rel), cand)
            row.update(status="ok", candidate_ids=cand["candidate_ids"],
                       elapsed=round(cand["elapsed_seconds"], 2),
                       audit=audit_pair(cand, recs, hist_results))
        except Exception as e:  # keep going; OOM on one long video should not kill the run
            row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])
            print(f"[step1][error] {pk}: {e}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        append_jsonl(status_path, row)

    if args.num_shards <= 1:
        build_manifest(args, meta, hist)
    else:
        print("[step1] shard finished. After all shards finish run again with --manifest_only.")


if __name__ == "__main__":
    main()
