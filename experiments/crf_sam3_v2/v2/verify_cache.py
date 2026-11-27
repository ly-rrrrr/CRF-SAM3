#!/usr/bin/env python3
"""Stage 4 (GPU): Verify training caches, built with the SAME candidate pool as inference.

Per expression of the official train split (training_set_spec.md, "Verify 训练缓存"):
  1. word   : Ground v2 word of the model that did NOT train on this video (cf0 / cf1 for Ground
              training videos, full otherwise; stage 3 wrote one categories json + heat dir per model).
  2. recall : the repo's CRFSAM3Pipeline.predict from the shared mainline yaml, i.e. the extractor's
              num_targets -> SAM3-DMS recall 24 -> keep 7 by SAM3 score sum. Tube rendering and Verify
              scoring are switched off here (they do not change the pool).
  3. seeds  : v2.mainline.point_seeds with the same heatmap / thresholds as inference.
  4. labels : J over ALL frames, eval-style (v2.gt.j_scores). iou_map = J_best, so the trainer's own
              pos_iou 0.5 / neg_iou 0.1 give the spec's labels. j_union is stored for reporting.
  5. no positive (max J_best < 0.5): not written into the training cache (the trainer then never
              samples it, so force_pos can never fire); recorded in excluded.jsonl. Holdout caches keep
              every expression (they are for tuning, not training).
Output layout = what crf_sam3_attn/dataset.py::_get_offline_listwise_item reads:
  <out>/<part>_root/train/<video>/<exp>/meta.json, masks/obj_<id>/<pos:05d>.png
(the split dir is always "train": these are train-split videos; <part> separates train from holdout).
done.jsonl (one line per finished expression, written last) makes the run resumable; --shard i/n
splits the videos between processes / GPUs.

  python -m v2.verify_cache --dataset refytvos --part train --splits runs/data/splits.json \
      --members runs/data/ground/members.json --words runs/ground_infer/refytvos \
      --repo $REPO --config runs/verify/mainline.yaml --out runs/verify/cache/refytvos --shard 0/2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from v2.gt import TrainSet, j_scores, label_of  # noqa: E402
from v2.validation import validate_heat
from v2 import mainline as ml  # noqa: E402

VD = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas"


def word_model(ds: str, vid: str, members: Dict[str, Dict[str, int]]) -> str:
    """Which Ground model's word / heatmap a video's cache uses (cross-fitting)."""
    f = members.get(ds, {}).get(vid)
    return "full" if f is None else f"cf{int(f)}"


def shard_of(vid: str, n: int) -> int:
    return int(hashlib.md5(vid.encode()).hexdigest(), 16) % n


def merge_words(words_dir: Path, out: Path) -> Path:
    """One categories json for the pipeline's override cache: records of every model, keyed by (video, expression)."""
    recs = []
    for m in ("full", "cf0", "cf1"):
        p = words_dir / m / "categories.json"
        if p.is_file():
            for r in json.loads(p.read_text(encoding="utf-8"))["records"]:
                recs.append(dict(r, ground_model=m))
    if not recs:
        raise SystemExit(f"no Ground words under {words_dir}/<full|cf0|cf1>/categories.json; run stage 3 first")
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"summary": {"total": len(recs)}, "records": recs}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    return out


def save_candidate(d: Path, oid: int, tube: Dict[int, np.ndarray]) -> List[int]:
    from PIL import Image
    md = d / "masks" / f"obj_{oid}"
    md.mkdir(parents=True, exist_ok=True)
    pos = []
    for t in sorted(tube):
        m = np.asarray(tube[t]).astype(bool)
        if m.any():
            Image.fromarray(m.astype(np.uint8) * 255).save(md / f"{int(t):05d}.png")
            pos.append(int(t))
    return pos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["mevis", "davis17", "refytvos", "revos"])
    ap.add_argument("--part", default="train", choices=["train", "holdout"])
    ap.add_argument("--video-datas", default=VD)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--members", required=True, help="runs/data/ground/members.json (Ground training videos + fold)")
    ap.add_argument("--words", required=True, help="stage-3 dir of this dataset: <full|cf0|cf1>/{categories.json,npz/}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="first N videos of this shard (smoke test)")
    ml.add_pool_args(ap)
    a = ap.parse_args(argv)
    a.adapter, a.num_tube_frames = "", 0
    si, sn = (int(x) for x in a.shard.split("/"))

    ds = TrainSet(a.dataset, a.video_datas)
    sp = json.loads(Path(a.splits).read_text(encoding="utf-8"))["datasets"][a.dataset]
    members = json.loads(Path(a.members).read_text(encoding="utf-8"))
    vids = [v for v in sorted(sp[a.part]) if shard_of(v, sn) == si]
    if a.limit:
        vids = vids[: a.limit]
    words_dir = Path(a.words).resolve()
    out = Path(a.out).resolve()
    root = out / f"{a.part}_root" / "train"
    root.mkdir(parents=True, exist_ok=True)
    done_path = out / f"done_{a.part}_{si}of{sn}.jsonl"
    done = set()
    for p in out.glob(f"done_{a.part}_*.jsonl"):
        done |= {json.loads(x)["key"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [(v, e) for v in vids for e in sorted(ds.videos[v]["expressions"]) if f"{v}/{e}" not in done]
    print(f"[cache] {a.dataset}/{a.part} shard {si}/{sn}: {len(vids)} videos, {len(todo)} expressions to do", flush=True)
    if not todo:
        return 0
    a.categories = str(merge_words(words_dir, out / f"categories_all_{si}of{sn}.json"))
    have = {(str(r["video_id"]), str(r.get("expression", "")).strip())
            for r in json.loads(Path(a.categories).read_text(encoding="utf-8"))["records"] if r.get("pred_category")}
    miss = [k for k in ((v, str(ds.videos[v]["expressions"][e].get("exp", "")).strip()) for v, e in todo) if k not in have]
    if miss:
        raise SystemExit(f"{len(miss)} expressions have no Ground word (first {miss[0]}); finish stage 3 first")
    a.out = str(out)
    for k in ("repo", "config"):
        setattr(a, k, str(Path(getattr(a, k)).resolve()))

    pipe, demo, offload = ml.build_pipeline(a)
    # the pool does not depend on tube images or Verify scores: skip both
    pipe.tube_builder.build_tubes = lambda video_path, masks, expr: {int(o): {"frame_paths": []}
                                                                     for fm in masks.values() for o in fm}
    pipe.discriminator.score_and_filter = lambda tubes, expr, k, return_details=False, **kw: ([], {}) if return_details else []
    from p1a.common import list_frames
    import torch

    t_start, n_err = time.time(), 0
    for n, (vid, eid) in enumerate(todo, 1):
        key = f"{vid}/{eid}"
        t0 = time.time()
        try:
            video_path = str(ds.root / "JPEGImages" / vid)
            frames = list_frames(video_path)
            if frames != ds.frames(vid):
                raise RuntimeError(f"JPEG list != meta frames for {vid}")
            pos_of = {s: i for i, s in enumerate(frames)}
            T = len(frames)
            expr = ds.videos[vid]["expressions"][eid].get("exp", "")
            gmodel = word_model(a.dataset, vid, members)
            targets = ds.targets(vid, eid)
            rec = {"key": key, "video_id": vid, "exp_id": int(eid), "ground_model": gmodel}
            if not targets:
                rec["status"] = "no_gt"
            else:
                _, _, det = pipe.predict(video_path, expr, return_details=True)
                cands = ml.candidates_of(det.get("all_masks") or {})
                ssum = {int(k): float(v) for k, v in (det.get("candidate_score_sums") or {}).items()}
                category = det.get("category") or ""
                heat = words_dir / gmodel / "npz" / vid / f"{eid}.npz"
                validate_heat(heat)
                new, info = ml.point_seeds(a, pipe, video_path, cands, heat, pos_of, category, offload)
                tubes = dict(cands)
                pids = [ml.POINT_BASE + i for i in range(len(new))]
                tubes.update(zip(pids, new))
                shape = next((np.asarray(m).shape for tb in tubes.values() for m in tb.values()), None)
                jm, ju = {}, {}
                for o, tb in tubes.items():
                    seq = [tb.get(t) for t in range(T)]
                    ju[o], jm[o] = j_scores(seq, targets, shape)
                order = sorted(cands, key=lambda o: (-ssum.get(o, 0.0), o)) + pids
                labels = {o: label_of(jm[o]) for o in order}
                rec.update(category=category, num_targets=int(det.get("num_targets") or 1), n_word=len(cands),
                           n_seed=len(pids), max_j=max(jm.values(), default=0.0),
                           n_pos=sum(1 for v in labels.values() if v == 1), seed_info=info)
                if a.part == "train" and rec["n_pos"] == 0:
                    rec["status"] = "no_positive" if tubes else "no_candidate"
                else:
                    d = root / vid / str(eid)
                    fidx = {str(o): save_candidate(d, o, tubes[o]) for o in order}
                    meta = {"video_id": vid, "exp_id": int(eid), "expression": expr, "category": category,
                            "num_targets": rec["num_targets"], "sam3_prompt_source": "ground_v2",
                            "ground_model": gmodel, "frame_pad": 5, "num_frames_video": T,
                            "obj_ids": [o for o in order if fidx[str(o)]],
                            "iou_map": {str(o): round(jm[o], 6) for o in order},
                            "j_union": {str(o): round(ju[o], 6) for o in order},
                            "source": {str(o): ("seed" if o >= ml.POINT_BASE else "word") for o in order},
                            "score_sum": {str(o): ssum.get(o) for o in order},
                            "frame_indices": fidx, "video_path": video_path,
                            "j_definition": "mean IoU over all frames, empty-vs-empty = 1; iou_map = J_best"}
                    tmp = d / "meta.json.tmp"
                    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                    os.replace(tmp, d / "meta.json")      # meta.json last: the trainer only sees complete samples
                    rec["status"] = "ok"
            rec["sec"] = round(time.time() - t0, 1)
            with open(done_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            el = time.time() - t_start
            print(f"[{n}/{len(todo)}] {key} {rec['status']} '{rec.get('category', '')}' word={rec.get('n_word', 0)} "
                  f"seed={rec.get('n_seed', 0)} pos={rec.get('n_pos', 0)} maxJ={rec.get('max_j', 0):.2f} {rec['sec']}s "
                  f"(eta {el / n * (len(todo) - n) / 3600:.1f} h)", flush=True)
        except Exception as e:  # not in done.jsonl -> retried on re-run (OOM is recorded and skipped)
            if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with open(done_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "video_id": vid, "exp_id": int(eid), "status": "oom",
                                        "frames": len(ds.frames(vid))}) + "\n")
                print(f"[oom] {key} ({len(ds.frames(vid))} frames): recorded as skipped", flush=True)
                continue
            n_err += 1
            import traceback
            print(f"[error] {key}: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if n_err:
        print(f"[cache] {n_err} expressions failed; re-run the same command to retry them", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
