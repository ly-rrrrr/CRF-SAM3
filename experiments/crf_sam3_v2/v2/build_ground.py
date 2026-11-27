#!/usr/bin/env python3
"""Stage 2a (CPU): Ground v2 training data for ReVOS + Ref-YTVOS, built from the raw official data.

Replaces the chain v1 builder -> gv2.build_data / gv2.build_refytvos. Same record fields as gv2
(gv2/data.py reads them unchanged), but
  * GT comes from v2.gt.TrainSet (Sa2VA mask_dict order, sorted-frame positions);
  * splits / folds come from runs/data/splits.json;
  * joint mixing is Sa2VA-style (v2/mix.py). Each of the r_d copies of an expression gets its OWN
    3 frames (copy 0 = evenly spaced, as v1; copy k>0 = a jittered evenly spaced draw), which is what
    Sa2VA's per-sample random frame draw does;
  * Ref-YTVOS targets are checked against the palette PNG (gate: IoU<0.9 on <=0.5% of checked frames).

Categories
  Ref-YTVOS : train/meta.json (every object has one).
  ReVOS     : API category cache (--revos-categories, records of video_id / exp_id / category). An object's
              category = majority over the single-target expressions that refer to it; objects without
              one are plain background (weight 1), as in gv2.build_data. Expressions without an API
              category are left out (counted as no_category).
  Both go through gv2's CategoryNormalizer.

Outputs (--out, default runs/data/ground):
  full/train.jsonl   all train videos            -> Ground model "full" (MeViS, DAVIS, holdout, valid)
  cf0/train.jsonl    train videos of fold 1 only -> model cf0, used for fold-0 videos' Verify caches
  cf1/train.jsonl    train videos of fold 0 only -> model cf1, used for fold-1 videos
  val.jsonl          holdout videos, copy 0 only (Ground val for every model)
  members.json       which (dataset, video) entered Ground training, and its fold
  summary.json       counts, skip reasons, repeats, PNG check, manifest
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from v2.gt import TrainSet, decode, encode, normalize_rle  # noqa: E402
from v2.mix import repeats  # noqa: E402

VD = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas"
REVOS_API_CATS = "category_grounding_mvp/inferera_category_api/results/revos_train_qwen37plus_categories_normalized_20260717.json"


def even(n: int, k: int) -> List[int]:
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [n // 2]


def jittered(n: int, k: int, rng: random.Random) -> List[int]:
    """k positions out of n: one uniform draw inside each of k equal bins."""
    if n <= k:
        return list(range(n))
    edges = np.linspace(0, n, k + 1)
    return sorted({min(n - 1, int(rng.uniform(edges[i], edges[i + 1]))) for i in range(k)})


def merge(rles: List[Optional[dict]]) -> Optional[dict]:
    ms = [m for m in (decode(r) for r in rles if r) if m is not None]
    if not ms:
        return None
    return encode(np.logical_or.reduce(ms))


def area(rle: Optional[dict]) -> int:
    if not rle:
        return 0
    from pycocotools import mask as cm
    return int(cm.area(dict(rle, counts=rle["counts"].encode("ascii"))))


def load_api_categories(path: Path) -> Dict[tuple, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recs = payload.get("records", payload) if isinstance(payload, dict) else payload
    out = {}
    for r in recs:
        if isinstance(r, dict) and r.get("video_id") is not None and r.get("exp_id") is not None and r.get("category"):
            out[(str(r["video_id"]), str(int(r["exp_id"])))] = r
    return out


class Builder:
    def __init__(self, a, splits, norm):
        self.a, self.splits, self.norm = a, splits, norm
        self.stats = defaultdict(Counter)
        self.png_ious: List[float] = []
        self.ds: Dict[str, TrainSet] = {}
        self._png_vid, self._pngs = None, {}

    def png(self, b: dict, f: int):
        """PNG of frame position f, cached for the current video only (records are grouped by video)."""
        key = (b["dataset"], b["video_id"])
        if key != self._png_vid:
            self._png_vid, self._pngs = key, {}
        if f not in self._pngs:
            self._pngs[f] = self.ds[b["dataset"]].png(b["video_id"], b["_frames"][f])
        return self._pngs[f]

    # ------------------------------------------------------------ per dataset: base expressions
    def base(self, name: str) -> List[dict]:
        ds = self.ds[name] = TrainSet(name, self.a.video_datas)
        sp = self.splits["datasets"][name]
        part = {v: "train" for v in sp["train"]}
        part.update({v: "holdout" for v in sp["holdout"]})
        st = self.stats[name]
        api = None
        if name == "revos":
            api = load_api_categories(Path(self.a.revos_categories))
            st["api_records"] = len(api)
        out = []
        by_vid = defaultdict(list)
        for vid, eid, e in ds.expressions():
            by_vid[vid].append((eid, e))
        for vid, exps in by_vid.items():
            if vid not in part:
                st["video_not_in_splits"] += 1
                continue
            frames, paths = ds.frames(vid), ds.frame_paths(vid)
            T = len(frames)
            # target RLEs of every expression of this video
            tr = {eid: ds.target_rles(vid, eid) for eid, _ in exps}
            if name == "refytvos":
                cats = {o: self.norm(c) for o, c in ds.object_categories(vid).items()}
                tgt_obj = {eid: [int(e.get("obj_id"))] for eid, e in exps}
                obj_rles = None  # objects come from the PNGs
            else:
                # ReVOS: objects = anno_ids; category by majority of single-target expressions
                votes = defaultdict(Counter)
                for eid, e in exps:
                    r = api.get((vid, eid))
                    ids = ds.target_ids(vid, eid)
                    if r and len(ids) == 1:
                        votes[ids[0]][self.norm(r["category"])] += 1
                cats = {o: c.most_common(1)[0][0] for o, c in votes.items()}
                tgt_obj = {eid: ds.target_ids(vid, eid) for eid, _ in exps}
                obj_rles = {}
                for eid, _ in exps:
                    for o, seq in zip(ds.target_ids(vid, eid), tr[eid]):
                        obj_rles.setdefault(o, seq)
            for eid, e in exps:
                expr = str(e.get("exp", "")).strip()
                if name == "refytvos":
                    cat_raw = ds.object_categories(vid).get(tgt_obj[eid][0], "")
                else:
                    r = api.get((vid, eid))
                    cat_raw = str(r["category"]) if r else ""
                    if r and r.get("expression") and str(r["expression"]).strip() != expr:
                        st["api_expression_text_mismatch"] += 1
                if not expr:
                    st["skip_no_expression"] += 1
                    continue
                if not cat_raw:
                    st["skip_no_category"] += 1
                    continue
                if not tr[eid]:
                    st["skip_no_mask"] += 1
                    continue
                union = TrainSet.union_rles(tr[eid])
                union += [None] * (T - len(union))
                visible = [f for f in range(T) if area(union[f]) >= self.a.min_area]
                if not visible:
                    st["skip_no_visible_frame"] += 1
                    continue
                group = self.norm(cat_raw)
                others = [o for o in (cats if name == "refytvos" else obj_rles) if o not in tgt_obj[eid]]
                neg = [o for o in others if cats.get(o) is not None and cats[o] != group]
                ign = [o for o in others if cats.get(o) is not None and cats[o] == group]
                st["unknown_other_objects"] += sum(1 for o in others if cats.get(o) is None)
                out.append(dict(dataset=name, video_id=vid, exp_id=int(eid), split=part[vid],
                                fold=sp["fold"].get(vid), expression=expr, category=cat_raw, target_group=group,
                                obj_id=",".join(str(o) for o in tgt_obj[eid]), _union=union, _visible=visible,
                                _neg=neg, _ign=ign, _frames=frames, _paths=paths,
                                _obj_rles=obj_rles, _T=T))
                st["expressions_" + part[vid]] += 1
        st["videos_train"] = len({r["video_id"] for r in out if r["split"] == "train"})
        st["videos_holdout"] = len({r["video_id"] for r in out if r["split"] == "holdout"})
        return out

    # ------------------------------------------------------------ one materialised copy
    def others_rle(self, b: dict, ids: List[int], f: int) -> Optional[dict]:
        if not ids:
            return None
        if b["dataset"] == "refytvos":
            p = self.png(b, f)
            return encode(np.isin(p, ids)) if p is not None else None
        return merge([b["_obj_rles"][o][f] for o in ids if f < len(b["_obj_rles"][o])])

    def record(self, b: dict, k: int, used: Optional[set] = None) -> dict:
        """Copy k of an expression. Copies k>0 redraw (up to 20 times) to avoid frame sets already used."""
        vis = b["_visible"]
        rng = random.Random(f"{b['dataset']}|{b['video_id']}|{b['exp_id']}|{k}|{self.a.seed}")
        if k == 0:
            pos = even(len(vis), self.a.frames)
        else:
            for _ in range(20):
                pos = jittered(len(vis), self.a.frames, rng)
                if used is None or tuple(pos) not in used:
                    break
        if used is not None:
            used.add(tuple(pos))
        sel = [vis[i] for i in pos]
        mask_rles = [b["_union"][f] for f in sel]
        if b["dataset"] == "refytvos" and k == 0:
            tgt = int(b["obj_id"])
            for f, r in zip(sel, mask_rles):
                p = self.png(b, f)
                m = decode(r)
                if p is not None and m is not None and m.shape == p.shape:
                    q = p == tgt
                    self.png_ious.append(float((m.astype(bool) & q).sum()) / max(float((m.astype(bool) | q).sum()), 1.0))
        absent_pool = [f for f in range(b["_T"]) if b["_union"][f] is None]
        absent = []
        for i in even(len(absent_pool), self.a.max_absent):
            f = absent_pool[i]
            absent.append({"frame_index": f, "frame_path": b["_paths"][f],
                           "neg_rle": self.others_rle(b, b["_neg"], f), "ign_rle": self.others_rle(b, b["_ign"], f)})
        return {
            "sample_id": f"{b['dataset']}:{b['video_id']}:{b['exp_id']}:r{k}",
            "dataset": b["dataset"], "video_id": b["video_id"], "exp_id": b["exp_id"], "obj_id": b["obj_id"],
            "expression": b["expression"], "category": b["category"], "target_group": b["target_group"],
            "frames": [b["_frames"][f] for f in sel], "frame_paths": [b["_paths"][f] for f in sel],
            "mask_rles": mask_rles, "mask_frame_indices": sel, "mask_areas": [area(r) for r in mask_rles],
            "neg_rles": [self.others_rle(b, b["_neg"], f) for f in sel],
            "ign_rles": [self.others_rle(b, b["_ign"], f) for f in sel],
            "absent_frames": absent, "repeat_k": k, "fold": b["fold"],
        }


def write_jsonl(p: Path, recs: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-datas", default=VD)
    ap.add_argument("--repo", default="", help="$REPO, to find the ReVOS API category cache")
    ap.add_argument("--revos-categories", default="")
    ap.add_argument("--gv2", default=str(HERE.parent / "ground_v2"), help="ground_v2 package (CategoryNormalizer)")
    ap.add_argument("--splits", default=str(HERE / "runs/data/splits.json"))
    ap.add_argument("--datasets", default="revos,refytvos")
    ap.add_argument("--synonyms", default="")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--min-area", type=int, default=1)
    ap.add_argument("--max-absent", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--png-gate", type=float, default=0.005, help="max share of checked frames with target-vs-PNG IoU < 0.9")
    ap.add_argument("--out", default=str(HERE / "runs/data/ground"))
    a = ap.parse_args(argv)
    if not a.revos_categories and a.repo:
        a.revos_categories = str(Path(a.repo) / REVOS_API_CATS)
    sys.path.insert(0, a.gv2)
    from gv2.common import CategoryNormalizer  # the server's (patched) gv2

    t0 = time.time()
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    b = Builder(a, splits, CategoryNormalizer.from_file(a.synonyms))
    names = [n for n in a.datasets.split(",") if n]
    base = {n: b.base(n) for n in names}
    for n in names:
        print(f"[ground] {n}: {dict(b.stats[n])}", flush=True)

    counts = {n: (len({r["video_id"] for r in base[n] if r["split"] == "train"}),
                  sum(1 for r in base[n] if r["split"] == "train")) for n in names}
    rep = repeats(counts)
    print(f"[ground] repeats: { {n: rep[n]['r'] for n in rep} }", flush=True)

    out = Path(a.out)
    sets = {"full": [], "cf0": [], "cf1": []}
    val = []
    for n in names:
        r_d = rep.get(n, {}).get("r", 1)
        for e in base[n]:
            if e["split"] == "holdout":
                val.append(b.record(e, 0))
                continue
            used = set()
            for k in range(r_d):
                rec = b.record(e, k, used)
                sets["full"].append(rec)
                sets["cf0" if e["fold"] == 1 else "cf1"].append(rec)   # cf<k> never sees fold k
    ious = b.png_ious
    bad = int(sum(x < 0.9 for x in ious))
    png = {"frames_checked": len(ious), "below_0.9": bad, "share_below_0.9": round(bad / max(1, len(ious)), 5),
           "mean": round(float(np.mean(ious)), 4) if ious else None}
    summary = {"stats": {n: dict(b.stats[n]) for n in names}, "repeats": rep,
               "records": {k: len(v) for k, v in sets.items()} | {"val": len(val)},
               "png_check_refytvos": png,
               "manifest": {"script": "v2/build_ground.py", "args": vars(a), "splits": a.splits,
                            "splits_time": splits.get("manifest", {}).get("time"),
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("records", "png_check_refytvos")}, ensure_ascii=False), flush=True)
    if "refytvos" in names and (not ious or png["share_below_0.9"] > a.png_gate):
        print(f"[ground] FAIL: Ref-YTVOS target vs PNG IoU<0.9 on {png['share_below_0.9']:.2%} of frames "
              f"(gate {a.png_gate:.2%}). Do not train; look at the data first.")
        return 1
    if any(b.stats[n].get("api_expression_text_mismatch") for n in names):
        print("[ground] FAIL: ReVOS API category records whose expression text differs from meta; "
              "the cache and revos_prepared disagree about exp ids.")
        return 1
    rng = random.Random(a.seed)
    for name, recs in sets.items():
        rng.shuffle(recs)
        write_jsonl(out / name / "train.jsonl", recs)
    write_jsonl(out / "val.jsonl", val)
    members = {n: {e["video_id"]: e["fold"] for e in base[n] if e["split"] == "train"} for n in names}
    (out / "members.json").write_text(json.dumps(members, ensure_ascii=False), encoding="utf-8")

    (out / "COMPLETE").write_text("validated\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
