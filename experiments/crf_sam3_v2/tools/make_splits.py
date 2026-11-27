#!/usr/bin/env python3
"""Stage 1 (CPU, read-only on the data): one splits file for the whole pipeline.

For every dataset's official train split, by VIDEO (never by expression):
  holdout : the v1 Ground val videos, kept exactly (Ref-YTVOS: refytvos_query_grounding/val.jsonl;
            ReVOS: revos_query_grounding_api_20260717/val.jsonl, i.e. ~18 of the 175 videos with API
            categories), PLUS 10% (seed 42) of the videos that are in neither v1 train nor v1 val
            (ReVOS: the ~451 videos without categories, which still enter Verify training).
            MeViS / DAVIS have no v1 split, so this is just 10% of their videos.
            Thresholds, Verify tuning and Ground val use only holdout videos.
  train   : the rest.
  fold    : for the Ground datasets (ReVOS, Ref-YTVOS) every train video gets fold 0 or 1 for
            cross-fitting: the Verify cache of a fold-k video takes words / heatmaps from the Ground
            model trained WITHOUT fold k.

Also checks, per video, that meta frames == sorted JPEG stems (audit A) and records the result.

  python tools/make_splits.py --video-datas $VD --repo $REPO --out runs/data/splits.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from v2.gt import GROUND_SETS, VERIFY_SETS, TrainSet  # noqa: E402

VD = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas"


V1_GROUND = {"refytvos": "category_grounding_mvp/data/refytvos_query_grounding",
             "revos": "category_grounding_mvp/data/revos_query_grounding_api_20260717"}


def jsonl_videos(p):
    if not Path(p).is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return {str(json.loads(x)["video_id"]) for x in f if x.strip()}


def split_videos(vids, ratio, seed, folds, v1_train=(), v1_val=()):
    """holdout = v1 val (kept) + ratio of the videos v1 never saw; folds over the rest."""
    vids = set(vids)
    keep = sorted(vids & set(v1_val))
    fresh = sorted(vids - set(v1_train) - set(v1_val))
    rng = random.Random(seed)
    rng.shuffle(fresh)
    n_new = int(round(len(fresh) * ratio)) if len(fresh) > 1 else 0
    if not keep and len(fresh) > 1:
        n_new = max(1, n_new)
    hold = sorted(keep + fresh[:n_new])
    train = sorted(vids - set(hold))
    rng = random.Random(seed + 1)
    order = list(train)
    rng.shuffle(order)
    fold = {v: i % folds for i, v in enumerate(order)} if folds else {}
    return hold, train, fold


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-datas", default=VD)
    ap.add_argument("--repo", default="", help="$REPO: the v1 Ground jsonl splits are kept from here")
    ap.add_argument("--datasets", default=",".join(VERIFY_SETS))
    ap.add_argument("--holdout-ratio", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "runs/data/splits.json"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits file")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"[splits] {out} exists; it is the single source of truth for every later stage. "
              f"Pass --force to rebuild it (and then rebuild everything after it).")
        return 0
    res = {"rule": {"by": "video", "holdout": "v1 Ground val kept + ratio of videos outside v1",
                    "holdout_ratio": a.holdout_ratio, "folds": a.folds, "seed": a.seed,
                    "fold_datasets": list(GROUND_SETS)},
           "datasets": {}}
    bad_total = 0
    for name in a.datasets.split(","):
        ds = TrainSet(name, a.video_datas)
        bad = {v: r for v in ds.videos if (r := ds.check_frames(v))}
        bad_total += len(bad)
        v1_train = v1_val = set()
        if name in V1_GROUND:
            d = Path(a.repo) / V1_GROUND[name]
            v1_train, v1_val = jsonl_videos(d / "train.jsonl"), jsonl_videos(d / "val.jsonl")
            if v1_train is None or v1_val is None:
                print(f"[splits] FAIL: {name}: v1 split {d}/{{train,val}}.jsonl not found (pass --repo)")
                return 1
        hold, train, fold = split_videos(list(ds.videos), a.holdout_ratio, a.seed,
                                         a.folds if name in GROUND_SETS else 0, v1_train, v1_val)
        n_expr = {v: len(ds.videos[v].get("expressions", {})) for v in ds.videos}
        res["datasets"][name] = {
            "meta": str(ds.meta_path), "mask_dict": str(ds.md_path), "key_mode": ds.mode,
            "videos": len(ds.videos), "expressions": ds.n_expressions(),
            "holdout": hold, "train": train, "fold": fold,
            "v1_kept": {"val_videos_kept": len(set(hold) & set(v1_val)),
                        "v1_train_videos_in_train": len(set(train) & set(v1_train)),
                        "v1_videos_missing_from_meta": len((set(v1_train) | set(v1_val)) - set(ds.videos)),
                        "fresh_videos": len(set(ds.videos) - set(v1_train) - set(v1_val))} if name in V1_GROUND else None,
            "holdout_expressions": sum(n_expr[v] for v in hold),
            "train_expressions": sum(n_expr[v] for v in train),
            "frame_mismatch": bad,
        }
        print(f"[splits] {name}: {len(ds.videos)} videos -> holdout {len(hold)} / train {len(train)}"
              + (f" (fold0 {sum(1 for f in fold.values() if f == 0)}, fold1 {sum(1 for f in fold.values() if f == 1)})"
                 if fold else "") + (f"; FRAME MISMATCH {len(bad)}" if bad else ""), flush=True)
    res["manifest"] = {"script": "tools/make_splits.py", "args": vars(a), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if bad_total:
        raise SystemExit(f"Frame mismatch in {bad_total} videos; splits not written")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[splits] wrote {out}")
    if bad_total:
        print(f"[splits] FAIL: {bad_total} videos whose meta frames differ from the JPEGs; see frame_mismatch")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
