#!/usr/bin/env python3
"""Stage 3 helper (CPU): which Ground model predicts which video, as gv2.predict jobs.

Cross-fitting rule (same as v2.verify_cache.word_model): a train-split video that entered Ground
training (runs/data/ground/members.json) gets the model trained WITHOUT its fold (cf0 / cf1); every
other video (holdout, ReVOS videos without categories, all MeViS / DAVIS, every valid split) gets "full".

  plan  : writes <out>/<set>/<model>/meta_p<k>.json (subsets of the official meta, split in --parts
          parts by video for the GPUs) and prints one job per line:
          <set> <model> <part> <data_root> <meta> <output_json> <heat_dir>
  merge : joins <out>/<set>/<model>/categories_p*.json into categories.json (heatmaps already share npz/)

<set> = <dataset> for train + holdout videos, <dataset>_valid for the evaluation split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from v2.gt import TrainSet, VERIFY_SETS  # noqa: E402

VD = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas"


def part_of(vid: str, n: int) -> int:
    return int(hashlib.md5(vid.encode()).hexdigest(), 16) % n


def valid_sets(vd: Path, refyt: Path, revos_root: Path):
    """(name, data_root for gv2.predict, meta path) of the four evaluation splits."""
    return [("mevis_valid", vd / "mevis" / "valid_u" / "JPEGImages", vd / "mevis" / "valid_u" / "meta_expressions.json"),
            ("davis17_valid", vd / "davis17" / "valid" / "JPEGImages",
             vd / "davis17" / "meta_expressions" / "valid" / "meta_expressions.json"),
            ("refytvos_valid", refyt / "valid" / "JPEGImages", refyt / "valid" / "meta_expressions_challenge.json"),
            ("revos_valid", revos_root, revos_root / "meta_expressions_valid_.json")]


def write_jobs(out: Path, name: str, model: str, data_root: Path, videos: dict, parts: int, jobs: list):
    d = out / name / model
    d.mkdir(parents=True, exist_ok=True)
    for k in range(parts):
        sub = {v: x for v, x in videos.items() if part_of(v, parts) == k}
        if not sub:
            continue
        mp = d / f"meta_p{k}.json"
        mp.write_text(json.dumps({"videos": sub}, ensure_ascii=False), encoding="utf-8")
        jobs.append((name, model, k, str(data_root), str(mp), str(d / f"categories_p{k}.json"), str(d / "npz"),
                     sum(len(x.get("expressions", {})) for x in sub.values())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["plan", "merge"])
    ap.add_argument("--video-datas", default=VD)
    ap.add_argument("--refyt", default="")
    ap.add_argument("--revos-root", default="")
    ap.add_argument("--splits", default=str(HERE / "runs/data/splits.json"))
    ap.add_argument("--members", default=str(HERE / "runs/data/ground/members.json"))
    ap.add_argument("--out", default=str(HERE / "runs/ground_infer"))
    ap.add_argument("--parts", type=int, default=2)
    ap.add_argument("--no-valid", action="store_true")
    ap.add_argument("--datasets", default=",".join(VERIFY_SETS))
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.step == "merge":
        bad = 0
        for d in sorted(p for p in out.glob("*/*") if p.is_dir()):
            if d.parent.name.removesuffix("_valid") not in a.datasets.split(","):
                continue
            parts = sorted(d.glob("categories_p*.json"))
            metas = sorted(d.glob("meta_p*.json"))
            if not metas:
                continue
            expected = {(str(vid), str(eid)) for m in metas
                        for vid,v in json.loads(m.read_text())["videos"].items()
                        for eid in v.get("expressions", {})}
            recs = [r for p in parts for r in json.loads(p.read_text())["records"]]
            keys = [(str(r["video_id"]), str(r["exp_id"])) for r in recs if r.get("pred_category")]
            missing_heat = [k for k in expected if not (d / "npz" / k[0] / f"{k[1]}.npz").is_file()]
            invalid = set(keys) != expected or len(keys) != len(expected) or bool(missing_heat)
            print(f"[ground-infer] {d.parent.name}/{d.name}: words {len(keys)}/{len(expected)}, missing heat {len(missing_heat)}")
            if invalid:
                bad += 1
                continue
            from v2.validation import validate_heat
            for vid,eid in sorted(expected):
                validate_heat(d / "npz" / vid / f"{eid}.npz")
            (d / "categories.json").write_text(json.dumps({"summary": {"total": len(recs)}, "records": recs}, ensure_ascii=False))
        return 1 if bad else 0

    sp = json.loads(Path(a.splits).read_text(encoding="utf-8"))["datasets"]
    members = json.loads(Path(a.members).read_text(encoding="utf-8"))
    jobs = []
    for name in a.datasets.split(","):
        if name not in sp:
            continue
        ds = TrainSet(name, a.video_datas)
        by_model = {}
        for vid in sp[name]["train"] + sp[name]["holdout"]:
            f = members.get(name, {}).get(vid)
            m = "full" if f is None else f"cf{int(f)}"
            v = dict(ds.videos[vid])
            v["frames"] = ds.frames(vid)
            by_model.setdefault(m, {})[vid] = v
        for m, videos in sorted(by_model.items()):
            write_jobs(out, name, m, ds.root / "JPEGImages", videos, a.parts, jobs)
    if not a.no_valid:
        vd = Path(a.video_datas)
        refyt = Path(a.refyt) if a.refyt else vd / "Ref-Youtube-VOS"
        revos_root = Path(a.revos_root) if a.revos_root else vd / "revos"
        for name, root, meta in valid_sets(vd, refyt, revos_root):
            if name.removesuffix("_valid") not in a.datasets.split(","):
                continue
            if not meta.is_file():
                print(f"[ground-infer] skip {name}: no {meta}", file=sys.stderr)
                continue
            write_jobs(out, name, "full", root, json.loads(meta.read_text(encoding="utf-8"))["videos"], a.parts, jobs)
    for j in jobs:
        print(" ".join(str(x) for x in j))
    print(f"[ground-infer] {len(jobs)} jobs, {sum(j[-1] for j in jobs)} expressions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
