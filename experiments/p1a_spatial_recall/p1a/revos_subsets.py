#!/usr/bin/env python3
"""Pick the ReVOS valid samples for heatmap-assisted recall, from ground_v2's step-4 per_expression.csv.

  zero : the v2 word got 0 SAM3 candidates             (the heatmap is the only evidence left)
  fail : candidates exist but none has region J >= 0.5 (target missing from the pool)
  ok   : random --ok-sample of the available ones      (false-trigger cost: how many tubes get added)
  all  : every expression                              (point tubes for the Verify re-eval)

  python -m p1a.revos_subsets --per-expression ../ground_v2/runs/pool_revos/per_expression.csv --out runs/revos_v2
"""
import argparse
import csv
import json
import random
from pathlib import Path


def subset_of(r) -> str:
    if int(r["new_n"]) == 0:
        return "zero"
    return "fail" if float(r["new_oracle_j"]) < 0.5 else "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-expression", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="0,1,2", help="type_id kept (0 Referring, 1 Reasoning, 2 other)")
    ap.add_argument("--ok-sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    types = {int(x) for x in a.types.split(",")}
    rows = [r for r in csv.DictReader(open(a.per_expression, encoding="utf-8")) if int(r["type_id"]) in types]
    groups = {"zero": [], "fail": [], "ok": []}
    for r in rows:
        groups[subset_of(r)].append({"video_id": r["video_id"], "exp_id": r["exp_id"], "type_id": int(r["type_id"]),
                                     "expression": r["expression"], "word": r["new_word"],
                                     "pool_j": float(r["new_oracle_j"]), "pool_n": int(r["new_n"])})
    groups["all"] = groups["zero"] + groups["fail"] + groups["ok"]  # every expression (for the Verify re-eval)
    random.Random(a.seed).shuffle(groups["ok"])
    groups["ok"] = groups["ok"][: a.ok_sample]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in groups.items():
        (out / f"samples_{k}.json").write_text(json.dumps({"samples": v}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{k}: {len(v)} samples -> {out / f'samples_{k}.json'}")


if __name__ == "__main__":
    main()
