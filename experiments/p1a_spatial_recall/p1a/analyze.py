#!/usr/bin/env python3
"""Summarise results.jsonl from run_p1a into a Markdown table.

If the run covers only the J*<0.5 failure subset of a cache, pass
``--total-n`` (all expressions in that cache) and ``--total-recall`` (its
Recall@0.5 in %) to extrapolate the full-set Recall@0.5:
    R_new = R_old + rescued / total_n
Run the same command on a random sample of *successful* expressions to measure
how many extra candidates the gate adds when nothing was missing.

If the run contains ``det_probe`` (and ideally ``oracle_gt``), every failure
(j_old < thr) is also assigned one cause, judged in this order:
  lost_after_detection  the detector finds the target (IoU>=0.5) with the main
                        word at score >= 0.7, yet no cached tube covers it
                        -> tracking / keep-top-k / video heuristics lose it
  below_threshold       found with the main word, but only at score < 0.7
                        -> new_det_thresh / score_threshold_detection hide it
  word                  not found with the main word, found with --alt-word-json
  detector_blind        not found by either word, oracle tracker seed reaches J>=thr
                        -> SAM3 can segment+track it, but text detection cannot
  seg_or_track_limit    oracle tracker seed also < thr (or no oracle run)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def bucket(n: int) -> str:
    if n == 0:
        return "0 cand"
    if n == 1:
        return "1 cand"
    if n >= 7:
        return ">=7 cand"
    return "2-6 cand"


CAUSES = ("lost_after_detection", "below_threshold", "word", "detector_blind", "seg_or_track_limit")


def diagnose(r, thr: float, hit_iou: float = 0.5, hi: float = 0.7) -> str:
    probes = r["det_probe"]["info"]
    main = [p for p in probes if p["word"] == r.get("category") and p["best_iou"] >= hit_iou]
    if main:
        return "lost_after_detection" if max(p["best_score"] for p in main) >= hi else "below_threshold"
    if any(p["word"] != r.get("category") and p["best_iou"] >= hit_iou for p in probes):
        return "word"
    if "oracle_gt" in r and r["oracle_gt"]["j_star"] >= thr:
        return "detector_blind"
    return "seg_or_track_limit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--total-n", type=int, default=0)
    ap.add_argument("--total-recall", type=float, default=float("nan"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = []
    for p in a.results:
        rows += [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    modes = [m for m in ("rescue", "point", "exemplar", "text_cfg", "text_word", "repropagate", "oracle_gt") if any(m in r for r in rows)]
    N = len(rows)
    j_old = np.array([r["j_old"] for r in rows])
    lines = [f"# P1-A results ({N} expressions)", ""]
    gated = [r for r in rows if "triggered" in r]
    if gated:
        lines.append(f"- triggered (uncovered evidence or zero candidates): {np.mean([r['triggered'] for r in gated]) * 100:.1f}%")
        lines.append(f"- mean uncovered blobs: {np.mean([r['n_uncovered'] for r in gated]):.2f}")
    lines.append(f"- baseline mean J*: {j_old.mean() * 100:.2f}, Recall@{a.thr}: {(j_old >= a.thr).mean() * 100:.2f}%")
    lines.append("")
    lines.append("| mode | mean added tubes | mean J* | Recall@%.1f | rescued (old<thr<=new) | sec/expr |" % a.thr)
    lines.append("|---|---:|---:|---:|---:|---:|")
    summary = {}
    for m in modes:
        rr = [r for r in rows if m in r]
        jn = np.array([r[m]["j_star"] for r in rr])
        jo = np.array([r["j_old"] for r in rr])
        rescued = int(((jo < a.thr) & (jn >= a.thr)).sum())
        added = np.mean([r[m]["added"] for r in rr])
        summary[m] = {"n": len(rr), "rescued": rescued, "mean_added": float(added), "mean_j": float(jn.mean())}
        lines.append(
            f"| {m} | {added:.2f} | {jn.mean() * 100:.2f} | {(jn >= a.thr).mean() * 100:.2f}% | {rescued} | {np.mean([r['sec'] for r in rr]):.0f} |"
        )
    lines.append("")
    lines.append("By candidate count (rescued / n):")
    lines.append("")
    lines.append("| bucket | n | " + " | ".join(modes) + " |")
    lines.append("|---|---:|" + "---:|" * len(modes))
    groups = defaultdict(list)
    for r in rows:
        groups[bucket(r["num_candidates"])].append(r)
    for b in ("0 cand", "1 cand", "2-6 cand", ">=7 cand"):
        g = groups.get(b, [])
        if not g:
            continue
        cells = []
        for m in modes:
            k = sum(1 for r in g if m in r and r["j_old"] < a.thr <= r[m]["j_star"])
            cells.append(f"{k}")
        lines.append(f"| {b} | {len(g)} | " + " | ".join(cells) + " |")
    for m in ("text_cfg", "text_word"):
        rr = [r for r in rows if m in r and r[m]["info"] and "j_star_replace" in r[m]["info"][0]]
        if rr:
            jr = np.array([r[m]["info"][0]["j_star_replace"] for r in rr])
            jo = np.array([r["j_old"] for r in rr])
            lines.append("")
            lines.append(f"{m} used ALONE (replace, not union): mean J* {jr.mean() * 100:.2f}, "
                         f"Recall@{a.thr} {(jr >= a.thr).mean() * 100:.2f}%, "
                         f"newly >=thr {int(((jo < a.thr) & (jr >= a.thr)).sum())}, lost {int(((jo >= a.thr) & (jr < a.thr)).sum())}")
    diag = [(r, diagnose(r, a.thr)) for r in rows if "det_probe" in r and r["j_old"] < a.thr]
    if diag:
        cnt = defaultdict(int)
        for _, c in diag:
            cnt[c] += 1
        lines.append("")
        lines.append(f"Failure causes from det_probe/oracle_gt ({len(diag)} failures):")
        lines.append("")
        lines.append("| cause | n | % |")
        lines.append("|---|---:|---:|")
        for c in CAUSES:
            if cnt.get(c):
                lines.append(f"| {c} | {cnt[c]} | {100 * cnt[c] / len(diag):.1f}% |")
        if a.out:
            with open(Path(a.out).with_suffix(".causes.jsonl"), "w", encoding="utf-8") as f:
                for r, c in diag:
                    f.write(json.dumps({"key": r["key"], "cause": c, "j_old": r["j_old"]}) + "\n")
    if a.total_n and not np.isnan(a.total_recall):
        lines.append("")
        lines.append(f"Full-set extrapolation (total_n={a.total_n}, Recall@{a.thr}={a.total_recall:.2f}%):")
        lines.append("")
        for m in modes:
            lines.append(f"- {m}: {a.total_recall:.2f}% -> {a.total_recall + 100.0 * summary[m]['rescued'] / a.total_n:.2f}%")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
