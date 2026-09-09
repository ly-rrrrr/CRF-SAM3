#!/usr/bin/env python3
"""Index ReVOS offline-cache samples whose best candidate J is below a threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        default="crf_sam3_attn/offline_cache_revos_api_categories_recall24_keep7_20260717",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        default="crf_sam3_attn/cache_reports/revos_train_api_categories_recall24_keep7_max_j_lt_0p5_20260718.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root).resolve() / args.split
    all_scores: list[float] = []
    failures: list[dict] = []

    for meta_path in sorted(cache_root.rglob("meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            score = float(meta.get("max_j", 0.0))
            video_id = str(meta["video_id"])
            exp_id = int(meta["exp_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[skip] {meta_path}: {exc}")
            continue

        all_scores.append(score)
        if score >= args.threshold:
            continue
        failures.append(
            {
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": str(meta.get("expression", "")),
                "category": str(meta.get("category", "")),
                "sam3_prompt_source": str(meta.get("sam3_prompt_source", "")),
                "sam3_prompt_text": str(meta.get("sam3_prompt_text", "")),
                "candidate_filter": str(meta.get("candidate_filter", "")),
                "num_candidates": int(meta.get("num_candidates", 0)),
                "num_candidates_before_filter": int(meta.get("num_candidates_before_filter", 0)),
                "max_j": score,
                "mean_j": float(meta.get("mean_j", 0.0)),
                "old_max_j": score,
                "new_max_j": score,
                "new_max_j_filled": score,
                "delta": 0.0,
                "status": "same",
                "new_num_candidates": int(meta.get("num_candidates", 0)),
            }
        )

    scores = np.asarray(all_scores, dtype=np.float64)
    summary = {
        "name": "ReVOS API-category offline-cache failures",
        "cache_root": str(cache_root.parent),
        "split": args.split,
        "threshold": args.threshold,
        "num_generated_meta": int(scores.size),
        "num_max_j_lt_threshold": len(failures),
        "rate_max_j_lt_threshold": float(len(failures) / scores.size) if scores.size else 0.0,
        "mean_max_j": float(scores.mean()) if scores.size else 0.0,
        "median_max_j": float(np.median(scores)) if scores.size else 0.0,
        "num_max_j_zero": int(np.count_nonzero(scores <= 1e-12)),
        "num_samples": len(failures),
        "num_improved": 0,
        "num_worse": 0,
        "num_same": len(failures),
        "old_mean_max_j": float(scores.mean()) if scores.size else 0.0,
        "new_mean_max_j_missing_as_zero": float(scores.mean()) if scores.size else 0.0,
        "new_num_ge_0p5_missing_as_zero": 0,
        "new_num_eq_0_missing_as_zero": int(np.count_nonzero(scores <= 1e-12)),
    }
    failures.sort(key=lambda row: (row["max_j"], row["video_id"], row["exp_id"]))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "samples": failures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"[done] wrote {len(failures)} samples to {output}")


if __name__ == "__main__":
    main()
