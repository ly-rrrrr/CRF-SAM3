#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


def _load_subset_pairs(path: Optional[str]) -> Optional[Set[Tuple[str, int]]]:
    if not path:
        return None
    subset_path = Path(path).resolve()
    with subset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("samples", payload)
    if not isinstance(items, list):
        raise TypeError(f"Unsupported subset JSON format: {subset_path}")
    pairs: Set[Tuple[str, int]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if "video_id" not in item or "exp_id" not in item:
            continue
        pairs.add((str(item["video_id"]), int(item["exp_id"])))
    return pairs


def _iter_sample_dirs(split_root: Path):
    if not split_root.is_dir():
        return
    for video_dir in sorted(split_root.iterdir()):
        if not video_dir.is_dir():
            continue
        for exp_dir in sorted(video_dir.iterdir()):
            if exp_dir.is_dir():
                yield exp_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy per-sample offline-cache directories from one cache root into another.")
    parser.add_argument("--src-cache-root", required=True)
    parser.add_argument("--dst-cache-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset-json", default="")
    args = parser.parse_args()

    src_root = Path(args.src_cache_root).resolve()
    dst_root = Path(args.dst_cache_root).resolve()
    src_split = src_root / args.split
    dst_split = dst_root / args.split
    if not src_split.is_dir():
        raise FileNotFoundError(f"Source split root not found: {src_split}")
    dst_split.mkdir(parents=True, exist_ok=True)

    subset_pairs = _load_subset_pairs(args.subset_json)
    copied = 0
    skipped = 0

    for src_exp_dir in _iter_sample_dirs(src_split):
        video_id = src_exp_dir.parent.name
        exp_id = int(src_exp_dir.name)
        if subset_pairs is not None and (video_id, exp_id) not in subset_pairs:
            skipped += 1
            continue
        dst_exp_dir = dst_split / video_id / str(exp_id)
        dst_exp_dir.parent.mkdir(parents=True, exist_ok=True)
        if dst_exp_dir.exists():
            shutil.rmtree(dst_exp_dir)
        shutil.copytree(src_exp_dir, dst_exp_dir)
        copied += 1

    print(
        json.dumps(
            {
                "src_cache_root": str(src_root),
                "dst_cache_root": str(dst_root),
                "split": args.split,
                "copied": copied,
                "skipped": skipped,
                "subset_filter": bool(subset_pairs is not None),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
