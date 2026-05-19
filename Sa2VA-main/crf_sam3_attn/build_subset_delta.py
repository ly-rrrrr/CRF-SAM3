#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_samples(path: Path) -> Tuple[Dict, List[Dict]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples", payload)
    if not isinstance(samples, list):
        raise TypeError(f"Unsupported subset JSON format: {path}")
    return payload if isinstance(payload, dict) else {}, samples


def _key(sample: Dict) -> Tuple[str, int]:
    return str(sample["video_id"]), int(sample["exp_id"])


def _index(samples: Iterable[Dict]) -> Dict[Tuple[str, int], Dict]:
    out: Dict[Tuple[str, int], Dict] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if "video_id" not in sample or "exp_id" not in sample:
            continue
        out[_key(sample)] = sample
    return out


def _write(path: Path, summary: Dict, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": samples}, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build added/removed delta subsets from two failure-subset JSON files.")
    parser.add_argument("--old-json", required=True)
    parser.add_argument("--new-json", required=True)
    parser.add_argument("--added-json", required=True)
    parser.add_argument("--removed-json", default="")
    parser.add_argument("--union-json", default="")
    args = parser.parse_args()

    old_path = Path(args.old_json).resolve()
    new_path = Path(args.new_json).resolve()
    _, old_samples = _load_samples(old_path)
    _, new_samples = _load_samples(new_path)

    old_index = _index(old_samples)
    new_index = _index(new_samples)

    old_keys = set(old_index.keys())
    new_keys = set(new_index.keys())

    added_keys = sorted(new_keys - old_keys)
    removed_keys = sorted(old_keys - new_keys)
    union_keys = sorted(new_keys | old_keys)

    added_samples = [new_index[k] for k in added_keys]
    removed_samples = [old_index[k] for k in removed_keys]
    union_samples = [new_index[k] if k in new_index else old_index[k] for k in union_keys]

    summary = {
        "old_json": str(old_path),
        "new_json": str(new_path),
        "old_count": len(old_keys),
        "new_count": len(new_keys),
        "added_count": len(added_samples),
        "removed_count": len(removed_samples),
        "union_count": len(union_samples),
    }

    _write(Path(args.added_json).resolve(), {**summary, "subset_type": "added"}, added_samples)
    if args.removed_json:
        _write(Path(args.removed_json).resolve(), {**summary, "subset_type": "removed"}, removed_samples)
    if args.union_json:
        _write(Path(args.union_json).resolve(), {**summary, "subset_type": "union"}, union_samples)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
