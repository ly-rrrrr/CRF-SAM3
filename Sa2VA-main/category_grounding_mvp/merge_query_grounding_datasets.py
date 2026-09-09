#!/usr/bin/env python3
"""Create a deterministic mixed-domain query-grounding JSONL dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_records(path: Path, dataset: str) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("dataset", dataset)
            records.append(record)
    return records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refytvos-dir", required=True)
    parser.add_argument("--revos-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    refytvos_dir = Path(args.refytvos_dir)
    revos_dir = Path(args.revos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"seed": args.seed, "splits": {}}
    for split in ("train", "val"):
        records = read_records(refytvos_dir / f"{split}.jsonl", "refytvos")
        records.extend(read_records(revos_dir / f"{split}.jsonl", "revos"))
        random.Random(args.seed + (0 if split == "train" else 1)).shuffle(records)
        write_records(output_dir / f"{split}.jsonl", records)
        summary["splits"][split] = {
            "records": len(records),
            "by_dataset": dict(Counter(record["dataset"] for record in records)),
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
