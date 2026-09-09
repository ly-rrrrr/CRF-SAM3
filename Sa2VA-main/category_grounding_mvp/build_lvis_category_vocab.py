#!/usr/bin/env python3
"""Build a visual noun vocabulary from LVIS category annotations.

This script intentionally extracts category names and synonyms only. It does
not use LVIS images or instance masks; the output is a lightweight vocabulary
for SAM3 text-space prototype construction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_prompt_text(text: str) -> str:
    return " ".join(text.replace("_", " ").split()).strip()


def iter_lvis_categories(annotation_path: Path) -> list[dict[str, Any]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"LVIS annotation has no categories list: {annotation_path}")

    records: list[dict[str, Any]] = []
    for category in categories:
        raw_name = str(category["name"])
        synonyms = category.get("synonyms") or []
        if not isinstance(synonyms, list):
            synonyms = []

        variants = []
        seen = set()
        for item in [raw_name, *synonyms]:
            raw = str(item).strip()
            prompt_text = normalize_prompt_text(raw)
            if not prompt_text:
                continue
            key = prompt_text.lower()
            if key in seen:
                continue
            seen.add(key)
            variants.append({"raw": raw, "prompt_text": prompt_text})

        records.append(
            {
                "source": "lvis_v1",
                "category_id": category.get("id"),
                "name": raw_name,
                "prompt_text": normalize_prompt_text(raw_name),
                "synset": category.get("synset"),
                "definition": category.get("def"),
                "frequency": category.get("frequency"),
                "instance_count": category.get("instance_count"),
                "image_count": category.get("image_count"),
                "variants": variants,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        default="/mnt/csip-500/share/kongjiawei/Sa2VA/data/datasets/lvis/annotations/lvis_v1_train.json",
        help="Path to LVIS v1 annotation json.",
    )
    parser.add_argument(
        "--output-dir",
        default="category_grounding_mvp/data/visual_vocab/lvis_v1",
        help="Directory for vocabulary outputs.",
    )
    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = iter_lvis_categories(annotation_path)

    jsonl_path = output_dir / "categories.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "annotation": str(annotation_path),
        "num_categories": len(records),
        "num_variants": sum(len(r["variants"]) for r in records),
        "frequency_counts": dict(Counter(r.get("frequency") for r in records)),
        "output_jsonl": str(jsonl_path),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
