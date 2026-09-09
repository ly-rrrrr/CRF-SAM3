#!/usr/bin/env python3
"""Freeze valid Inferera category predictions into an auditable training cache."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PERSON_ALIASES = {
    "adult",
    "adults",
    "baby",
    "babies",
    "boy",
    "boys",
    "child",
    "children",
    "girl",
    "girls",
    "human",
    "humans",
    "individual",
    "individuals",
    "kid",
    "kids",
    "man",
    "men",
    "people",
    "person",
    "woman",
    "women",
}


def normalize_category(value: Any) -> tuple[str, str]:
    category = re.sub(r"\s+", " ", str(value or "").strip().lower())
    category = category.rstrip(".,;:!? ")
    if category in PERSON_ALIASES:
        return "person", "person_ontology"
    return category, "surface_form"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_json).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    source_records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(source_records, list):
        raise TypeError("Inferera result must contain a records list")

    records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    skipped = Counter()
    normalization_counts = Counter()
    for source in source_records:
        if not isinstance(source, dict) or source.get("status") != "ok":
            skipped["status_not_ok"] += 1
            continue
        video_id = source.get("video_id")
        exp_id = source.get("exp_id")
        raw_category = str(source.get("pred_category", "")).strip()
        if video_id is None or exp_id is None or not raw_category:
            skipped["missing_required_field"] += 1
            continue
        category, normalization = normalize_category(raw_category)
        if not category or len(category) > 80 or any(ch in category for ch in "{}[]\n\r"):
            skipped["invalid_category"] += 1
            continue
        key = (str(video_id), int(exp_id))
        temporal_views = source.get("temporal_views", [])
        frame_indices = [
            int(view["frame_index"])
            for view in temporal_views
            if isinstance(view, dict) and view.get("frame_index") is not None
        ]
        records_by_key[key] = {
            "sample_id": f"revos:{key[0]}:{key[1]}",
            "video_id": key[0],
            "exp_id": key[1],
            "expression": str(source.get("expression", "")),
            "raw_category": raw_category,
            "category": category,
            "normalization": normalization,
            "num_targets": 1,
            "anno_id": source.get("anno_id", []),
            "type_id": source.get("type_id"),
            "frame_indices": frame_indices,
            "source": "inferera_qwen3.7_plus_bbox_3views",
        }
        normalization_counts[normalization] += 1

    records = [records_by_key[key] for key in sorted(records_by_key)]
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_json": str(input_path),
        "source_records": len(source_records),
        "records": records,
        "summary": {
            "usable_records": len(records),
            "skipped": dict(skipped),
            "normalization_counts": dict(normalization_counts),
            "num_targets_policy": "single_expression_target_defaults_to_1",
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(output_path)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
