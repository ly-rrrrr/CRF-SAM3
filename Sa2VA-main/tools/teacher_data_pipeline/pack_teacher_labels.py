#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_short_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if "\n" in text:
        return None
    if len(text) > 128:
        return None
    return text


def validate_output(output: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    expression = normalize_short_text(output.get("expression"))
    masked_expression = normalize_short_text(output.get("masked_expression"))
    category = normalize_short_text(output.get("category"))
    backup_categories = output.get("backup_categories")

    if expression is None:
        return None, "invalid_expression"
    if masked_expression is None:
        return None, "invalid_masked_expression"
    if masked_expression.count("[MASK]") != 1:
        return None, "masked_expression_must_contain_exactly_one_mask"
    if category is None:
        return None, "invalid_category"
    if not isinstance(backup_categories, list):
        return None, "backup_categories_not_list"
    if len(backup_categories) > 2:
        return None, "too_many_backup_categories"

    normalized_backups: List[str] = []
    for item in backup_categories:
        backup = normalize_short_text(item)
        if backup is None:
            return None, "invalid_backup_category"
        normalized_backups.append(backup)

    return {
        "expression": expression,
        "masked_expression": masked_expression,
        "category": category,
        "backup_categories": normalized_backups,
    }, None


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and pack local teacher outputs into final labels.")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejects", required=True)
    args = parser.parse_args()

    request_map = {row["sample_id"]: row for row in load_jsonl(Path(args.requests))}

    accepted = 0
    rejected = 0
    for response_row in load_jsonl(Path(args.responses)):
        sample_id = response_row.get("sample_id")
        if not sample_id:
            append_jsonl(Path(args.rejects), {"sample_id": None, "reason": "missing_sample_id", "row": response_row})
            rejected += 1
            continue

        request_row = request_map.get(sample_id)
        if request_row is None:
            append_jsonl(Path(args.rejects), {"sample_id": sample_id, "reason": "sample_id_not_found_in_requests", "row": response_row})
            rejected += 1
            continue

        output = response_row.get("output")
        if not isinstance(output, dict):
            append_jsonl(Path(args.rejects), {"sample_id": sample_id, "reason": "missing_output_object", "row": response_row})
            rejected += 1
            continue

        normalized, reason = validate_output(output)
        if normalized is None:
            append_jsonl(Path(args.rejects), {"sample_id": sample_id, "reason": reason, "row": response_row})
            rejected += 1
            continue

        final_row = {
            "sample_id": sample_id,
            "source": request_row["source"],
            "teacher_label": normalized,
        }
        append_jsonl(Path(args.output), final_row)
        accepted += 1

    print(
        json.dumps(
            {
                "requests": args.requests,
                "responses": args.responses,
                "output": args.output,
                "rejects": args.rejects,
                "accepted": accepted,
                "rejected": rejected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
