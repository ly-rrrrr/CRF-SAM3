#!/usr/bin/env python3
"""Build a compact OpenImages hierarchy vocabulary for category prompting.

The output keeps the original OpenImages label, hierarchy paths, and a compact
SAM3-oriented prompt. The compact prompt is intentionally reversible: original
labels and prompt candidates are kept so later stages can audit or change the
choice without rebuilding the hierarchy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_LABELS = "category_grounding_mvp/data/openimages_vocab/raw/oidv7-class-descriptions-boxable.csv"
DEFAULT_HIERARCHY = "category_grounding_mvp/data/openimages_vocab/raw/bbox_labels_600_hierarchy.json"
DEFAULT_OUTPUT = "category_grounding_mvp/data/openimages_vocab/processed_20260703"

PERSON_SUBTYPES = {"man", "woman", "boy", "girl"}
HUMAN_PREFIX = "human "

# Multi-word concepts where the final token alone changes the category meaning.
# This is a vocabulary-level lexicalization list, not a sample-specific fix.
LEXICALIZED_KEEP = {
    "alarm clock",
    "balance beam",
    "bell pepper",
    "billiard table",
    "bow and arrow",
    "christmas tree",
    "cocktail shaker",
    "convenience store",
    "cutting board",
    "fire hydrant",
    "french fries",
    "frying pan",
    "gas stove",
    "hot dog",
    "ice cream",
    "light bulb",
    "parking meter",
    "picture frame",
    "traffic light",
    "traffic sign",
}

# Generic first tokens that usually specify domain/material/source rather than
# the object noun SAM3 should receive as the primary prompt.
DROP_LEADING_MODIFIERS = {
    "baseball",
    "bathroom",
    "bicycle",
    "computer",
    "cricket",
    "football",
    "golf",
    "hand",
    "kitchen",
    "tennis",
}


def clean_label(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = text.replace("_", " ")
    return " ".join(text.lower().split())


def load_labels(path: Path) -> Dict[str, str]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == "LabelName":
                continue
            rows[row[0]] = row[1]
    return rows


def collect_paths(node: Dict[str, Any], path: List[str], paths: Dict[str, List[List[str]]]) -> None:
    mid = str(node["LabelName"])
    current = path + [mid]
    paths[mid].append(current)
    for child in node.get("Subcategory", []) or []:
        collect_paths(child, current, paths)


def path_names(path: Iterable[str], labels: Dict[str, str]) -> List[str]:
    return [labels.get(mid, mid) for mid in path if mid in labels]


def has_ancestor(paths: List[List[str]], labels: Dict[str, str], ancestor_name: str) -> bool:
    target = ancestor_name.lower()
    for path in paths:
        if any(labels.get(mid, "").lower() == target for mid in path[:-1]):
            return True
    return False


def parent_prompts(paths: List[List[str]], labels: Dict[str, str]) -> List[str]:
    prompts = []
    seen = set()
    for path in paths:
        if len(path) < 2:
            continue
        parent = labels.get(path[-2])
        if not parent:
            continue
        prompt = clean_label(parent)
        if prompt and prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def choose_sam3_prompt(display_name: str, paths: List[List[str]], labels: Dict[str, str]) -> str:
    prompt = clean_label(display_name)
    if not prompt:
        return prompt

    tokens = prompt.split()
    if prompt in PERSON_SUBTYPES:
        return "person"
    if prompt.startswith(HUMAN_PREFIX) and len(tokens) > 1:
        return " ".join(tokens[1:])
    if prompt in LEXICALIZED_KEEP:
        return prompt

    # For animals, keep the visual species/category label. This avoids collapsing
    # rabbit/dog/cat into the too-broad "animal" prompt.
    if has_ancestor(paths, labels, "Animal"):
        return prompt

    parents = set(parent_prompts(paths, labels))
    if len(tokens) > 1:
        head = tokens[-1]
        if head in parents:
            return head
        if tokens[0] in DROP_LEADING_MODIFIERS:
            return head
    return prompt


def build_records(labels: Dict[str, str], hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths_by_mid: Dict[str, List[List[str]]] = defaultdict(list)
    collect_paths(hierarchy, [], paths_by_mid)

    child_map: Dict[str, set[str]] = defaultdict(set)
    parent_map: Dict[str, set[str]] = defaultdict(set)

    def walk(node: Dict[str, Any]) -> None:
        mid = str(node["LabelName"])
        for child in node.get("Subcategory", []) or []:
            child_mid = str(child["LabelName"])
            child_map[mid].add(child_mid)
            parent_map[child_mid].add(mid)
            walk(child)

    walk(hierarchy)

    records = []
    for mid, display_name in sorted(labels.items(), key=lambda kv: clean_label(kv[1])):
        paths = paths_by_mid.get(mid, [[mid]])
        display_prompt = clean_label(display_name)
        sam3_prompt = choose_sam3_prompt(display_name, paths, labels)
        parents = sorted(parent_map.get(mid, set()), key=lambda x: labels.get(x, x))
        children = sorted(child_map.get(mid, set()), key=lambda x: labels.get(x, x))

        candidates = []
        for item in [sam3_prompt, display_prompt, *parent_prompts(paths, labels)]:
            if item and item not in candidates:
                candidates.append(item)

        records.append(
            {
                "source": "openimages_v7_boxable",
                "mid": mid,
                "display_name": display_name,
                "display_prompt_text": display_prompt,
                "sam3_prompt_text": sam3_prompt,
                "prompt_candidates": candidates,
                "parent_mids": parents,
                "parent_names": [labels.get(x, x) for x in parents],
                "child_mids": children,
                "child_names": [labels.get(x, x) for x in children],
                "is_leaf": len(children) == 0,
                "paths": [
                    {
                        "mids": path,
                        "names": path_names(path, labels),
                        "prompt_names": [clean_label(x) for x in path_names(path, labels)],
                    }
                    for path in paths
                ],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", default=DEFAULT_LABELS)
    parser.add_argument("--hierarchy-json", default=DEFAULT_HIERARCHY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    labels = load_labels(Path(args.labels_csv))
    hierarchy = json.loads(Path(args.hierarchy_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(labels, hierarchy)

    categories_path = output_dir / "categories.jsonl"
    with categories_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    coarse_to_fine = {
        record["mid"]: {
            "display_name": record["display_name"],
            "sam3_prompt_text": record["sam3_prompt_text"],
            "children": record["child_mids"],
            "child_names": record["child_names"],
            "parents": record["parent_mids"],
            "parent_names": record["parent_names"],
        }
        for record in records
    }
    (output_dir / "coarse_to_fine.json").write_text(
        json.dumps(coarse_to_fine, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    changed = [r for r in records if r["sam3_prompt_text"] != r["display_prompt_text"]]
    summary = {
        "labels_csv": args.labels_csv,
        "hierarchy_json": args.hierarchy_json,
        "num_categories": len(records),
        "num_leaf_categories": sum(1 for r in records if r["is_leaf"]),
        "num_prompt_normalized": len(changed),
        "sample_prompt_normalizations": [
            {
                "display_name": r["display_name"],
                "display_prompt_text": r["display_prompt_text"],
                "sam3_prompt_text": r["sam3_prompt_text"],
                "parents": r["parent_names"],
            }
            for r in changed[:80]
        ],
        "outputs": {
            "categories_jsonl": str(categories_path),
            "coarse_to_fine_json": str(output_dir / "coarse_to_fine.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
