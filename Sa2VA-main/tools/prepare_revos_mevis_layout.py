#!/usr/bin/env python3
"""Build a MeViS-like lightweight view for ReVOS.

The CRF-SAM3 offline dataset loader expects:
  <data_root>/<split>/meta_expressions.json
  <data_root>/<split>/JPEGImages/<video_id>/*.jpg

ReVOS stores frames directly under <source_root>/<video_id>, where video_id
already contains dataset subdirectories such as UVO/all/xxx.  This script keeps
the original files in place and creates only metadata copies plus symlinks.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_SOURCE_ROOT = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos"
DEFAULT_OUTPUT_ROOT = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos_prepared"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_symlink(target: Path, link: Path, overwrite: bool = False) -> Tuple[bool, str]:
    if not target.exists():
        return False, "missing_target"

    if link.is_symlink():
        current = Path(os.readlink(link))
        if not current.is_absolute():
            current = (link.parent / current).resolve()
        if current == target.resolve():
            return False, "exists"
        if not overwrite:
            return False, "conflict"
        link.unlink()
    elif link.exists():
        return False, "conflict"

    link.parent.mkdir(parents=True, exist_ok=True)
    rel_target = os.path.relpath(target, start=link.parent)
    os.symlink(rel_target, link)
    return True, "created"


def prepare_split(
    *,
    source_root: Path,
    output_root: Path,
    split: str,
    meta_file: str,
    overwrite_links: bool,
) -> Dict:
    source_meta_path = source_root / meta_file
    if not source_meta_path.exists():
        raise FileNotFoundError(f"Missing ReVOS meta file: {source_meta_path}")

    split_root = output_root / split
    jpeg_root = split_root / "JPEGImages"
    split_root.mkdir(parents=True, exist_ok=True)
    jpeg_root.mkdir(parents=True, exist_ok=True)

    meta = _load_json(source_meta_path)
    _write_json(split_root / "meta_expressions.json", meta)

    stats = {
        "split": split,
        "source_meta": str(source_meta_path),
        "meta": str(split_root / "meta_expressions.json"),
        "videos": len(meta.get("videos", {})),
        "expressions": sum(
            len(video_meta.get("expressions", {}))
            for video_meta in meta.get("videos", {}).values()
        ),
        "frame_links_created": 0,
        "frame_links_existing": 0,
        "missing_frame_dirs": 0,
        "link_conflicts": 0,
    }

    for video_id in meta.get("videos", {}).keys():
        created, status = _safe_symlink(
            target=source_root / video_id,
            link=jpeg_root / video_id,
            overwrite=overwrite_links,
        )
        if created:
            stats["frame_links_created"] += 1
        elif status == "exists":
            stats["frame_links_existing"] += 1
        elif status == "missing_target":
            stats["missing_frame_dirs"] += 1
        elif status == "conflict":
            stats["link_conflicts"] += 1

    for name in ("mask_dict.json", "mask_dict_foreground.json"):
        target = source_root / name
        link = split_root / name
        created, status = _safe_symlink(target=target, link=link, overwrite=overwrite_links)
        stats[f"{name}_status"] = "created" if created else status

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ReVOS as a MeViS-like symlink layout.")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-meta", default="meta_expressions_train_.json")
    parser.add_argument("--valid-meta", default="meta_expressions_valid_.json")
    parser.add_argument("--splits", default="train,valid", help="Comma-separated subset of: train,valid")
    parser.add_argument("--overwrite-links", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    split_to_meta = {
        "train": args.train_meta,
        "valid": args.valid_meta,
    }

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "layout": "<output_root>/<split>/JPEGImages/<video_id> -> <source_root>/<video_id>",
        "splits": {},
    }

    requested = [x.strip() for x in args.splits.split(",") if x.strip()]
    for split in requested:
        if split not in split_to_meta:
            raise ValueError(f"Unsupported split={split!r}; expected one of {sorted(split_to_meta)}")
        stats = prepare_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            meta_file=split_to_meta[split],
            overwrite_links=bool(args.overwrite_links),
        )
        manifest["splits"][split] = stats
        print(
            f"[{split}] videos={stats['videos']} expressions={stats['expressions']} "
            f"created={stats['frame_links_created']} existing={stats['frame_links_existing']} "
            f"missing={stats['missing_frame_dirs']} conflicts={stats['link_conflicts']}"
        )

    _write_json(output_root / "manifest.json", manifest)
    print(f"[manifest] {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
