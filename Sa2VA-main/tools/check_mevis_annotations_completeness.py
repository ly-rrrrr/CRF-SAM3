#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_meta(meta_path: Path) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "videos" in meta and isinstance(meta["videos"], dict):
        return meta["videos"]
    if isinstance(meta, dict):
        return meta
    raise ValueError(f"Unsupported meta_expressions format: {meta_path}")


def expected_frame_names(frames: List[str]) -> List[str]:
    return [Path(frame).stem + ".png" for frame in frames]


def scan_missing(pred_root: Path, videos: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    missing_items: List[Dict[str, Any]] = []

    total_expr = 0
    total_expected_png = 0
    total_found_png = 0
    missing_exp_dir_count = 0

    for video_id, video_info in videos.items():
        frames = video_info["frames"]
        expressions = video_info["expressions"]
        expected_pngs = expected_frame_names(frames)

        for exp_id in expressions.keys():
            exp_id = str(exp_id)
            total_expr += 1
            exp_dir = pred_root / video_id / exp_id

            if not exp_dir.is_dir():
                missing_exp_dir_count += 1

            missing_pngs = [name for name in expected_pngs if not (exp_dir / name).is_file()]
            found_pngs = len(expected_pngs) - len(missing_pngs)

            total_expected_png += len(expected_pngs)
            total_found_png += found_pngs

            if missing_pngs:
                missing_items.append(
                    {
                        "video_id": video_id,
                        "exp_id": exp_id,
                        "missing_count": len(missing_pngs),
                        "expected_count": len(expected_pngs),
                        "found_count": found_pngs,
                        "sample_missing": missing_pngs[:10],
                        "exp_dir_exists": exp_dir.is_dir(),
                    }
                )

    summary = {
        "pred_root": str(pred_root),
        "total_expressions": total_expr,
        "total_expected_png": total_expected_png,
        "total_found_png": total_found_png,
        "total_missing_png": total_expected_png - total_found_png,
        "expressions_with_missing": len(missing_items),
        "missing_expression_dirs": missing_exp_dir_count,
    }
    return summary, missing_items


def scan_extras(pred_root: Path, videos: Dict[str, Any]) -> List[Dict[str, Any]]:
    extras: List[Dict[str, Any]] = []

    for video_id, video_info in videos.items():
        expected_frames = set(expected_frame_names(video_info["frames"]))
        expected_exp_ids = {str(k) for k in video_info["expressions"].keys()}
        video_dir = pred_root / video_id

        if not video_dir.is_dir():
            continue

        actual_exp_ids = {p.name for p in video_dir.iterdir() if p.is_dir()}
        extra_exp_ids = sorted(actual_exp_ids - expected_exp_ids)
        if extra_exp_ids:
            extras.append(
                {
                    "video_id": video_id,
                    "extra_exp_ids": extra_exp_ids,
                }
            )

        for exp_id in sorted(actual_exp_ids & expected_exp_ids):
            exp_dir = video_dir / exp_id
            actual_pngs = {p.name for p in exp_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
            extra_pngs = sorted(actual_pngs - expected_frames)
            if extra_pngs:
                extras.append(
                    {
                        "video_id": video_id,
                        "exp_id": exp_id,
                        "extra_png_count": len(extra_pngs),
                        "sample_extra_pngs": extra_pngs[:10],
                    }
                )

    return extras


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether a MEVIS-style Annotations directory has missing or extra PNG files."
    )
    parser.add_argument(
        "--pred-root",
        required=True,
        help="Prediction root, typically .../Annotations",
    )
    parser.add_argument(
        "--meta-path",
        required=True,
        help="Path to meta_expressions.json",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to save the full report JSON",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=50,
        help="How many missing/extra items to print to stdout",
    )
    parser.add_argument(
        "--check-extra",
        action="store_true",
        help="Also check for extra exp_id directories and extra PNG files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    pred_root = Path(args.pred_root).expanduser().resolve()
    meta_path = Path(args.meta_path).expanduser().resolve()

    videos = load_meta(meta_path)
    summary, missing_items = scan_missing(pred_root, videos)
    extras = scan_extras(pred_root, videos) if args.check_extra else []

    report: Dict[str, Any] = {
        "summary": summary,
        "missing_items": missing_items,
        "extra_items": extras,
    }

    print("=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if missing_items:
        print(f"\n=== Missing Items (first {min(args.show_limit, len(missing_items))}) ===")
        for item in missing_items[: args.show_limit]:
            print(json.dumps(item, ensure_ascii=False))
    else:
        print("\nNo missing files detected.")

    if args.check_extra:
        if extras:
            print(f"\n=== Extra Items (first {min(args.show_limit, len(extras))}) ===")
            for item in extras[: args.show_limit]:
                print(json.dumps(item, ensure_ascii=False))
        else:
            print("\nNo extra files detected.")

    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
