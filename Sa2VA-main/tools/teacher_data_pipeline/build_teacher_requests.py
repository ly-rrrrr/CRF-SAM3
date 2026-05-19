#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


DEFAULT_DATA_ROOT = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas"
DEFAULT_OUTPUT = "data/teacher/teacher_requests_train.jsonl"
DEFAULT_PROMPT = "tools/teacher_data_pipeline/teacher_extract_prompt.txt"
FIXED_CATEGORY_STAGE_NUM_FRAMES = 3


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}




def sample_category_stage_frames(frame_paths: Sequence[Path], num_frames: int) -> List[Path]:
    """
    Unified frame sampler for the category-extraction stage.

    Teacher data generation must use the same temporal sampling policy as the
    current category extractor. Keep all category-stage frame sampling here.
    """
    if num_frames <= 0:
        raise ValueError("resolved category-stage frame count must be > 0")
    if not frame_paths:
        return []

    frames = list(frame_paths)
    n = len(frames)
    if n <= num_frames:
        return frames

    indices: List[int] = []
    for i in range(num_frames):
        idx = round(i * (n - 1) / (num_frames - 1))
        indices.append(idx)

    dedup_indices: List[int] = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            dedup_indices.append(idx)
            seen.add(idx)

    if len(dedup_indices) < num_frames:
        for idx in range(n):
            if idx not in seen:
                dedup_indices.append(idx)
                seen.add(idx)
            if len(dedup_indices) == num_frames:
                break

    dedup_indices = sorted(dedup_indices[:num_frames])
    return [frames[i] for i in dedup_indices]


def _extract_expression_text(exp_info) -> Optional[str]:
    if isinstance(exp_info, str):
        text = exp_info.strip()
        return text or None
    if isinstance(exp_info, dict):
        for key in ("exp", "expression", "text", "sentence", "sent"):
            value = exp_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _iter_video_expressions(video_info) -> Iterable[Tuple[str, str]]:
    if not isinstance(video_info, dict):
        return []
    expressions = (
        video_info.get("expressions")
        or video_info.get("sentences")
        or video_info.get("refs")
        or {}
    )
    if isinstance(expressions, dict):
        for exp_id, exp_info in expressions.items():
            text = _extract_expression_text(exp_info)
            if text:
                yield str(exp_id), text
    elif isinstance(expressions, list):
        for idx, exp_info in enumerate(expressions):
            text = _extract_expression_text(exp_info)
            if text:
                yield str(idx), text


def iter_meta_expressions(meta: Dict) -> Iterable[Tuple[str, str, str]]:
    videos = meta.get("videos")
    if isinstance(videos, dict):
        for video_id, video_info in videos.items():
            for exp_id, expression in _iter_video_expressions(video_info):
                yield str(video_id), str(exp_id), expression
        return

    for key in ("expressions", "refs", "items"):
        items = meta.get(key)
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            video_id = (
                item.get("video")
                or item.get("video_id")
                or item.get("vid")
                or item.get("video_name")
            )
            expression = _extract_expression_text(item)
            if video_id is None or not expression:
                continue
            exp_id = item.get("exp_id", item.get("expression_id", idx))
            yield str(video_id), str(exp_id), expression
        return

    raise ValueError("Unsupported metadata structure: cannot find expression entries")


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sorted_frames(frame_dir: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames: List[Path] = []
    for pattern in exts:
        frames.extend(frame_dir.glob(pattern))
    frames = sorted({p.resolve() for p in frames})
    return frames


class DatasetResolver:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def resolve(self, dataset_key: str, split: str) -> Tuple[Path, Path, str]:
        key = dataset_key.lower()
        if key in {"davis", "davis17"}:
            dataset_root = self.data_root / "davis17"
            meta_path = dataset_root / "meta_expressions" / split / "meta_expressions.json"
            return dataset_root, meta_path, "davis17"

        if key in {"mevis"}:
            dataset_root = self.data_root / "mevis"
            meta_path = dataset_root / split / "meta_expressions.json"
            return dataset_root, meta_path, "mevis"

        if key in {"refytvos", "ref-youtube-vos", "ref_youtube_vos"}:
            candidates = [
                self.data_root / "Ref-Youtube-VOS",
                self.data_root / "ref-youtube-vos",
                self.data_root / "ref_youtube_vos",
                self.data_root / "ref-youtube-vos-21" / "21",
            ]
            dataset_root = next((p for p in candidates if p.exists()), None)
            if dataset_root is None:
                raise FileNotFoundError(
                    "Cannot find Ref-Youtube-VOS dataset root under data_root. "
                    f"Looked for: {', '.join(str(p) for p in candidates)}"
                )

            meta_candidates = [
                dataset_root / f"meta_expressions_{split}_.json",
                dataset_root / f"meta_expressions_{split}.json",
                dataset_root / split / "meta_expressions.json",
                dataset_root / "meta_expressions" / split / "meta_expressions.json",
            ]
            meta_path = next((p for p in meta_candidates if p.exists()), None)
            if meta_path is None:
                raise FileNotFoundError(
                    "Cannot find Ref-Youtube-VOS metadata file. "
                    f"Tried: {', '.join(str(p) for p in meta_candidates)}"
                )
            return dataset_root, meta_path, "ref-youtube-vos"

        raise ValueError(f"Unsupported dataset key: {dataset_key}")

    def find_frame_dir(self, canonical_name: str, dataset_root: Path, split: str, video_id: str) -> Path:
        if canonical_name == "davis17":
            candidates = [
                dataset_root / split / "JPEGImages" / video_id,
                dataset_root / "JPEGImages" / video_id,
            ]
        elif canonical_name == "mevis":
            candidates = [
                dataset_root / split / "JPEGImages" / video_id,
                dataset_root / "JPEGImages" / video_id,
            ]
        elif canonical_name == "ref-youtube-vos":
            candidates = [
                dataset_root / video_id,
                dataset_root / "JPEGImages" / video_id,
                dataset_root / split / video_id,
                dataset_root / split / "JPEGImages" / video_id,
            ]
        else:
            raise ValueError(f"Unknown canonical dataset name: {canonical_name}")

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        raise FileNotFoundError(
            f"Cannot find frame directory for dataset={canonical_name} split={split} video_id={video_id}. "
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )


def make_user_prompt(expression: str) -> str:
    return (
        f"Expression: {expression}\n"
        "Return one JSON object only."
    )


def build_response_schema() -> Dict:
    return {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "masked_expression": {"type": "string"},
            "category": {"type": "string"},
            "backup_categories": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 2,
            },
        },
        "required": [
            "expression",
            "masked_expression",
            "category",
            "backup_categories",
        ],
        "additionalProperties": False,
    }


def parse_spec(spec_items: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for item in spec_items:
        if ":" not in item:
            raise ValueError(f"Invalid spec '{item}'. Expected dataset:split.")
        dataset, split = item.split(":", 1)
        parsed.append((dataset.strip(), split.strip()))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build teacher labeling requests for local Gemma inference.")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Dataset root, e.g. /mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas",
    )
    parser.add_argument(
        "--category-config",
        default="configs/crf_sam3_config.yaml",
        help="Retained for compatibility. Teacher requests currently use a fixed 3-frame category-stage policy.",
    )
    parser.add_argument(
        "--spec",
        nargs="+",
        default=["davis17:train", "mevis:train", "refytvos:train"],
        help="Dataset specs in dataset:split form. This pipeline is intended for train splits only.",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT,
        help="Teacher system prompt text file.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on total samples written.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    prompt_path = Path(args.prompt_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_prompt(prompt_path)
    _ = load_yaml(Path(args.category_config))
    frames_per_sample = FIXED_CATEGORY_STAGE_NUM_FRAMES

    resolver = DatasetResolver(data_root)
    response_schema = build_response_schema()

    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for dataset_key, split in parse_spec(args.spec):
            dataset_root, meta_path, canonical_name = resolver.resolve(dataset_key, split)
            meta = read_json(meta_path)
            for video_id, exp_id, expression in iter_meta_expressions(meta):
                if args.limit and written >= args.limit:
                    break

                frame_dir = resolver.find_frame_dir(canonical_name, dataset_root, split, video_id)
                frames = sorted_frames(frame_dir)
                if not frames:
                    continue

                selected_frames = sample_category_stage_frames(frames, frames_per_sample)
                sample_id = f"{canonical_name}:{split}:{video_id}:{exp_id}"

                row = {
                    "sample_id": sample_id,
                    "source": {
                        "dataset": canonical_name,
                        "split": split,
                        "video_id": video_id,
                        "exp_id": exp_id,
                        "expression": expression,
                        "frame_dir": str(frame_dir),
                        "frame_names": [p.name for p in selected_frames],
                        "frame_paths": [str(p) for p in selected_frames],
                    },
                    "teacher_request": {
                        "task": "masked_referent_category_extraction",
                        "system_prompt": system_prompt,
                        "user_prompt": make_user_prompt(expression),
                        "response_schema": response_schema,
                        "images": [str(p) for p in selected_frames],
                    },
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            if args.limit and written >= args.limit:
                break

    print(
        json.dumps(
            {
                "output": str(output_path),
                "count": written,
                "frames_per_sample": frames_per_sample,
                "category_config": args.category_config,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
