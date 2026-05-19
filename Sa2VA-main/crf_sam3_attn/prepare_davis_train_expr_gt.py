import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
from tqdm import tqdm


def _to_obj_ids(v) -> List[int]:
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(v)]


def _is_exp_complete(exp_dir: Path, frame_names: Iterable[str]) -> bool:
    if not exp_dir.is_dir():
        return False
    for fr in frame_names:
        if not (exp_dir / f"{fr}.png").exists():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DAVIS train expression-level GT masks for CRF-SAM3 attn training."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/davis17",
        help="DAVIS17 root path",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid"])
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root for expression-level GT masks. "
        "Default: <data_root>/gt_masks_<split>_expr",
    )
    parser.add_argument(
        "--copy_meta_to_split",
        action="store_true",
        help="Copy meta_expressions/<split>/meta_expressions.json to <data_root>/<split>/meta_expressions.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing expression directories",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    split = args.split

    meta_src = data_root / "meta_expressions" / split / "meta_expressions.json"
    if not meta_src.exists():
        raise FileNotFoundError(f"meta file not found: {meta_src}")

    if args.copy_meta_to_split:
        split_meta = data_root / split / "meta_expressions.json"
        split_meta.parent.mkdir(parents=True, exist_ok=True)
        split_meta.write_bytes(meta_src.read_bytes())
        print(f"[meta] copied to {split_meta}")

    output_root = Path(args.output_root).resolve() if args.output_root else (data_root / f"gt_masks_{split}_expr")
    output_root.mkdir(parents=True, exist_ok=True)

    with open(meta_src, "r", encoding="utf-8") as f:
        meta = json.load(f)
    videos = meta.get("videos", {})

    total_exp = 0
    total_saved = 0
    total_skipped = 0

    for video_id, video_info in tqdm(videos.items(), desc=f"DAVIS-{split} videos"):
        frame_names = [str(x) for x in video_info.get("frames", [])]
        anno_dir = data_root / split / "Annotations" / video_id
        if not anno_dir.exists():
            continue

        expressions = video_info.get("expressions", {})
        for exp_id, exp_info in expressions.items():
            total_exp += 1
            exp_dir = output_root / video_id / str(exp_id)

            if exp_dir.exists() and not args.overwrite:
                if _is_exp_complete(exp_dir, frame_names):
                    total_skipped += 1
                    continue

            exp_dir.mkdir(parents=True, exist_ok=True)
            obj_ids = _to_obj_ids(exp_info.get("obj_id"))

            saved = 0
            for frame_name in frame_names:
                anno_path = anno_dir / f"{frame_name}.png"
                if not anno_path.exists():
                    continue
                anno = np.array(Image.open(anno_path))
                mask = np.isin(anno, obj_ids).astype(np.uint8) * 255
                Image.fromarray(mask).save(exp_dir / f"{frame_name}.png")
                saved += 1

            if saved > 0:
                total_saved += 1

    print(f"[done] expressions total={total_exp}, saved={total_saved}, skipped={total_skipped}")
    print(f"[done] gt root: {output_root}")


if __name__ == "__main__":
    main()

