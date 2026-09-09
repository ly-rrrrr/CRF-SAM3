import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAW_FRAMES_PER_SAMPLE = 8
EVAL_TOTAL = 200
TRAIN_TOTAL = 200

DATASET_SPECS = {
    "mevis": {
        "eval": {
            "meta": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/mevis/valid_u/meta_expressions.json",
            "frame_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/mevis/valid_u/JPEGImages",
            "gt_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/mevis/valid_u/Annotations",
            "output_root": "/mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_mevis_valid_u_joint_train_poswin_mergedprompt",
        },
        "train": {
            "meta": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/mevis/train/meta_expressions.json",
            "frame_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/mevis/train/JPEGImages",
            "gt_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/mevis/train/Annotations",
        },
    },
    "davis17": {
        "eval": {
            "meta": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/davis17/meta_expressions/valid/meta_expressions.json",
            "frame_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/davis17/valid/JPEGImages",
            "gt_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/davis17/valid/Annotations",
            "output_root": "/mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_davis17_valid_joint_train_poswin_mergedprompt",
        },
        "train": {
            "meta": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/davis17/meta_expressions/train/meta_expressions.json",
            "frame_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/davis17/train/JPEGImages",
            "gt_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/davis17/train/Annotations",
        },
    },
    "refytros": {
        "eval": {
            "meta": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/valid/meta_expressions_challenge.json",
            "frame_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/valid/JPEGImages",
            "gt_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/valid/Annotations",
            "output_root": "/mnt/csip-500/share/kongjiawei/Sa2VA-main/crf_sam3_refytros_valid_joint_train_poswin_mergedprompt",
        },
        "train": {
            "meta": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/train/meta_expressions.json",
            "frame_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/train/JPEGImages",
            "gt_root": "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/Ref-Youtube-VOS/train/Annotations",
        },
    },
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    safe_mkdir(dst.parent)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    safe_mkdir(dst.parent)
    shutil.copytree(src, dst)


def load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("videos", data)


def build_meta_items(meta_path: Path) -> List[Dict]:
    videos = load_meta(meta_path)
    items = []
    for video_id, video_info in videos.items():
        expressions = video_info.get("expressions", {})
        for exp_id, exp_info in expressions.items():
            items.append(
                {
                    "video_id": str(video_id),
                    "exp_id": str(exp_id),
                    "expression": exp_info.get("exp") or exp_info.get("expression") or "",
                }
            )
    return items


def build_eval_items(output_root: Path) -> List[Dict]:
    details_root = output_root / "expression_details"
    items = []
    if not details_root.exists():
        return items
    for video_dir in sorted(p for p in details_root.iterdir() if p.is_dir()):
        for json_path in sorted(video_dir.glob("*.json"), key=lambda p: p.stem):
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items.append(
                {
                    "video_id": str(data.get("video_id", video_dir.name)),
                    "exp_id": str(data.get("exp_id", json_path.stem)),
                    "expression": data.get("expression", ""),
                }
            )
    return items


def stable_video_order(video_ids: List[str]) -> List[str]:
    return sorted(video_ids)


def spread_select(items: List[Dict], target_count: int) -> List[Dict]:
    by_video: Dict[str, List[Dict]] = {}
    for item in items:
        by_video.setdefault(item["video_id"], []).append(item)
    for lst in by_video.values():
        lst.sort(key=lambda x: (str(x["exp_id"])))
    order = stable_video_order(list(by_video.keys()))
    selected = []
    round_idx = 0
    while len(selected) < target_count:
        added = False
        for vid in order:
            lst = by_video[vid]
            if round_idx < len(lst):
                selected.append(lst[round_idx])
                added = True
                if len(selected) >= target_count:
                    break
        if not added:
            break
        round_idx += 1
    return selected


def evenly_spaced_names(frame_dir: Path, limit: int) -> List[str]:
    names = sorted(p.name for p in frame_dir.iterdir() if p.is_file())
    if len(names) <= limit:
        return names
    idxs = []
    for i in range(limit):
        pos = round(i * (len(names) - 1) / (limit - 1))
        idxs.append(pos)
    out = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            out.append(names[idx])
            seen.add(idx)
    return out


def copy_sampled_frames(frame_dir: Path, dst_dir: Path, limit: int) -> List[str]:
    if not frame_dir.exists():
        return []
    names = evenly_spaced_names(frame_dir, limit)
    safe_mkdir(dst_dir)
    for name in names:
        copy_file(frame_dir / name, dst_dir / name)
    return names


def copy_matching_gt(gt_dir: Path, dst_dir: Path, frame_names: List[str]) -> List[str]:
    if not gt_dir.exists():
        return []
    copied = []
    safe_mkdir(dst_dir)
    for frame_name in frame_names:
        stem = Path(frame_name).stem
        candidates = [gt_dir / frame_name, gt_dir / f"{stem}.png", gt_dir / f"{stem}.jpg"]
        for src in candidates:
            if src.exists():
                copy_file(src, dst_dir / src.name)
                copied.append(src.name)
                break
    return copied


def copy_output_common_files(output_root: Path, dst_root: Path) -> List[str]:
    copied = []
    for name in sorted(os.listdir(output_root)):
        src = output_root / name
        if src.is_file() and src.suffix.lower() in {".json", ".yaml", ".yml", ".txt"}:
            copy_file(src, dst_root / name)
            copied.append(name)
    return copied


def allocate_counts(total: int, dataset_keys: List[str]) -> Dict[str, int]:
    base = total // len(dataset_keys)
    rem = total % len(dataset_keys)
    out = {}
    for i, key in enumerate(dataset_keys):
        out[key] = base + (1 if i < rem else 0)
    return out


def build_package(out_dir: Path) -> Dict:
    dataset_order = ["mevis", "davis17", "refytros"]
    eval_counts = allocate_counts(EVAL_TOTAL, dataset_order)
    train_counts = allocate_counts(TRAIN_TOTAL, dataset_order)
    manifest = {
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "output_dir": str(out_dir),
        "raw_frames_per_sample": RAW_FRAMES_PER_SAMPLE,
        "eval_target_total": EVAL_TOTAL,
        "train_target_total": TRAIN_TOTAL,
        "datasets": {},
    }

    for dataset in dataset_order:
        manifest["datasets"][dataset] = {"eval": {}, "train": {}}
        spec_eval = DATASET_SPECS[dataset]["eval"]
        spec_train = DATASET_SPECS[dataset]["train"]

        eval_items = build_eval_items(Path(spec_eval["output_root"]))
        selected_eval = spread_select(eval_items, eval_counts[dataset])
        eval_dst = out_dir / dataset / "eval"
        raw_eval_root = eval_dst / "raw"
        results_eval_root = eval_dst / "results"
        safe_mkdir(raw_eval_root)
        safe_mkdir(results_eval_root)

        copied_common = copy_output_common_files(Path(spec_eval["output_root"]), results_eval_root)
        eval_entries = []
        copied_raw_videos = set()
        copied_gt_videos = set()
        for item in selected_eval:
            video_id = item["video_id"]
            exp_id = item["exp_id"]
            frame_dir = Path(spec_eval["frame_root"]) / video_id
            raw_frames_dst = raw_eval_root / "JPEGImages" / video_id
            frame_names = []
            if video_id not in copied_raw_videos:
                frame_names = copy_sampled_frames(frame_dir, raw_frames_dst, RAW_FRAMES_PER_SAMPLE)
                copied_raw_videos.add(video_id)
            else:
                frame_names = sorted(p.name for p in raw_frames_dst.iterdir()) if raw_frames_dst.exists() else []

            gt_copied = []
            gt_root = Path(spec_eval["gt_root"])
            gt_video_dir = gt_root / video_id
            raw_gt_dst = raw_eval_root / "Annotations" / video_id
            if gt_video_dir.exists() and video_id not in copied_gt_videos:
                gt_copied = copy_matching_gt(gt_video_dir, raw_gt_dst, frame_names)
                copied_gt_videos.add(video_id)
            elif raw_gt_dst.exists():
                gt_copied = sorted(p.name for p in raw_gt_dst.iterdir())

            output_root = Path(spec_eval["output_root"])
            copy_tree(output_root / "Annotations" / video_id / exp_id, results_eval_root / "Annotations" / video_id / exp_id)
            copy_tree(output_root / "visual_tubes" / video_id / exp_id, results_eval_root / "visual_tubes" / video_id / exp_id)
            copy_file(output_root / "expression_details" / video_id / f"{exp_id}.json", results_eval_root / "expression_details" / video_id / f"{exp_id}.json")

            eval_entries.append(
                {
                    **item,
                    "sampled_raw_frames": frame_names,
                    "sampled_gt_frames": gt_copied,
                }
            )

        subset_meta = {
            "dataset": dataset,
            "split": "eval",
            "source_meta": spec_eval["meta"],
            "source_output_root": spec_eval["output_root"],
            "selected_expressions": eval_entries,
            "copied_output_common_files": copied_common,
        }
        safe_mkdir(eval_dst)
        with (eval_dst / "subset_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(subset_meta, f, ensure_ascii=False, indent=2)
        manifest["datasets"][dataset]["eval"] = {
            "requested_count": eval_counts[dataset],
            "selected_count": len(eval_entries),
            "unique_video_count": len({x['video_id'] for x in eval_entries}),
            "subset_manifest": str(eval_dst / "subset_manifest.json"),
        }

        train_items = build_meta_items(Path(spec_train["meta"]))
        selected_train = spread_select(train_items, train_counts[dataset])
        train_dst = out_dir / dataset / "train"
        raw_train_root = train_dst / "raw"
        safe_mkdir(raw_train_root)
        train_entries = []
        copied_train_raw = set()
        copied_train_gt = set()
        for item in selected_train:
            video_id = item["video_id"]
            frame_dir = Path(spec_train["frame_root"]) / video_id
            raw_frames_dst = raw_train_root / "JPEGImages" / video_id
            frame_names = []
            if video_id not in copied_train_raw:
                frame_names = copy_sampled_frames(frame_dir, raw_frames_dst, RAW_FRAMES_PER_SAMPLE)
                copied_train_raw.add(video_id)
            else:
                frame_names = sorted(p.name for p in raw_frames_dst.iterdir()) if raw_frames_dst.exists() else []

            gt_copied = []
            gt_root = Path(spec_train["gt_root"])
            gt_video_dir = gt_root / video_id
            raw_gt_dst = raw_train_root / "Annotations" / video_id
            if gt_video_dir.exists() and video_id not in copied_train_gt:
                gt_copied = copy_matching_gt(gt_video_dir, raw_gt_dst, frame_names)
                copied_train_gt.add(video_id)
            elif raw_gt_dst.exists():
                gt_copied = sorted(p.name for p in raw_gt_dst.iterdir())

            train_entries.append(
                {
                    **item,
                    "sampled_raw_frames": frame_names,
                    "sampled_gt_frames": gt_copied,
                }
            )

        subset_meta = {
            "dataset": dataset,
            "split": "train",
            "source_meta": spec_train["meta"],
            "selected_expressions": train_entries,
        }
        with (train_dst / "subset_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(subset_meta, f, ensure_ascii=False, indent=2)
        manifest["datasets"][dataset]["train"] = {
            "requested_count": train_counts[dataset],
            "selected_count": len(train_entries),
            "unique_video_count": len({x['video_id'] for x in train_entries}),
            "subset_manifest": str(train_dst / "subset_manifest.json"),
        }

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/mnt/csip-500/share/kongjiawei/Sa2VA-main/transfer_subset_crf_sam3_attn_20260518",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    safe_mkdir(out_dir)
    manifest = build_package(out_dir)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
