#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 video text-prompt inference on an MP4 or JPEG-frame folder."
    )
    parser.add_argument("--video", required=True, help="Path to input MP4 video or JPEG-frame directory.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to SAM3 checkpoint file or checkpoint directory containing sam3.pt.",
    )
    parser.add_argument(
        "--sam3-root",
        default="/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main",
        help="Path to the local SAM3 repository root.",
    )
    parser.add_argument("--text", default="zebra", help="Text prompt passed to SAM3.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index used to add the text prompt.")
    parser.add_argument(
        "--output-dir",
        "--output-path",
        dest="output_dir",
        default="",
        help="Directory for outputs. Defaults to demo/<video_stem>_sam3_<text>/",
    )
    parser.add_argument(
        "--save-mask-pngs",
        action="store_true",
        help="Save one union-mask PNG per frame.",
    )
    parser.add_argument(
        "--save-mask-video",
        action="store_true",
        help="Save a binary union-mask MP4.",
    )
    parser.add_argument(
        "--save-overlay-video",
        action="store_true",
        help="Save an RGB overlay MP4 with the union mask painted in red.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Optional CUDA device index. If set, the script exports CUDA_VISIBLE_DEVICES before importing SAM3.",
    )
    return parser.parse_args()


def resolve_checkpoint(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        candidate = path / "sam3.pt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Checkpoint directory does not contain sam3.pt: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    return path


def ensure_import_path(sam3_root: Path) -> None:
    sam3_root = sam3_root.expanduser().resolve()
    if not sam3_root.exists():
        raise FileNotFoundError(f"sam3 root does not exist: {sam3_root}")
    sam3_root_str = str(sam3_root)
    if sam3_root_str not in sys.path:
        sys.path.insert(0, sam3_root_str)


def default_output_dir(video_path: Path, text_prompt: str) -> Path:
    safe_text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text_prompt.strip())
    safe_text = safe_text or "prompt"
    return video_path.parent / f"{video_path.stem}_sam3_{safe_text}"


def list_frame_files(frame_root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frame_files = [p for p in frame_root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    frame_files.sort()
    return frame_files


def inspect_frame_source(video_path: Path) -> Tuple[str, List[Path]]:
    if video_path.is_dir():
        frame_files = list_frame_files(video_path)
        if not frame_files:
            raise FileNotFoundError(f"No image frames found under directory: {video_path}")
        return "frames_dir", frame_files
    return "video_file", []


def to_numpy_bool(mask: Any) -> "numpy.ndarray":
    import numpy as np

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    return mask.astype(bool)


def to_python_list(values: Any) -> List[Any]:
    import numpy as np

    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def save_union_png(mask: "numpy.ndarray", path: Path) -> None:
    import cv2
    import numpy as np

    img = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(path), img)


def overlay_frame(frame_bgr: "numpy.ndarray", union_mask: "numpy.ndarray") -> "numpy.ndarray":
    import numpy as np

    union_mask = np.asarray(union_mask).astype(bool)
    if union_mask.ndim > 2:
        union_mask = union_mask.squeeze()
    if union_mask.size == 0 or not union_mask.any():
        return frame_bgr.copy()

    overlay = frame_bgr.copy()
    color = np.zeros_like(frame_bgr, dtype=np.float32)
    color[..., 2] = 255.0
    alpha = 0.45

    blended = (
        frame_bgr.astype(np.float32) * (1.0 - alpha)
        + color * alpha
    ).clip(0, 255).astype(frame_bgr.dtype)
    overlay[union_mask] = blended[union_mask]
    return overlay


def infer_frame_source_stats(
    source_kind: str,
    video_path: Path,
    frame_files: List[Path],
) -> Tuple[float, int, int, int]:
    import cv2
    import numpy as np

    if source_kind == "video_file":
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for visualization: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return float(fps), width, height, total_frames

    first_frame = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame image: {frame_files[0]}")
    height, width = first_frame.shape[:2]
    total_frames = len(frame_files)
    return 25.0, int(width), int(height), int(total_frames)


def main() -> None:
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    source_kind, frame_files = inspect_frame_source(video_path)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    sam3_root = Path(args.sam3_root)
    ensure_import_path(sam3_root)

    import cv2
    import numpy as np
    import torch
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(video_path, args.text)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_png_dir = output_dir / "union_masks"
    if args.save_mask_pngs:
        mask_png_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "video": str(video_path),
        "source_kind": source_kind,
        "checkpoint": str(checkpoint_path),
        "sam3_root": str(sam3_root.resolve()),
        "text": args.text,
        "frame_index": args.frame_index,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))

    predictor = Sam3VideoPredictor(
        checkpoint_path=str(checkpoint_path),
        has_presence_token=True,
        apply_temporal_disambiguation=True,
    )

    session = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(video_path),
        }
    )
    session_id = session["session_id"]
    print(f"[SAM3] session started: {session_id}")

    add_prompt_response = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": args.frame_index,
            "text": args.text,
        }
    )
    initial_outputs = add_prompt_response.get("outputs", {})

    frame_outputs: Dict[int, Dict[str, Any]] = {}
    all_obj_ids = set()

    try:
        for response in predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
            }
        ):
            frame_idx = int(response["frame_index"])
            outputs = response.get("outputs", {}) or {}
            obj_ids = [int(x) for x in to_python_list(outputs.get("out_obj_ids", None))]
            frame_outputs[frame_idx] = outputs
            for obj_id in obj_ids:
                all_obj_ids.add(obj_id)
            print(
                json.dumps(
                    {
                        "frame_index": frame_idx,
                        "num_objects": len(obj_ids),
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        predictor.handle_request(request={"type": "close_session", "session_id": session_id})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fps, width, height, total_frames = infer_frame_source_stats(source_kind, video_path, frame_files)

    mask_video_writer = None
    overlay_video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if args.save_mask_video:
        mask_video_writer = cv2.VideoWriter(str(output_dir / "union_mask.mp4"), fourcc, fps, (width, height), False)
    if args.save_overlay_video:
        overlay_video_writer = cv2.VideoWriter(str(output_dir / "overlay.mp4"), fourcc, fps, (width, height), True)

    frames_with_masks = 0
    per_frame_counts: List[Dict[str, Any]] = []

    try:
        if source_kind == "video_file":
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video for visualization: {video_path}")
        else:
            cap = None

        frame_idx = 0
        while True:
            if source_kind == "video_file":
                assert cap is not None
                ok, frame_bgr = cap.read()
                if not ok:
                    break
            else:
                if frame_idx >= len(frame_files):
                    break
                frame_bgr = cv2.imread(str(frame_files[frame_idx]), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise RuntimeError(f"Failed to read frame image: {frame_files[frame_idx]}")

            outputs = frame_outputs.get(frame_idx, {})
            raw_masks = outputs.get("out_binary_masks", None)
            obj_ids = [int(x) for x in to_python_list(outputs.get("out_obj_ids", None))]
            union_mask = np.zeros((height, width), dtype=bool)

            if raw_masks is not None and len(obj_ids) > 0:
                for mask in raw_masks:
                    mask_bool = to_numpy_bool(mask)
                    if mask_bool.shape != union_mask.shape:
                        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    union_mask |= mask_bool

            if union_mask.any():
                frames_with_masks += 1

            if args.save_mask_pngs:
                save_union_png(union_mask, mask_png_dir / f"{frame_idx:05d}.png")

            if mask_video_writer is not None:
                mask_frame = (union_mask.astype(np.uint8) * 255)
                mask_video_writer.write(mask_frame)

            if overlay_video_writer is not None:
                overlay_video_writer.write(overlay_frame(frame_bgr, union_mask))

            per_frame_counts.append(
                {
                    "frame_index": frame_idx,
                    "num_objects": len(obj_ids),
                    "obj_ids": obj_ids,
                    "has_union_mask": bool(union_mask.any()),
                    "union_area": int(union_mask.sum()),
                }
            )
            frame_idx += 1
    finally:
        if cap is not None:
            cap.release()
        if mask_video_writer is not None:
            mask_video_writer.release()
        if overlay_video_writer is not None:
            overlay_video_writer.release()

    summary = {
        "video": str(video_path),
        "source_kind": source_kind,
        "checkpoint": str(checkpoint_path),
        "sam3_root": str(sam3_root.resolve()),
        "text": args.text,
        "frame_index": args.frame_index,
        "session_id": session_id,
        "total_video_frames": total_frames,
        "propagated_frames": len(frame_outputs),
        "frames_with_union_mask": frames_with_masks,
        "detected_object_ids": sorted(all_obj_ids),
        "initial_prompt_object_ids": [int(x) for x in to_python_list(initial_outputs.get("out_obj_ids", None))],
        "initial_prompt_num_objects": len(to_python_list(initial_outputs.get("out_obj_ids", None))),
        "saved_union_pngs": bool(args.save_mask_pngs),
        "saved_mask_video": bool(args.save_mask_video),
        "saved_overlay_video": bool(args.save_overlay_video),
        "per_frame": per_frame_counts,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[SAM3] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
