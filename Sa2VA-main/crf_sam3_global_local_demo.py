"""
CRF-SAM3 Global-Local demo

This variant uses a Global + Local (crop) two-stream input for the MLLM discriminator.
The local crop is derived from the candidate mask bounding box, removing the need
for green mask overlays that can confuse the model.
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

from crf_sam3_demo import ConceptExtractor, SAM3ConceptRecall, load_mevis_sample, load_video_frames


def _mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _pad_bbox(
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_ratio: float,
    min_size: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)

    pad_w = max(int(w * pad_ratio), 0)
    pad_h = max(int(h * pad_ratio), 0)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_w - 1, x2 + pad_w)
    y2 = min(img_h - 1, y2 + pad_h)

    # Enforce minimum size
    if (x2 - x1 + 1) < min_size:
        extra = min_size - (x2 - x1 + 1)
        x1 = max(0, x1 - extra // 2)
        x2 = min(img_w - 1, x2 + extra - extra // 2)
    if (y2 - y1 + 1) < min_size:
        extra = min_size - (y2 - y1 + 1)
        y1 = max(0, y1 - extra // 2)
        y2 = min(img_h - 1, y2 + extra - extra // 2)

    return x1, y1, x2, y2


def _resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=image.dtype)
    scale = size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=image.dtype)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _resize_with_pad_to_shape(
    image: np.ndarray,
    target_h: int,
    target_w: int,
    max_scale: float | None = None,
    min_fill_ratio: float | None = None,
    center_xy: Tuple[float, float] | None = None,
) -> np.ndarray:
    if target_h <= 0 or target_w <= 0:
        return image
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        if image.ndim == 2:
            return np.zeros((target_h, target_w), dtype=image.dtype)
        return np.zeros((target_h, target_w, 3), dtype=image.dtype)
    fit_scale = min(target_w / w, target_h / h)
    scale = fit_scale
    if min_fill_ratio is not None:
        min_side = min(target_h, target_w)
        required = (min_fill_ratio * min_side) / max(h, w)
        scale = max(scale, required)
    scale = min(scale, fit_scale)
    if max_scale is not None:
        scale = min(scale, max_scale)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if image.ndim == 2:
        canvas = np.zeros((target_h, target_w), dtype=image.dtype)
    else:
        canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    if center_xy is None:
        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    cx, cy = center_xy
    x0 = int(round(cx - new_w / 2))
    y0 = int(round(cy - new_h / 2))

    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(target_w, x0 + new_w)
    y2 = min(target_h, y0 + new_h)

    if x2 <= x1 or y2 <= y1:
        return canvas

    crop_x1 = x1 - x0
    crop_y1 = y1 - y0
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)

    canvas[y1:y2, x1:x2] = resized[crop_y1:crop_y2, crop_x1:crop_x2]
    return canvas


def _compute_resize_pad_params(height: int, width: int, size: int) -> Tuple[float, int, int, int, int]:
    scale = size / max(height, width)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    return scale, new_w, new_h, x0, y0


def _blend_color_mask(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
    outline_only: bool,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if outline_only:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        region = edges > 0
    else:
        region = mask > 0

    if not np.any(region):
        return canvas

    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    canvas_f = canvas.astype(np.float32)
    canvas_f[region] = canvas_f[region] * (1 - alpha) + color_arr * alpha
    return canvas_f.astype(np.uint8)


class GlobalLocalVisualTubeBuilder:
    """Build Global + Local frame pairs for each candidate object."""

    def __init__(
        self,
        num_keyframes: int = 8,
        temp_dir: str = "visual_tubes_global_local",
        sampling_strategy: str = "uniform",
        custom_frame_indices: List[int] | None = None,
        qframe_config: Dict | None = None,
        local_size: int = 448,
        local_pad_ratio: float = 0.1,
        min_box_size: int = 16,
        global_size: int | None = None,
        align_local_to_global: bool = False,
        local_max_scale: float | None = None,
        local_min_fill_ratio: float | None = None,
        local_use_mask: bool = True,
        local_composite: bool = False,
        local_composite_frames: int = 3,
        local_motion_threshold: float = 0.02,
        local_motion_min_gap: int = 1,
        local_composite_strategy: str = "top_motion",
        local_overlap_threshold: float = 0.0,
        local_composite_max_frames: int | None = None,
        local_interval_stride: int = 2,
        local_interval_include_end: bool = True,
        local_composite_alpha: float = 0.5,
        local_composite_outline_only: bool = True,
        local_composite_mode: str = "mask",
    ):
        self.num_keyframes = num_keyframes
        self.temp_dir = temp_dir
        self.sampling_strategy = sampling_strategy
        self.custom_frame_indices = custom_frame_indices or []
        self.qframe_config = qframe_config or {}
        self.local_size = local_size
        self.local_pad_ratio = local_pad_ratio
        self.min_box_size = min_box_size
        self.global_size = global_size
        self.align_local_to_global = align_local_to_global
        self.local_max_scale = local_max_scale
        self.local_min_fill_ratio = local_min_fill_ratio
        self.local_use_mask = local_use_mask
        self.local_composite = local_composite
        self.local_composite_frames = max(1, local_composite_frames)
        self.local_motion_threshold = local_motion_threshold
        self.local_motion_min_gap = max(1, local_motion_min_gap)
        self.local_composite_strategy = local_composite_strategy
        self.local_overlap_threshold = local_overlap_threshold
        self.local_composite_max_frames = local_composite_max_frames
        self.local_interval_stride = max(1, local_interval_stride)
        self.local_interval_include_end = local_interval_include_end
        self.local_composite_alpha = local_composite_alpha
        self.local_composite_outline_only = local_composite_outline_only
        self.local_composite_mode = local_composite_mode

        self.qframe_selector = None
        if sampling_strategy == "qframe":
            from mapp_sam3_demo import QFrameSelector
            clip_model_path = self.qframe_config.get("clip_model_path")
            device = self.qframe_config.get("device", "cuda")
            if not clip_model_path:
                raise ValueError("Q-Frame sampling requires 'clip_model_path' in qframe_config")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.qframe_selector = QFrameSelector(clip_model_path, device, apply_rebalancing=False)

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "global"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "local"), exist_ok=True)

        print(f"[GlobalLocal] Sampling strategy: {sampling_strategy}")
        if self.align_local_to_global:
            print(f"[GlobalLocal] Local crops aligned to global frame size and position")
            if self.local_max_scale is not None:
                print(f"[GlobalLocal] Local max upscaling: {self.local_max_scale}x")
            if self.local_min_fill_ratio is not None:
                print(f"[GlobalLocal] Local min fill ratio: {self.local_min_fill_ratio:.2f}")
        else:
            print(f"[GlobalLocal] Local crop size: {local_size}")
        print(f"[GlobalLocal] Local pad ratio: {local_pad_ratio}")
        print(f"[GlobalLocal] Local mask applied: {self.local_use_mask}")
        if self.local_composite:
            print(
                "[GlobalLocal] Local composite enabled: "
                f"strategy={self.local_composite_strategy}, "
                f"frames={self.local_composite_frames}, "
                f"motion_thr={self.local_motion_threshold}, "
                f"overlap_thr={self.local_overlap_threshold}, "
                f"stride={self.local_interval_stride}, "
                f"outline_only={self.local_composite_outline_only}, "
                f"mode={self.local_composite_mode}"
            )
        print(f"[GlobalLocal] Temporary frames will be saved to {temp_dir}")

    def _select_frames(
        self,
        frames_with_obj: List[int],
        all_masks: Dict[int, Dict[int, np.ndarray]],
        video_frames: List[np.ndarray] | None,
        referring_expression: str | None,
    ) -> List[int]:
        if self.sampling_strategy == "qframe":
            if self.qframe_selector is None or video_frames is None:
                raise ValueError("Q-Frame selector not initialized or video frames missing")
            if not referring_expression:
                raise ValueError("Q-Frame sampling requires referring_expression")
            qframe_indices = self.qframe_selector.select_keyframes(
                video_frames,
                referring_expression,
                num_candidates=self.num_keyframes,
                tau=self.qframe_config.get("tau", 0.8),
            )
            selected = [idx for idx in qframe_indices if idx in frames_with_obj]
            if not selected:
                if len(frames_with_obj) <= self.num_keyframes:
                    selected = frames_with_obj
                else:
                    step = len(frames_with_obj) // self.num_keyframes
                    selected = [frames_with_obj[i * step] for i in range(self.num_keyframes)]
            return selected

        if self.sampling_strategy == "custom" and self.custom_frame_indices:
            selected = [idx for idx in self.custom_frame_indices if idx in frames_with_obj and idx in all_masks]
            return selected

        if self.sampling_strategy == "all":
            return frames_with_obj

        if len(frames_with_obj) <= self.num_keyframes:
            return frames_with_obj
        step = len(frames_with_obj) // self.num_keyframes
        return [frames_with_obj[i * step] for i in range(self.num_keyframes)]

    def _select_motion_frames(
        self,
        frames: List[int],
        all_masks: Dict[int, Dict[int, np.ndarray]],
        obj_id: int,
        max_frames: int,
    ) -> List[int]:
        if not frames:
            return []

        sample_mask = all_masks[frames[0]][obj_id]
        diag = float(np.hypot(sample_mask.shape[0], sample_mask.shape[1]))

        centroids = {}
        for f in frames:
            centroids[f] = _mask_centroid(all_masks[f][obj_id])

        scored = []
        prev_c = centroids[frames[0]]
        for f in frames[1:]:
            cur_c = centroids[f]
            if prev_c is None or cur_c is None:
                prev_c = cur_c
                continue
            dist = float(np.linalg.norm(np.array(cur_c) - np.array(prev_c))) / max(diag, 1.0)
            if dist >= self.local_motion_threshold:
                scored.append((f, dist))
            prev_c = cur_c

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = []
        for f, _ in scored:
            if all(abs(f - s) >= self.local_motion_min_gap for s in selected):
                selected.append(f)
            if len(selected) >= max_frames:
                break

        if len(selected) < max_frames:
            remaining = [f for f in frames if f not in selected]
            if remaining:
                step = max(1, len(remaining) // max(1, (max_frames - len(selected))))
                selected.extend(remaining[::step][: max_frames - len(selected)])

        return sorted(selected)

    def _select_non_overlap_frames(
        self,
        frames: List[int],
        all_masks: Dict[int, Dict[int, np.ndarray]],
        obj_id: int,
    ) -> List[int]:
        if not frames:
            return []

        first = None
        for f in frames:
            if f in all_masks and obj_id in all_masks[f]:
                first = f
                break
        if first is None:
            return []

        selected = [first]
        ref_mask = all_masks[first][obj_id] > 0

        for f in frames[frames.index(first) + 1:]:
            if f not in all_masks or obj_id not in all_masks[f]:
                continue
            cur_mask = all_masks[f][obj_id] > 0
            if not ref_mask.any() or not cur_mask.any():
                continue
            overlap = np.logical_and(ref_mask, cur_mask)
            if self.local_overlap_threshold <= 0.0:
                if not np.any(overlap):
                    selected.append(f)
                    ref_mask = cur_mask
            else:
                inter = overlap.sum()
                union = np.logical_or(ref_mask, cur_mask).sum()
                iou = inter / union if union > 0 else 1.0
                if iou <= self.local_overlap_threshold:
                    selected.append(f)
                    ref_mask = cur_mask

            if self.local_composite_max_frames and len(selected) >= self.local_composite_max_frames:
                break

        return selected

    def build_tubes(
        self,
        video_path: str,
        all_masks: Dict[int, Dict[int, np.ndarray]],
        referring_expression: str | None = None,
    ) -> Dict[int, Dict[str, any]]:
        if not all_masks:
            return {}

        video_frames = None
        if self.sampling_strategy == "qframe":
            video_frames = load_video_frames(video_path)

        frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        if not frame_files:
            return {}

        all_obj_ids = sorted({obj_id for frame in all_masks.values() for obj_id in frame.keys()})
        print(f"\n[GlobalLocal] Building tubes for {len(all_obj_ids)} objects")

        visual_tubes = {}

        for obj_id in all_obj_ids:
            frames_with_obj = [idx for idx in sorted(all_masks.keys()) if obj_id in all_masks[idx]]
            if not frames_with_obj:
                continue

            selected_frames = self._select_frames(frames_with_obj, all_masks, video_frames, referring_expression)
            if not selected_frames:
                continue

            global_paths = []
            local_paths = []
            frame_indices = []
            global_frames = []
            global_meta = []

            obj_global_dir = os.path.join(self.temp_dir, "global", f"obj_{obj_id}")
            obj_local_dir = os.path.join(self.temp_dir, "local", f"obj_{obj_id}")
            os.makedirs(obj_global_dir, exist_ok=True)
            os.makedirs(obj_local_dir, exist_ok=True)

            for frame_idx in selected_frames:
                if frame_idx >= len(frame_files):
                    continue

                frame = np.array(Image.open(frame_files[frame_idx]).convert("RGB"))
                if self.global_size:
                    scale, _, _, x0, y0 = _compute_resize_pad_params(
                        frame.shape[0], frame.shape[1], self.global_size
                    )
                    global_frame = _resize_with_pad(frame, self.global_size)
                else:
                    scale, x0, y0 = 1.0, 0, 0
                    global_frame = frame

                global_path = os.path.join(obj_global_dir, f"frame_{frame_idx:05d}.jpg")
                Image.fromarray(global_frame).save(global_path)

                global_paths.append(global_path)
                global_frames.append(global_frame)
                global_meta.append((scale, x0, y0))

            if not global_paths:
                continue

            for i, g_idx in enumerate(selected_frames):
                if i >= len(global_frames):
                    break

                g_frame = global_frames[i]
                scale, x0, y0 = global_meta[i]

                if i + 1 < len(selected_frames):
                    seg_end = selected_frames[i + 1]
                else:
                    seg_end = frames_with_obj[-1]

                if self.local_composite_strategy == "non_overlap":
                    segment_frames = list(range(g_idx, seg_end + 1))
                elif self.local_composite_strategy == "interval_step":
                    segment_frames = list(range(g_idx, seg_end + 1, self.local_interval_stride))
                    if self.local_interval_include_end and segment_frames:
                        if segment_frames[-1] != seg_end:
                            segment_frames.append(seg_end)
                else:
                    segment_frames = [g_idx] + [f for f in frames_with_obj if g_idx < f <= seg_end]

                segment_frames = [
                    f for f in segment_frames if f in all_masks and obj_id in all_masks[f]
                ]
                if not segment_frames:
                    segment_frames = [g_idx]

                if self.local_composite:
                    if self.local_composite_strategy == "non_overlap":
                        local_indices = self._select_non_overlap_frames(
                            segment_frames,
                            all_masks,
                            obj_id,
                        )
                    elif self.local_composite_strategy == "interval_step":
                        local_indices = segment_frames
                    else:
                        local_indices = self._select_motion_frames(
                            segment_frames,
                            all_masks,
                            obj_id,
                            self.local_composite_frames,
                        )
                    if not local_indices:
                        local_indices = [segment_frames[len(segment_frames) // 2]]

                    if self.local_composite_max_frames and len(local_indices) > self.local_composite_max_frames:
                        step = max(1, len(local_indices) // self.local_composite_max_frames)
                        local_indices = local_indices[::step][: self.local_composite_max_frames]
                else:
                    local_indices = [g_idx]

                if self.local_composite:
                    local_canvas = np.zeros_like(g_frame)
                    filled_mask = np.zeros(g_frame.shape[:2], dtype=np.uint8)
                    num_local = len(local_indices)
                    for j, lf in enumerate(local_indices):
                        if lf >= len(frame_files):
                            continue
                        if lf not in all_masks or obj_id not in all_masks[lf]:
                            continue
                        frame = np.array(Image.open(frame_files[lf]).convert("RGB"))
                        mask = all_masks[lf][obj_id]
                        bbox = _mask_to_bbox(mask)
                        if bbox is None:
                            continue

                        x1, y1, x2, y2 = _pad_bbox(
                            bbox,
                            img_w=frame.shape[1],
                            img_h=frame.shape[0],
                            pad_ratio=self.local_pad_ratio,
                            min_size=self.min_box_size,
                        )

                        mask_crop = (mask[y1:y2 + 1, x1:x2 + 1] > 0).astype(np.uint8) * 255

                        bbox_cx = (x1 + x2) / 2.0
                        bbox_cy = (y1 + y2) / 2.0
                        mapped_cx = bbox_cx * scale + x0
                        mapped_cy = bbox_cy * scale + y0

                        if self.align_local_to_global:
                            placed_mask = _resize_with_pad_to_shape(
                                mask_crop,
                                target_h=g_frame.shape[0],
                                target_w=g_frame.shape[1],
                                max_scale=self.local_max_scale,
                                min_fill_ratio=self.local_min_fill_ratio,
                                center_xy=(mapped_cx, mapped_cy),
                            )
                        else:
                            placed_mask = _resize_with_pad(mask_crop, self.local_size)

                        if self.local_composite_mode == "rgb":
                            if self.local_use_mask:
                                masked_frame = frame.copy()
                                masked_frame[mask <= 0] = 0
                            else:
                                masked_frame = frame

                            local_crop = masked_frame[y1:y2 + 1, x1:x2 + 1]
                            if self.align_local_to_global:
                                placed_rgb = _resize_with_pad_to_shape(
                                    local_crop,
                                    target_h=g_frame.shape[0],
                                    target_w=g_frame.shape[1],
                                    max_scale=self.local_max_scale,
                                    min_fill_ratio=self.local_min_fill_ratio,
                                    center_xy=(mapped_cx, mapped_cy),
                                )
                            else:
                                placed_rgb = _resize_with_pad(local_crop, self.local_size)

                            region = placed_mask > 0
                            if self.local_composite_alpha < 1.0:
                                local_canvas[region] = (
                                    local_canvas[region] * (1 - self.local_composite_alpha)
                                    + placed_rgb[region] * self.local_composite_alpha
                                )
                            else:
                                write_region = region & (filled_mask == 0)
                                local_canvas[write_region] = placed_rgb[write_region]
                            filled_mask[region] = 1
                        else:
                            t = 0.0 if num_local == 1 else j / max(1, num_local - 1)
                            color = (int(255 * t), 0, int(255 * (1 - t)))
                            local_canvas = _blend_color_mask(
                                local_canvas,
                                placed_mask,
                                color=color,
                                alpha=self.local_composite_alpha,
                                outline_only=self.local_composite_outline_only,
                            )

                    local_image = local_canvas
                else:
                    frame = np.array(Image.open(frame_files[g_idx]).convert("RGB"))
                    mask = all_masks[g_idx][obj_id]
                    bbox = _mask_to_bbox(mask)
                    if bbox is None:
                        continue

                    x1, y1, x2, y2 = _pad_bbox(
                        bbox,
                        img_w=frame.shape[1],
                        img_h=frame.shape[0],
                        pad_ratio=self.local_pad_ratio,
                        min_size=self.min_box_size,
                    )

                    if self.local_use_mask:
                        masked_frame = frame.copy()
                        masked_frame[mask <= 0] = 0
                    else:
                        masked_frame = frame

                    local_crop = masked_frame[y1:y2 + 1, x1:x2 + 1]
                    bbox_cx = (x1 + x2) / 2.0
                    bbox_cy = (y1 + y2) / 2.0
                    mapped_cx = bbox_cx * scale + x0
                    mapped_cy = bbox_cy * scale + y0
                    if self.align_local_to_global:
                        local_crop = _resize_with_pad_to_shape(
                            local_crop,
                            target_h=g_frame.shape[0],
                            target_w=g_frame.shape[1],
                            max_scale=self.local_max_scale,
                            min_fill_ratio=self.local_min_fill_ratio,
                            center_xy=(mapped_cx, mapped_cy),
                        )
                    else:
                        local_crop = _resize_with_pad(local_crop, self.local_size)
                    local_image = local_crop

                local_path = os.path.join(obj_local_dir, f"frame_{g_idx:05d}.jpg")
                Image.fromarray(local_image).save(local_path)
                local_paths.append(local_path)
                frame_indices.append(g_idx)

            if not global_paths or not local_paths:
                continue

            visual_tubes[obj_id] = {
                "global_frame_paths": global_paths,
                "local_frame_paths": local_paths,
                "frame_indices": frame_indices,
            }
            print(f"[GlobalLocal] Object {obj_id}: {len(frame_indices)} pairs, indices: {frame_indices}")

        return visual_tubes


class GlobalLocalDiscriminator:
    """MLLM discriminator with Global + Local frame pairs."""

    def __init__(
        self,
        mllm_path: str,
        prompt_template_path: str,
        prompt_type: str,
        model_type: str,
    ):
        from transformers import AutoProcessor

        self.model_type = model_type.lower()
        self.prompt_type = prompt_type

        print(f"[GlobalLocal] Loading {self.model_type.upper()} from {mllm_path}")

        if self.model_type in ["qwen3", "qwen3-thinking"]:
            from transformers import Qwen3VLForConditionalGeneration
            self.mllm = Qwen3VLForConditionalGeneration.from_pretrained(
                mllm_path,
                dtype="auto",
                device_map="auto",
            ).eval()
            self.process_vision_info = None
        elif self.model_type == "qwen2.5":
            from transformers import Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mllm_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            self.process_vision_info = process_vision_info
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.processor = AutoProcessor.from_pretrained(
            mllm_path,
            trust_remote_code=(self.model_type == "qwen2.5")
        )

        with open(prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        print(f"[GlobalLocal] Prompt template loaded from {prompt_template_path}")

    def score_and_filter(
        self,
        visual_tubes: Dict[int, Dict[str, any]],
        referring_expression: str,
        num_targets: int,
        return_details: bool = False,
        **kwargs,
    ):
        print(f"\n[GlobalLocal] Scoring {len(visual_tubes)} candidates")
        print(f"[GlobalLocal] Target expression: '{referring_expression}'")
        print(f"[GlobalLocal] Will select Top-{num_targets} objects")

        scores = {}
        detailed_scores = {}

        for obj_id, tube_data in visual_tubes.items():
            result = self._score_tube(obj_id, tube_data, referring_expression)
            scores[obj_id] = result["score"]
            detailed_scores[obj_id] = result

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        selected_ids = sorted_ids[:num_targets]

        print(f"\n[GlobalLocal] ===== Scoring Results =====")
        for obj_id in sorted_ids:
            status = "✓ SELECTED" if obj_id in selected_ids else "✗ FILTERED"
            print(f"  Object {obj_id}: {scores[obj_id]:.3f} {status}")
        print(f"[GlobalLocal] ===========================\n")

        if return_details:
            return selected_ids, detailed_scores
        return selected_ids

    def _build_messages(self, image_paths: List[str], prompt: str):
        content = [{"type": "image", "image": p} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, messages):
        if self.model_type in ["qwen3", "qwen3-thinking"]:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_sample_frames=False,
            )
            return inputs.to(self.mllm.device)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(
            messages, return_video_kwargs=True
        )
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.mllm.device)

    def _score_tube(self, obj_id: int, tube_data: Dict[str, any], referring_expression: str):
        global_paths = tube_data.get("global_frame_paths", [])
        local_paths = tube_data.get("local_frame_paths", [])
        if not global_paths or not local_paths:
            return {"score": 0.0, "prob_yes": 0.0, "prob_partial": 0.0, "prob_no": 1.0, "prediction": "No"}

        num_pairs = min(len(global_paths), len(local_paths))
        image_paths = []
        for i in range(num_pairs):
            image_paths.append(global_paths[i])
            image_paths.append(local_paths[i])

        prompt = self.prompt_template.format(
            referring_expression=referring_expression,
            num_pairs=num_pairs,
        )
        messages = self._build_messages(image_paths, prompt)
        inputs = self._prepare_inputs(messages)

        if self.prompt_type != "three_level":
            return self._score_cot(obj_id, inputs)
        return self._score_three_level(obj_id, inputs)

    def _score_cot(self, obj_id: int, inputs):
        with torch.no_grad():
            outputs = self.mllm.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        print(f"\n[GlobalLocal] Object {obj_id} Raw Response:\n{response}\n")

        import re
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                json_str = re.sub(r",(\\s*[}\\]])", r"\\1", json_match.group())
                result = yaml.safe_load(json_str)
                confidence_score = result.get("confidence_score", 0)
                score = confidence_score / 100.0
                return {"score": score}
            except Exception:
                pass
        return {"score": 0.0}

    def _score_three_level(self, obj_id: int, inputs):
        yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        partial_token_id = self.processor.tokenizer.encode("Partial", add_special_tokens=False)[0]
        no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]

        with torch.no_grad():
            outputs = self.mllm.generate(
                **inputs,
                max_new_tokens=5,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        first_token_logits = outputs.scores[0][0]
        yes_logit = first_token_logits[yes_token_id].item()
        partial_logit = first_token_logits[partial_token_id].item()
        no_logit = first_token_logits[no_token_id].item()

        import torch.nn.functional as F
        three_logits = torch.tensor([yes_logit, partial_logit, no_logit])
        three_probs = F.softmax(three_logits, dim=0)

        prob_yes = three_probs[0].item()
        prob_partial = three_probs[1].item()
        prob_no = three_probs[2].item()
        score = prob_yes * 1.0 + prob_partial * 0.5

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs.sequences)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print(f"[GlobalLocal] Object {obj_id}: Response='{response}', Probs=[Yes:{prob_yes:.3f}, Partial:{prob_partial:.3f}, No:{prob_no:.3f}], Score={score:.3f}")

        return {
            "score": score,
            "prob_yes": prob_yes,
            "prob_partial": prob_partial,
            "prob_no": prob_no,
            "prediction": response,
        }


class GlobalLocalRerankerDiscriminator:
    """Reranker discriminator that consumes Global + Local frame pairs."""

    def __init__(
        self,
        model_path: str,
        instruction: str,
        doc_text: str,
        num_frames: int = 8,
        max_frames: int = 8,
        embedding_repo: str = "Qwen3-VL-Embedding",
    ):
        if not model_path:
            raise ValueError("GlobalLocalRerankerDiscriminator requires a valid model_path.")

        if num_frames % 2 != 0:
            num_frames = max(2, num_frames - 1)
        if max_frames is None:
            max_frames = num_frames

        self.num_frames = num_frames
        self.max_frames = max_frames
        self.instruction = instruction
        self.doc_text = doc_text
        self.reranker = self._load_reranker(embedding_repo, model_path, num_frames, max_frames)

        print(
            "[GlobalLocal] Reranker enabled. "
            f"model={model_path}, num_frames={num_frames}, max_frames={max_frames}"
        )

    def _load_reranker(self, embedding_repo: str, model_path: str, num_frames: int, max_frames: int):
        repo_src = os.path.join(os.path.dirname(__file__), embedding_repo, "src")
        if not os.path.exists(repo_src):
            raise FileNotFoundError(
                f"Qwen3-VL-Embedding src not found at: {repo_src}. "
                f"Set discrimination.reranker.embedding_repo accordingly."
            )
        if repo_src not in sys.path:
            sys.path.append(repo_src)

        try:
            from models.qwen3_vl_reranker import Qwen3VLReranker, sample_frames
        except Exception as exc:
            raise ImportError(f"Failed to import Qwen3VLReranker: {exc}") from exc

        reranker = Qwen3VLReranker(
            model_name_or_path=model_path,
            num_frames=num_frames,
            max_frames=max_frames,
        )

        if not hasattr(reranker, "_sample_frames"):
            reranker._sample_frames = sample_frames

        return reranker

    def _build_frame_sequence(self, global_paths: List[str], local_paths: List[str]) -> List[str]:
        pair_count = min(len(global_paths), len(local_paths))
        if pair_count == 0:
            return []

        max_pairs = max(1, self.num_frames // 2)
        use_pairs = min(pair_count, max_pairs)
        frames = []
        for idx in range(use_pairs):
            frames.append(global_paths[idx])
            frames.append(local_paths[idx])

        # Pad to even length = num_frames to preserve global/local pairing
        while len(frames) < self.num_frames:
            frames.append(frames[-2])
            frames.append(frames[-1])

        return frames[:self.num_frames]

    def score_and_filter(
        self,
        visual_tubes: Dict[int, Dict[str, any]],
        referring_expression: str,
        num_targets: int,
        return_details: bool = False,
        **kwargs,
    ):
        obj_ids = sorted(visual_tubes.keys())
        documents = []
        valid_obj_ids = []
        for obj_id in obj_ids:
            global_paths = visual_tubes[obj_id].get("global_frame_paths", [])
            local_paths = visual_tubes[obj_id].get("local_frame_paths", [])
            frames = self._build_frame_sequence(global_paths, local_paths)
            if not frames:
                continue
            documents.append({"video": frames, "text": self.doc_text})
            valid_obj_ids.append(obj_id)

        if not documents:
            return ([], {}) if return_details else []

        print(f"[GlobalLocal] Target expression: '{referring_expression}'")
        print(f"[GlobalLocal] Will select Top-{num_targets} objects")
        print(f"[GlobalLocal] Scoring {len(documents)} candidates with reranker...")

        inputs = {
            "instruction": self.instruction,
            "query": {"text": referring_expression},
            "documents": documents,
        }

        with torch.no_grad():
            scores = self.reranker.process(inputs)

        score_map = {obj_id: float(score) for obj_id, score in zip(valid_obj_ids, scores)}
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
        selected_ids = sorted_ids[:num_targets]

        print(f"\n[GlobalLocal] ===== Scoring Results =====")
        for obj_id in sorted_ids:
            status = "✓ SELECTED" if obj_id in selected_ids else "✗ FILTERED"
            print(f"  Object {obj_id}: {score_map[obj_id]:.6f} {status}")
        print(f"[GlobalLocal] ===========================\n")

        if return_details:
            details = {obj_id: {"score": score_map[obj_id]} for obj_id in sorted_ids}
            return selected_ids, details
        return selected_ids


class CRFSAM3GlobalLocalPipeline:
    """CRF-SAM3 pipeline with Global + Local discriminator."""

    def __init__(
        self,
        mllm_path: str,
        sam3_checkpoint: str,
        num_tube_frames: int,
        model_type: str,
        prompt_type: str,
        prompt_template_path: str,
        sampling_strategy: str,
        custom_frame_indices: List[int] | None,
        qframe_config: Dict | None,
        local_cfg: Dict,
        concept_use_qframe: bool = False,
        discriminator_backend: str = "mllm",
        reranker_cfg: Dict | None = None,
    ):
        self.concept_extractor = ConceptExtractor(
            mllm_path,
            model_type,
            use_qframe=concept_use_qframe,
            qframe_config=qframe_config,
        )
        self.sam3_recall = SAM3ConceptRecall(sam3_checkpoint)
        local_cfg = dict(local_cfg)
        if discriminator_backend == "reranker":
            local_cfg["align_local_to_global"] = True
            print("[GlobalLocal] Reranker requires uniform frame sizes; align local crops to global frames")

        self.tube_builder = GlobalLocalVisualTubeBuilder(
            num_keyframes=num_tube_frames,
            temp_dir=local_cfg.get("temp_dir", "visual_tubes_global_local"),
            sampling_strategy=sampling_strategy,
            custom_frame_indices=custom_frame_indices,
            qframe_config=qframe_config,
            local_size=local_cfg.get("local_size", 448),
            local_pad_ratio=local_cfg.get("local_pad_ratio", 0.1),
            min_box_size=local_cfg.get("min_box_size", 16),
            global_size=local_cfg.get("global_size"),
            align_local_to_global=local_cfg.get("align_local_to_global", False),
            local_max_scale=local_cfg.get("local_max_scale"),
            local_min_fill_ratio=local_cfg.get("local_min_fill_ratio"),
            local_use_mask=local_cfg.get("local_use_mask", True),
            local_composite=local_cfg.get("local_composite", False),
            local_composite_frames=local_cfg.get("local_composite_frames", 3),
            local_motion_threshold=local_cfg.get("local_motion_threshold", 0.02),
            local_motion_min_gap=local_cfg.get("local_motion_min_gap", 1),
            local_composite_strategy=local_cfg.get("local_composite_strategy", "top_motion"),
            local_overlap_threshold=local_cfg.get("local_overlap_threshold", 0.0),
            local_composite_max_frames=local_cfg.get("local_composite_max_frames"),
            local_interval_stride=local_cfg.get("local_interval_stride", 2),
            local_interval_include_end=local_cfg.get("local_interval_include_end", True),
            local_composite_alpha=local_cfg.get("local_composite_alpha", 0.5),
            local_composite_outline_only=local_cfg.get("local_composite_outline_only", True),
            local_composite_mode=local_cfg.get("local_composite_mode", "mask"),
        )
        if discriminator_backend == "reranker":
            reranker_cfg = reranker_cfg or {}
            self.discriminator = GlobalLocalRerankerDiscriminator(
                model_path=reranker_cfg.get("model_path"),
                instruction=reranker_cfg.get(
                    "instruction",
                    "Each pair includes a global scene and a local crop of the candidate. "
                    "Judge whether the cropped object matches the referring expression."
                ),
                doc_text=reranker_cfg.get(
                    "doc_text",
                    "Document video is ordered as [Global, Local] pairs for the same candidate object."
                ),
                num_frames=reranker_cfg.get("num_frames", 8),
                max_frames=reranker_cfg.get("max_frames", 8),
                embedding_repo=reranker_cfg.get("embedding_repo", "Qwen3-VL-Embedding"),
            )
            print("[GlobalLocal] Discriminator backend: reranker")
        else:
            self.discriminator = GlobalLocalDiscriminator(
                mllm_path=mllm_path,
                prompt_template_path=prompt_template_path,
                prompt_type=prompt_type,
                model_type=model_type,
            )
            print("[GlobalLocal] Discriminator backend: mllm")

        print("[GlobalLocal] Pipeline initialized")

    def predict(self, video_path: str, referring_expression: str, return_details: bool = False):
        print(f"\n{'=' * 60}")
        print("[GlobalLocal] Starting pipeline")
        print(f"[GlobalLocal] Expression: '{referring_expression}'")
        print(f"[GlobalLocal] Video path: {video_path}")
        print(f"{'=' * 60}\n")

        category, num_targets, category_reasoning = self.concept_extractor.extract_category(
            referring_expression, video_path
        )
        all_masks = self.sam3_recall.recall_all_instances(video_path, category)

        if len(all_masks) == 0:
            print("[GlobalLocal] No objects recalled, returning empty result")
            if return_details:
                return {}, {}, {
                    "category": category,
                    "num_targets": num_targets,
                    "sub_conditions": category_reasoning.get("sub_conditions", []),
                    "final_rule_applied": category_reasoning.get("final_rule_applied", "N/A"),
                    "all_candidates": [],
                    "selected_candidates": [],
                    "scores": {},
                }
            return {}, {}

        visual_tubes = self.tube_builder.build_tubes(video_path, all_masks, referring_expression)
        all_candidate_ids = sorted(visual_tubes.keys())

        if return_details:
            selected_obj_ids, detailed_scores = self.discriminator.score_and_filter(
                visual_tubes,
                referring_expression,
                num_targets,
                return_details=True,
            )
        else:
            selected_obj_ids = self.discriminator.score_and_filter(
                visual_tubes,
                referring_expression,
                num_targets,
                return_details=False,
            )

        final_masks = {}
        for frame_idx, frame_masks in all_masks.items():
            final_masks[frame_idx] = {
                obj_id: mask for obj_id, mask in frame_masks.items() if obj_id in selected_obj_ids
            }

        print(f"\n[GlobalLocal] Pipeline complete!")
        print(f"[GlobalLocal] Selected {len(selected_obj_ids)} objects: {selected_obj_ids}")

        if return_details:
            return final_masks, visual_tubes, {
                "category": category,
                "num_targets": num_targets,
                "sub_conditions": category_reasoning.get("sub_conditions", []),
                "final_rule_applied": category_reasoning.get("final_rule_applied", "N/A"),
                "all_candidates": all_candidate_ids,
                "selected_candidates": selected_obj_ids,
                "scores": detailed_scores,
                "all_masks": all_masks,
            }
        return final_masks, visual_tubes


def main():
    parser = argparse.ArgumentParser(description="CRF-SAM3 Global-Local demo")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading configuration from {args.config}")

    mllm_path = config["model"]["mllm_path"]
    sam3_checkpoint = config["model"]["sam3_checkpoint"]
    model_type = config["model"].get("model_type", "qwen3")
    num_tube_frames = config["pipeline"]["num_tube_frames"]

    demo_cfg = config.get("demo", {})
    mevis_root = demo_cfg["mevis_root"]
    video_id = demo_cfg["video_id"]
    exp_id = demo_cfg["exp_id"]
    output_dir = demo_cfg.get("output_dir", "crf_sam3_demo_output")

    visual_cfg = config.get("visual_tube", {})
    sampling_strategy = visual_cfg.get("sampling_strategy", "uniform")
    custom_frame_indices = visual_cfg.get("custom_frame_indices", None)
    qframe_config = visual_cfg.get("qframe", None)
    concept_use_qframe = config.get("concept_extraction", {}).get("use_qframe", False)

    gl_cfg = config.get("global_local", {})
    discrimination_cfg = config.get("discrimination", {})
    prompt_type = discrimination_cfg.get("prompt_type", "three_level")
    discriminator_backend = discrimination_cfg.get("backend", "mllm")
    reranker_cfg = discrimination_cfg.get("reranker", {})
    mllm_cfg = discrimination_cfg.get("mllm", {})

    prompt_template_path = gl_cfg.get("prompt_template_path")
    if not prompt_template_path:
        prompt_templates = {
            "cot": "configs/discrimination_prompt.txt",
            "three_level": "configs/three_level_ttrl_cot_global_local.txt",
        }
        prompt_template_path = prompt_templates.get(prompt_type, "configs/three_level_ttrl_cot_global_local.txt")
    if discriminator_backend == "mllm":
        prompt_type = mllm_cfg.get("prompt_type", prompt_type)
        prompt_template_path = mllm_cfg.get("prompt_template_path", prompt_template_path)

    video_path, expression, num_frames = load_mevis_sample(mevis_root, video_id, exp_id)
    print("=" * 60)
    print("CRF-SAM3 Global-Local Demo")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    print(f"MLLM: {mllm_path}")
    print(f"SAM3: {sam3_checkpoint}")
    print(f"Visual Tube Frames: {num_tube_frames}")
    print(f"Sampling Strategy: {sampling_strategy}")
    print(f"Discriminator Backend: {discriminator_backend}")
    print(f"Prompt Template: {prompt_template_path}")
    print(f"Video: {video_path}")
    print(f"Expression: {expression}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    pipeline = CRFSAM3GlobalLocalPipeline(
        mllm_path=mllm_path,
        sam3_checkpoint=sam3_checkpoint,
        num_tube_frames=num_tube_frames,
        model_type=model_type,
        prompt_type=prompt_type,
        prompt_template_path=prompt_template_path,
        sampling_strategy=sampling_strategy,
        custom_frame_indices=custom_frame_indices,
        qframe_config=qframe_config,
        local_cfg=gl_cfg,
        concept_use_qframe=concept_use_qframe,
        discriminator_backend=discriminator_backend,
        reranker_cfg=reranker_cfg,
    )

    masks_dict, visual_tubes, discrimination_details = pipeline.predict(
        video_path, expression, return_details=True
    )

    # Optional: save masks for inspection
    if masks_dict:
        exp_output_dir = os.path.join(output_dir, video_id, str(exp_id))
        os.makedirs(exp_output_dir, exist_ok=True)
        for frame_idx, frame_masks in masks_dict.items():
            if not frame_masks:
                continue
            first_mask = next(iter(frame_masks.values()))
            h, w = first_mask.shape
            instance_mask = np.zeros((h, w), dtype=np.uint8)
            for obj_id, mask in frame_masks.items():
                gray_value = ((obj_id + 1) * 50) % 256
                instance_mask = np.bitwise_or(instance_mask, (mask > 0).astype(np.uint8) * gray_value)
            mask_file = os.path.join(exp_output_dir, f"{frame_idx:05d}.png")
            Image.fromarray(instance_mask).save(mask_file)

        print(f"\nSaved {len(masks_dict)} frames to {exp_output_dir}")


if __name__ == "__main__":
    main()
