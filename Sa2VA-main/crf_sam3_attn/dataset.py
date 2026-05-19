import json
import glob
import os
import random
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

from crf_sam3_demo import (
    ConceptExtractor,
    SAM3ConceptRecall,
    load_gt_masks,
    compute_j_score_sequence,
)
try:
    from pycocotools import mask as cocomask
except Exception:  # pragma: no cover - optional dependency for train split
    cocomask = None


def _is_oom_runtime_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _load_meta_expressions(split_root: str) -> Dict:
    split_root = os.path.abspath(split_root)
    data_root = os.path.dirname(split_root)
    split = os.path.basename(split_root)
    candidates = [
        os.path.join(split_root, "meta_expressions.json"),
        os.path.join(data_root, "meta_expressions", split, "meta_expressions.json"),
        os.path.join(data_root, "meta_expressions.json"),
    ]
    for meta_file in candidates:
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        "meta_expressions.json not found. Tried: " + ", ".join(candidates)
    )


def _list_frame_files(video_path: str) -> List[str]:
    frame_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        frame_files.extend(glob.glob(os.path.join(video_path, ext)))
    frame_files.sort()
    return frame_files


def _get_frame_pad(video_path: str) -> int:
    frame_files = _list_frame_files(video_path)
    if not frame_files:
        return 0
    stem = os.path.splitext(os.path.basename(frame_files[0]))[0]
    return len(stem) if stem.isdigit() else 0


def _overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay_color = np.array([0, 220, 0], dtype=np.uint8)
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return frame
    out = frame.copy()
    out[mask_bool] = (out[mask_bool] * (1 - alpha) + overlay_color * alpha).astype(np.uint8)
    return out


def _draw_bbox(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return frame
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def _sample_indices(indices: List[int], k: int) -> List[int]:
    if len(indices) <= k:
        return indices
    step = len(indices) / float(k)
    sel = [indices[int(i * step)] for i in range(k)]
    return sel


def summarize_iou_map(iou_map: Dict[int, float] | Dict[str, float]) -> Dict[str, float]:
    values = []
    if isinstance(iou_map, dict):
        for value in iou_map.values():
            try:
                values.append(float(value))
            except Exception:
                continue
    if not values:
        return {
            'max_j': 0.0,
            'sum_j': 0.0,
            'mean_j': 0.0,
            'num_candidates': 0,
        }
    return {
        'max_j': float(max(values)),
        'sum_j': float(sum(values)),
        'mean_j': float(sum(values) / len(values)),
        'num_candidates': int(len(values)),
    }


class CRFSAM3AttnDataset(Dataset):
    def __init__(
        self,
        base_cfg: Dict,
        train_cfg: Dict,
    ) -> None:
        super().__init__()

        self.base_cfg = base_cfg
        self.train_cfg = train_cfg

        dataset_cfg = base_cfg["dataset"]
        data_root = dataset_cfg["data_root"]
        split = train_cfg.get("split") or dataset_cfg["split"]
        self.split = split
        self.mevis_root = os.path.join(data_root, split)
        self.gt_masks_root = dataset_cfg.get("gt_masks_root", "")
        self.dataset_name = str(
            train_cfg.get("dataset_name")
            or dataset_cfg.get("name")
            or os.path.basename(os.path.normpath(str(data_root)))
        )

        self.num_frames = int(train_cfg.get("num_frames", base_cfg["pipeline"]["num_tube_frames"]))
        self.mask_style = train_cfg.get("mask_style", base_cfg["visual_tube"]["visualization_style"])
        self.pos_iou = float(train_cfg.get("pos_iou", 0.5))
        self.neg_iou = float(train_cfg.get("neg_iou", 0.1))
        self.sample_neg_prob = float(train_cfg.get("sample_neg_prob", 0.5))
        self.listwise = bool(train_cfg.get("listwise", False))
        self.listwise_max_candidates = train_cfg.get("listwise_max_candidates")
        self.listwise_neg_policy = train_cfg.get("listwise_neg_policy")
        self.hard_neg_weight = train_cfg.get("hard_neg_weight")
        self.hard_neg_margin = train_cfg.get("hard_neg_margin")
        self.hard_neg_pos_strategy = train_cfg.get("hard_neg_pos_strategy")
        self.cache_sam3 = bool(train_cfg.get("cache_sam3", False))
        self.category_source = train_cfg.get("category_source", "concept")
        self.max_video_frames = train_cfg.get("max_video_frames")
        self.mask_dict_path = train_cfg.get("mask_dict_path") or dataset_cfg.get("mask_dict_path")
        self.gt_source = str(
            train_cfg.get("gt_source") or dataset_cfg.get("gt_source") or "auto"
        ).strip().lower()
        if self.gt_source not in {"auto", "gt_masks_root", "mask_dict"}:
            raise ValueError(
                f"Unsupported gt_source={self.gt_source!r}. Expected one of: auto, gt_masks_root, mask_dict"
            )

        model_cfg = base_cfg["model"]
        self.model_type = model_cfg.get("model_type", "qwen3")
        self.mllm_path = model_cfg["mllm_path"]
        self.sam3_checkpoint = model_cfg["sam3_checkpoint"]

        self._repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.category_prompt_path = (
            train_cfg.get("category_prompt_path")
            or base_cfg.get("concept_extraction", {}).get("category_prompt_path")
            or "configs/category_num_targets_inference.txt"
        )
        self.offline_mode = bool(train_cfg.get("offline_mode", False))
        self.offline_strict = bool(train_cfg.get("offline_strict", True))
        self.offline_subset_json = train_cfg.get("offline_subset_json")
        self.offline_cache_root = train_cfg.get("offline_cache_root", "crf_sam3_attn/offline_cache")
        if not os.path.isabs(self.offline_cache_root):
            self.offline_cache_root = os.path.join(self._repo_root, self.offline_cache_root)
        self.offline_cache_root = os.path.abspath(self.offline_cache_root)
        self.offline_split_root = os.path.join(self.offline_cache_root, str(split))

        self._meta = _load_meta_expressions(self.mevis_root)
        self._offline_subset_pairs = self._load_offline_subset_pairs(self.offline_subset_json)
        self._samples = self._build_samples(
            max_videos=train_cfg.get("max_videos"),
            max_expressions=train_cfg.get("max_expressions"),
        )

        self._concept_extractor = None
        self._sam3 = None
        self._sam3_cache: Dict[Tuple[str, int], Dict] = {}
        self._sam3_prompt_meta: Dict[Tuple[str, int], Dict] = {}
        self._category_cache: Dict[Tuple[str, int], str] = {}
        self._category_info_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._offline_meta_cache: Dict[Tuple[str, int], Dict] = {}
        self._mask_dict = None
        self._mask_dict_mode = "obj_or_anno"
        self._expr_global_index: Optional[Dict[Tuple[str, int], int]] = None

        if self.offline_mode:
            if not os.path.isdir(self.offline_split_root):
                if self.offline_strict:
                    raise FileNotFoundError(
                        f"Offline cache root not found: {self.offline_split_root}. "
                        "Run offline preparation first or disable offline_mode."
                    )
            if not self._samples and self.offline_strict:
                raise RuntimeError(
                    "Offline mode enabled but no cached samples found. "
                    "Run offline preparation first or disable offline_mode."
                )
        else:
            if self.category_source == "concept":
                concept_cfg = base_cfg.get("concept_extraction", {})
                self._concept_extractor = ConceptExtractor(
                    self.mllm_path,
                    model_type=self.model_type,
                    use_qframe=bool(concept_cfg.get("use_qframe", False)),
                    qframe_config={
                        "clip_model_path": base_cfg["model"].get("clip_model_path"),
                        "num_frames": self.num_frames,
                        "tau": base_cfg["visual_tube"].get("qframe", {}).get("tau", 0.8),
                        "device": base_cfg["pipeline"].get("device", "auto"),
                    },
                    category_prompt_path=self.category_prompt_path,
                    teacher_pipeline_compatible=bool(concept_cfg.get("teacher_pipeline_compatible", False)),
                    teacher_pipeline_num_frames=int(concept_cfg.get("teacher_pipeline_num_frames", 3)),
                    teacher_pipeline_deterministic=bool(concept_cfg.get("teacher_pipeline_deterministic", True)),
                )

            self._sam3 = SAM3ConceptRecall(self.sam3_checkpoint)

    def _load_offline_subset_pairs(self, subset_json: Optional[str]) -> Optional[set[tuple[str, int]]]:
        if not subset_json:
            return None
        subset_path = subset_json
        if not os.path.isabs(subset_path):
            subset_path = os.path.join(self._repo_root, subset_path)
        subset_path = os.path.abspath(subset_path)
        with open(subset_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        items = payload.get("samples") if isinstance(payload, dict) and isinstance(payload.get("samples"), list) else payload
        if not isinstance(items, list):
            raise TypeError(f"Unsupported subset json format: {subset_path}")
        pairs = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            video_id = item.get("video_id")
            exp_id = item.get("exp_id")
            if video_id is None or exp_id is None:
                continue
            pairs.add((str(video_id), int(exp_id)))
        return pairs

    def _build_samples(self, max_videos=None, max_expressions=None) -> List[Dict]:
        samples = []
        videos = list(self._meta.get("videos", {}).items())
        if max_videos is not None and max_videos > 0:
            videos = videos[: int(max_videos)]
        for video_id, video_meta in videos:
            expressions = list(video_meta.get("expressions", {}).items())
            if max_expressions is not None and max_expressions > 0:
                expressions = expressions[: int(max_expressions)]
            for exp_id, exp_data in expressions:
                exp_id_int = int(exp_id)
                if self._offline_subset_pairs is not None and (video_id, exp_id_int) not in self._offline_subset_pairs:
                    continue
                if self.offline_mode:
                    meta_path = self._offline_meta_path(video_id, exp_id_int)
                    if not os.path.exists(meta_path):
                        continue
                samples.append(
                    {
                        "video_id": video_id,
                        "exp_id": exp_id_int,
                        "expression": exp_data.get("exp", ""),
                    }
                )
        return samples

    def _load_mask_dict(self):
        if self._mask_dict is not None:
            return self._mask_dict

        if self.mask_dict_path:
            mask_dict_path = self.mask_dict_path
            if not os.path.isabs(mask_dict_path):
                mask_dict_path = os.path.join(self.mevis_root, mask_dict_path)
            candidates = [mask_dict_path]
        else:
            candidates = [
                os.path.join(self.mevis_root, "mask_dict.json"),
                os.path.join(self.mevis_root, "mask_dict.pkl"),
            ]

        mask_dict_path = None
        for p in candidates:
            if os.path.exists(p):
                mask_dict_path = p
                break

        if mask_dict_path is None:
            self._mask_dict = {}
            return self._mask_dict

        if cocomask is None:
            raise RuntimeError(
                "pycocotools is required to decode mask_dict but is not available."
            )

        if mask_dict_path.endswith(".pkl"):
            with open(mask_dict_path, "rb") as f:
                self._mask_dict = pickle.load(f)
        else:
            with open(mask_dict_path, "r", encoding="utf-8") as f:
                self._mask_dict = json.load(f)
        self._mask_dict_mode = self._infer_mask_dict_mode(self._mask_dict)
        print(
            f"[mask_dict] Loaded {os.path.basename(mask_dict_path)} | "
            f"entries={len(self._mask_dict)} | mode={self._mask_dict_mode}"
        )
        return self._mask_dict

    def _build_expression_global_index(self) -> Dict[Tuple[str, int], int]:
        index_map: Dict[Tuple[str, int], int] = {}
        idx = 0
        for video_id, video_meta in self._meta.get("videos", {}).items():
            for exp_id in video_meta.get("expressions", {}).keys():
                index_map[(video_id, int(exp_id))] = idx
                idx += 1
        return index_map

    def _infer_mask_dict_mode(self, mask_dict: Dict[Any, Any]) -> str:
        if not isinstance(mask_dict, dict) or not mask_dict:
            return "obj_or_anno"
        total_expr = sum(
            len(video_meta.get("expressions", {}))
            for video_meta in self._meta.get("videos", {}).values()
        )
        int_keys = []
        for key in mask_dict.keys():
            try:
                int_keys.append(int(key))
            except (TypeError, ValueError):
                return "obj_or_anno"
        if (
            total_expr > 0
            and len(int_keys) == total_expr
            and min(int_keys) == 0
            and max(int_keys) == total_expr - 1
        ):
            return "expression_index"
        return "obj_or_anno"

    def _decode_mask_rle(self, mask_rle: Any) -> Optional[np.ndarray]:
        if not mask_rle:
            return None

        if isinstance(mask_rle, (list, tuple)):
            merged = None
            for item in mask_rle:
                decoded = self._decode_mask_rle(item)
                if decoded is None:
                    continue
                if merged is None:
                    merged = decoded.astype(np.uint8)
                else:
                    merged = np.logical_or(merged > 0, decoded > 0).astype(np.uint8)
            return merged

        if isinstance(mask_rle, dict):
            counts = mask_rle.get("counts")
            if isinstance(counts, str):
                mask_rle = dict(mask_rle)
                mask_rle["counts"] = counts.encode("utf-8")

        decoded = cocomask.decode(mask_rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2).astype(np.uint8)
        return decoded.astype(np.uint8)

    def _load_gt_masks_from_mask_dict(self, video_id: str, exp_id: int) -> Dict[int, np.ndarray]:
        mask_dict = self._load_mask_dict()
        if not mask_dict:
            return {}
        vid_info = self._meta.get("videos", {}).get(video_id)
        if vid_info is None:
            return {}
        exp_info = vid_info.get("expressions", {}).get(str(exp_id))
        if exp_info is None:
            return {}

        num_frames = len(vid_info.get("frames", []))

        # ref-youtube-ros style: mask_dict keys are global expression indices (0..N-1)
        if self._mask_dict_mode == "expression_index":
            if self._expr_global_index is None:
                self._expr_global_index = self._build_expression_global_index()
            expr_idx = self._expr_global_index.get((video_id, int(exp_id)))
            if expr_idx is None:
                return {}
            mask_seq = mask_dict.get(str(expr_idx))
            if mask_seq is None:
                mask_seq = mask_dict.get(expr_idx)
            if mask_seq is None:
                return {}

            gt_masks: Dict[int, np.ndarray] = {}
            upper = len(mask_seq) if num_frames <= 0 else min(num_frames, len(mask_seq))
            for frame_idx in range(upper):
                decoded = self._decode_mask_rle(mask_seq[frame_idx])
                if decoded is not None:
                    gt_masks[frame_idx] = decoded
            return gt_masks

        # Different datasets use either anno_id or obj_id in meta_expressions.
        anno_ids = exp_info.get("anno_id", [])
        if not anno_ids:
            anno_ids = exp_info.get("obj_id", [])
        if isinstance(anno_ids, (int, str)):
            anno_ids = [anno_ids]
        anno_ids = [int(x) for x in anno_ids]
        gt_masks: Dict[int, np.ndarray] = {}
        for frame_idx in range(num_frames):
            merged = None
            for anno_id in anno_ids:
                mask_seq = mask_dict.get(str(anno_id))
                if mask_seq is None:
                    mask_seq = mask_dict.get(anno_id)
                if mask_seq is None or frame_idx >= len(mask_seq):
                    continue
                decoded = self._decode_mask_rle(mask_seq[frame_idx])
                if decoded is None:
                    continue
                if merged is None:
                    merged = decoded.astype(np.uint8)
                else:
                    merged = np.logical_or(merged > 0, decoded > 0).astype(np.uint8)
            if merged is not None:
                gt_masks[frame_idx] = merged
        return gt_masks

    def _load_expression_gt_masks(self, video_id: str, exp_id: int) -> Dict[int, np.ndarray]:
        if self.gt_source == "mask_dict":
            return self._load_gt_masks_from_mask_dict(video_id, exp_id)

        if self.gt_source == "gt_masks_root":
            if not self.gt_masks_root:
                return {}
            return load_gt_masks(self.gt_masks_root, video_id, exp_id)

        gt_masks = {}
        if self.gt_masks_root:
            gt_masks = load_gt_masks(self.gt_masks_root, video_id, exp_id)
        if gt_masks:
            return gt_masks
        return self._load_gt_masks_from_mask_dict(video_id, exp_id)

    def __len__(self) -> int:
        return len(self._samples)

    def _offline_exp_dir(self, video_id: str, exp_id: int) -> str:
        return os.path.join(self.offline_split_root, video_id, str(exp_id))

    def _offline_meta_path(self, video_id: str, exp_id: int) -> str:
        return os.path.join(self._offline_exp_dir(video_id, exp_id), "meta.json")

    def _load_offline_meta(self, video_id: str, exp_id: int) -> Dict:
        cache_key = (video_id, exp_id)
        if cache_key in self._offline_meta_cache:
            return self._offline_meta_cache[cache_key]
        meta_path = self._offline_meta_path(video_id, exp_id)
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._offline_meta_cache[cache_key] = meta
        return meta

    def _get_category_info(self, video_id: str, exp_id: int, expression: str, video_path: str) -> Dict[str, Any]:
        cache_key = (video_id, exp_id)
        if cache_key in self._category_info_cache:
            return self._category_info_cache[cache_key]
        if self._concept_extractor is None:
            category = expression.split()[0] if expression else "object"
            info = {
                "category": category,
                "num_targets": 1,
                "backup_categories": [],
                "reasoning": {},
            }
        else:
            category, num_targets, reasoning = self._concept_extractor.extract_category(expression, video_path)
            backup_categories = []
            if isinstance(reasoning, dict):
                raw_backups = reasoning.get("backup_categories", [])
                if isinstance(raw_backups, (list, tuple)):
                    for item in raw_backups:
                        text = str(item).strip()
                        if not text:
                            continue
                        if text.lower() == str(category).strip().lower():
                            continue
                        if text.lower() in [x.lower() for x in backup_categories]:
                            continue
                        backup_categories.append(text)
            info = {
                "category": category,
                "num_targets": num_targets,
                "backup_categories": backup_categories,
                "reasoning": reasoning if isinstance(reasoning, dict) else {},
            }
        self._category_info_cache[cache_key] = info
        self._category_cache[cache_key] = info["category"]
        return info

    def _get_category(self, video_id: str, exp_id: int, expression: str, video_path: str) -> str:
        cache_key = (video_id, exp_id)
        if cache_key in self._category_cache:
            return self._category_cache[cache_key]
        info = self._get_category_info(video_id, exp_id, expression, video_path)
        category = info["category"]
        self._category_cache[cache_key] = category
        return category

    def _get_all_masks(
        self,
        video_id: str,
        exp_id: int,
        video_path: str,
        category: str,
        backup_categories: Optional[List[str]] = None,
        expression: str = None,
    ) -> Dict:
        cache_key = (video_id, exp_id)
        if self.cache_sam3 and cache_key in self._sam3_cache:
            return self._sam3_cache[cache_key]

        def _is_recall_empty(mask_dict: Dict[int, Dict[int, np.ndarray]]) -> bool:
            if not mask_dict:
                return True
            for frame_masks in mask_dict.values():
                if frame_masks and len(frame_masks) > 0:
                    return False
            return True

        query_candidates: List[str] = []
        for q in [category] + list(backup_categories or []):
            qn = str(q).strip()
            if not qn:
                continue
            if qn.lower() in [x.lower() for x in query_candidates]:
                continue
            query_candidates.append(qn)
        if not query_candidates:
            query_candidates = [category]

        prompt_source = "category"
        prompt_text = query_candidates[0]
        all_masks = {}
        for i, recall_query in enumerate(query_candidates):
            try:
                recall_masks = self._sam3.recall_all_instances(video_path, recall_query)
            except RuntimeError as err:
                # Fallback path for extreme crowded-category OOM:
                # retry the same sample with full expression prompt.
                if (not _is_oom_runtime_error(err)) or (not expression):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"[SAM3Recall][OOM] {video_id}/{exp_id} with query='{recall_query}'. "
                    "Retry with full expression prompt."
                )
                recall_masks = self._sam3.recall_all_instances(video_path, expression)
                prompt_source = "full_expression_fallback"
                prompt_text = expression
                all_masks = recall_masks
                break

            if not _is_recall_empty(recall_masks):
                all_masks = recall_masks
                prompt_source = "category" if i == 0 else "backup_category"
                prompt_text = recall_query
                break
            print(f"[SAM3Recall][empty] {video_id}/{exp_id} query='{recall_query}' returned no masks.")

        self._sam3_prompt_meta[cache_key] = {
            "source": prompt_source,
            "text": prompt_text,
        }
        if self.cache_sam3:
            self._sam3_cache[cache_key] = all_masks
        return all_masks

    def _compute_iou_map(self, all_masks: Dict[int, Dict[int, np.ndarray]], gt_masks: Dict[int, np.ndarray]) -> Dict[int, float]:
        obj_ids = set()
        for frame_masks in all_masks.values():
            obj_ids.update(frame_masks.keys())
        iou_map = {}
        for obj_id in obj_ids:
            obj_masks = {}
            obj_gt_masks = {}
            # Evaluate every candidate on the full GT sequence.
            # If a candidate is missing on a GT frame, count it as an all-zero prediction
            # instead of skipping the frame, otherwise sequence IoU is systematically inflated.
            for frame_idx, gt_mask in gt_masks.items():
                frame_masks = all_masks.get(frame_idx, {})
                pred_mask = frame_masks.get(obj_id)
                if pred_mask is None:
                    obj_masks[frame_idx] = np.zeros_like(gt_mask, dtype=np.uint8)
                    obj_gt_masks[frame_idx] = gt_mask
                    continue

                if pred_mask.shape != gt_mask.shape:
                    gt_mask = cv2.resize(
                        (gt_mask > 0).astype(np.uint8),
                        (pred_mask.shape[1], pred_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                obj_masks[frame_idx] = pred_mask
                obj_gt_masks[frame_idx] = gt_mask
            if obj_masks:
                iou_map[obj_id] = compute_j_score_sequence(obj_masks, obj_gt_masks)
            else:
                iou_map[obj_id] = 0.0
        return iou_map

    def _select_candidate(self, iou_map: Dict[int, float]) -> Tuple[int, int]:
        pos_ids = [obj_id for obj_id, iou in iou_map.items() if iou >= self.pos_iou]
        neg_ids = [obj_id for obj_id, iou in iou_map.items() if iou <= self.neg_iou]
        if pos_ids and (not neg_ids or random.random() > self.sample_neg_prob):
            return random.choice(pos_ids), 1
        if neg_ids:
            return random.choice(neg_ids), 0
        # fallback to any candidate
        obj_id = random.choice(list(iou_map.keys()))
        label = 1 if iou_map[obj_id] >= self.pos_iou else 0
        return obj_id, label

    def _trim_candidate_ids(self, obj_ids: List[int], iou_map: Dict[int, float]) -> List[int]:
        if not self.listwise_max_candidates:
            return obj_ids
        try:
            max_candidates = int(self.listwise_max_candidates)
        except (TypeError, ValueError):
            return obj_ids
        if max_candidates <= 0 or len(obj_ids) <= max_candidates:
            return obj_ids
        obj_ids_sorted = sorted(obj_ids, key=lambda oid: iou_map.get(int(oid), 0.0), reverse=True)
        return obj_ids_sorted[:max_candidates]

    def __getitem__(self, idx: int) -> Dict:
        if self.offline_mode:
            if self.listwise:
                return self._get_offline_listwise_item(idx)
            return self._get_offline_item(idx)
        if self.listwise:
            return self._get_online_listwise_item(idx)
        attempts = 0
        while attempts < 5:
            sample = self._samples[idx]
            video_id = sample["video_id"]
            exp_id = sample["exp_id"]
            expression = sample["expression"]

            video_path = os.path.join(self.mevis_root, "JPEGImages", video_id)
            if not os.path.exists(video_path):
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            category_info = self._get_category_info(video_id, exp_id, expression, video_path)
            category = category_info["category"]
            all_masks = self._get_all_masks(
                video_id,
                exp_id,
                video_path,
                category,
                backup_categories=category_info.get("backup_categories", []),
                expression=expression,
            )
            if not all_masks:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            gt_masks = self._load_expression_gt_masks(video_id, exp_id)
            if not gt_masks:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            iou_map = self._compute_iou_map(all_masks, gt_masks)
            if not iou_map:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            obj_id, label = self._select_candidate(iou_map)
            obj_iou = float(iou_map.get(obj_id, 0.0))

            frames_with_obj = [f for f in sorted(all_masks.keys()) if obj_id in all_masks[f]]
            if not frames_with_obj:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            frame_indices = _sample_indices(frames_with_obj, self.num_frames)

            frame_files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                frame_files.extend(list(sorted(glob.glob(os.path.join(video_path, ext)))))
            if not frame_files:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            frames = []
            masks = []
            for frame_idx in frame_indices:
                if frame_idx >= len(frame_files):
                    continue
                frame = np.array(Image.open(frame_files[frame_idx]).convert("RGB"))
                mask = all_masks[frame_idx][obj_id]
                masks.append(mask)
                if self.mask_style == "bbox":
                    annotated = _draw_bbox(frame, mask)
                else:
                    annotated = _overlay_mask(frame, mask, alpha=0.5)
                frames.append(Image.fromarray(annotated))

            if not frames:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            return {
                "dataset_name": self.dataset_name,
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": expression,
                "frames": frames,
                "masks": masks,
                "label": label,
                "obj_id": obj_id,
                "obj_iou": obj_iou,
                "frame_indices": frame_indices,
            }

        raise RuntimeError("Failed to sample a valid training item after multiple attempts.")

    def _get_offline_item(self, idx: int) -> Dict:
        attempts = 0
        while attempts < 5:
            sample = self._samples[idx]
            video_id = sample["video_id"]
            exp_id = sample["exp_id"]

            meta = self._load_offline_meta(video_id, exp_id)
            if not meta:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            expression = meta.get("expression", sample.get("expression", ""))
            iou_map_raw = meta.get("iou_map", {})
            iou_map = {int(k): float(v) for k, v in iou_map_raw.items()}
            if not iou_map:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            obj_id, label = self._select_candidate(iou_map)
            obj_iou = float(iou_map.get(obj_id, 0.0))
            frame_indices_map = meta.get("frame_indices", {})
            frames_with_obj = frame_indices_map.get(str(obj_id)) or frame_indices_map.get(obj_id)
            if not frames_with_obj:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            frame_indices = _sample_indices([int(x) for x in frames_with_obj], self.num_frames)
            video_path = meta.get("video_path") or os.path.join(self.mevis_root, "JPEGImages", video_id)
            frame_files = _list_frame_files(video_path)
            if not frame_files:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            frame_pad = int(meta.get("frame_pad", 0))
            mask_dir = os.path.join(self._offline_exp_dir(video_id, exp_id), "masks", f"obj_{obj_id}")
            if not os.path.isdir(mask_dir):
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            frames = []
            masks = []
            for frame_idx in frame_indices:
                if frame_idx >= len(frame_files):
                    continue
                if frame_pad > 0:
                    mask_name = f"{frame_idx:0{frame_pad}d}.png"
                else:
                    mask_name = f"{frame_idx}.png"
                mask_path = os.path.join(mask_dir, mask_name)
                if not os.path.exists(mask_path):
                    continue
                frame = np.array(Image.open(frame_files[frame_idx]).convert("RGB"))
                mask = np.array(Image.open(mask_path))
                mask = (mask > 0).astype(np.uint8)
                masks.append(mask)
                if self.mask_style == "bbox":
                    annotated = _draw_bbox(frame, mask)
                else:
                    annotated = _overlay_mask(frame, mask, alpha=0.5)
                frames.append(Image.fromarray(annotated))

            if not frames:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            return {
                "dataset_name": self.dataset_name,
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": expression,
                "frames": frames,
                "masks": masks,
                "label": label,
                "obj_id": obj_id,
                "obj_iou": obj_iou,
                "frame_indices": frame_indices,
            }

        raise RuntimeError("Failed to sample a valid offline item after multiple attempts.")

    def _get_offline_listwise_item(self, idx: int) -> Dict:
        attempts = 0
        while attempts < 5:
            sample = self._samples[idx]
            video_id = sample["video_id"]
            exp_id = sample["exp_id"]

            meta = self._load_offline_meta(video_id, exp_id)
            if not meta:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            expression = meta.get("expression", sample.get("expression", ""))
            iou_map_raw = meta.get("iou_map", {})
            iou_map = {int(k): float(v) for k, v in iou_map_raw.items()}
            if not iou_map:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            obj_ids = meta.get("obj_ids") or list(iou_map.keys())
            obj_ids = [int(obj_id) for obj_id in obj_ids]
            obj_ids = self._trim_candidate_ids(obj_ids, iou_map)
            if not obj_ids:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            video_path = meta.get("video_path") or os.path.join(self.mevis_root, "JPEGImages", video_id)
            frame_files = _list_frame_files(video_path)
            if not frame_files:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            frame_indices_map = meta.get("frame_indices", {})
            frame_pad = int(meta.get("frame_pad", 0))
            frame_cache: Dict[int, np.ndarray] = {}
            candidates = []

            for obj_id in obj_ids:
                frames_with_obj = frame_indices_map.get(str(obj_id)) or frame_indices_map.get(obj_id)
                if not frames_with_obj:
                    continue
                frame_indices = _sample_indices([int(x) for x in frames_with_obj], self.num_frames)
                mask_dir = os.path.join(self._offline_exp_dir(video_id, exp_id), "masks", f"obj_{obj_id}")
                if not os.path.isdir(mask_dir):
                    continue

                frames = []
                masks = []
                for frame_idx in frame_indices:
                    if frame_idx >= len(frame_files):
                        continue
                    if frame_pad > 0:
                        mask_name = f"{frame_idx:0{frame_pad}d}.png"
                    else:
                        mask_name = f"{frame_idx}.png"
                    mask_path = os.path.join(mask_dir, mask_name)
                    if not os.path.exists(mask_path):
                        continue
                    frame = frame_cache.get(frame_idx)
                    if frame is None:
                        frame = np.array(Image.open(frame_files[frame_idx]).convert("RGB"))
                        frame_cache[frame_idx] = frame
                    mask = np.array(Image.open(mask_path))
                    mask = (mask > 0).astype(np.uint8)
                    masks.append(mask)
                    if self.mask_style == "bbox":
                        annotated = _draw_bbox(frame, mask)
                    else:
                        annotated = _overlay_mask(frame, mask, alpha=0.5)
                    frames.append(Image.fromarray(annotated))

                if not frames:
                    continue

                candidates.append(
                    {
                        "obj_id": obj_id,
                        "frames": frames,
                        "masks": masks,
                        "obj_iou": float(iou_map.get(obj_id, 0.0)),
                        "frame_indices": frame_indices,
                    }
                )

            if not candidates:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            pos_obj_ids = [c["obj_id"] for c in candidates if c["obj_iou"] >= self.pos_iou]
            neg_obj_ids = [c["obj_id"] for c in candidates if c["obj_iou"] <= self.neg_iou]

            return {
                "dataset_name": self.dataset_name,
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": expression,
                "candidates": candidates,
                "pos_obj_ids": pos_obj_ids,
                "neg_obj_ids": neg_obj_ids,
                "iou_map": iou_map,
                "listwise_neg_policy": self.listwise_neg_policy,
                "hard_neg_weight": self.hard_neg_weight,
                "hard_neg_margin": self.hard_neg_margin,
                "hard_neg_pos_strategy": self.hard_neg_pos_strategy,
            }

        raise RuntimeError("Failed to sample a valid offline listwise item after multiple attempts.")

    def _get_online_listwise_item(self, idx: int) -> Dict:
        attempts = 0
        while attempts < 5:
            sample = self._samples[idx]
            video_id = sample["video_id"]
            exp_id = sample["exp_id"]
            expression = sample["expression"]

            video_path = os.path.join(self.mevis_root, "JPEGImages", video_id)
            if not os.path.exists(video_path):
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            category_info = self._get_category_info(video_id, exp_id, expression, video_path)
            category = category_info["category"]
            all_masks = self._get_all_masks(
                video_id,
                exp_id,
                video_path,
                category,
                backup_categories=category_info.get("backup_categories", []),
                expression=expression,
            )
            if not all_masks:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            gt_masks = self._load_expression_gt_masks(video_id, exp_id)
            if not gt_masks:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            iou_map = self._compute_iou_map(all_masks, gt_masks)
            if not iou_map:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            obj_ids = self._trim_candidate_ids(list(iou_map.keys()), iou_map)
            if not obj_ids:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            frame_files = _list_frame_files(video_path)
            if not frame_files:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            frame_cache: Dict[int, np.ndarray] = {}
            candidates = []
            for obj_id in obj_ids:
                frames_with_obj = [f for f in sorted(all_masks.keys()) if obj_id in all_masks[f]]
                if not frames_with_obj:
                    continue
                frame_indices = _sample_indices(frames_with_obj, self.num_frames)
                frames = []
                masks = []
                for frame_idx in frame_indices:
                    if frame_idx >= len(frame_files):
                        continue
                    frame = frame_cache.get(frame_idx)
                    if frame is None:
                        frame = np.array(Image.open(frame_files[frame_idx]).convert("RGB"))
                        frame_cache[frame_idx] = frame
                    mask = all_masks[frame_idx][obj_id]
                    masks.append(mask)
                    if self.mask_style == "bbox":
                        annotated = _draw_bbox(frame, mask)
                    else:
                        annotated = _overlay_mask(frame, mask, alpha=0.5)
                    frames.append(Image.fromarray(annotated))

                if not frames:
                    continue

                candidates.append(
                    {
                        "obj_id": obj_id,
                        "frames": frames,
                        "masks": masks,
                        "obj_iou": float(iou_map.get(obj_id, 0.0)),
                        "frame_indices": frame_indices,
                    }
                )

            if not candidates:
                idx = random.randint(0, len(self._samples) - 1)
                attempts += 1
                continue

            pos_obj_ids = [c["obj_id"] for c in candidates if c["obj_iou"] >= self.pos_iou]
            neg_obj_ids = [c["obj_id"] for c in candidates if c["obj_iou"] <= self.neg_iou]

            return {
                "dataset_name": self.dataset_name,
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": expression,
                "candidates": candidates,
                "pos_obj_ids": pos_obj_ids,
                "neg_obj_ids": neg_obj_ids,
                "iou_map": iou_map,
                "listwise_neg_policy": self.listwise_neg_policy,
                "hard_neg_weight": self.hard_neg_weight,
                "hard_neg_margin": self.hard_neg_margin,
                "hard_neg_pos_strategy": self.hard_neg_pos_strategy,
            }

        raise RuntimeError("Failed to sample a valid online listwise item after multiple attempts.")

    def prepare_offline_cache(self, cache_root: str = None, overwrite: bool = False) -> int:
        if self._sam3 is None:
            raise RuntimeError("Offline cache preparation requires online mode (offline_mode=false).")

        cache_root = cache_root or self.offline_cache_root
        if not os.path.isabs(cache_root):
            cache_root = os.path.join(self._repo_root, cache_root)
        cache_root = os.path.abspath(cache_root)
        split_root = os.path.join(cache_root, str(self.split))
        os.makedirs(split_root, exist_ok=True)

        prepared = 0
        skip_counts: Dict[str, int] = {}

        def _mark_skip(reason: str, video_id: str, exp_id: int) -> None:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            print(f"[OfflinePrepare][skip:{reason}] {video_id}/{exp_id}")

        for sample in self._samples:
            video_id = sample["video_id"]
            exp_id = sample["exp_id"]
            expression = sample["expression"]

            exp_dir = os.path.join(split_root, video_id, str(exp_id))
            meta_path = os.path.join(exp_dir, "meta.json")
            if os.path.exists(meta_path) and not overwrite:
                continue

            video_path = os.path.join(self.mevis_root, "JPEGImages", video_id)
            if not os.path.exists(video_path):
                _mark_skip("missing_video_path", video_id, exp_id)
                continue
            if self.max_video_frames is not None:
                frame_files = _list_frame_files(video_path)
                if len(frame_files) > int(self.max_video_frames):
                    _mark_skip("exceeds_max_video_frames", video_id, exp_id)
                    continue

            category_info = self._get_category_info(video_id, exp_id, expression, video_path)
            category = category_info["category"]
            backup_categories = category_info.get("backup_categories", [])
            try:
                all_masks = self._get_all_masks(
                    video_id,
                    exp_id,
                    video_path,
                    category,
                    backup_categories=backup_categories,
                    expression=expression,
                )
            except RuntimeError as err:
                if _is_oom_runtime_error(err):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        f"[OfflinePrepare][oom-skip] Skip {video_id}/{exp_id} "
                        "after fallback retry."
                    )
                    _mark_skip("sam3_oom_after_retry", video_id, exp_id)
                    continue
                raise
            if not all_masks:
                _mark_skip("empty_sam3_masks", video_id, exp_id)
                continue

            gt_masks = self._load_expression_gt_masks(video_id, exp_id)
            if not gt_masks:
                _mark_skip("empty_gt_masks", video_id, exp_id)
                continue

            iou_map = self._compute_iou_map(all_masks, gt_masks)
            if not iou_map:
                _mark_skip("empty_iou_map", video_id, exp_id)
                continue

            frame_pad = _get_frame_pad(video_path)
            mask_root = os.path.join(exp_dir, "masks")
            os.makedirs(mask_root, exist_ok=True)

            frame_indices_map: Dict[int, List[int]] = {}
            for obj_id in sorted(iou_map.keys()):
                frames_with_obj = [f for f in sorted(all_masks.keys()) if obj_id in all_masks[f]]
                if not frames_with_obj:
                    continue
                frame_indices_map[obj_id] = frames_with_obj
                obj_dir = os.path.join(mask_root, f"obj_{obj_id}")
                os.makedirs(obj_dir, exist_ok=True)
                for frame_idx in frames_with_obj:
                    mask = all_masks[frame_idx][obj_id]
                    mask_img = (mask > 0).astype(np.uint8) * 255
                    if frame_pad > 0:
                        mask_name = f"{frame_idx:0{frame_pad}d}.png"
                    else:
                        mask_name = f"{frame_idx}.png"
                    Image.fromarray(mask_img).save(os.path.join(obj_dir, mask_name))

            if not frame_indices_map:
                _mark_skip("empty_frame_indices_map", video_id, exp_id)
                continue

            iou_summary = summarize_iou_map(iou_map)
            meta = {
                "dataset_name": self.dataset_name,
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": expression,
                "category": category,
                "sam3_prompt_source": self._sam3_prompt_meta.get((video_id, exp_id), {}).get("source", "category"),
                "sam3_prompt_text": self._sam3_prompt_meta.get((video_id, exp_id), {}).get("text", category),
                "video_path": video_path,
                "frame_pad": frame_pad,
                "obj_ids": sorted([int(x) for x in frame_indices_map.keys()]),
                "iou_map": {str(k): float(v) for k, v in iou_map.items()},
                "max_j": iou_summary["max_j"],
                "sum_j": iou_summary["sum_j"],
                "mean_j": iou_summary["mean_j"],
                "num_candidates": iou_summary["num_candidates"],
                "frame_indices": {str(k): v for k, v in frame_indices_map.items()},
            }
            os.makedirs(exp_dir, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            prepared += 1

        if skip_counts:
            print(f"[OfflinePrepare] prepared={prepared} skipped={sum(skip_counts.values())} details={skip_counts}")

        return prepared
