#!/usr/bin/env python3
"""Build ReVOS pseudo categories from GT-mask-guided region retrieval.

This is an offline label-construction utility. It does not ask a VLM to freely
name objects. Instead, each visible GT mask crop is encoded by SigLIP/CLIP-like
image features and retrieved against a fixed LVIS visual noun vocabulary.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


DEFAULT_REVOS_ROOT = "/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos"
DEFAULT_META = f"{DEFAULT_REVOS_ROOT}/meta_expressions_train_.json"
DEFAULT_MASK_DICT = f"{DEFAULT_REVOS_ROOT}/mask_dict.json"
DEFAULT_VOCAB = "category_grounding_mvp/data/visual_vocab/lvis_v1/categories.jsonl"
DEFAULT_MODEL = "siglip2-base-patch16-224"
DEFAULT_OUTPUT = "category_grounding_mvp/data/revos_siglip2_lvis_pseudo_category_20260703"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_rle(rle: Dict[str, Any] | None) -> np.ndarray | None:
    if not rle:
        return None
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def mask_area_from_rle(rle: Dict[str, Any] | None) -> int:
    if not rle:
        return 0
    try:
        return int(mask_utils.area(rle))
    except Exception:
        decoded = decode_rle(rle)
        return int(decoded.sum()) if decoded is not None else 0


def sample_evenly(indices: Sequence[int], k: int) -> List[int]:
    if not indices:
        return []
    if len(indices) <= k:
        return list(indices)
    return [indices[round(i * (len(indices) - 1) / (k - 1))] for i in range(k)]


def mask_visibility_score(mask: np.ndarray) -> float:
    """Score whether a GT mask frame is a clear crop candidate.

    This is not a semantic rule. It only prefers frames where the target mask is
    visibly large, compact inside its bbox, and not truncated by image borders.
    """
    bbox = mask_bbox(mask)
    if bbox is None:
        return 0.0
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    area = float(mask.sum())
    bbox_area = float(bw * bh)
    fill_ratio = area / max(1.0, bbox_area)
    area_ratio = area / max(1.0, float(h * w))
    edge_touch = float(x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1)

    return math.log1p(area) + 2.0 * math.sqrt(max(area_ratio, 0.0)) + 0.5 * fill_ratio - 0.75 * edge_touch


def select_visible_frames(
    visible_masks: Sequence[tuple[int, np.ndarray]],
    k: int,
    mode: str,
) -> tuple[List[int], Dict[int, float]]:
    if not visible_masks:
        return [], {}
    if mode == "even":
        selected = sample_evenly([idx for idx, _ in visible_masks], k)
        selected_set = set(selected)
        return selected, {idx: mask_visibility_score(mask) for idx, mask in visible_masks if idx in selected_set}

    scored = [(idx, mask_visibility_score(mask)) for idx, mask in visible_masks]
    scored.sort(key=lambda x: x[1], reverse=True)
    if mode == "best_quality":
        selected = [scored[0][0]]
    elif mode == "topk_quality":
        selected = [idx for idx, _ in scored[:k]]
    else:
        raise ValueError(f"unsupported frame selection mode: {mode}")
    selected.sort()
    return selected, dict(scored)


def frame_path(root: Path, video_id: str, frame_name: str) -> Path:
    video_dir = root / video_id
    for suffix in (".jpg", ".jpeg", ".png"):
        path = video_dir / f"{frame_name}{suffix}"
        if path.exists():
            return path
    return video_dir / f"{frame_name}.jpg"


def merge_mask_at_frame(mask_dict: Dict[str, Any], anno_ids: Sequence[int], frame_idx: int) -> np.ndarray | None:
    merged = None
    for anno_id in anno_ids:
        seq = mask_dict.get(str(anno_id))
        if seq is None:
            seq = mask_dict.get(anno_id)
        if seq is None or frame_idx >= len(seq):
            continue
        decoded = decode_rle(seq[frame_idx])
        if decoded is None:
            continue
        merged = decoded if merged is None else np.logical_or(merged, decoded)
    return merged


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def make_mask_aware_crop(
    image: Image.Image,
    mask: np.ndarray,
    padding_ratio: float,
    background: str,
) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb).copy()
    h, w = arr.shape[:2]
    bbox = mask_bbox(mask)
    if bbox is None:
        return rgb

    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad = int(round(max(bw, bh) * padding_ratio))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)

    crop = arr[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    if background == "gray":
        crop[~crop_mask] = 127
    elif background == "dark":
        crop[~crop_mask] = (crop[~crop_mask] * 0.15).astype(np.uint8)
    elif background == "original":
        pass
    else:
        raise ValueError(f"unsupported background mode: {background}")
    return Image.fromarray(crop)


def iter_vocab(vocab_jsonl: Path, include_variants: bool, text_field: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with vocab_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            variants = item.get("variants") or []
            category_id = item.get("category_id")
            if category_id is None:
                category_id = item.get("mid")
            category_name = item.get("name")
            if category_name is None:
                category_name = item.get("display_name")
            category_prompt = (
                item.get("sam3_prompt_text")
                or item.get("prompt_text")
                or item.get("display_prompt_text")
                or category_name
                or ""
            )
            if include_variants and variants:
                texts = [str(v.get("prompt_text", "")).strip() for v in variants]
            else:
                texts = [
                    str(
                        item.get(text_field)
                        or item.get("prompt_text")
                        or item.get("display_prompt_text")
                        or item.get("name")
                        or item.get("display_name")
                        or ""
                    ).strip()
                ]
            seen = set()
            for text in texts:
                key = text.lower()
                if not text or key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "category_id": category_id,
                        "category_name": category_name,
                        "category_prompt": category_prompt,
                        "display_prompt_text": item.get("display_prompt_text"),
                        "sam3_prompt_text": item.get("sam3_prompt_text"),
                        "variant_text": text,
                        "prompt_candidates": item.get("prompt_candidates"),
                        "parent_names": item.get("parent_names"),
                        "child_names": item.get("child_names"),
                        "frequency": item.get("frequency"),
                    }
                )
    return records


@torch.inference_mode()
def encode_texts(
    model: torch.nn.Module,
    processor: Any,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    prompt_template: str,
) -> torch.Tensor:
    features = []
    prompted = [prompt_template.format(t) for t in texts]
    for start in tqdm(range(0, len(prompted), batch_size), desc="Encoding LVIS texts"):
        batch = prompted[start : start + batch_size]
        inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)


@torch.inference_mode()
def encode_images(
    model: torch.nn.Module,
    processor: Any,
    images: Sequence[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(images=list(images), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(feats.float(), dim=-1)


def top_category_results(
    image_features: torch.Tensor,
    text_features_cpu: torch.Tensor,
    vocab: Sequence[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    text_features = text_features_cpu.to(image_features.device)
    sims = image_features @ text_features.T
    variant_scores = sims.mean(dim=0).detach().cpu().numpy()

    category_best: Dict[str, Dict[str, Any]] = {}
    for idx, score in enumerate(variant_scores.tolist()):
        item = vocab[idx]
        cat_key = str(item["category_id"])
        prev = category_best.get(cat_key)
        if prev is None or score > prev["score"]:
            category_best[cat_key] = {
                "category_id": item["category_id"],
                "category_name": item["category_name"],
                "category_prompt": item["category_prompt"],
                "display_prompt_text": item.get("display_prompt_text"),
                "sam3_prompt_text": item.get("sam3_prompt_text"),
                "variant_text": item["variant_text"],
                "prompt_candidates": item.get("prompt_candidates"),
                "parent_names": item.get("parent_names"),
                "child_names": item.get("child_names"),
                "score": float(score),
                "frequency": item.get("frequency"),
            }

    top_categories = sorted(category_best.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    top_variant_indices = np.argsort(-variant_scores)[:top_k].tolist()
    top_variants = []
    for idx in top_variant_indices:
        item = vocab[idx]
        top_variants.append(
            {
                "category_id": item["category_id"],
                "category_name": item["category_name"],
                "category_prompt": item["category_prompt"],
                "display_prompt_text": item.get("display_prompt_text"),
                "sam3_prompt_text": item.get("sam3_prompt_text"),
                "variant_text": item["variant_text"],
                "prompt_candidates": item.get("prompt_candidates"),
                "parent_names": item.get("parent_names"),
                "child_names": item.get("child_names"),
                "score": float(variant_scores[idx]),
                "frequency": item.get("frequency"),
            }
        )

    margin = 0.0
    if len(top_categories) >= 2:
        margin = float(top_categories[0]["score"] - top_categories[1]["score"])
    best = top_categories[0] if top_categories else {}
    return {
        # Strict pseudo label: one of the OpenImages boxable categories.
        "pseudo_category": best.get("category_name", ""),
        "pseudo_category_text": best.get("display_prompt_text") or best.get("variant_text", ""),
        "pseudo_category_id": best.get("category_id"),
        # SAM3-facing prompt is kept separate; it is not the pseudo label.
        "pseudo_sam3_prompt": best.get("sam3_prompt_text") or best.get("category_prompt", ""),
        "pseudo_variant": best.get("variant_text", ""),
        "score": best.get("score"),
        "margin_top1_top2": margin,
        "top_categories": top_categories,
        "top_variants": top_variants,
    }


def normalize_ids(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        return [int(value)]
    return [int(x) for x in value]


def load_done_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            sample_id = item.get("sample_id")
            if sample_id:
                done.add(sample_id)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-expressions", default=DEFAULT_META)
    parser.add_argument("--mask-dict", default=DEFAULT_MASK_DICT)
    parser.add_argument("--image-root", default=DEFAULT_REVOS_ROOT)
    parser.add_argument("--vocab-jsonl", default=DEFAULT_VOCAB)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-per-sample", type=int, default=5)
    parser.add_argument(
        "--frame-selection",
        choices=["even", "best_quality", "topk_quality"],
        default="even",
        help="How to choose visible GT-mask frames before crop retrieval.",
    )
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--padding-ratio", type=float, default=0.15)
    parser.add_argument("--background", choices=["gray", "dark", "original"], default="gray")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument(
        "--vocab-text-field",
        default="prompt_text",
        help="JSONL field used as the text retrieval label when variants are disabled.",
    )
    parser.add_argument("--prompt-template", default="a photo of a {}.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "pseudo_categories.jsonl"
    done_ids = load_done_ids(output_jsonl) if args.resume else set()

    meta = read_json(Path(args.meta_expressions))
    mask_dict = read_json(Path(args.mask_dict))
    vocab = iter_vocab(
        Path(args.vocab_jsonl),
        include_variants=args.include_variants,
        text_field=args.vocab_text_field,
    )
    if not vocab:
        raise ValueError(f"empty vocabulary: {args.vocab_jsonl}")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    print(
        json.dumps(
            {
                "event": "load_model",
                "model_path": args.model_path,
                "device": str(device),
                "dtype": str(dtype),
                "num_vocab_entries": len(vocab),
                "include_variants": args.include_variants,
                "vocab_text_field": args.vocab_text_field,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True, torch_dtype=dtype)
    model.to(device)
    model.eval()

    text_cache = output_dir / f"text_features_{Path(args.vocab_jsonl).stem}_{'variants' if args.include_variants else 'category'}.pt"
    if text_cache.exists():
        cache = torch.load(text_cache, map_location="cpu")
        text_features = cache["text_features"]
        cached_texts = cache["texts"]
        if cached_texts != [x["variant_text"] for x in vocab]:
            raise ValueError(f"text cache does not match current vocabulary: {text_cache}")
        print(json.dumps({"event": "loaded_text_cache", "path": str(text_cache)}, ensure_ascii=False), flush=True)
    else:
        text_features = encode_texts(
            model=model,
            processor=processor,
            texts=[x["variant_text"] for x in vocab],
            device=device,
            batch_size=args.text_batch_size,
            prompt_template=args.prompt_template,
        )
        torch.save({"texts": [x["variant_text"] for x in vocab], "text_features": text_features}, text_cache)
        print(json.dumps({"event": "saved_text_cache", "path": str(text_cache)}, ensure_ascii=False), flush=True)

    videos = meta.get("videos", {})
    total_expr = sum(len(v.get("expressions", {})) for v in videos.values())
    stats = defaultdict(int)
    processed = 0

    mode = "a" if args.resume else "w"
    with output_jsonl.open(mode, encoding="utf-8") as out:
        pbar = tqdm(videos.items(), desc="ReVOS videos")
        for video_id, video_meta in pbar:
            frames = [str(x) for x in video_meta.get("frames", [])]
            expressions = video_meta.get("expressions", {})
            for exp_id, exp_data in expressions.items():
                sample_id = f"revos:{video_id}:{exp_id}"
                if sample_id in done_ids:
                    stats["skipped_resume"] += 1
                    continue
                if args.max_samples and processed >= args.max_samples:
                    break

                expression = str(exp_data.get("exp", "")).strip()
                anno_ids = normalize_ids(exp_data.get("anno_id") or exp_data.get("obj_id"))
                record: Dict[str, Any] = {
                    "sample_id": sample_id,
                    "video_id": video_id,
                    "exp_id": str(exp_id),
                    "expression": expression,
                    "anno_ids": anno_ids,
                    "type_id": exp_data.get("type_id"),
                }

                if not frames or not anno_ids:
                    record.update({"status": "skipped", "skip_reason": "missing_frames_or_anno_ids"})
                    stats["missing_fields"] += 1
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                visible_masks = []
                for idx in range(len(frames)):
                    merged = merge_mask_at_frame(mask_dict, anno_ids, idx)
                    if merged is not None and int(merged.sum()) >= args.min_area:
                        visible_masks.append((idx, merged))
                selected, visibility_scores = select_visible_frames(
                    visible_masks,
                    args.frames_per_sample,
                    args.frame_selection,
                )
                if not selected:
                    record.update({"status": "skipped", "skip_reason": "no_visible_mask"})
                    stats["no_visible_mask"] += 1
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                crops: List[Image.Image] = []
                selected_frames: List[str] = []
                selected_areas: List[int] = []
                for idx in selected:
                    merged = merge_mask_at_frame(mask_dict, anno_ids, idx)
                    if merged is None:
                        continue
                    path = frame_path(Path(args.image_root), video_id, frames[idx])
                    if not path.exists():
                        continue
                    image = Image.open(path)
                    crops.append(
                        make_mask_aware_crop(
                            image=image,
                            mask=merged,
                            padding_ratio=args.padding_ratio,
                            background=args.background,
                        )
                    )
                    selected_frames.append(frames[idx])
                    selected_areas.append(int(merged.sum()))

                if not crops:
                    record.update({"status": "skipped", "skip_reason": "missing_images"})
                    stats["missing_images"] += 1
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                image_features = encode_images(model=model, processor=processor, images=crops, device=device)
                retrieval = top_category_results(
                    image_features=image_features,
                    text_features_cpu=text_features,
                    vocab=vocab,
                    top_k=args.top_k,
                )
                record.update(
                    {
                        "status": "ok",
                        "selected_frame_indices": selected,
                        "selected_frames": selected_frames,
                        "selected_mask_areas": selected_areas,
                        "selected_visibility_scores": [visibility_scores.get(idx) for idx in selected],
                        "frame_selection": args.frame_selection,
                        "num_selected_frames": len(crops),
                        **retrieval,
                    }
                )
                stats["ok"] += 1
                processed += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

            if args.max_samples and processed >= args.max_samples:
                break

    summary = {
        "mode": "revos_mask_crop_open_vocab_retrieval",
        "meta_expressions": args.meta_expressions,
        "mask_dict": args.mask_dict,
        "image_root": args.image_root,
        "vocab_jsonl": args.vocab_jsonl,
        "model_path": args.model_path,
        "output_jsonl": str(output_jsonl),
        "total_expressions": total_expr,
        "processed_ok": stats["ok"],
        "stats": dict(stats),
        "frames_per_sample": args.frames_per_sample,
        "frame_selection": args.frame_selection,
        "min_area": args.min_area,
        "padding_ratio": args.padding_ratio,
        "background": args.background,
        "top_k": args.top_k,
        "include_variants": args.include_variants,
        "vocab_text_field": args.vocab_text_field,
        "prompt_template": args.prompt_template,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
