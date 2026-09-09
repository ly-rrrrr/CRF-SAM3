#!/usr/bin/env python3
"""Evaluate mask-guided SigLIP patch retrieval on selected ReVOS samples.

This tests scheme B:
  full image -> SigLIP patch tokens -> mask-weighted region feature -> text retrieval.
It is a diagnostic script, not part of the offline cache pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModel, AutoProcessor

DEFAULT_REVOS_ROOT = Path("/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos")
DEFAULT_META = DEFAULT_REVOS_ROOT / "meta_expressions_train_.json"
DEFAULT_MASK_DICT = DEFAULT_REVOS_ROOT / "mask_dict.json"
DEFAULT_VOCAB = Path("category_grounding_mvp/data/openimages_vocab/processed_20260703/categories.jsonl")
DEFAULT_MODEL = Path("siglip2-base-patch16-224")
DEFAULT_OUTPUT = Path("category_grounding_mvp/outputs/siglip_mask_patch_retrieval_debug")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def decode_rle(rle: Dict[str, Any] | None) -> np.ndarray | None:
    if not rle:
        return None
    x = mask_utils.decode(rle)
    if x.ndim == 3:
        x = x.any(axis=2)
    return x.astype(bool)


def merge_mask(mask_dict: Dict[str, Any], anno_ids: Sequence[int], frame_idx: int) -> np.ndarray | None:
    out = None
    for anno_id in anno_ids:
        seq = mask_dict.get(str(anno_id)) or mask_dict.get(anno_id)
        if seq is None or frame_idx >= len(seq):
            continue
        m = decode_rle(seq[frame_idx])
        if m is None:
            continue
        out = m if out is None else np.logical_or(out, m)
    return out


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def visibility_score(mask: np.ndarray) -> float:
    box = mask_bbox(mask)
    if box is None:
        return 0.0
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    area = float(mask.sum())
    fill = area / max(1.0, float(bw * bh))
    area_ratio = area / max(1.0, float(h * w))
    edge = float(x0 <= 1 or y0 <= 1 or x1 >= w - 1 or y1 >= h - 1)
    return math.log1p(area) + 2.0 * math.sqrt(max(area_ratio, 0.0)) + 0.5 * fill - 0.75 * edge


def parse_sample(sample: str) -> tuple[str, str]:
    parts = sample.strip().split("/")
    if len(parts) < 2:
        raise ValueError(f"sample should be video_id/exp_id, got: {sample}")
    return "/".join(parts[:-1]), parts[-1]


def frame_path(image_root: Path, video_id: str, frame_name: str) -> Path:
    video_dir = image_root / video_id
    for suffix in (".jpg", ".jpeg", ".png"):
        p = video_dir / f"{frame_name}{suffix}"
        if p.exists():
            return p
    return video_dir / f"{frame_name}.jpg"


def make_crop(image: Image.Image, mask: np.ndarray, padding_ratio: float) -> Image.Image:
    box = mask_bbox(mask)
    if box is None:
        return image.convert("RGB")
    x0, y0, x1, y1 = box
    w, h = image.size
    pad = int(round(max(x1 - x0, y1 - y0) * padding_ratio))
    return image.convert("RGB").crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))


def save_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    arr = np.asarray(image.convert("RGB")).copy().astype(np.float32)
    arr[mask] = arr[mask] * 0.45 + np.array([255, 48, 48], np.float32) * 0.55
    box = mask_bbox(mask)
    if box is not None:
        x0, y0, x1, y1 = box
        h, w = arr.shape[:2]
        arr[max(0, y0 - 2):min(h, y0 + 2), x0:x1] = [255, 255, 0]
        arr[max(0, y1 - 2):min(h, y1 + 2), x0:x1] = [255, 255, 0]
        arr[y0:y1, max(0, x0 - 2):min(w, x0 + 2)] = [255, 255, 0]
        arr[y0:y1, max(0, x1 - 2):min(w, x1 + 2)] = [255, 255, 0]
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def load_vocab(path: Path, text_field: str) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            text = (x.get(text_field) or x.get("display_prompt_text") or x.get("display_name") or "").strip()
            if not text:
                continue
            rows.append({**x, "retrieval_text": text})
    return rows


@torch.inference_mode()
def encode_texts(model, processor, texts: Sequence[str], device: torch.device, template: str, batch_size: int) -> torch.Tensor:
    feats = []
    prompted = [template.format(t) for t in texts]
    for s in range(0, len(prompted), batch_size):
        inp = processor(text=prompted[s:s + batch_size], padding=True, truncation=True, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        feat = model.get_text_features(**inp).float()
        feats.append(torch.nn.functional.normalize(feat, dim=-1).cpu())
    return torch.cat(feats, dim=0)


def mask_patch_weights(mask: np.ndarray, image_size: tuple[int, int], grid: int) -> torch.Tensor:
    # SigLIP processor stretches images to 224x224, so resize mask the same way.
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(mask_img).astype(np.float32) / 255.0
    patch_h = image_size[1] // grid
    patch_w = image_size[0] // grid
    arr = arr[:grid * patch_h, :grid * patch_w]
    weights = arr.reshape(grid, patch_h, grid, patch_w).mean(axis=(1, 3))
    flat = torch.from_numpy(weights.reshape(-1)).float()
    if float(flat.sum()) <= 1e-8:
        # fallback to bbox-center nearest patch if soft resize vanishes
        box = mask_bbox(mask)
        if box is not None:
            x0, y0, x1, y1 = box
            ow, oh = mask.shape[1], mask.shape[0]
            cx = ((x0 + x1) / 2.0) / max(1, ow) * grid
            cy = ((y0 + y1) / 2.0) / max(1, oh) * grid
            idx = min(grid - 1, max(0, int(cy))) * grid + min(grid - 1, max(0, int(cx)))
            flat[idx] = 1.0
    return flat


def top_scores(scores: torch.Tensor, vocab: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    vals, idxs = torch.topk(scores.detach().cpu(), k=min(top_k, scores.numel()))
    out = []
    for val, idx in zip(vals.tolist(), idxs.tolist()):
        item = vocab[idx]
        out.append({
            "category_name": item.get("display_name"),
            "retrieval_text": item.get("retrieval_text"),
            "sam3_prompt_text": item.get("sam3_prompt_text"),
            "mid": item.get("mid"),
            "score": float(val),
        })
    return out


def rank_of(scores: torch.Tensor, vocab: Sequence[Dict[str, Any]], names: Sequence[str]) -> List[Dict[str, Any]]:
    order = torch.argsort(scores.detach().cpu(), descending=True)
    ranks = []
    for name in names:
        idx = next((i for i, x in enumerate(vocab) if x.get("display_name") == name), None)
        if idx is None:
            continue
        pos = int((order == idx).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append({"category_name": name, "rank": pos, "score": float(scores.detach().cpu()[idx])})
    return ranks


@torch.inference_mode()
def evaluate_sample(args, model, processor, text_features, vocab, meta, mask_dict, sample: str, device: torch.device) -> Dict[str, Any]:
    video_id, exp_id = parse_sample(sample)
    video = meta["videos"][video_id]
    exp = video["expressions"][str(exp_id)]
    frames = [str(x) for x in video.get("frames", [])]
    anno_ids = [int(x) for x in (exp.get("anno_id") or exp.get("obj_id") or [])]
    visible = []
    for idx in range(len(frames)):
        mask = merge_mask(mask_dict, anno_ids, idx)
        if mask is not None and int(mask.sum()) >= args.min_area:
            visible.append((idx, mask, visibility_score(mask)))
    if not visible:
        raise RuntimeError(f"no visible mask for {sample}")
    visible.sort(key=lambda x: x[2], reverse=True)
    frame_idx, mask, qscore = visible[0]
    frame_name = frames[frame_idx]
    image = Image.open(frame_path(args.image_root, video_id, frame_name)).convert("RGB")
    crop = make_crop(image, mask, args.padding_ratio)

    img_inputs = processor(images=[image], return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(device)
    vision_out = model.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=args.interpolate_pos_encoding)
    tokens = vision_out.last_hidden_state.float().squeeze(0)
    grid = int(round(math.sqrt(tokens.shape[0])))
    if grid * grid != tokens.shape[0]:
        raise RuntimeError(f"unexpected token count: {tokens.shape[0]}")
    proc_size = processor.image_processor.size
    if isinstance(proc_size, dict):
        mask_size = (int(proc_size.get("width", args.image_size)), int(proc_size.get("height", args.image_size)))
    else:
        mask_size = (int(args.image_size), int(args.image_size))
    weights = mask_patch_weights(mask, mask_size, grid).to(device)
    weights = weights / weights.sum().clamp_min(1e-6)
    token_norm = torch.nn.functional.normalize(tokens, dim=-1)
    avg_feat = torch.nn.functional.normalize((token_norm * weights[:, None]).sum(dim=0, keepdim=True), dim=-1)
    sim_per_patch = token_norm @ text_features.to(device).T
    avg_scores = (avg_feat @ text_features.to(device).T).squeeze(0)
    max_scores = sim_per_patch[weights > args.patch_weight_threshold].max(dim=0).values if bool((weights > args.patch_weight_threshold).any()) else sim_per_patch.max(dim=0).values
    k = min(args.mask_topk_patches, int((weights > 0).sum().item()))
    top_patch_idx = torch.topk(weights, k=max(1, k)).indices
    topk_scores = sim_per_patch[top_patch_idx].mean(dim=0)

    crop_inputs = processor(images=[crop], return_tensors="pt")
    crop_inputs = {k: v.to(device) for k, v in crop_inputs.items()}
    crop_feat = torch.nn.functional.normalize(model.get_image_features(**crop_inputs, interpolate_pos_encoding=args.interpolate_pos_encoding).float(), dim=-1)
    crop_scores = (crop_feat @ text_features.to(device).T).squeeze(0)
    full_feat = torch.nn.functional.normalize(model.get_image_features(pixel_values=pixel_values, interpolate_pos_encoding=args.interpolate_pos_encoding).float(), dim=-1)
    full_scores = (full_feat @ text_features.to(device).T).squeeze(0)

    box = mask_bbox(mask)
    out_dir = args.output_dir / sample.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_overlay(image, mask, out_dir / "overlay.jpg")
    crop.save(out_dir / "crop.jpg")

    watch = [
        "Person", "Man", "Woman", "Boy", "Girl", "Human body", "Human head",
        "Dog", "Cat", "Motorcycle", "Bicycle", "Bird", "Duck",
        "Wood-burning stove", "Salt and pepper shakers",
    ]
    result = {
        "sample": sample,
        "video_id": video_id,
        "exp_id": str(exp_id),
        "expression": exp.get("exp"),
        "anno_ids": anno_ids,
        "selected_frame_index": frame_idx,
        "selected_frame": frame_name,
        "visibility_score": qscore,
        "image_size": list(image.size),
        "bbox_xyxy": list(box) if box else None,
        "mask_area": int(mask.sum()),
        "mask_area_ratio": float(mask.sum() / max(1, mask.size)),
        "patch_grid": [grid, grid],
        "mask_patch_nonzero": int((weights > 0).sum().item()),
        "mask_patch_effective": float(1.0 / (weights.square().sum().item() + 1e-12)),
        "top_weighted_patches": [(int(i), float(weights[i])) for i in torch.topk(weights.detach().cpu(), k=min(10, weights.numel())).indices.tolist()],
        "methods": {
            "scheme_b_masked_avg_patch": {
                "top": top_scores(avg_scores, vocab, args.top_k),
                "watch_ranks": rank_of(avg_scores, vocab, watch),
            },
            "scheme_b_masked_max_patch": {
                "top": top_scores(max_scores, vocab, args.top_k),
                "watch_ranks": rank_of(max_scores, vocab, watch),
            },
            "scheme_b_masked_topk_patch_mean": {
                "top": top_scores(topk_scores, vocab, args.top_k),
                "watch_ranks": rank_of(topk_scores, vocab, watch),
            },
            "baseline_crop_pooled": {
                "top": top_scores(crop_scores, vocab, args.top_k),
                "watch_ranks": rank_of(crop_scores, vocab, watch),
            },
            "baseline_full_image_pooled": {
                "top": top_scores(full_scores, vocab, args.top_k),
                "watch_ranks": rank_of(full_scores, vocab, watch),
            },
        },
        "debug_overlay": str(out_dir / "overlay.jpg"),
        "debug_crop": str(out_dir / "crop.jpg"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--meta-expressions", type=Path, default=DEFAULT_META)
    ap.add_argument("--mask-dict", type=Path, default=DEFAULT_MASK_DICT)
    ap.add_argument("--image-root", type=Path, default=DEFAULT_REVOS_ROOT)
    ap.add_argument("--vocab-jsonl", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--vocab-text-field", default="display_prompt_text")
    ap.add_argument("--prompt-template", default="a photo of a {}.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=224, help="SigLIP square image size for full-image patch tokens.")
    ap.add_argument("--text-batch-size", type=int, default=256)
    ap.add_argument("--min-area", type=int, default=16)
    ap.add_argument("--padding-ratio", type=float, default=0.15)
    ap.add_argument("--mask-topk-patches", type=int, default=4)
    ap.add_argument("--patch-weight-threshold", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    ap.add_argument("--interpolate-pos-encoding", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    vocab = load_vocab(args.vocab_jsonl, args.vocab_text_field)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.image_processor.size = {"height": args.image_size, "width": args.image_size}
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    texts = [x["retrieval_text"] for x in vocab]
    text_features = encode_texts(model, processor, texts, device, args.prompt_template, args.text_batch_size)
    meta = read_json(args.meta_expressions)
    mask_dict = read_json(args.mask_dict)
    results = []
    for sample in args.samples:
        result = evaluate_sample(args, model, processor, text_features, vocab, meta, mask_dict, sample, device)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
