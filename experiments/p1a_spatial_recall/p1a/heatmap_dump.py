#!/usr/bin/env python3
"""Stage 1 (GPU, Qwen3-VL only): dump Ground response maps for a list of samples.

Reuses the repository code unchanged (``category_grounding_mvp``):
``load_query_model`` / ``build_query_messages`` / ``build_query_context`` and a
forward hook on ``mask_head`` (same capture as visualize_ground_responses.py).
GT is never read here.

Output per sample: ``<out>/<video_id>/<exp_id>.npz`` with
  responses (K, N) float16 sigmoid maps, image_grid_thw (M, 3), merge,
  frame_indices (M,) SAM3 frame indices, frame_stems (M,), orig_hw (2,).

Example
  python -m p1a.heatmap_dump --repo /path/Sa2VA-main \
     --model-path .../Qwen3-VL-4B-Instruct --query-module-dir .../checkpoint-900 \
     --samples cache_reports/davis17_..._max_j_lt_0p5.json \
     --video-root .../davis17/train/JPEGImages --cache-root ... --split train \
     --frames-per-sample 3 --max-image-side 448 --out runs/heat_davis --bf16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .common import frame_path, list_frames, load_cache_meta, load_sample_list, sample_frame_indices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="Sa2VA-main root of the CRF-SAM3 repository")
    p.add_argument("--model-path", required=True)
    p.add_argument("--query-module-dir", required=True)
    p.add_argument("--samples", required=True, help="cache_reports JSON / JSONL with video_id, exp_id")
    p.add_argument("--video-root", required=True, help="folder containing <video_id>/ JPEG frames")
    p.add_argument("--cache-root", default="", help="offline cache root (to read the expression from meta.json)")
    p.add_argument("--split", default="train")
    p.add_argument("--frames-per-sample", type=int, default=3)
    p.add_argument("--max-image-side", type=int, default=0, help="0 = keep original (Ref-YTVOS/DAVIS scripts); 448 for ReVOS")
    p.add_argument("--resize-cache", default="runs/_frame_cache")
    p.add_argument("--out", required=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device-map", default="auto", choices=["", "auto"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def maybe_resize(path: Path, max_side: int, cache: Path) -> str:
    if max_side <= 0:
        return str(path)
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(1.0, max_side / float(max(w, h)))
        if s >= 1.0:
            return str(path)
        out = cache / f"side{max_side}" / path.as_posix().replace("/", "__")
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BICUBIC).save(out, quality=95)
        return str(out)


def main():
    a = parse_args()
    grounding_dir = Path(a.repo) / "category_grounding_mvp"
    sys.path.insert(0, str(grounding_dir))
    import torch
    from eval_refytvos_prompt_vs_query_grounding import build_query_context, build_query_messages, resolve_device
    from predict_query_grounding_refytros_meta import load_query_model

    samples = load_sample_list(a.samples)
    if a.limit:
        samples = samples[: a.limit]
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    processor, qg = load_query_model(
        SimpleNamespace(model_path=a.model_path, query_module_dir=a.query_module_dir, bf16=a.bf16,
                        device_map=a.device_map, device=a.device)
    )
    merge = int(qg.qwen.model.visual.spatial_merge_size)
    captured = {}
    handle = qg.mask_head.register_forward_hook(lambda m, i, o: captured.__setitem__("logits", o.detach().float().cpu()))

    for n, s in enumerate(samples, 1):
        vid, eid = str(s["video_id"]), str(s["exp_id"])
        dst = out_root / vid / f"{eid}.npz"
        if dst.is_file() and not a.overwrite:
            continue
        expression = s.get("expression")
        if not expression and a.cache_root:
            expression = load_cache_meta(a.cache_root, a.split, vid, eid)["expression"]
        video_dir = Path(a.video_root) / vid
        stems = list_frames(video_dir)
        idxs = sample_frame_indices(len(stems), a.frames_per_sample)
        paths = [maybe_resize(frame_path(video_dir, stems[i]), a.max_image_side, Path(a.resize_cache)) for i in idxs]
        record = {"expression": expression, "frame_paths": paths}
        with torch.inference_mode():
            prompt = processor.apply_chat_template(
                build_query_messages(record), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(resolve_device(qg.qwen))
            expr = processor.tokenizer(expression, return_tensors="pt", add_special_tokens=True)
            batch = {
                "input_ids": prompt["input_ids"],
                "attention_mask": prompt["attention_mask"],
                "expression_input_ids": expr["input_ids"].to(prompt["input_ids"].device),
                "expression_attention_mask": expr["attention_mask"].to(prompt["input_ids"].device),
            }
            for k in ("pixel_values", "image_grid_thw"):
                if k in prompt:
                    batch[k] = prompt[k]
            build_query_context(qg, batch)
        logits = captured.pop("logits")[0]          # (K, N)
        resp = torch.sigmoid(logits).numpy().astype(np.float16)
        grid = prompt["image_grid_thw"].detach().cpu().numpy()
        with Image.open(frame_path(video_dir, stems[idxs[0]])) as im0:
            orig_hw = np.array([im0.size[1], im0.size[0]])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dst, responses=resp, image_grid_thw=grid, merge=np.array(merge),
            frame_indices=np.array(idxs), frame_stems=np.array(stems)[idxs], orig_hw=orig_hw,
            expression=np.array(expression),
        )
        print(f"[{n}/{len(samples)}] {vid}/{eid} M={len(idxs)} tokens={resp.shape[1]} max={float(resp.max()):.3f}", flush=True)
        del batch, prompt
        torch.cuda.empty_cache()
    handle.remove()


if __name__ == "__main__":
    main()
