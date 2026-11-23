#!/usr/bin/env python3
"""Ground v2 inference on ReVOS-style meta (RGB frames only, no GT).

Writes
  --output-json : category cache, same schema as predict_query_grounding_revos_meta.py
                  (records[].pred_category / raw_response / status ...), so CRFSAM3Pipeline
                  can read it unchanged;
  --heat-dir    : <video_id>/<exp_id>.npz, same fields as p1a.heatmap_dump
                  (responses (1,N) float16, image_grid_thw, merge, frame_indices,
                  frame_stems, orig_hw, expression), so p1a.run_p1a reads it unchanged.

  python -m gv2.predict --model-path .../Qwen3-VL-4B-Instruct --ckpt runs/gv2_revos \
      --data-root .../revos --expression-file meta_expressions_valid_.json \
      --output-json runs/gv2_revos/revos_valid_categories.json --heat-dir runs/gv2_revos/heat_valid --bf16
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from .common import build_messages, sample_frame_indices
from .data import resize_cached


def parse_category(text: str) -> str:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("category", "")).strip()
        except Exception:
            pass
    m = re.search(r'"category"\s*:\s*"([^"]*)"', text)
    return m.group(1).strip() if m else text.strip().strip('{}" ')


def load_records(data_root: str, expression_file: str, k: int) -> List[Dict[str, Any]]:
    root = Path(data_root)
    meta_path = Path(expression_file)
    meta_path = meta_path if meta_path.is_absolute() else root / meta_path
    videos = json.loads(meta_path.read_text(encoding="utf-8"))["videos"]
    recs = []
    for vid in sorted(videos):
        v = videos[vid]
        frames = [str(x) for x in v.get("frames", [])]
        idx = sample_frame_indices(len(frames), k)
        paths = [str(root / vid / f"{frames[i]}.jpg") for i in idx]
        for eid in sorted(v.get("expressions", {}), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            e = v["expressions"][eid]
            recs.append({"sample_id": f"revos:{vid}:{eid}", "video_id": vid, "exp_id": str(eid),
                         "expression": str(e.get("exp", "")), "type_id": e.get("type_id"),
                         "frames": [frames[i] for i in idx], "frame_indices": idx, "frame_paths": paths})
    return recs


@torch.inference_mode()
def predict_one(processor, model, rec: Dict[str, Any], max_new_tokens: int, max_image_side: int, frame_cache: str):
    r = dict(rec)
    r["frame_paths"] = [resize_cached(p, max_image_side, frame_cache) for p in rec["frame_paths"]]
    inputs = processor.apply_chat_template(build_messages(r, False), tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    dev = model.qwen.get_input_embeddings().weight.device
    inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    # Some processor releases emit this field although transformers 4.57's
    # Qwen3-VL generation interface rejects it during model-kwargs validation.
    # The corresponding model forward does not consume it either.
    generate_inputs = dict(inputs)
    generate_inputs.pop("mm_token_type_ids", None)
    model.set_context(inputs["input_ids"])
    try:
        gen = model.qwen.generate(**generate_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        logits = model._ctx["logits"][0]
    finally:
        model.clear_context()
    if logits is None:
        raise RuntimeError("heatmap hook did not run during prefill")
    raw = processor.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    heat = torch.sigmoid(logits.float()).cpu().numpy().astype(np.float16)
    return parse_category(raw), raw, heat, inputs["image_grid_thw"].cpu().numpy()


def write_payload(path: str, records: List[Dict[str, Any]], args) -> None:
    payload = {"summary": {"total": len(records), "valid": sum(bool(r.get("pred_category")) for r in records),
                           "data_root": args.data_root, "expression_file": args.expression_file,
                           "query_module_dir": args.ckpt, "model": "ground_v2"},
               "records": records}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--ckpt", required=True, help="output dir (or checkpoint-XXX) of gv2.train")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--expression-file", default="meta_expressions_valid_.json")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--heat-dir", default="")
    ap.add_argument("--frames-per-sample", type=int, default=3)
    ap.add_argument("--max-image-side", type=int, default=0, help="must match training (0 = original)")
    ap.add_argument("--frame-cache", default="runs/_frame_cache")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--drop-summary", action="store_true", help="ablation row E': run without the summary tokens")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device-map", default="auto", choices=["", "auto"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from .model import base_model_of, load_for_inference

    recs = load_records(args.data_root, args.expression_file, args.frames_per_sample)
    if args.max_samples > 0:
        recs = recs[: args.max_samples]
    done, out_recs = set(), []
    if args.resume and Path(args.output_json).is_file():
        out_recs = [r for r in json.loads(Path(args.output_json).read_text(encoding="utf-8"))["records"] if r.get("pred_category")]
        done = {r["sample_id"] for r in out_recs}
    processor, model = load_for_inference(args.model_path, args.ckpt, torch.bfloat16 if args.bf16 else torch.float32,
                                          args.device_map, args.device, args.drop_summary)
    merge = int(base_model_of(model.qwen).model.visual.spatial_merge_size)
    for n, rec in enumerate(recs, 1):
        if rec["sample_id"] in done:
            continue
        pred, raw, heat, grid = predict_one(processor, model, rec, args.max_new_tokens, args.max_image_side, args.frame_cache)
        out = dict(rec)
        out.update(pred_category=pred, raw_response=raw, status="ok" if pred else "error")
        out_recs.append(out)
        if args.heat_dir:
            dst = Path(args.heat_dir) / rec["video_id"] / f"{rec['exp_id']}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(rec["frame_paths"][0]) as im:
                orig_hw = np.array([im.size[1], im.size[0]])
            np.savez_compressed(dst, responses=heat[None], image_grid_thw=grid, merge=np.array(merge),
                                frame_indices=np.array(rec["frame_indices"]), frame_stems=np.array(rec["frames"]),
                                orig_hw=orig_hw, expression=np.array(rec["expression"]))
        print(f"[{n}/{len(recs)}] {rec['video_id']}/{rec['exp_id']} type={rec.get('type_id')} pred={pred or '<empty>'} "
              f"heat_max={float(heat.max()):.3f}", flush=True)
        if args.save_every > 0 and len(out_recs) % args.save_every == 0:
            write_payload(args.output_json, out_recs, args)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_payload(args.output_json, out_recs, args)
    print(f"[output] {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
