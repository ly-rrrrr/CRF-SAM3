#!/usr/bin/env python3
"""Ground v2 step 0: does a FROZEN Qwen3-VL already look at the right object?

For each record (expression + sampled frames + optional GT RLEs):
  1. frozen Qwen3-VL-4B-Instruct reads the expression (first) and the frames
     at ORIGINAL resolution (no 336 pre-resize) and generates
     {"category": "<word>"}                                  -> zero-shot word
  2. one teacher-forced forward over prompt + answer; for every text layer we
     record the attention from the answer's category-word tokens (and from the
     position that generated the word) to every image token  -> per-layer maps
  3. same answer forced after a generic prompt ("Describe these frames.");
     word / generic attention removes expression-independent hot spots
     (idea from "MLLMs Know Where to Look", ICLR 2025)
  4. per frame: does the peak fall on GT, top-3/5 hits, attention mass on GT,
     lift = mass share / GT area share; the same metrics for Ground v1's MAX
     map when --ground-npz-dir is given (the 2026-11-19 failure package).

No training, nothing in the repo is modified. Outputs: results.jsonl,
summary.md, fig/<i>.png, maps/<i>.npz.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics import decode_rle, desink, frame_metrics, save_figure, summarize, token_fraction

SYSTEM = "You are a precise visual grounding assistant."
QUESTION = ('Which object in these frames does the referring expression describe? '
            'Answer with exactly one JSON object: {"category": "<short noun naming that object>"}.')
GENERIC_EXPR = "Describe these frames."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="Qwen3-VL-4B-Instruct directory (frozen, no Ground modules)")
    p.add_argument("--input-jsonl", required=True,
                   help="records with expression, frames/frame_paths, optional mask_rles and pred_category "
                        "(e.g. input_8_failures.jsonl from the failure package)")
    p.add_argument("--video-root", default="",
                   help="if set, frames are read from <video-root>/<video_id>/<frame>.jpg|png (original resolution)")
    p.add_argument("--max-side", type=int, default=0, help="0 = original resolution (default)")
    p.add_argument("--ground-npz-dir", default="", help="failure package dir with case_XX.npz (Ground v1 maps), matched by order")
    p.add_argument("--no-generic", action="store_true", help="skip the generic-prompt normalisation pass")
    p.add_argument("--words-only", action="store_true",
                   help="only generate the zero-shot word (fast); writes categories.json for SAM3 recall")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


# ----------------------------------------------------------------------------- inputs
def original_path_from_cache(p: str) -> Optional[str]:
    """frame cache files are named like __mnt__a__b__video__00000.jpg -> /mnt/a/b/video/00000.jpg"""
    name = Path(p).name
    if not name.startswith("__"):
        return None
    cand = "/" + name[2:].replace("__", "/")
    return cand if Path(cand).is_file() else None


def resolve_frames(rec: Dict[str, Any], video_root: str) -> List[str]:
    if video_root:
        out = []
        for f in rec["frames"]:
            base = Path(video_root) / str(rec["video_id"]) / str(f)
            hit = next((str(base) + e for e in (".jpg", ".jpeg", ".png", "") if Path(str(base) + e).is_file()), None)
            if hit is None:
                raise FileNotFoundError(f"{base}.jpg/.png not found")
            out.append(hit)
        return out
    out = []
    for p in rec.get("frame_paths", []):
        orig = original_path_from_cache(p)
        out.append(orig or p)
    if any("/side" in x for x in out):
        print(f"[warn] {rec.get('sample_id')}: using a resized frame cache; pass --video-root for original frames", flush=True)
    return out


def load_images(paths: List[str], max_side: int):
    from PIL import Image

    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if max_side > 0 and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
        imgs.append(im)
    return imgs


def build_inputs(processor, imgs, expression: str):
    content = [{"type": "text", "text": f"Referring expression:\n{expression}\n"}]
    content += [{"type": "image"} for _ in imgs]
    content.append({"type": "text", "text": QUESTION})
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=imgs, return_tensors="pt")


# ----------------------------------------------------------------------------- answer tokens
def word_token_offsets(tokenizer, ids: List[int], answer: str) -> Tuple[List[int], str]:
    """Indices (within the answer) of the tokens that spell the category value."""
    m = re.search(r'"category"\s*:\s*"([^"]+)"', answer)
    word = m.group(1).strip() if m else answer.strip()
    if not m:
        return list(range(len(ids))), word
    vs, ve = m.start(1), m.end(1)
    ends = [len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)) for i in range(len(ids))]
    starts = [0] + ends[:-1]
    idx = [i for i in range(len(ids)) if starts[i] < ve and ends[i] > vs]
    return (idx or list(range(len(ids)))), word


def forced_inputs(inputs, answer_ids, device):
    import torch

    L = inputs["input_ids"].shape[1]
    ans = torch.tensor(answer_ids, dtype=inputs["input_ids"].dtype).unsqueeze(0)
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == L:
            pad = ans if k == "input_ids" else (torch.ones_like(ans, dtype=v.dtype) if k == "attention_mask"
                                                 else torch.zeros_like(ans, dtype=v.dtype))
            out[k] = torch.cat([v, pad], dim=1)
        else:
            out[k] = v
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}, L


def frame_layout(model, processor, inputs) -> List[Tuple[int, int]]:
    merge = int(getattr(processor.image_processor, "merge_size", 2))
    grids = inputs["image_grid_thw"].tolist()
    return [(int(h) // merge * int(t), int(w) // merge) for t, h, w in grids]


def split_maps(flat: np.ndarray, layout: List[Tuple[int, int]]) -> List[np.ndarray]:
    out, off = [], 0
    for gh, gw in layout:
        out.append(flat[off: off + gh * gw].reshape(gh, gw))
        off += gh * gw
    return out


def attention_for(model, processor, imgs, expression, answer_ids, word_idx, device):
    """Return (per-layer word map [Ly, Ncols], per-layer pre map [Ly, Ncols], layout)."""
    import torch

    from .attn_capture import layer_maps

    inputs = build_inputs(processor, imgs, expression)
    full, L = forced_inputs(inputs, answer_ids, device)
    img_id = int(model.config.image_token_id)
    cols = torch.nonzero(full["input_ids"][0] == img_id).flatten().cpu()
    rows = [L + i for i in word_idx] + [L + word_idx[0] - 1]
    maps = layer_maps(model, full, rows, cols).numpy()  # Ly, n_rows, Ncols
    word = maps[:, :-1, :].mean(axis=1)
    pre = maps[:, -1, :]
    layout = frame_layout(model, processor, inputs)
    if sum(h * w for h, w in layout) != word.shape[1]:
        raise RuntimeError(f"image tokens {word.shape[1]} != grid layout {layout}")
    return word, pre, layout


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    import torch
    import transformers
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        sys.exit(f"transformers {transformers.__version__} has no Qwen3-VL; use the Sa2VA venv")

    dt = getattr(torch, a.dtype)
    processor = AutoProcessor.from_pretrained(a.model_path, local_files_only=True)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(a.model_path, dtype=dt, attn_implementation="sdpa",
                                                                local_files_only=True)
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(a.model_path, torch_dtype=dt, attn_implementation="sdpa",
                                                                local_files_only=True)
    model.to(a.device).eval()
    tok = processor.tokenizer
    stop_ids = {i for i in [tok.eos_token_id, tok.pad_token_id, getattr(model.generation_config, "eos_token_id", None)]
                if isinstance(i, int)}
    ge = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(ge, (list, tuple)):
        stop_ids |= set(int(x) for x in ge)

    recs = [json.loads(l) for l in Path(a.input_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        recs = recs[: a.limit]
    out = Path(a.out)
    (out / "fig").mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
    res_f = open(out / "results.jsonl", "w", encoding="utf-8")
    all_rows: Dict[str, List[Dict]] = {}
    per_layer_hits: List[List[float]] = []
    md_rows = []
    cat_records: List[Dict] = []

    for i, rec in enumerate(recs):
        t0 = time.time()
        expr = rec["expression"]
        paths = resolve_frames(rec, a.video_root)
        imgs = load_images(paths, a.max_side)
        inputs = build_inputs(processor, imgs, expr)
        L = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            gen = model.generate(**{k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in inputs.items()},
                                 max_new_tokens=a.max_new_tokens, do_sample=False)
        ans_ids = [int(x) for x in gen[0, L:].tolist()]
        while ans_ids and ans_ids[-1] in stop_ids:
            ans_ids.pop()
        answer = tok.decode(ans_ids, skip_special_tokens=True)
        word_idx, word = word_token_offsets(tok, ans_ids, answer)
        if a.words_only:
            cat_records.append({"video_id": rec.get("video_id"), "exp_id": rec.get("exp_id"),
                                "sample_id": rec.get("sample_id"), "expression": expr,
                                "pred_category": word, "raw_response": answer,
                                "ground_v1_category": rec.get("pred_category") or rec.get("category")})
            res_f.write(json.dumps(cat_records[-1], ensure_ascii=False) + "\n")
            res_f.flush()
            md_rows.append((i, expr, cat_records[-1]["ground_v1_category"], word, {}))
            print(f"[{i + 1}/{len(recs)}] {rec.get('sample_id')} v1={cat_records[-1]['ground_v1_category']} zs={word} "
                  f"{time.time() - t0:.1f}s", flush=True)
            continue

        w_layers, pre_layers, layout = attention_for(model, processor, imgs, expr, ans_ids, word_idx, a.device)
        nly = w_layers.shape[0]
        mid = slice(nly // 4, (3 * nly) // 4)
        variants = {
            "word·all layers": w_layers.mean(0),
            "word·all layers, sinks removed": None,
            "word·middle layers": w_layers[mid].mean(0),
            "pre-word·middle layers": pre_layers[mid].mean(0),
        }
        if not a.no_generic:
            g_layers, _, _ = attention_for(model, processor, imgs, GENERIC_EXPR, ans_ids, word_idx, a.device)
            gm = g_layers[mid].mean(0)
            variants["word/generic·middle layers"] = variants["word·middle layers"] / (gm + 0.1 * gm.mean() + 1e-12)
        fmaps = {k: split_maps(v, layout) for k, v in variants.items() if v is not None}
        fmaps["word·all layers, sinks removed"] = [desink(m) for m in split_maps(w_layers.mean(0), layout)]

        # Ground v1 (optional)
        v1 = None
        if a.ground_npz_dir:
            npz = Path(a.ground_npz_dir) / f"case_{i:02d}.npz"
            if npz.is_file():
                z = np.load(npz)
                g = z["image_grid_thw"]
                agg = z["aggregate"] if "aggregate" in z.files else z["responses"].max(0)
                lay = [(int(r[1]) // 2, int(r[2]) // 2) for r in g]
                v1 = (split_maps(agg, lay), split_maps(z["gt_token_mask"].astype(np.float32), lay))

        gts = [decode_rle(r) for r in rec.get("mask_rles", [None] * len(imgs))]
        if len(gts) != len(imgs):
            gts = [None] * len(imgs)
        frame_rows: Dict[str, List[Dict]] = {k: [] for k in fmaps}
        if v1:
            frame_rows["Ground v1 MAX (336)"] = []
        for f, gt in enumerate(gts):
            if gt is None:
                continue
            for k, ms in fmaps.items():
                gh, gw = ms[f].shape
                frame_rows[k].append(frame_metrics(ms[f], token_fraction(gt, gh, gw)))
            if v1 and f < len(v1[0]):
                frame_rows["Ground v1 MAX (336)"].append(frame_metrics(v1[0][f], v1[1][f]))
        # per-layer peak hit (word rows)
        hits = []
        for ly in range(nly):
            fl = split_maps(w_layers[ly], layout)
            r = [frame_metrics(fl[f], token_fraction(gt, *fl[f].shape)) for f, gt in enumerate(gts) if gt is not None]
            r = [x["peak_in_gt"] for x in r if "peak_in_gt" in x]
            hits.append(float(np.mean(r)) if r else float("nan"))
        per_layer_hits.append(hits)

        summ = {k: summarize(v) for k, v in frame_rows.items()}
        for k, v in frame_rows.items():
            all_rows.setdefault(k, []).extend(v)
        rec_out = {
            "index": i, "sample_id": rec.get("sample_id"), "expression": expr,
            "ground_v1_word": rec.get("pred_category") or rec.get("category"),
            "zero_shot_answer": answer, "zero_shot_word": word,
            "grid": layout, "frames": paths, "metrics": summ, "per_frame": frame_rows,
            "per_layer_peak_hit": hits, "sec": round(time.time() - t0, 1),
        }
        res_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
        res_f.flush()
        np.savez_compressed(out / "maps" / f"{i:02d}.npz", word_layers=w_layers.astype(np.float16),
                            pre_layers=pre_layers.astype(np.float16), layout=np.array(layout))
        title = f'[{i}] "{expr}" | Ground v1: {rec_out["ground_v1_word"]} | frozen zero-shot: {word}'
        save_figure(str(out / "fig" / f"{i:02d}.png"), [np.asarray(im) for im in imgs], gts, fmaps, title)
        md_rows.append((i, expr, rec_out["ground_v1_word"], word,
                        {k: v.get("peak_in_gt", float("nan")) for k, v in summ.items()}))
        print(f"[{i + 1}/{len(recs)}] {rec.get('sample_id')} v1={rec_out['ground_v1_word']} zs={word} grid={layout[0]} "
              + " ".join(f"{k}:{v.get('peak_in_gt', float('nan')):.2f}" for k, v in summ.items())
              + f" {rec_out['sec']}s", flush=True)
    res_f.close()
    if a.words_only:
        (out / "categories.json").write_text(json.dumps({"records": cat_records}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
        same = sum(1 for r in cat_records if (r["ground_v1_category"] or "").lower() == r["pred_category"].lower())
        print(f"wrote {out / 'categories.json'}: {len(cat_records)} records, {same} identical to Ground v1")
        return
    write_summary(out, md_rows, all_rows, per_layer_hits)


def write_summary(out: Path, md_rows, all_rows, per_layer_hits):
    lines = ["# Ground v2 step 0: frozen Qwen3-VL attention probe", ""]
    lines += ["## Words", "", "| # | expression | Ground v1 | frozen zero-shot |", "|---|---|---|---|"]
    for i, e, w1, w0, _ in md_rows:
        lines.append(f"| {i} | {e} | {w1} | {w0} |")
    lines += ["", "## Localisation over all GT-visible frames", "",
              "| map | frames | peak on GT | top-3 hit | top-5 hit | attention mass on GT | lift (mass / area) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k, rows in all_rows.items():
        s = summarize(rows)
        lines.append(f"| {k} | {s['frames_with_gt']} | {s['peak_in_gt']:.2f} | {s['top3_hit']:.2f} | {s['top5_hit']:.2f} "
                     f"| {s['mass_in_gt']:.3f} | {s['lift']:.1f} |")
    lines += ["", "## Peak-on-GT per sample (word·middle layers / Ground v1)", ""]
    for i, e, _, _, hits in md_rows:
        lines.append(f"- [{i}] {e}: " + ", ".join(f"{k} {v:.2f}" for k, v in hits.items()))
    if per_layer_hits:
        arr = np.nanmean(np.array(per_layer_hits, dtype=float), axis=0)
        lines += ["", "## Peak-on-GT by layer (word tokens, mean over samples)", "",
                  " ".join(f"L{j}:{v:.2f}" for j, v in enumerate(arr))]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
