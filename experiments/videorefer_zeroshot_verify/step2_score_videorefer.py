#!/usr/bin/env python
"""Step 2 (GPU, VideoRefer environment): zero-shot candidate scoring with VideoRefer-7B.

For every contested expression (more candidates than the number of targets the
Qwen3-VL verifier had to pick), each replayed SAM3 candidate tube is given to an
*unmodified* VideoRefer-7B checkpoint:

  * global video context : `--video_frames` frames sampled uniformly over the WHOLE
                           video (VideoRefer's own video branch, no mask drawn on it)
  * object reference     : the candidate mask on `--mask_frames` frames sampled
                           uniformly over the frames where the candidate exists
                           (the same rule as the CRF-SAM3 4-frame visual tube), fed
                           through VideoRefer's region encoder ([<region>] tokens)

Scoring modes
  yesno  : one forward per candidate; score = P(Yes) / (P(Yes) + P(No)) from the
           next-token logits (prefill only, no generation) -> directly comparable
           with the P(Yes) of the current verifier.
  choice : one forward per expression with all candidates as (A)(B)(C)... options
           (VideoRefer-Bench-Q multiple-choice format); score = softmax over the
           option letters, optionally averaged over the forward and reversed order
           to reduce position bias.

No training, no LoRA, no weight change. The model is only used in inference mode.
Output: one JSON line per expression in --out (resumable).
"""
import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict, defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.io_utils import (  # noqa: E402
    append_jsonl, candidate_frame_mask, candidate_present_frames, read_json_gz,
    read_jsonl, shard_list, uniform_object_frames, uniform_video_frames,
)
from common.prompts import LETTERS, build_choice, build_yesno  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="manifest.jsonl from step 1")
    p.add_argument("--cand_dir", default=None, help="default: <manifest dir>/candidates")
    p.add_argument("--image_root", required=True, help="ReVOS root with <video_id>/<frame>.jpg")
    p.add_argument("--frame_ext", default=".jpg")
    p.add_argument("--videorefer_repo", default=None,
                   help="path to the cloned repo dir that contains the `videorefer` package "
                        "(i.e. .../VideoRefer/VideoRefer)")
    p.add_argument("--model_path", default="DAMO-NLP-SG/VideoRefer-7B", help="local dir or HF id")
    p.add_argument("--vision_tower", default=None,
                   help="override mm_vision_tower (local siglip-so400m-patch14-384 dir; "
                        "the path must contain the word 'siglip')")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none",
                   help="none = fp16 (~17GB), 8bit (~10GB), 4bit (~7GB). Needs bitsandbytes for 8/4bit")
    p.add_argument("--attn", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    p.add_argument("--mode", choices=["yesno", "choice"], default="yesno")
    p.add_argument("--prompt_id", default=None, help="yesno: p0/p1/p2 (default p0); choice: c0")
    p.add_argument("--mask_frames", type=int, default=8,
                   help="frames per candidate for the region encoder (4 = same frames as the Qwen verifier)")
    p.add_argument("--video_frames", type=int, default=0,
                   help="global video frames; 0 = VideoRefer default (videorefer.constants.NUM_FRAMES)")
    p.add_argument("--mask_max_side", type=int, default=0,
                   help="downsample masks (nearest) so that max(H,W)<=this before sending to GPU; 0=off")
    p.add_argument("--choice_orders", choices=["fixed", "fixed+reverse"], default="fixed+reverse")
    p.add_argument("--out", required=True, help="output jsonl")
    p.add_argument("--include_id_mismatch", action="store_true",
                   help="also score expressions whose replayed ids differ from the historical ids")
    p.add_argument("--include_uncontested", action="store_true")
    p.add_argument("--max_expressions", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--dry_run", action="store_true",
                   help="no model: random logits, checks the data path only")
    return p.parse_args()


# ----------------------------------------------------------------------------
# VideoRefer wrapper
# ----------------------------------------------------------------------------

class VideoReferScorer:
    """Prefill-only scorer around the official VideoRefer-7B implementation."""

    def __init__(self, args):
        import torch
        self.torch = torch
        if args.videorefer_repo:
            sys.path.insert(0, os.path.abspath(args.videorefer_repo))
        from transformers import AutoConfig, AutoTokenizer
        from videorefer.constants import DEFAULT_VIDEO_TOKEN, NUM_FRAMES
        from videorefer.mm_utils import tokenizer_multimodal_token
        from videorefer.model.videorefer_qwen2 import VideoReferQwen2ForCausalLM  # registers config
        from videorefer.utils import disable_torch_init

        disable_torch_init()
        self.tokenizer_multimodal_token = tokenizer_multimodal_token
        self.video_token = DEFAULT_VIDEO_TOKEN
        self.default_video_frames = NUM_FRAMES

        config = AutoConfig.from_pretrained(args.model_path)
        if args.vision_tower:
            config.mm_vision_tower = args.vision_tower
        tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        kw = dict(low_cpu_mem_usage=True, config=config, device_map={"": 0})
        if args.attn != "eager":
            kw["attn_implementation"] = args.attn
        if args.quant == "none":
            kw["torch_dtype"] = torch.float16
        else:
            from transformers import BitsAndBytesConfig
            skip = ["vision_tower", "mm_projector", "region_encoder", "lm_head"]
            if args.quant == "4bit":
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=skip)
            else:
                kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
            kw["torch_dtype"] = torch.float16
        model = VideoReferQwen2ForCausalLM.from_pretrained(args.model_path, **kw)
        vt = model.get_vision_tower()
        if not vt.is_loaded:
            vt.load_model()
        vt.to(device="cuda", dtype=torch.float16)
        for name in ("mm_projector", "region_encoder"):
            mod = getattr(model.get_model(), name, None)
            if mod is not None:
                mod.to(device="cuda", dtype=torch.float16)
        if tok.pad_token is None and tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        for m in model.modules():  # required by prepare_inputs (looks up the <region> id)
            m.tokenizer = tok
        region_id = tok.convert_tokens_to_ids(["<region>"])[0]
        if region_id is None or region_id == tok.unk_token_id:
            raise RuntimeError("<region> token missing in the tokenizer: wrong checkpoint?")
        model.eval()
        self.model, self.tok, self.processor = model, tok, vt.image_processor
        self._orig_encode = model.encode_images_or_videos
        self._video_feats = None
        self.yes_ids = self._first_token_ids(["Yes", "yes", " Yes", "YES"])
        self.no_ids = self._first_token_ids(["No", "no", " No", "NO"])
        self.letter_ids = [self._first_token_ids([L])[0] for L in LETTERS]
        print(f"[VideoRefer] loaded {args.model_path} quant={args.quant} "
              f"model.config.num_frames={getattr(model.config, 'num_frames', None)} "
              f"image_aspect_ratio={getattr(model.config, 'image_aspect_ratio', None)} "
              f"yes_ids={self.yes_ids} no_ids={self.no_ids}")

    def _first_token_ids(self, words):
        ids = []
        for w in words:
            t = self.tok.encode(w, add_special_tokens=False)
            if t and t[0] not in ids:
                ids.append(t[0])
        return ids

    # pixel preprocessing identical to process_video(aspect_ratio='square')
    def preprocess(self, pil_images):
        return self.processor.preprocess(pil_images, return_tensors="pt")["pixel_values"]

    def set_video(self, video_pix):
        """Encode the global video once and reuse it for every candidate/expression."""
        torch = self.torch
        self.model.encode_images_or_videos = self._orig_encode
        with torch.inference_mode():
            feats = self._orig_encode([(video_pix.half().cuda(), "video")])
        self._video_feats = feats
        self._video_dummy = [(video_pix[:1].half().cuda(), "video")]  # placeholder, not re-encoded
        self.model.encode_images_or_videos = lambda images: self._video_feats

    def _prompt_ids(self, instruct):
        torch = self.torch
        message = [{"role": "user", "content": self.video_token + "\n" + instruct}]
        prompt = self.tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer_multimodal_token(prompt, self.tok, self.video_token, return_tensors="pt")
        return ids.unsqueeze(0).long().cuda()

    def next_token_logits(self, instruct, frame_pix, masks, ann_indices):
        """frame_pix: (F,3,H,W) region frames; masks: (1, sum_i len(ann_i), H0, W0) bool."""
        torch = self.torch
        input_ids = self._prompt_ids(instruct)
        attn = input_ids.ne(self.tok.pad_token_id).long().cuda()
        frame_nums = [int(frame_pix.shape[0])]
        with torch.inference_mode():
            _, attn2, _, embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                input_ids, attn, None, None, self._video_dummy, masks.cuda(),
                [frame_pix.half().cuda()], [ann_indices], frame_nums)
            out = self.model.model(inputs_embeds=embeds, attention_mask=attn2, use_cache=False, return_dict=True)
            logits = self.model.lm_head(out.last_hidden_state[:, -1, :]).float()[0]
        return logits

    def yesno(self, instruct, frame_pix, masks, ann_indices):
        torch = self.torch
        lg = self.next_token_logits(instruct, frame_pix, masks, ann_indices)
        ly = torch.logsumexp(lg[self.yes_ids], 0).item()
        ln = torch.logsumexp(lg[self.no_ids], 0).item()
        top = int(lg.argmax().item())
        return ly, ln, self.tok.decode([top])

    def choice(self, instruct, frame_pix, masks, ann_indices, n):
        lg = self.next_token_logits(instruct, frame_pix, masks, ann_indices)
        letters = lg[self.letter_ids[:n]]
        top = int(lg.argmax().item())
        return letters.tolist(), self.tok.decode([top])


class DryRunScorer:
    """No model. Same interface; random logits; validates shapes/bookkeeping."""

    default_video_frames = 8

    def __init__(self, args):
        self.rng = np.random.default_rng(0)

    def preprocess(self, pil_images):
        import torch
        arr = [np.asarray(im.resize((384, 384))).transpose(2, 0, 1) / 255.0 for im in pil_images]
        return torch.tensor(np.stack(arr), dtype=torch.float32)

    def set_video(self, video_pix):
        assert video_pix.ndim == 4 and video_pix.shape[1] == 3

    def _check(self, instruct, frame_pix, masks, ann_indices):
        n_region = instruct.count("[<region>]")
        assert n_region == len(ann_indices), (n_region, len(ann_indices))
        flat = [i for a in ann_indices for i in a]
        assert masks.shape[1] == len(flat), (masks.shape, len(flat))
        assert max(flat) < frame_pix.shape[0]

    def yesno(self, instruct, frame_pix, masks, ann_indices):
        self._check(instruct, frame_pix, masks, ann_indices)
        return float(self.rng.normal()), float(self.rng.normal()), "Yes"

    def choice(self, instruct, frame_pix, masks, ann_indices, n):
        self._check(instruct, frame_pix, masks, ann_indices)
        return self.rng.normal(size=n).tolist(), "A"


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------

class FrameCache:
    def __init__(self, image_root, ext, scorer):
        self.image_root, self.ext, self.scorer = image_root, ext, scorer
        self.video_id = None
        self.frames = []
        self.pix = {}

    def reset(self, video_id, frames):
        self.video_id, self.frames, self.pix = video_id, frames, {}

    def pil(self, t):
        path = os.path.join(self.image_root, self.video_id, self.frames[t] + self.ext)
        return Image.open(path).convert("RGB")

    def pixels(self, ts):
        import torch
        need = [t for t in ts if t not in self.pix]
        if need:
            out = self.scorer.preprocess([self.pil(t) for t in need])
            for t, x in zip(need, out):
                self.pix[t] = x
        return torch.stack([self.pix[t] for t in ts], 0)


def resize_mask(m, max_side):
    if not max_side or max(m.shape) <= max_side:
        return m
    s = max_side / float(max(m.shape))
    new = (max(1, int(round(m.shape[1] * s))), max(1, int(round(m.shape[0] * s))))
    return np.asarray(Image.fromarray(m * 255).resize(new, Image.NEAREST)) > 0


def build_region_inputs(cand, obj_ids, mask_frames, fc, max_side):
    """Return frame tensor, mask tensor, ann_indices, per-object frame lists."""
    import torch
    per_obj = []
    for oid in obj_ids:
        present = candidate_present_frames(cand, oid)
        per_obj.append(uniform_object_frames(present, mask_frames))
    all_frames = sorted({t for fr in per_obj for t in fr})
    pos = {t: i for i, t in enumerate(all_frames)}
    ann_indices, masks = [], []
    for oid, fr in zip(obj_ids, per_obj):
        ann_indices.append([pos[t] for t in fr])
        for t in fr:
            m = candidate_frame_mask(cand, oid, t)
            masks.append(resize_mask(m.astype(np.uint8), max_side).astype(bool))
    frame_pix = fc.pixels(all_frames)
    masks_t = torch.from_numpy(np.stack(masks, 0)).unsqueeze(0)  # (1, N, H, W) bool
    return frame_pix, masks_t, ann_indices, per_obj


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.prompt_id is None:
        args.prompt_id = "p0" if args.mode == "yesno" else "c0"
    cand_dir = args.cand_dir or os.path.join(os.path.dirname(os.path.abspath(args.manifest)), "candidates")

    rows = read_jsonl(args.manifest)
    sel = []
    for r in rows:
        if not r.get("cand_file"):
            continue
        if not args.include_uncontested and not r["contested"]:
            continue
        if not args.include_id_mismatch and not r.get("ids_match"):
            continue
        sel.append(r)
    sel.sort(key=lambda r: (r["video_id"], r["category"], str(r["exp_id"])))
    if args.max_expressions is not None:
        sel = sel[: args.max_expressions]
    # shard by video so that the global-video cache is reused
    vids = sorted({r["video_id"] for r in sel})
    keep_vids = set(shard_list(vids, args.num_shards, args.shard_id))
    sel = [r for r in sel if r["video_id"] in keep_vids]

    done = {r["key"] for r in read_jsonl(args.out) if r.get("status") == "ok"}
    todo = [r for r in sel if r["key"] not in done]
    print(f"[step2] mode={args.mode} prompt={args.prompt_id} mask_frames={args.mask_frames} "
          f"expressions={len(sel)} done={len(sel) - len(todo)} todo={len(todo)}")
    if not todo:
        return

    scorer = DryRunScorer(args) if args.dry_run else VideoReferScorer(args)
    n_video = args.video_frames or scorer.default_video_frames
    fc = FrameCache(args.image_root, args.frame_ext, scorer)
    cand_cache = OrderedDict()

    by_video = defaultdict(list)
    for r in todo:
        by_video[r["video_id"]].append(r)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(todo), desc="videorefer")
    except Exception:
        pbar = None

    for vid in sorted(by_video):
        vrows = by_video[vid]
        first = read_json_gz(os.path.join(cand_dir, vrows[0]["cand_file"]))
        frames = first["frames"]
        fc.reset(vid, frames)
        try:
            vframes = uniform_video_frames(len(frames), n_video)
            scorer.set_video(fc.pixels(vframes))
        except Exception as e:
            for r in vrows:
                append_jsonl(args.out, {"key": r["key"], "status": "error",
                                        "error": f"video encode: {type(e).__name__}: {e}"})
            _empty_cache()
            continue

        for r in vrows:
            t0 = time.time()
            rec = {"key": r["key"], "video_id": vid, "exp_id": r["exp_id"], "mode": args.mode,
                   "prompt_id": args.prompt_id, "mask_frames": args.mask_frames,
                   "video_frames": n_video, "video_frame_ids": vframes}
            try:
                cf = r["cand_file"]
                if cf not in cand_cache:
                    cand_cache[cf] = read_json_gz(os.path.join(cand_dir, cf))
                    while len(cand_cache) > 4:
                        cand_cache.popitem(last=False)
                cand = cand_cache[cf]
                ids = [int(x) for x in cand["candidate_ids"] if candidate_present_frames(cand, x)]
                if args.mode == "yesno":
                    instruct = build_yesno(args.prompt_id, r["expression"])
                    scores, raw = {}, {}
                    for oid in ids:
                        fp, mk, ann, per_obj = build_region_inputs(cand, [oid], args.mask_frames, fc,
                                                                   args.mask_max_side)
                        ly, ln, top = scorer.yesno(instruct, fp, mk, ann)
                        p = float(1.0 / (1.0 + np.exp(ln - ly)))
                        scores[str(oid)] = p
                        raw[str(oid)] = {"logit_yes": ly, "logit_no": ln, "top_token": top,
                                         "frames": per_obj[0]}
                    rec.update(scores=scores, raw=raw, instruct=instruct)
                else:
                    orders = [ids] if args.choice_orders == "fixed" or len(ids) < 2 else [ids, ids[::-1]]
                    acc = defaultdict(list)
                    raw = []
                    for order in orders:
                        instruct = build_choice(args.prompt_id, r["expression"], len(order))
                        fp, mk, ann, per_obj = build_region_inputs(cand, order, args.mask_frames, fc,
                                                                   args.mask_max_side)
                        lg, top = scorer.choice(instruct, fp, mk, ann, len(order))
                        pr = softmax(lg)
                        for oid, p in zip(order, pr):
                            acc[str(oid)].append(float(p))
                        raw.append({"order": order, "letter_logits": lg, "top_token": top})
                    scores = {k: float(np.mean(v)) for k, v in acc.items()}
                    rec.update(scores=scores, raw=raw, instruct=instruct)
                rec.update(status="ok", seconds=round(time.time() - t0, 3))
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
                print(f"[step2][error] {r['key']}: {e}")
                _empty_cache()
            append_jsonl(args.out, rec)
            if pbar is not None:
                pbar.update(1)
        _empty_cache()
    if pbar is not None:
        pbar.close()


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
