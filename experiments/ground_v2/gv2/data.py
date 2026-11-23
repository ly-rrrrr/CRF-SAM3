"""Dataset + collator for Ground v2.

A training record (output of ``gv2.build_data``) has
  expression, category, frame_paths, mask_rles, mask_frame_indices,
  neg_rles   (per frame, other-category annotated objects; may be null),
  ign_rles   (per frame, same-category other instances; may be null),
  absent_frames: [{frame_index, frame_path, neg_rle, ign_rle}]  (target not visible).
Records without neg/ign fields still work (plain BCE, like v1).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .common import build_messages, frame_labels, read_jsonl


class JsonlDataset(Dataset):
    def __init__(self, path: str, max_samples: int = 0) -> None:
        self.records = read_jsonl(path)
        if max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.records[i]


def resize_cached(path: str, max_side: int, cache_dir: str) -> str:
    if max_side <= 0:
        return path
    with Image.open(path) as im:
        w, h = im.size
        s = min(1.0, max_side / float(max(w, h)))
        if s >= 1.0:
            return path
        out = Path(cache_dir) / f"side{max_side}" / Path(path).as_posix().replace("/", "__")
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BICUBIC).save(out, quality=95)
        return str(out)


class GroundV2Collator:
    def __init__(self, processor: Any, merge: int, image_token_id: int, positive_threshold: float = 0.05,
                 neg_weight: float = 3.0, absent_prob: float = 0.2, max_image_side: int = 0,
                 frame_cache: str = "runs/_frame_cache", train: bool = True, max_frames: int = 0) -> None:
        self.max_frames = max_frames
        self.processor = processor
        self.merge = merge
        self.image_token_id = image_token_id
        self.thr = positive_threshold
        self.neg_weight = neg_weight
        self.absent_prob = absent_prob if train else 0.0
        self.max_image_side = max_image_side
        self.frame_cache = frame_cache
        self.pad_id = int(processor.tokenizer.pad_token_id)
        self.stats = {"samples": 0, "absent_inserted": 0, "neg_tokens": 0, "ign_tokens": 0}

    def frames_of(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        n = len(rec["frame_paths"])
        idx = rec.get("mask_frame_indices") or list(range(n))
        neg = rec.get("neg_rles") or [None] * n
        ign = rec.get("ign_rles") or [None] * n
        frames = [{"index": int(idx[i]), "path": rec["frame_paths"][i], "target": rec["mask_rles"][i],
                   "neg": neg[i], "ign": ign[i]} for i in range(n)]
        if self.max_frames > 0 and len(frames) > self.max_frames:
            keep = sorted(random.sample(range(len(frames)), self.max_frames))
            frames = [frames[i] for i in keep]
        absent = rec.get("absent_frames") or []
        if absent and len(frames) >= 2 and random.random() < self.absent_prob:
            a = random.choice(absent)
            frames[random.randrange(len(frames))] = {"index": int(a["frame_index"]), "path": a["frame_path"],
                                                     "target": None, "neg": a.get("neg_rle"), "ign": a.get("ign_rle")}
            frames.sort(key=lambda f: f["index"])
            self.stats["absent_inserted"] += 1
        return frames

    def encode(self, rec: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = self.frames_of(rec)
        r = dict(rec)
        r["frame_paths"] = [resize_cached(f["path"], self.max_image_side, self.frame_cache) for f in frames]
        full = self.processor.apply_chat_template(build_messages(r, True), tokenize=True, add_generation_prompt=False,
                                                  return_dict=True, return_tensors="pt")
        prompt = self.processor.apply_chat_template(build_messages(r, False), tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
        out: Dict[str, torch.Tensor] = {}
        for k, v in full.items():
            if torch.is_tensor(v):
                out[k] = v[0] if (k in ("input_ids", "attention_mask", "mm_token_type_ids") and v.dim() == 2) else v
        labels = out["input_ids"].clone()
        labels[: int(prompt["input_ids"].shape[-1])] = -100
        out["labels"] = labels

        lab, wts = [], []
        for f, grid in zip(frames, out["image_grid_thw"].tolist()):
            _, gh, gw = [int(x) for x in grid]
            l, w = frame_labels(f["target"], f["neg"], f["ign"], gh, gw, self.merge, self.thr, self.neg_weight)
            lab.append(l)
            wts.append(w)
        out["token_labels"] = torch.cat(lab)
        out["token_weights"] = torch.cat(wts)
        n_img = int((out["input_ids"] == self.image_token_id).sum())
        if n_img != out["token_labels"].numel():
            raise ValueError(f"{rec.get('sample_id')}: image tokens {n_img} != labels {out['token_labels'].numel()}")
        self.stats["samples"] += 1
        self.stats["neg_tokens"] += int((out["token_weights"] > 1).sum())
        self.stats["ign_tokens"] += int((out["token_weights"] == 0).sum())
        return out

    def __call__(self, records: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        enc = [self.encode(r) for r in records]
        b: Dict[str, torch.Tensor] = {
            "input_ids": pad_sequence([e["input_ids"] for e in enc], batch_first=True, padding_value=self.pad_id),
            "attention_mask": pad_sequence([e["attention_mask"] for e in enc], batch_first=True, padding_value=0),
            "labels": pad_sequence([e["labels"] for e in enc], batch_first=True, padding_value=-100),
            "token_labels": pad_sequence([e["token_labels"] for e in enc], batch_first=True, padding_value=0.0),
            "token_weights": pad_sequence([e["token_weights"] for e in enc], batch_first=True, padding_value=0.0),
            "token_valid": pad_sequence([torch.ones_like(e["token_labels"], dtype=torch.bool) for e in enc],
                                        batch_first=True, padding_value=False),
        }
        if any("mm_token_type_ids" in e for e in enc):
            b["mm_token_type_ids"] = pad_sequence(
                [e.get("mm_token_type_ids", torch.zeros_like(e["input_ids"])) for e in enc], batch_first=True, padding_value=0)
        for k in ("pixel_values", "image_grid_thw"):
            b[k] = torch.cat([e[k] for e in enc], dim=0)
        return b
