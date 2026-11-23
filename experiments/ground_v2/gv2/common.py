"""Shared helpers: prompts, category normalisation, RLE -> token-grid labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

TGT_TOKEN = "<TGT>"
NUM_TGT = 2

SYSTEM_PROMPT = """You are a referring-expression target category grounding module.

Given a referring expression and original frames from the same video, identify
the visual object category of the target referred to by the expression. Return
exactly one JSON object with key "category".
"""


def expression_text(expression: str) -> str:
    # Expression first: every visual token can read the whole sentence (causal attention).
    return "Referring expression:\n" f"{expression}\n\n" "The frames are from the same video."


def evidence_text() -> str:
    return "Target evidence: " + TGT_TOKEN * NUM_TGT


def output_text() -> str:
    return 'Return exactly one JSON object with key "category" only.'


def build_messages(record: Dict[str, Any], with_answer: bool) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": expression_text(record["expression"])}]
    for path in record["frame_paths"]:
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": evidence_text() + "\n" + output_text()})
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT.strip()}]},
        {"role": "user", "content": content},
    ]
    if with_answer:
        target = json.dumps({"category": record["category"]}, ensure_ascii=False)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


# ----------------------------------------------------------------------------- categories

DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid",
               "baby", "human", "player", "rider", "lady", "guy", "adult", "pedestrian", "skier", "surfer"],
    "car": ["car", "sedan", "vehicle", "suv", "taxi", "automobile"],
    "dog": ["dog", "puppy"],
    "cat": ["cat", "kitten"],
    "bird": ["bird", "parrot", "pigeon", "sparrow", "duck", "goose", "chicken", "hen"],
    "monkey": ["monkey", "ape", "chimpanzee", "gorilla"],
}


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class CategoryNormalizer:
    """Maps a category string to a group key; same key == same category."""

    def __init__(self, synonyms: Optional[Dict[str, Iterable[str]]] = None) -> None:
        groups = dict(DEFAULT_SYNONYMS)
        if synonyms:
            groups.update({k: list(v) for k, v in synonyms.items()})
        self.lookup: Dict[str, str] = {}
        for key, words in groups.items():
            for w in list(words) + [key]:
                self.lookup[self._clean(w)] = self._clean(key)

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text).lower().strip().replace("_", " ").split())

    def __call__(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        t = self._clean(text)
        if t in self.lookup:
            return self.lookup[t]
        s = " ".join(_singular(w) for w in t.split())
        if s in self.lookup:
            return self.lookup[s]
        head = s.split()[-1]  # "brown deer" -> "deer"
        return self.lookup.get(head, s)

    @classmethod
    def from_file(cls, path: str) -> "CategoryNormalizer":
        if not path:
            return cls()
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------- masks

def normalize_rle(rle: Dict[str, Any]) -> Dict[str, Any]:
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(x) for x in rle["size"]], "counts": counts}


def _to_coco(rle: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(rle)
    if isinstance(r.get("counts"), str):
        r["counts"] = r["counts"].encode("ascii")
    return r


def decode_rle(rle: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not rle:
        return None
    from pycocotools import mask as mask_utils

    m = mask_utils.decode(_to_coco(rle))
    if m.ndim == 3:
        m = m.any(axis=2)
    return m.astype(bool)


def merge_rles(rles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    from pycocotools import mask as mask_utils

    rles = [_to_coco(r) for r in rles if r and r.get("counts")]
    if not rles:
        return None
    merged = rles[0] if len(rles) == 1 else mask_utils.merge(rles)
    if int(mask_utils.area(merged)) <= 0:
        return None
    return normalize_rle(merged)


def token_fraction(mask: Optional[np.ndarray], grid_h: int, grid_w: int, merge: int) -> np.ndarray:
    """Share of each merged token cell covered by ``mask`` (patch grid -> merged token grid)."""
    th, tw = int(grid_h) // merge, int(grid_w) // merge
    if mask is None or mask.size == 0:
        return np.zeros(th * tw, dtype=np.float32)
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize((int(grid_w), int(grid_h)), Image.Resampling.NEAREST)
    arr = (np.asarray(img) > 0).astype(np.float32)[: th * merge, : tw * merge]
    return arr.reshape(th, merge, tw, merge).mean(axis=(1, 3)).reshape(-1)


def frame_labels(target_rle, neg_rle, ign_rle, grid_h: int, grid_w: int, merge: int, thr: float,
                 neg_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token label and weight for one frame.

    target -> label 1, weight 1 ; other-category object -> label 0, weight ``neg_weight``;
    same-category other instance -> weight 0 (not in the loss) ; background -> label 0, weight 1.
    Target wins over the other two where they overlap.
    """
    tgt = token_fraction(decode_rle(target_rle), grid_h, grid_w, merge) >= thr
    neg = token_fraction(decode_rle(neg_rle), grid_h, grid_w, merge) >= thr
    ign = token_fraction(decode_rle(ign_rle), grid_h, grid_w, merge) >= thr
    label = tgt.astype(np.float32)
    weight = np.ones_like(label)
    weight[neg] = neg_weight
    weight[ign] = 0.0
    weight[tgt] = 1.0
    return torch.from_numpy(label), torch.from_numpy(weight)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_frame_indices(n: int, k: int) -> List[int]:
    """Same uniform sampling as predict_query_grounding_refytros_meta.sample_frames."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    idxs, seen = [], set()
    for i in range(k):
        idx = round(i * (n - 1) / max(k - 1, 1))
        if idx not in seen:
            idxs.append(idx)
            seen.add(idx)
    for idx in range(n):
        if len(idxs) >= k:
            break
        if idx not in seen:
            idxs.append(idx)
            seen.add(idx)
    return sorted(idxs[:k])
