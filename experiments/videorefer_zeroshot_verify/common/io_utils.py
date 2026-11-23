"""Shared I/O helpers for the VideoRefer zero-shot Verify experiment.

Candidate file format (one file per (video_id, recall category), gzip JSON):
{
  "video_id": "OVIS/train/6a6e0e3b",
  "category": "cat",
  "frames": ["img_0000001", ...],         # ReVOS frame names (meta order)
  "height": 1080, "width": 1920,
  "candidate_ids": [0, 1, 2],              # kept candidates (recall24 -> keep7)
  "native_score_sums": {"0": 40.2, ...},   # SAM3 native score sums of kept ids
  "all_score_sums": {...},                 # before keep (for the record)
  "masks": {"0": {"5": <coco RLE>, ...}},  # frame index -> RLE, empty frames omitted
  "config": {...}
}
"""
import gzip
import json
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from pycocotools import mask as cocomask
except Exception:  # pragma: no cover
    cocomask = None


def exp_key(video_id: str, exp_id) -> str:
    return f"{video_id}||{exp_id}"


def pair_key(video_id: str, category: str) -> str:
    return f"{video_id}||{str(category).strip().lower()}"


def safe_name(text: str) -> str:
    text = str(text).strip().lower().replace("/", "__")
    return re.sub(r"[^0-9a-zA-Z_.\-]+", "_", text)


def cand_relpath(video_id: str, category: str) -> str:
    return os.path.join(safe_name(video_id), safe_name(category) + ".json.gz")


def write_json_gz(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_json_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate a truncated last line after a crash
                continue
    return rows


def append_jsonl(path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def encode_rle(mask: np.ndarray) -> Dict:
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_rle(rle: Dict) -> np.ndarray:
    r = dict(rle)
    if isinstance(r.get("counts"), str):
        r["counts"] = r["counts"].encode("utf-8")
    return cocomask.decode(r).astype(np.uint8)


def candidate_frame_mask(cand: Dict, obj_id, t: int) -> Optional[np.ndarray]:
    """Return the binary mask of candidate obj_id at frame t, or None if empty."""
    per_obj = cand["masks"].get(str(obj_id), {})
    rle = per_obj.get(str(t))
    if rle is None:
        return None
    return decode_rle(rle)


def candidate_present_frames(cand: Dict, obj_id) -> List[int]:
    return sorted(int(t) for t in cand["masks"].get(str(obj_id), {}).keys())


def candidate_union_video(cand: Dict, obj_ids: Iterable, n_frames: int, h: int, w: int) -> np.ndarray:
    out = np.zeros((n_frames, h, w), dtype=np.uint8)
    for oid in obj_ids:
        for t_str, rle in cand["masks"].get(str(oid), {}).items():
            t = int(t_str)
            if 0 <= t < n_frames:
                out[t] |= decode_rle(rle)
    return out


def uniform_object_frames(present: List[int], k: int) -> List[int]:
    """Same rule as CRF-SAM3 VisualTubeBuilder 'uniform' sampling.

    if len(frames) <= k: use all; else step = len // k; frames[i * step].
    With k=4 this reproduces the frames the Qwen3-VL verifier saw.
    """
    if k <= 0 or len(present) <= k:
        return list(present)
    step = len(present) // k
    return [present[i * step] for i in range(k)]


def uniform_video_frames(n_frames: int, k: int) -> List[int]:
    """Same rule as VideoRefer mm_utils.frame_sample(mode='uniform')."""
    seg = float(n_frames - 1) / k
    ids = [(seg * i + seg * (i + 1)) / 2 for i in range(k)]
    return [int(x) for x in np.round(np.array(ids) + 1e-6).astype(int)]


def shard_list(items: List, num_shards: int, shard_id: int) -> List:
    if num_shards <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % num_shards == shard_id]
