"""Shared I/O helpers: frame lists, candidate caches, GT loaders, RLE and J.

Frame indexing convention (same as SAM3-DMS and the CRF-SAM3 offline caches):
frame index ``i`` is the i-th JPEG of the video folder after sorting by the
integer value of the file stem (lexicographic fallback). Every tube in this
package is a ``Dict[int, np.ndarray(bool, H, W)]`` keyed by that index.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

Tube = Dict[int, np.ndarray]


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------
def list_frames(video_dir: str | Path) -> List[str]:
    """Return frame stems sorted exactly like sam3/model/io_utils.py."""
    video_dir = Path(video_dir)
    names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in {".jpg", ".jpeg"}
    ]
    try:
        names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        names.sort()
    return [os.path.splitext(p)[0] for p in names]


def frame_path(video_dir: str | Path, stem: str) -> Path:
    video_dir = Path(video_dir)
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        p = video_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"frame {stem} not found in {video_dir}")


def sample_frame_indices(num_frames: int, m: int) -> List[int]:
    """Uniform sampling identical to category_grounding_mvp.sample_frames."""
    if num_frames <= 0:
        return []
    if num_frames <= m:
        return list(range(num_frames))
    idxs: List[int] = []
    seen = set()
    for i in range(m):
        idx = round(i * (num_frames - 1) / max(m - 1, 1))
        if idx not in seen:
            idxs.append(idx)
            seen.add(idx)
    for idx in range(num_frames):
        if len(idxs) >= m:
            break
        if idx not in seen:
            idxs.append(idx)
            seen.add(idx)
    return sorted(idxs[:m])


# --------------------------------------------------------------------------
# RLE
# --------------------------------------------------------------------------
def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def decode_rle(rle: Any) -> Optional[np.ndarray]:
    from pycocotools import mask as mask_utils

    if not rle:
        return None
    if isinstance(rle, (list, tuple)):
        out = None
        for item in rle:
            m = decode_rle(item)
            if m is None:
                continue
            out = m if out is None else (out | m)
        return out
    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m.any(axis=2)
    return m.astype(bool)


def tube_to_rle(tube: Tube) -> Dict[str, Any]:
    return {str(k): encode_rle(v) for k, v in sorted(tube.items()) if v is not None and v.any()}


def tube_from_rle(payload: Dict[str, Any]) -> Tube:
    out: Tube = {}
    for k, v in payload.items():
        m = decode_rle(v)
        if m is not None:
            out[int(k)] = m
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _resize_bool(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    import cv2

    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def frame_iou(pred: Optional[np.ndarray], gt: np.ndarray) -> float:
    gt = gt.astype(bool)
    if pred is None:
        return 1.0 if not gt.any() else 0.0
    pred = _resize_bool(pred.astype(bool), gt.shape)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def video_j(tube: Tube, gt: Tube) -> float:
    """Region similarity J averaged over the GT-annotated frames.

    A frame where the tube has no mask counts as an empty prediction. Frames
    where both GT and prediction are empty count as 1 (DAVIS convention).
    """
    if not gt:
        return float("nan")
    vals = [frame_iou(tube.get(t), g) for t, g in gt.items()]
    return float(np.mean(vals))


def tube_iou(a: Tube, b: Tube) -> float:
    """Spatio-temporal IoU between two tubes (sum of intersections / unions)."""
    inter = 0
    union = 0
    for t in set(a) | set(b):
        ma, mb = a.get(t), b.get(t)
        if ma is None and mb is None:
            continue
        if ma is None:
            union += int(mb.sum())
            continue
        if mb is None:
            union += int(ma.sum())
            continue
        if ma.shape != mb.shape:
            mb = _resize_bool(mb, ma.shape)
        inter += int(np.logical_and(ma, mb).sum())
        union += int(np.logical_or(ma, mb).sum())
    return float(inter / union) if union else 0.0


# --------------------------------------------------------------------------
# candidate caches (crf_sam3_attn offline cache layout)
# --------------------------------------------------------------------------
def cache_exp_dir(cache_root: str | Path, split: str, video_id: str, exp_id: Any) -> Path:
    return Path(cache_root) / split / str(video_id) / str(exp_id)


def load_cache_meta(cache_root: str | Path, split: str, video_id: str, exp_id: Any) -> Dict[str, Any]:
    p = cache_exp_dir(cache_root, split, video_id, exp_id) / "meta.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_cache_candidates(cache_root: str | Path, split: str, video_id: str, exp_id: Any) -> Dict[int, Tube]:
    """Load ``masks/obj_<id>/<frame_idx padded>.png`` written by crf_sam3_attn/dataset.py."""
    exp_dir = cache_exp_dir(cache_root, split, video_id, exp_id)
    mask_root = exp_dir / "masks"
    out: Dict[int, Tube] = {}
    if not mask_root.is_dir():
        return out
    for obj_dir in sorted(mask_root.glob("obj_*")):
        obj_id = int(obj_dir.name.split("_", 1)[1])
        tube: Tube = {}
        for png in obj_dir.glob("*.png"):
            try:
                t = int(png.stem)
            except ValueError:
                continue
            m = np.array(Image.open(png)) > 0
            if m.ndim == 3:
                m = m.any(axis=2)
            if m.any():
                tube[t] = m
        out[obj_id] = tube
    return out


def iter_cache_samples(cache_root: str | Path, split: str) -> Iterable[Tuple[str, str]]:
    root = Path(cache_root) / split
    for meta in sorted(root.rglob("meta.json")):
        yield meta.parent.parent.name, meta.parent.name


def load_sample_list(path: str | Path) -> List[Dict[str, Any]]:
    """Accept crf_sam3_attn/cache_reports/*.json (``samples``/``failures``) or JSONL."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    for key in ("samples", "failures", "records"):
        if key in payload:
            return list(payload[key])
    raise ValueError(f"cannot find a sample list in {path}")


# --------------------------------------------------------------------------
# GT loaders
# --------------------------------------------------------------------------
class GTLoader:
    """Return ``{frame_idx: bool mask}`` for one expression."""

    def __call__(self, video_id: str, exp_id: Any, frames: List[str]) -> Tube:  # pragma: no cover
        raise NotImplementedError


class PaletteGT(GTLoader):
    """DAVIS17 / Ref-YouTube-VOS: ``<root>/Annotations/<vid>/<frame>.png`` + meta obj_id."""

    def __init__(self, annotation_root: str, meta_expressions: str):
        self.annotation_root = Path(annotation_root)
        meta = json.loads(Path(meta_expressions).read_text(encoding="utf-8"))
        self.videos = meta["videos"]

    def obj_ids(self, video_id: str, exp_id: Any) -> List[int]:
        info = self.videos[video_id]["expressions"][str(exp_id)]
        v = info.get("obj_id", info.get("obj_ids"))
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        return [int(v)]

    def __call__(self, video_id: str, exp_id: Any, frames: List[str]) -> Tube:
        ids = self.obj_ids(video_id, exp_id)
        out: Tube = {}
        for t, stem in enumerate(frames):
            p = self.annotation_root / video_id / f"{stem}.png"
            if not p.is_file():
                continue
            anno = np.array(Image.open(p))
            out[t] = np.isin(anno, ids)
        return out


class PngDirGT(GTLoader):
    """Per-expression binary PNGs, e.g. ``gt_masks_train_expr/<vid>/<exp>/<frame>.png``."""

    def __init__(self, template: str):
        self.template = template

    def __call__(self, video_id: str, exp_id: Any, frames: List[str]) -> Tube:
        out: Tube = {}
        for t, stem in enumerate(frames):
            p = Path(self.template.format(video_id=video_id, exp_id=exp_id, frame=stem, index=t))
            if p.is_file():
                m = np.array(Image.open(p)) > 0
                out[t] = m.any(axis=2) if m.ndim == 3 else m
        return out


class ReVOSGT(GTLoader):
    """ReVOS: ``mask_dict.json`` + ``meta_expressions_*.json`` (anno_id list, per-frame RLE)."""

    def __init__(self, mask_dict: str, meta_expressions: str):
        self.mask_dict = json.loads(Path(mask_dict).read_text(encoding="utf-8"))
        meta = json.loads(Path(meta_expressions).read_text(encoding="utf-8"))
        self.videos = meta["videos"]

    def __call__(self, video_id: str, exp_id: Any, frames: List[str]) -> Tube:
        info = self.videos[video_id]["expressions"][str(exp_id)]
        anno_ids = info.get("anno_id", [])
        if not isinstance(anno_ids, (list, tuple)):
            anno_ids = [anno_ids]
        meta_frames = [str(x) for x in self.videos[video_id].get("frames", frames)]
        index_of = {stem: i for i, stem in enumerate(frames)}
        out: Tube = {}
        shape = None
        for j, stem in enumerate(meta_frames):
            t = index_of.get(stem, j)
            merged = None
            for aid in anno_ids:
                seq = self.mask_dict.get(str(aid))
                if seq is None or j >= len(seq):
                    continue
                m = decode_rle(seq[j])
                if m is None:
                    continue
                shape = m.shape
                merged = m if merged is None else (merged | m)
            if merged is not None:
                out[t] = merged
            else:
                out[t] = None  # filled below once the shape is known
        for t in list(out):
            if out[t] is None:
                if shape is None:
                    del out[t]
                else:
                    out[t] = np.zeros(shape, dtype=bool)
        return out


def build_gt_loader(kind: str, **kw: str) -> GTLoader:
    if kind == "palette":
        return PaletteGT(kw["annotation_root"], kw["meta_expressions"])
    if kind == "png_dir":
        return PngDirGT(kw["template"])
    if kind == "revos":
        return ReVOSGT(kw["mask_dict"], kw["meta_expressions"])
    raise ValueError(f"unknown GT loader {kind!r}")


# --------------------------------------------------------------------------
# candidate pools written by ground_v2's gv2.recall_pool (step 4)
# --------------------------------------------------------------------------
def norm_word(w: Any) -> str:
    return " ".join(str(w or "").strip().lower().split())


def pool_file(pool_dir: str | Path, video_id: str, word: str) -> Path:
    import hashlib

    key = hashlib.sha1(f"{video_id}\t{norm_word(word)}".encode("utf-8")).hexdigest()[:20]
    return Path(pool_dir) / f"{key}.json.gz"


def load_pool_candidates(pool_dir: str | Path, video_id: str, word: str) -> Optional[Dict[int, Tube]]:
    """Top-7 text-recall tubes for (video, word); None if that pool was never run, {} for an empty word."""
    import gzip

    if not norm_word(word):
        return {}
    p = pool_file(pool_dir, video_id, word)
    if not p.is_file():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[int, Tube] = {}
    for k, seq in d.get("rles", {}).items():
        tube = {}
        for t, r in enumerate(seq):
            m = decode_rle(r)
            if m is not None and m.any():
                tube[t] = m
        if tube:
            out[int(k)] = tube
    return out
