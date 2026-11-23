"""Turn a (category-level) Ground response map into recall seeds.

Design (see 设计说明.md): the response map is NOT treated as an instance box.
It is used as a *region gate* that answers "where is there category evidence
that none of the current SAM3 candidates explains?". Instance separation is
left to SAM3 (low-threshold detections or a tracker point seed).

All geometry is in normalised [0, 1] image coordinates, so it can be passed to
SAM3 (which expects relative coordinates) regardless of frame resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage as nd


@dataclass
class Blob:
    frame_idx: int            # SAM3 frame index (video order)
    frame_slot: int           # position among the M Ground frames
    mass: float               # sum of response inside the blob
    peak: float               # max response inside the blob
    size: int                 # number of tokens
    box: List[float]          # [x0, y0, x1, y1] normalised, token-aligned
    peaks: List[List[float]]  # local maxima [x, y] normalised (>=1)
    coverage: float = 0.0     # fraction of blob mass explained by existing candidates
    token_mask: Optional[np.ndarray] = field(default=None, repr=False)

    def to_json(self) -> Dict:
        d = asdict(self)
        d.pop("token_mask", None)
        return d


# --------------------------------------------------------------------------
# response map handling
# --------------------------------------------------------------------------
def split_frames(flat: np.ndarray, image_grid_thw: Sequence[Sequence[int]], merge: int = 2) -> List[np.ndarray]:
    """Flat token vector (N,) or (K, N) -> list of per-frame (Ht, Wt) / (K, Ht, Wt)."""
    out = []
    off = 0
    for t, h, w in np.asarray(image_grid_thw).tolist():
        assert int(t) == 1, "image-mode grids expected (one grid per frame)"
        ht, wt = int(h) // merge, int(w) // merge
        n = ht * wt
        chunk = flat[..., off:off + n]
        out.append(chunk.reshape(*chunk.shape[:-1], ht, wt))
        off += n
    assert off == flat.shape[-1], f"token count mismatch {off} vs {flat.shape[-1]}"
    return out


def aggregate(responses: np.ndarray, how: str = "max") -> np.ndarray:
    """responses: (K, N) sigmoid maps. ``max`` is what Ground uses at inference.

    On the 2026-11-12 diagnostic, slots 4-7 were near-duplicates (corr > 0.9) and
    slot choice was unstable across cases, so ``max`` is the recall-safe default.
    """
    if how == "max":
        return responses.max(0)
    if how == "mean":
        return responses.mean(0)
    if how.startswith("slot"):
        return responses[int(how[4:])]
    raise ValueError(how)


def _local_peaks(m: np.ndarray, blob: np.ndarray, max_peaks: int, min_sep: int) -> List[tuple]:
    """Local maxima inside a blob, greedy NMS with Chebyshev distance >= min_sep."""
    vals = np.where(blob, m, -1.0)
    order = np.argsort(vals, axis=None)[::-1]
    picked: List[tuple] = []
    for flat in order:
        if vals.flat[flat] < 0:
            break
        r, c = divmod(int(flat), m.shape[1])
        if all(max(abs(r - pr), abs(c - pc)) >= min_sep for pr, pc in picked):
            picked.append((r, c))
            if len(picked) >= max_peaks:
                break
    return picked


def find_blobs(
    frame_map: np.ndarray,
    frame_idx: int,
    frame_slot: int,
    abs_thr: float = 0.5,
    rel_thr: float = 0.6,
    min_tokens: int = 2,
    max_peaks: int = 3,
    peak_min_sep: int = 3,
) -> List[Blob]:
    """Connected components of ``map >= max(abs_thr, rel_thr * max(map))``."""
    ht, wt = frame_map.shape
    if frame_map.max() < abs_thr:
        return []
    thr = max(abs_thr, rel_thr * float(frame_map.max()))
    lab, n = nd.label(frame_map >= thr)
    blobs: List[Blob] = []
    for c in range(1, n + 1):
        cm = lab == c
        size = int(cm.sum())
        if size < min_tokens:
            continue
        rows, cols = np.where(cm)
        box = [cols.min() / wt, rows.min() / ht, (cols.max() + 1) / wt, (rows.max() + 1) / ht]
        pk = _local_peaks(frame_map, cm, max_peaks, peak_min_sep)
        peaks = [[(c_ + 0.5) / wt, (r_ + 0.5) / ht] for r_, c_ in pk]
        blobs.append(
            Blob(
                frame_idx=int(frame_idx),
                frame_slot=int(frame_slot),
                mass=float((frame_map * cm).sum()),
                peak=float(frame_map[cm].max()),
                size=size,
                box=[float(x) for x in box],
                peaks=peaks,
                token_mask=cm,
            )
        )
    blobs.sort(key=lambda b: -b.mass)
    return blobs


def mask_to_token_grid(mask: np.ndarray, ht: int, wt: int) -> np.ndarray:
    """Area fraction of ``mask`` inside each token cell (proportional mapping)."""
    import cv2

    return cv2.resize(mask.astype(np.float32), (wt, ht), interpolation=cv2.INTER_AREA)


def blob_coverage(blob: Blob, frame_map: np.ndarray, candidate_masks: Sequence[np.ndarray]) -> float:
    """Fraction of the blob's response mass that falls on existing candidate masks."""
    if not candidate_masks:
        return 0.0
    ht, wt = frame_map.shape
    cov = np.zeros((ht, wt), dtype=np.float32)
    for m in candidate_masks:
        if m is None or not m.any():
            continue
        cov = np.maximum(cov, mask_to_token_grid(m, ht, wt))
    if not cov.any():
        return 0.0
    w = frame_map * blob.token_mask
    return float((w * cov).sum() / max(w.sum(), 1e-6))


def select_seeds(
    responses: np.ndarray,
    image_grid_thw: Sequence[Sequence[int]],
    frame_indices: Sequence[int],
    candidates: Dict[int, Dict[int, np.ndarray]],
    merge: int = 2,
    agg: str = "max",
    abs_thr: float = 0.5,
    rel_thr: float = 0.6,
    min_tokens: int = 2,
    coverage_thr: float = 0.5,
    budget: int = 2,
    max_peaks: int = 3,
) -> Dict[str, object]:
    """Return blobs per frame, the uncovered ones, and the top-``budget`` seeds.

    ``candidates`` maps obj_id -> tube (frame_idx -> bool mask) from text recall.
    A blob is "uncovered evidence" when < ``coverage_thr`` of its mass lies on
    existing candidates. Seeds are the uncovered blobs ranked by mass.
    """
    amap = aggregate(responses, agg)
    frames = split_frames(amap, image_grid_thw, merge)
    all_blobs: List[Blob] = []
    for slot, (fm, t) in enumerate(zip(frames, frame_indices)):
        blobs = find_blobs(fm, t, slot, abs_thr, rel_thr, min_tokens, max_peaks)
        cand_masks = [tube[t] for tube in candidates.values() if t in tube]
        for b in blobs:
            b.coverage = blob_coverage(b, fm, cand_masks)
        all_blobs.extend(blobs)
    uncovered = [b for b in all_blobs if b.coverage < coverage_thr]
    uncovered.sort(key=lambda b: -b.mass)
    return {
        "blobs": all_blobs,
        "uncovered": uncovered,
        "seeds": uncovered[:budget],
        "triggered": bool(uncovered) or not candidates,
    }
