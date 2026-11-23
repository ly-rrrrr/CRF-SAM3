"""GT handling, per-frame localisation metrics and figures (CPU only)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def decode_rle(rle: Any) -> Optional[np.ndarray]:
    if rle is None:
        return None
    from pycocotools import mask as mask_utils

    r = dict(rle)
    if isinstance(r.get("counts"), str):
        r["counts"] = r["counts"].encode("utf-8")
    return mask_utils.decode(r).astype(bool)


def token_fraction(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Share of each token cell covered by the mask (area average)."""
    import cv2

    return cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)


def desink(m: np.ndarray, ratio: float = 5.0) -> np.ndarray:
    """Replace isolated outlier tokens (attention sinks) by their 3x3 neighbourhood mean.

    A token counts as a sink when it is > ``ratio`` x the mean of its 8 neighbours;
    real objects span several tokens and are left untouched.
    """
    from scipy import ndimage as nd

    m = m.astype(np.float64)
    nb = (nd.uniform_filter(m, 3, mode="nearest") * 9 - m) / 8
    out = m.copy()
    bad = m > ratio * nb + 1e-12
    out[bad] = nb[bad]
    return out


def frame_metrics(att: np.ndarray, gt_frac: np.ndarray, tok_thr: float = 0.05) -> Dict[str, float]:
    """att: (gh, gw) non-negative map; gt_frac: (gh, gw) GT coverage per token."""
    gt_tok = gt_frac >= tok_thr
    out: Dict[str, float] = {"gt_tokens": int(gt_tok.sum()), "gt_area": float(gt_tok.mean())}
    if not gt_tok.any():
        return out
    a = np.clip(att.astype(np.float64), 0, None)
    tot = a.sum()
    order = np.argsort(-a.reshape(-1))
    flat_gt = gt_tok.reshape(-1)
    out["peak_in_gt"] = float(flat_gt[order[0]])
    out["top3_hit"] = float(flat_gt[order[:3]].any())
    out["top5_hit"] = float(flat_gt[order[:5]].any())
    rank = np.where(flat_gt[order])[0]
    out["first_gt_rank"] = int(rank[0]) + 1
    out["mass_in_gt"] = float((a * gt_tok).sum() / tot) if tot > 0 else 0.0
    out["lift"] = out["mass_in_gt"] / max(out["gt_area"], 1e-6)
    return out


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["peak_in_gt", "top3_hit", "top5_hit", "mass_in_gt", "lift"]
    vis = [r for r in rows if "peak_in_gt" in r]
    out = {"frames_with_gt": len(vis)}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in vis])) if vis else float("nan")
    return out


def save_figure(path: str, images: List[np.ndarray], gts: List[Optional[np.ndarray]], maps: Dict[str, List[np.ndarray]],
                title: str) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(maps)
    nf = len(images)
    fig, axes = plt.subplots(nf, 1 + len(names), figsize=(4.2 * (1 + len(names)), 2.6 * nf), squeeze=False)
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        axes[i][0].imshow(img)
        if gts[i] is not None and gts[i].any():
            g = cv2.resize(gts[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            axes[i][0].contour(g, levels=[0.5], colors="lime", linewidths=1.5)
        axes[i][0].set_title("RGB + GT" if i == 0 else "")
        for j, n in enumerate(names):
            m = maps[n][i].astype(np.float32)
            m = m / (m.max() + 1e-12)
            up = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            ax = axes[i][j + 1]
            ax.imshow(img)
            ax.imshow(up, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
            if gts[i] is not None and gts[i].any():
                ax.contour(g, levels=[0.5], colors="lime", linewidths=1.0)
            y, x = np.unravel_index(int(np.argmax(m)), m.shape)
            ax.plot((x + 0.5) / m.shape[1] * w, (y + 0.5) / m.shape[0] * h, marker="x", color="cyan", ms=10, mew=2)
            if i == 0:
                ax.set_title(n)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
