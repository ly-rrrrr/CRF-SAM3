"""v1 data: the fixed retina (ring sampling) applied to video clips, with ground truth kept only for probing.

The network trains on ring samples S [T,K,Nt,C] alone. Camera motion and object positions are stored
next to them so that, after training, we can ask whether the network learned them by itself.
"""
from __future__ import annotations

import glob
import os
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

from .ring import RingConfig, RingFilter
from .world import sample_world

MAX_OBJ = 3


def _render(args):
    seed, mode, dt, T, H, W = args
    rng = np.random.default_rng(seed)
    t0 = float(rng.uniform(0.0, 1.0))
    w = sample_world(seed, mode, H=H, W=W, T_max=t0 + dt * T + 0.1)
    times = t0 + dt * np.arange(T)
    vid = w.video(times).numpy()
    cam = np.stack([w.gt_camera(t) for t in times]).astype(np.float32)
    pos = np.full((T, MAX_OBJ, 2), np.nan, np.float32)
    rad = np.full((T, MAX_OBJ), np.nan, np.float32)
    for j, t in enumerate(times):
        p = w.obj_img_pos(t)
        for i in range(min(len(p), MAX_OBJ)):
            pos[j, i] = p[i]
            rad[j, i] = w.objs[i].radius * np.exp(w.cam_s * t)
    return vid, cam, pos, rad


def build(n: int, dt: float, T: int, seed0: int, mode: str = "mixed", cfg: RingConfig | None = None,
          workers: int = 4, keep_frames: int = 0):
    """Render n clips of T frames at interval dt and pass them through the fixed retina."""
    cfg = cfg or RingConfig()
    rf = RingFilter(cfg)
    jobs = [(seed0 + i, mode, dt, T, cfg.H, cfg.W) for i in range(n)]
    if workers > 1:
        with Pool(workers) as p:
            out = p.map(_render, jobs, chunksize=4)
    else:
        out = [_render(j) for j in jobs]
    S = torch.stack([rf.samples(torch.from_numpy(v)) for v, *_ in out]).half()  # [N,T,K,Nt,C]
    d = {"S": S, "dt": dt, "cfg": cfg.to_dict(), "mode": mode,
         "cam": torch.from_numpy(np.stack([o[1] for o in out])),
         "pos": torch.from_numpy(np.stack([o[2] for o in out])),
         "rad": torch.from_numpy(np.stack([o[3] for o in out]))}
    if keep_frames:
        d["frames"] = torch.stack([torch.from_numpy(o[0]) for o in out[:keep_frames]])
    return d


def cached(path: str, **kw):
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d = build(**kw)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(d, path)
    return d


def from_frames_dir(frames_dir: str, T: int = 16, stride: int = 8, cfg: RingConfig | None = None, max_frames: int = 100000):
    """Your own video, unlabeled: a folder of frames (sorted by name) -> overlapping clips of ring samples.

    Frames are center-cropped to a square and resized to cfg.H x cfg.W. Subfolders are treated as separate videos.
    """
    cfg = cfg or RingConfig()
    rf = RingFilter(cfg)
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(frames_dir, "**", "*"), recursive=True)
                   if p.lower().endswith((".png", ".jpg", ".jpeg"))})
    clips = []
    for d in dirs:
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:max_frames]
        S = []
        for f in files:
            im = Image.open(os.path.join(d, f)).convert("RGB")
            w, h = im.size; s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((cfg.W, cfg.H), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
            S.append(rf.samples(x[None])[0])
        S = torch.stack(S) if S else torch.zeros(0)
        for a in range(0, len(S) - T + 1, stride):
            clips.append(S[a:a + T])
    if not clips:
        raise ValueError(f"no clips of {T} frames found under {frames_dir}")
    return {"S": torch.stack(clips).half(), "cfg": cfg.to_dict(), "mode": "real"}
