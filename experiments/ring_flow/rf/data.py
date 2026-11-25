"""Build ring-coefficient clips from the synthetic world (cached to .pt)."""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import torch

from .ring import RingFilter, RingConfig
from .world import sample_world


def _render_clip(args):
    seed, dt, n_frames, mode, H, W = args
    rng = np.random.default_rng(seed)
    t0 = float(rng.uniform(0.0, 1.0))
    T_max = t0 + dt * n_frames + 0.1
    w = sample_world(seed, mode, H=H, W=W, T_max=T_max)
    times = t0 + dt * np.arange(n_frames)
    return w.video(times).numpy(), np.stack([w.gt_camera(t) for t in times])


def build_clips(n_clips: int, dt: float, n_frames: int, seed0: int, mode: str = "mixed",
                cfg: RingConfig | None = None, device="cpu", workers: int = 4, keep_frames: bool = False):
    cfg = cfg or RingConfig()
    rf = RingFilter(cfg, device=device)
    jobs = [(seed0 + i, dt, n_frames, mode, cfg.H, cfg.W) for i in range(n_clips)]
    if workers > 1:
        with Pool(workers) as p:
            out = p.map(_render_clip, jobs, chunksize=4)
    else:
        out = [_render_clip(j) for j in jobs]
    Zs, frames, cams = [], [], []
    for vid, cam in out:
        v = torch.from_numpy(vid).to(device)
        Zs.append(rf.encode(v).cpu())
        cams.append(torch.from_numpy(cam))
        if keep_frames:
            frames.append(torch.from_numpy(vid))
    res = {"Z": torch.stack(Zs), "dt": dt, "cam": torch.stack(cams), "cfg": cfg.to_dict()}  # Z [N,T,K,M+1,C]
    if keep_frames:
        res["frames"] = torch.stack(frames)
    return res


def cached(path: str, **kw):
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d = build_clips(**kw)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(d, path)
    return d


def temporal_triples(Z: torch.Tensor, dt: float):
    """Z [N,T,K,M1,C] complex -> (z, v, a), each [N*(T-2),K,M1,C]: value, backward rate, acceleration."""
    zm, z0, zp = Z[:, :-2], Z[:, 1:-1], Z[:, 2:]
    v = (z0 - zm) / dt
    a = (zp - 2 * z0 + zm) / dt ** 2
    f = lambda x: x.reshape(-1, *x.shape[2:])
    return f(z0), f(v), f(a)
