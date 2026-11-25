"""Representation check: fixed-center rings (+ detail splats) vs a uniform patch grid at the same number of floats.

Reports PSNR on the whole frame and inside the central disk (where the fixed center pays attention).
Saves a comparison grid: original | rings | rings + splats | uniform grid.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .ring import RingConfig, RingFilter, fit_splats, psnr, render_splats
from .train import save_grid
from .world import sample_world


def uniform_grid(img: torch.Tensor, n_floats: int) -> torch.Tensor:
    C, H, W = img.shape
    g = max(1, int(math.floor(math.sqrt(n_floats / C))))
    small = F.adaptive_avg_pool2d(img[None], (g, g))
    return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo_render")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--splats", default="0,100,300")
    ap.add_argument("--center_radius", type=float, default=24.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rf = RingFilter(RingConfig())
    c = rf.cfg
    ys, xs = torch.meshgrid(torch.arange(c.H), torch.arange(c.W), indexing="ij")
    center = (torch.sqrt((xs - rf.cx) ** 2 + (ys - rf.cy) ** 2) < args.center_radius).float()
    base_floats = rf.n_floats(3)
    res = {}
    vis = {"orig": [], "ring": [], "ring_splat": [], "grid": []}
    for ns in [int(s) for s in args.splats.split(",")]:
        budget = base_floats + 5 * ns  # a splat = x, y, rgb
        acc = {"ring_all": [], "ring_center": [], "grid_all": [], "grid_center": []}
        for i in range(args.n):
            w = sample_world(2000 + i, "static", n_obj=3)
            img = w.video([0.5])[0]
            R = rf.render(rf.encode(img[None]))[0]
            if ns > 0:
                cc, col, sg = fit_splats(img - R, ns, rf=rf)
                R = R + render_splats(cc, col, c.H, c.W, sg)
            G = uniform_grid(img, budget)
            acc["ring_all"].append(psnr(R, img)); acc["ring_center"].append(psnr(R, img, center))
            acc["grid_all"].append(psnr(G, img)); acc["grid_center"].append(psnr(G, img, center))
            if i < 6:
                if ns == 0:
                    vis["orig"].append(img); vis["ring"].append(R); vis["grid"].append(G)
                elif ns == max(int(s) for s in args.splats.split(",")):
                    vis["ring_splat"].append(R)
        res[f"splats={ns}"] = {"floats": budget, **{k: float(np.mean(v)) for k, v in acc.items()}}
        print(ns, res[f"splats={ns}"], flush=True)
    save_grid([torch.stack(vis[k]) for k in ["orig", "ring", "ring_splat", "grid"] if vis[k]],
              os.path.join(args.out, "compare.png"))
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
