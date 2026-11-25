"""Training-free check: can ring derivatives alone read camera motion and local motion?

Synthetic part (exact ground truth):
  for each mode in static / zoom / rotate / pan / mixed, render N clips with moving objects; estimate
  (s, omega, bx, by) from every consecutive frame pair; report the error against the analytic ground truth;
  locate the strongest local-motion peak and check that it lands on a moving object;
  integrate on demand (total zoom factor, total rotation) and compare with the truth.
Real part (optional): --frames_dir with sorted images -> per-step CSV of (s, omega, bx, by) and local-motion maps.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from PIL import Image

from .motion import estimate_motion, integrate, resid_peak
from .ring import RingConfig, RingFilter
from .world import sample_world


def synthetic(rf: RingFilter, n: int, frames: int, dt: float, seed0: int):
    out = {}
    for mode in ["static", "zoom", "rotate", "pan", "mixed"]:
        errs, hits, tries, zoom_err, rot_err = [], 0, 0, [], []
        scale = []
        for i in range(n):
            w = sample_world(seed0 + i, mode, n_obj=2, T_max=0.3 + dt * frames + 0.1)
            times = 0.3 + dt * np.arange(frames)
            vid = w.video(times)
            est, gts = [], []
            for j in range(frames - 1):
                r = estimate_motion(rf, vid[j], vid[j + 1], dt)
                tm = 0.5 * (times[j] + times[j + 1])
                gt = w.gt_camera(tm)
                est.append(r["params"]); gts.append(gt)
                # local motion: does the residual peak land on a moving object?
                P0, P1 = w.obj_img_pos(times[j]), w.obj_img_pos(times[j + 1])
                speed = np.linalg.norm(P1 - P0, axis=1) / dt if len(P0) else np.zeros(0)
                # object motion relative to the camera-induced motion at its position
                moving = []
                for o, (p0, p1) in enumerate(zip(P0, P1)):
                    pm = 0.5 * (p0 + p1)
                    xr = pm - np.array([rf.cx, rf.cy])
                    vcam = gt[0] * xr + gt[1] * np.array([-xr[1], xr[0]]) + gt[2:]
                    vobj = (p1 - p0) / dt
                    if np.linalg.norm(vobj - vcam) > 25 and 0 <= pm[0] < rf.cfg.W and 0 <= pm[1] < rf.cfg.H:
                        moving.append((pm, w.objs[o].radius * math.exp(w.cam(tm)[0])))
                if moving:
                    tries += 1
                    x, y, k, _, _ = resid_peak(rf, r["resid"])
                    tol = lambda R: 2.0 * R + 2.0 * float(rf.sig[k])
                    if any(np.hypot(x - p[0], y - p[1]) <= tol(R) for p, R in moving):
                        hits += 1
            est, gts = np.array(est), np.array(gts)
            errs.append(np.abs(est - gts).mean(0))
            scale.append(np.abs(gts).mean(0))
            # on-demand integration: total zoom factor and total rotation over the clip
            zoom_err.append(abs(math.exp(integrate(est[:, 0], dt)[-1]) - math.exp(integrate(gts[:, 0], dt)[-1])))
            rot_err.append(abs(math.degrees(integrate(est[:, 1], dt)[-1] - integrate(gts[:, 1], dt)[-1])))
        errs = np.mean(errs, 0); scale = np.mean(scale, 0)
        out[mode] = {
            "mae_s": float(errs[0]), "mae_omega": float(errs[1]), "mae_b": float(errs[2:].mean()),
            "mean_abs_gt_s": float(scale[0]), "mean_abs_gt_omega": float(scale[1]), "mean_abs_gt_b": float(scale[2:].mean()),
            "local_motion_hit_rate": hits / max(tries, 1), "local_motion_tries": tries,
            "zoom_factor_abs_err": float(np.mean(zoom_err)), "rotation_deg_abs_err": float(np.mean(rot_err)),
        }
        print(mode, json.dumps(out[mode]), flush=True)
    return out


def real(rf: RingFilter, frames_dir: str, dt: float, out_dir: str, max_frames: int):
    paths = sorted(p for p in glob.glob(os.path.join(frames_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))[:max_frames]
    if len(paths) < 2:
        raise SystemExit(f"need >= 2 frames in {frames_dir}")
    c = rf.cfg
    load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((c.W, c.H)), np.float32) / 255).permute(2, 0, 1)
    os.makedirs(out_dir, exist_ok=True)
    rows = ["step,s,omega,bx,by,peak_x,peak_y,peak_val"]
    prev = load(paths[0])
    est = []
    for i, p in enumerate(paths[1:]):
        cur = load(p)
        r = estimate_motion(rf, prev, cur, dt)
        x, y, k, _, val = resid_peak(rf, r["resid"])
        est.append(r["params"])
        rows.append(f"{i},{r['params'][0]:.5f},{r['params'][1]:.5f},{r['params'][2]:.4f},{r['params'][3]:.4f},{x:.1f},{y:.1f},{val:.3f}")
        R = r["resid"].numpy(); R = R / (R.max() + 1e-6)
        Image.fromarray((np.clip(R, 0, 1) * 255).astype(np.uint8)).resize((c.Nt * 4, c.K * 8), Image.NEAREST).save(
            os.path.join(out_dir, f"local_motion_{i:04d}.png"))
        prev = cur
    est = np.array(est)
    open(os.path.join(out_dir, "camera_motion.csv"), "w").write("\n".join(rows) + "\n")
    summary = {"frames": len(paths), "total_zoom_factor": float(math.exp(integrate(est[:, 0], dt)[-1])),
               "total_rotation_deg": float(math.degrees(integrate(est[:, 1], dt)[-1])),
               "total_pan_px": [float(integrate(est[:, 2], dt)[-1]), float(integrate(est[:, 3], dt)[-1])]}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print("real:", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/check_motion")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--dt", type=float, default=1 / 30)
    ap.add_argument("--frames_dir", default=None)
    ap.add_argument("--real_dt", type=float, default=1 / 30)
    ap.add_argument("--max_frames", type=int, default=200)
    args = ap.parse_args()
    rf = RingFilter(RingConfig())
    os.makedirs(args.out, exist_ok=True)
    res = synthetic(rf, args.n, args.frames, args.dt, seed0=500)
    json.dump(res, open(os.path.join(args.out, "synthetic.json"), "w"), indent=2)
    if args.frames_dir:
        real(rf, args.frames_dir, args.real_dt, os.path.join(args.out, "real"), args.max_frames)


if __name__ == "__main__":
    main()
