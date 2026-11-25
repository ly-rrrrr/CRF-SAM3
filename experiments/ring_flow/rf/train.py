"""Train the calculus engine (TemporalFlow + RadialFlow) on the synthetic world and evaluate.

Evaluation:
- temporal: roll out from (Z, Zdot) for `horizon` seconds at the training frame rate AND at unseen frame rates;
  compare with 'constant' and 'constant velocity' baselines. Frame-rate invariance = similar model/baseline ratios.
- radial: given the inner half of the rings, generate the outer half (completion); also unconditional samples.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from .data import cached, temporal_triples
from .model import RadialFlow, TemporalFlow, z_to_real, real_to_z
from .ring import RingConfig, RingFilter, psnr


def save_grid(rows, path, pad=2):
    """rows: list of [N,3,H,W] tensors in [0,1] -> png"""
    rows = [r.clamp(0, 1).cpu() for r in rows]
    N, _, H, W = rows[0].shape
    canvas = np.ones((len(rows) * (H + pad) + pad, N * (W + pad) + pad, 3), np.float32)
    for i, r in enumerate(rows):
        for j in range(N):
            y, x = pad + i * (H + pad), pad + j * (W + pad)
            canvas[y:y + H, x:x + W] = r[j].permute(1, 2, 0).numpy()
    Image.fromarray((canvas * 255).astype(np.uint8)).save(path)


def train_loop(model, batch_fn, steps, lr, log_every, tag, log):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=max(0.05, 2.0 / steps))
    t0, run = time.time(), 0.0
    for it in range(1, steps + 1):
        loss = batch_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % log_every == 0 or it == steps:
            msg = f"[{tag}] step {it}/{steps} loss {run:.4f} ({time.time() - t0:.0f}s)"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()


@torch.no_grad()
def rollout_eval(model: TemporalFlow, Z: torch.Tensor, dt: float, horizon: float, n_samp: int, device):
    """Z [N,T,K,M1,C] complex ground-truth clip at frame interval dt."""
    N, T = Z.shape[:2]
    steps = min(int(round(horizon / dt)), T - 2)
    zt = z_to_real(Z.reshape(N * T, *Z.shape[2:])).reshape(N, T, Z.shape[2], -1).float().to(device)
    z0, v0 = zt[:, 1], (zt[:, 1] - zt[:, 0]) / dt
    z = z0.repeat(n_samp, 1, 1); v = v0.repeat(n_samp, 1, 1)
    errs = {"model": [], "const": [], "constvel": []}
    preds = []
    for s in range(1, steps + 1):
        a = model.sample(z, v)
        v = v + a * dt
        z = z + v * dt
        zm = z.view(n_samp, N, *z.shape[1:]).mean(0)
        truth = zt[:, 1 + s]
        errs["model"].append(((zm - truth) ** 2).mean().item())
        errs["const"].append(((z0 - truth) ** 2).mean().item())
        errs["constvel"].append(((z0 + v0 * dt * s - truth) ** 2).mean().item())
        preds.append(zm)
    out = {k: float(np.mean(v)) for k, v in errs.items()}
    out["final"] = {k: v[-1] for k, v in errs.items()}
    out["ratio_model_vs_constvel"] = out["model"] / max(out["constvel"], 1e-12)
    out["ratio_model_vs_const"] = out["model"] / max(out["const"], 1e-12)
    out["steps"] = steps
    return out, preds[-1], zt[:, 1 + steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ringflow_v0")
    ap.add_argument("--cache", default="data")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--dt_train", type=float, default=1 / 15)
    ap.add_argument("--dt_test", default="0.0333333,0.0166667")
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--mode", default="mixed")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--radial_steps", type=int, default=None)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_samp", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_radial", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "a")
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    cfg = RingConfig()
    rf = RingFilter(cfg, device=dev)
    kw = dict(cfg=cfg, device=dev, workers=args.workers, mode=args.mode)

    # ---------------- data ----------------
    t0 = time.time()
    tr = cached(f"{args.cache}/train_{args.mode}_n{args.n_train}_f{args.frames}_dt{args.dt_train:.4f}.pt",
                n_clips=args.n_train, dt=args.dt_train, n_frames=args.frames, seed0=0, **kw)
    dts = [args.dt_train] + [float(x) for x in args.dt_test.split(",") if x]
    vals = {}
    for dt in dts:
        nf = int(round(args.horizon / dt)) + 3
        vals[dt] = cached(f"{args.cache}/val_{args.mode}_n{args.n_val}_dt{dt:.4f}_f{nf}.pt",
                          n_clips=args.n_val, dt=dt, n_frames=nf, seed0=1_000_000, **kw)
    print(f"data ready ({time.time() - t0:.0f}s)", flush=True)
    Z = tr["Z"]
    K, M1, C = Z.shape[2:]
    D = M1 * C * 2
    metrics = {"ring_cfg": cfg.to_dict(), "n_floats_per_frame": rf.n_floats(C)}

    # ---------------- temporal flow ----------------
    z, v, a = (z_to_real(x).float() for x in temporal_triples(Z, args.dt_train))
    tf = TemporalFlow(K, D, hidden=args.hidden, layers=args.layers).to(dev)
    tf.nz.fit(z.to(dev)); tf.nv.fit(v.to(dev)); tf.na.fit(a.to(dev))
    z, v, a = z.to(dev), v.to(dev), a.to(dev)
    print(f"temporal: {z.shape[0]} triples, params {sum(p.numel() for p in tf.parameters()) / 1e6:.2f}M", flush=True)

    def tbatch():
        idx = torch.randint(0, z.shape[0], (args.bs,), device=dev)
        return tf.loss(z[idx], v[idx], a[idx])

    tf.train()
    train_loop(tf, tbatch, args.steps, args.lr, 200, "temporal", log)
    tf.eval()
    torch.save(tf.state_dict(), os.path.join(args.out, "temporal.pt"))
    metrics["temporal"] = {}
    for dt in dts:
        m, zp, ztrue = rollout_eval(tf, vals[dt]["Z"], dt, args.horizon, args.n_samp, dev)
        metrics["temporal"][f"dt={dt:.4f}"] = m
        print(f"rollout dt={dt:.4f} ({m['steps']} steps): model {m['model']:.4g} const {m['const']:.4g} "
              f"constvel {m['constvel']:.4g} ratio(model/constvel) {m['ratio_model_vs_constvel']:.3f}", flush=True)
        n = min(8, zp.shape[0])
        save_grid([rf.render(real_to_z(ztrue[:n], M1, C)), rf.render(real_to_z(zp[:n], M1, C))],
                  os.path.join(args.out, f"rollout_dt{dt:.4f}.png"))

    # ---------------- radial flow ----------------
    if not args.skip_radial:
        zr = z_to_real(Z.reshape(-1, K, M1, C)).float().to(dev)
        rfl = RadialFlow(K, D, hidden=args.hidden, layers=args.layers).to(dev)
        rfl.nz.fit(zr)

        def rbatch():
            idx = torch.randint(0, zr.shape[0], (args.bs,), device=dev)
            return rfl.loss(zr[idx])

        rfl.train()
        train_loop(rfl, rbatch, args.radial_steps or args.steps, args.lr, 200, "radial", log)
        rfl.eval()
        torch.save(rfl.state_dict(), os.path.join(args.out, "radial.pt"))
        zv = z_to_real(vals[args.dt_train]["Z"][:, 0]).float().to(dev)[:64]
        k0 = K // 2
        comp = rfl.sample(zv.shape[0], prefix=zv[:, :k0], device=dev)
        base = zv.clone(); base[:, k0:] = rfl.nz.mean[k0:]
        truth_img = rf.render(real_to_z(zv, M1, C))
        comp_img = rf.render(real_to_z(comp, M1, C))
        base_img = rf.render(real_to_z(base, M1, C))
        ys, xs = torch.meshgrid(torch.arange(cfg.H, device=dev), torch.arange(cfg.W, device=dev), indexing="ij")
        rr = torch.sqrt((xs - rf.cx) ** 2 + (ys - rf.cy) ** 2)
        outer = (rr > float(rf.r[k0])).float()
        metrics["radial"] = {"k0": k0, "psnr_outer_completion": psnr(comp_img, truth_img, outer),
                             "psnr_outer_mean_baseline": psnr(base_img, truth_img, outer)}
        gen = rfl.sample(8, device=dev)
        save_grid([truth_img[:8], comp_img[:8], base_img[:8]], os.path.join(args.out, "radial_completion.png"))
        save_grid([rf.render(real_to_z(gen, M1, C))], os.path.join(args.out, "radial_samples.png"))
        print("radial:", metrics["radial"], flush=True)

    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print("done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
