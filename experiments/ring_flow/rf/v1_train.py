"""Train v1 (RingWorld) self-supervised, then evaluate.

What is trained: encoder + derivative + integrator + decoder, from ring samples only (no labels).
What is evaluated:
  1. reconstruction: can the state h hold one frame?                    (PSNR of decode(encode(S)) vs S)
  2. prediction: given 2 frames, roll out 0.4 s on its own, at the training intervals (1/30, 1/15 s) and at an
     unseen one (1/60 s). Compared with "copy the last frame" and "keep changing at the current rate".
  3. did it learn understanding by itself? (synthetic data only, ground truth used ONLY here)
     - camera probe: a linear readout from the motion code (g, m) to zoom / rotation / pan, R^2 on held-out clips.
     - object probe: do cells with a moving object get a larger motion code |m|? (AUC, 0.5 = chance)
     Both are also run on an untrained copy of the network, so we can tell learned structure from luck.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .ring import RingConfig, RingFilter
from .v1_data import cached, from_frames_dir
from .v1_model import RingWorld, V1Config
from .train import save_grid


def psnr_from_mse(m):
    return float(10 * math.log10(1.0 / max(m, 1e-12)))


def batches(S, bs, dt_base, strides, gen, device):
    N, T = S.shape[:2]
    while True:
        idx = torch.randint(0, N, (bs,), generator=gen)
        k = torch.tensor(strides)[torch.randint(0, len(strides), (bs,), generator=gen)]
        a = (torch.rand(bs, generator=gen) * (T - 3 * k)).long()
        fr = a[:, None] + k[:, None] * torch.arange(4)[None]
        x = S[idx[:, None], fr].float().to(device)
        yield x, (k.float() * dt_base).to(device)


@torch.no_grad()
def eval_rollout(model, S, dt, stride, horizon, device, bs=32):
    """S [N,T,...] clips at frame interval dt/stride... here: frames taken every `stride` -> interval dt."""
    steps = int(round(horizon / dt))
    need = stride * (steps + 1)
    assert S.shape[1] > need, f"clips too short: need {need + 1} frames"
    acc = {"model": np.zeros(steps), "copy": np.zeros(steps), "linear": np.zeros(steps)}
    n = 0
    first = None
    for i in range(0, S.shape[0], bs):
        x = S[i:i + bs].float().to(device)
        S0, S1 = x[:, 0], x[:, stride]
        truth = torch.stack([x[:, stride * (j + 2)] for j in range(steps)], 1)
        pred = model.rollout(S0, S1, dt, steps)
        lin = torch.stack([S1 + (j + 1) * (S1 - S0) for j in range(steps)], 1).clamp(0, 1)
        cp = S1[:, None].expand_as(truth)
        for k, p in (("model", pred), ("copy", cp), ("linear", lin)):
            acc[k] += ((p - truth) ** 2).mean(dim=(2, 3, 4)).sum(0).cpu().numpy()
        n += x.shape[0]
        if first is None:
            first = (truth[:6].cpu(), pred[:6].cpu(), cp[:6].cpu())
    res = {"steps": steps, "dt": dt}
    for k, v in acc.items():
        v = v / n
        res[f"{k}_mse_mean"] = float(v.mean())
        res[f"{k}_psnr_final"] = psnr_from_mse(float(v[-1]))
        res[f"{k}_psnr_step1"] = psnr_from_mse(float(v[0]))
    res["model_vs_copy_mse"] = res["model_mse_mean"] / res["copy_mse_mean"]
    res["model_vs_linear_mse"] = res["model_mse_mean"] / res["linear_mse_mean"]
    return res, first


@torch.no_grad()
def codes(model, d, device, times=(2, 6, 10, 14), bs=32):
    """motion codes at a few frames of each clip: g [n,dg], m [n,N,dm], plus camera and object ground truth."""
    G, Mc, cam, pos = [], [], [], []
    S = d["S"]
    for i in range(0, S.shape[0], bs):
        x = S[i:i + bs].float().to(device)
        for t in times:
            m, g = model.derivative(model.encode(x[:, t]), model.encode(x[:, t - 1]), d["dt"])
            G.append(g.cpu()); Mc.append(m.cpu())
            cam.append(d["cam"][i:i + bs, t]); pos.append(d["pos"][i:i + bs, t])
    return torch.cat(G), torch.cat(Mc), torch.cat(cam), torch.cat(pos)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    A = Xtr.T @ Xtr + lam * Xtr.shape[0] * 1e-3 * torch.eye(Xtr.shape[1])
    W = torch.linalg.solve(A, Xtr.T @ (Ytr - ym))
    P = Xte @ W + ym
    ss_res = ((P - Yte) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0).clamp(min=1e-12)
    return (1 - ss_res / ss_tot).tolist()


def object_auc(model_cfg: V1Config, ring_cfg: RingConfig, M, pos):
    """AUC of |m| for cells that contain a moving object's center vs cells that do not (0.5 = chance)."""
    rf = RingFilter(ring_cfg)
    K, P = model_cfg.K, model_cfg.P
    q = ring_cfg.q
    r = rf.r.numpy()
    lo, hi = r / math.sqrt(q), r * math.sqrt(q)
    score = M.norm(dim=-1).view(-1, K, P).numpy()
    aucs = []
    for n in range(score.shape[0]):
        lab = np.zeros((K, P), bool)
        for p in pos[n].numpy():
            if np.isnan(p).any():
                continue
            dx, dy = p[0] - rf.cx, p[1] - rf.cy
            rr = math.hypot(dx, dy); ang = math.atan2(dy, dx) % (2 * math.pi)
            ks = np.where((rr >= lo) & (rr < hi))[0]
            if len(ks):
                lab[ks[0], min(int(ang / (2 * math.pi) * P), P - 1)] = True
        if lab.sum() == 0 or lab.sum() == lab.size:
            continue
        s = score[n]
        pos_s, neg_s = s[lab], s[~lab]
        aucs.append(float((pos_s[:, None] > neg_s[None, :]).mean() + 0.5 * (pos_s[:, None] == neg_s[None, :]).mean()))
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def probes(model, untrained, d_fit, d_test, mcfg, rcfg, device):
    out = {}
    for name, net in (("trained", model), ("untrained", untrained)):
        net.eval()
        g1, m1, c1, _ = codes(net, d_fit, device)
        g2, m2, c2, p2 = codes(net, d_test, device)
        r_g = ridge_r2(g1, c1, g2, c2)
        r_all = ridge_r2(torch.cat([g1, m1.flatten(1)], 1), c1, torch.cat([g2, m2.flatten(1)], 1), c2, lam=10.0)
        auc, n_auc = object_auc(mcfg, rcfg, m2, p2)
        out[name] = {"camera_R2_from_g [s,omega,bx,by]": r_g, "camera_R2_from_g_and_m [s,omega,bx,by]": r_all,
                     "object_AUC_from_|m|": auc, "object_frames": n_auc}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--cache", default="data")
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--frames_dir", default=None, help="--data real: folder of frames (subfolders = videos)")
    ap.add_argument("--real_dt", type=float, default=1 / 30)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--mode", default="mixed")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=2)
    ap.add_argument("--der_layers", type=int, default=2)
    ap.add_argument("--int_layers", type=int, default=3)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--horizon", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strides", default="1,2", help="frame strides used in training; \"2\" = only 15 fps")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcfg = RingConfig()
    rf = RingFilter(rcfg)
    log = open(os.path.join(args.out, "log.txt"), "a")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    if args.data == "synthetic":
        base_dt = 1 / 30
        kw = dict(T=args.T, mode=args.mode, cfg=rcfg, workers=args.workers)
        tr = cached(f"{args.cache}/v1_train_{args.mode}_n{args.n_train}_T{args.T}.pt", n=args.n_train, dt=base_dt, seed0=0, **kw)
        va = cached(f"{args.cache}/v1_val_{args.mode}_n{args.n_val}_T{args.T}.pt", n=args.n_val, dt=base_dt, seed0=1_000_000, **kw)
        kw60 = dict(kw, T=2 * args.T)
        va60 = cached(f"{args.cache}/v1_val60_{args.mode}_n{args.n_val}_T{2 * args.T}.pt", n=args.n_val, dt=1 / 60, seed0=1_000_000, **kw60)
        S_tr = tr["S"]
    else:
        base_dt = args.real_dt
        allc = from_frames_dir(args.frames_dir, T=args.T, cfg=rcfg)["S"]
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(allc.shape[0], generator=g)
        nv = max(1, allc.shape[0] // 10)
        va = {"S": allc[perm[:nv]], "dt": base_dt}
        S_tr = allc[perm[nv:]]
        va60 = None
    say(f"data ready: train {tuple(S_tr.shape)} ({time.time() - t0:.0f}s), device {dev}")

    mcfg = V1Config(K=rcfg.K, Nt=rcfg.Nt, P=args.P, d=args.d, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                    der_layers=args.der_layers, int_layers=args.int_layers, dt_ref=base_dt)
    model = RingWorld(mcfg).to(dev)
    torch.manual_seed(args.seed + 1)
    untrained = RingWorld(mcfg).to(dev)
    say(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=max(0.05, 2.0 / args.steps))
    gen = torch.Generator().manual_seed(args.seed)
    it = batches(S_tr, args.bs, base_dt, [int(v) for v in args.strides.split(",")], gen, dev)

    def evaluate(final=False):
        model.eval()
        m = {"recon_psnr": None, "rollout": {}}
        with torch.no_grad():
            x = va["S"][:64, 1].float().to(dev)
            m["recon_psnr"] = psnr_from_mse(float(((model.decode(model.encode(x)) - x) ** 2).mean()))
        cases = [("dt=1/30 (trained)", va["S"], base_dt, 1), ("dt=1/15 (trained)", va["S"], 2 * base_dt, 2)]
        if va60 is not None:
            cases.append(("dt=1/60 (never seen)", va60["S"], base_dt / 2, 1))
        vis = None
        for name, S, dt, stride in cases:
            r, first = eval_rollout(model, S, dt, stride, args.horizon, dev)
            m["rollout"][name] = r
            if vis is None:
                vis = first
        if final and args.data == "synthetic":
            fit = {k: tr[k][:400] for k in ("S", "cam", "pos")}; fit["dt"] = base_dt
            m["probes"] = probes(model, untrained, fit, va, mcfg, rcfg, dev)
        model.train()
        return m, vis

    model.train()
    run, t1 = None, time.time()
    for step in range(1, args.steps + 1):
        x, dt = next(it)
        loss, parts = model.loss(x, dt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = parts if run is None else {k: 0.98 * run[k] + 0.02 * v for k, v in parts.items()}
        if step % 100 == 0:
            say(f"step {step}/{args.steps} " + " ".join(f"{k} {v:.5f}" for k, v in run.items()) + f" ({time.time() - t1:.0f}s)")
        if step % args.eval_every == 0 and step < args.steps:
            m, _ = evaluate()
            say(f"eval @ {step}: recon {m['recon_psnr']:.2f} dB | " + " | ".join(
                f"{k}: model/copy {v['model_vs_copy_mse']:.3f}" for k, v in m["rollout"].items()))
    torch.save({"cfg": mcfg.to_dict(), "state": model.state_dict()}, os.path.join(args.out, "ringworld.pt"))
    m, vis = evaluate(final=True)
    m["args"] = vars(args); m["train_seconds"] = time.time() - t1
    json.dump(m, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    say(json.dumps({k: v for k, v in m.items() if k != "args"}, indent=1))
    truth, pred, cp = vis
    cols = [0, truth.shape[1] // 2, truth.shape[1] - 1]
    rows = [torch.cat([rf.render_samples(a[:, j]) for j in cols]) for a in (truth, pred, cp)]
    save_grid(rows, os.path.join(args.out, "rollout.png"))
    say(f"done -> {args.out}")


if __name__ == "__main__":
    main()
