"""Train v2 (RingPDE) at ONE frame rate, then test at frame rates it never saw.

Training data: the 30 fps synthetic clips, but only every 2nd frame is used, so the network only ever sees
15 fps footage (intervals of 1/15 s, plus 2/15 s for the consistency term).

Tests (all predict the same physical moment, 4/15 s = 0.267 s after the last input frame):
  input interval 1/60, 1/30 (never seen), 1/15 (trained), 2/15 (never seen as input to prediction).
  Frame-rate independence = the error stays the same whatever the input interval.
Understanding, read DIRECTLY from the learned flow fields, no fitted probe:
  zoom  s     = trimmed mean of v_k * ln q          (cells/s -> 1/s)
  rotation w  = trimmed mean of v_n * 2*pi/N        (cells/s -> rad/s)
  pan   b     = mean image-plane velocity over full rings (zoom and rotation cancel around a full circle)
  objects     = cells whose flow differs from that camera flow (AUC vs object positions; 0.5 = chance)
Reference rows: "oracle camera flow" uses the TRUE camera motion with the same transport (no learning), and the
v1 network (latent next-frame predictor) if a checkpoint trained the same way is given.
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
from .v1_data import cached
from .v2_model import RingPDE, V2Config, transport
from .train import save_grid

H_PHYS = 4 / 15  # seconds ahead to predict


def psnr(m):
    return float(10 * math.log10(1.0 / max(m, 1e-12)))


def cam_fields(rf: RingFilter, cam, K, N):
    """true camera motion (s, w, bx, by) [B,4] -> flow fields (v_k, v_n) [B,1,K,N] in cells/s."""
    q = rf.cfg.q
    r = rf.r.view(1, K, 1).to(cam.device)
    ph = rf.phi.view(1, 1, N).to(cam.device)
    s, w, bx, by = (cam[:, i].view(-1, 1, 1) for i in range(4))
    vr = s * r + bx * torch.cos(ph) + by * torch.sin(ph)
    vt = w * r - bx * torch.sin(ph) + by * torch.cos(ph)
    vk = vr / (r * math.log(q))
    vn = vt / r * N / (2 * math.pi)
    return vk[:, None].float(), vn[:, None].float()


def read_camera(rf: RingFilter, vk, vn, k_lo=4, k_hi=20):
    """learned flow fields -> (s, w, bx, by) by fixed geometry, no fitting."""
    q = rf.cfg.q
    K, N = vk.shape[-2:]
    r = rf.r.view(1, K, 1).to(vk.device)
    ph = rf.phi.view(1, 1, N).to(vk.device)
    vr = vk[:, 0] * math.log(q) * r
    vt = vn[:, 0] * 2 * math.pi / N * r
    sl = slice(k_lo, k_hi + 1)

    def tmean(x):  # trimmed mean over cells, drops the 15% most extreme on each side (moving objects)
        x = x.flatten(1).sort(1).values
        a = int(0.15 * x.shape[1])
        return x[:, a:x.shape[1] - a].mean(1)

    s = tmean((vr / r)[:, sl])
    w = tmean((vt / r)[:, sl])
    vx = vr * torch.cos(ph) - vt * torch.sin(ph)
    vy = vr * torch.sin(ph) + vt * torch.cos(ph)
    bx = tmean(vx[:, sl]); by = tmean(vy[:, sl])
    return torch.stack([s, w, bx, by], 1)


def r2(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    return (1 - ((pred - gt) ** 2).sum(0) / ((gt - gt.mean(0)) ** 2).sum(0).clip(1e-12)).tolist()


def object_auc(rf, vk, vn, cam_hat, pos):
    """score = |learned flow - flow implied by the read-out camera| per cell; label = object center in the cell."""
    K, N = vk.shape[-2:]
    ck, cn = cam_fields(rf, cam_hat, K, N)
    sc = ((vk - ck) ** 2 * math.log(rf.cfg.q) ** 2 * rf.r.view(1, 1, K, 1) ** 2
          + (vn - cn) ** 2 * (2 * math.pi / N) ** 2 * rf.r.view(1, 1, K, 1) ** 2).sqrt()[:, 0]  # px/s
    return auc_from_score(rf, sc, pos)


def auc_from_score(rf, sc, pos):
    """sc [B,K,N] per-cell score -> AUC of object cells vs other cells on the same ring."""
    K, N = sc.shape[-2:]
    sc = sc * rf.valid[None]
    q = rf.cfg.q
    lo, hi = rf.r.numpy() / math.sqrt(q), rf.r.numpy() * math.sqrt(q)
    aucs = []
    for b in range(sc.shape[0]):
        lab = np.zeros((K, N), bool)
        for p in pos[b].numpy():
            if np.isnan(p).any():
                continue
            dx, dy = p[0] - rf.cx, p[1] - rf.cy
            rr, ang = math.hypot(dx, dy), math.atan2(dy, dx) % (2 * math.pi)
            ks = np.where((rr >= lo) & (rr < hi))[0]
            if len(ks):
                n0 = int(round(ang / (2 * math.pi) * N)) % N
                for dn in (-1, 0, 1):
                    lab[ks[0], (n0 + dn) % N] = True
        val = rf.valid.numpy() > 0
        s = sc[b].numpy()
        # compare each object cell only with non-object cells on the SAME ring, so radius alone cannot help
        for k in range(K):
            pk, nk = lab[k] & val[k], (~lab[k]) & val[k]
            if pk.sum() and nk.sum():
                ps, ns = s[k][pk], s[k][nk]
                aucs.append(float((ps[:, None] > ns[None, :]).mean() + 0.5 * (ps[:, None] == ns[None, :]).mean()))
    return float(np.mean(aucs)) if aucs else float("nan")


RF = None  # set in main: the fixed retina, used to find which ring samples a moving object covers


@torch.no_grad()
def eval_rates(predict_fns, sets, device, bs=32):
    """predict_fns: name -> f(S_prev, S_now, dt_in, horizons, cam) -> [B,len(h),K,N,C]."""
    out = {}
    for tag, d, stride in sets:
        fdt = d["dt"]
        dt_in = stride * fdt
        hs = [H_PHYS * f for f in (0.25, 0.5, 0.75, 1.0)]
        idx = [stride + int(round(h / fdt)) for h in hs]
        assert idx[-1] < d["S"].shape[1], (tag, idx)
        acc = {k: np.zeros(len(hs)) for k in list(predict_fns) + ["copy", "linear"]}
        obj = {k: [0.0, 0.0] for k in acc}  # squared error and count on samples covered by a moving object (final horizon)
        n = 0
        for i in range(0, d["S"].shape[0], bs):
            x = d["S"][i:i + bs].float().to(device)
            cam = d["cam"][i:i + bs, stride].to(device)
            S0, S1 = x[:, 0], x[:, stride]
            tgt = torch.stack([x[:, j] for j in idx], 1)
            preds = {k: f(S0, S1, dt_in, hs, cam) for k, f in predict_fns.items()}
            preds["copy"] = S1[:, None].expand_as(tgt)
            preds["linear"] = torch.stack([(S1 + (h / dt_in) * (S1 - S0)).clamp(0, 1) for h in hs], 1)
            for k, p in preds.items():
                acc[k] += ((p - tgt) ** 2).mean(dim=(2, 3, 4)).sum(0).cpu().numpy()
            if "pos" in d and RF is not None:
                pos = d["pos"][i:i + bs, idx[-1]].to(device); rad = d["rad"][i:i + bs, idx[-1]].to(device)
                px, py = RF.px.to(device), RF.py.to(device)
                dist = ((px[None, None] - pos[..., 0, None, None]) ** 2 + (py[None, None] - pos[..., 1, None, None]) ** 2).sqrt()
                inside = (dist <= rad[..., None, None]).nan_to_num(0).any(1).float()[..., None]  # [B,K,N,1]
                for k, p in preds.items():
                    obj[k][0] += float((((p[:, -1] - tgt[:, -1]) ** 2) * inside).sum())
                    obj[k][1] += float(inside.sum() * tgt.shape[-1])
            n += x.shape[0]
        res = {}
        for k, v in acc.items():
            v = v / n
            res[k] = {"psnr_at_0.267s": psnr(v[-1]), "mse_by_horizon": v.tolist()}
            if obj[k][1] > 0:
                res[k]["psnr_on_moving_objects_at_0.267s"] = psnr(obj[k][0] / obj[k][1])
        for k in predict_fns:
            res[k]["final_mse_vs_copy"] = float(acc[k][-1] / acc["copy"][-1])
        out[tag] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--cache", default="data")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--mode", default="mixed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lam_q", type=float, default=0.005)
    ap.add_argument("--lam_cons", type=float, default=0.1)
    ap.add_argument("--lam_loc", type=float, default=0.0)
    ap.add_argument("--cam_lsq", action="store_true", help="v2.3: solve the 4 camera numbers from the equation (robust least squares)")
    ap.add_argument("--lam_res", type=float, default=0.0, help="v2.2: weight of the equation residual at the present moment")
    ap.add_argument("--global_cam", action="store_true", help="v2.1: whole-frame camera head + local flow")
    ap.add_argument("--n_frames", type=int, default=5, help="frames per training sample at 15 fps (predict n_frames-2 ahead)")
    ap.add_argument("--v1_ckpt", default=None, help="optional: a v1 checkpoint trained with --strides 2, for comparison")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true", help="load <out>/ringpde.pt and only evaluate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcfg = RingConfig(); rf = RingFilter(rcfg)
    global RF
    RF = rf
    log = open(os.path.join(args.out, "log.txt"), "a")

    def say(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    kw = dict(T=args.T, mode=args.mode, cfg=rcfg, workers=args.workers)
    tr = cached(f"{args.cache}/v1_train_{args.mode}_n{args.n_train}_T{args.T}.pt", n=args.n_train, dt=1 / 30, seed0=0, **kw)
    va = cached(f"{args.cache}/v1_val_{args.mode}_n{args.n_val}_T{args.T}.pt", n=args.n_val, dt=1 / 30, seed0=1_000_000, **kw)
    va60 = cached(f"{args.cache}/v1_val60_{args.mode}_n{args.n_val}_T{2 * args.T}.pt", n=args.n_val, dt=1 / 60,
                  seed0=1_000_000, **dict(kw, T=2 * args.T))
    S = tr["S"]
    say(f"data: train {tuple(S.shape)}, device {dev}")

    cfg = V2Config(K=rcfg.K, Nt=rcfg.Nt, width=args.width, dt_ref=1 / 15, global_cam=args.global_cam, cam_lsq=args.cam_lsq, log_q=math.log(rcfg.q))
    model = RingPDE(cfg).to(dev)
    say(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=max(0.05, 2.0 / args.steps))
    g = torch.Generator().manual_seed(args.seed)
    N, T = S.shape[:2]
    STRIDE = 2  # 30 fps data, every 2nd frame -> the model only ever sees 15 fps

    def predict_v2(S0, S1, dt_in, hs, cam):
        return model.predict(S0, S1, dt_in, hs)

    def predict_oracle(S0, S1, dt_in, hs, cam):
        u = RingPDE.grid(S1)
        vk, vn = cam_fields(rf, cam, cfg.K, cfg.Nt)
        z = torch.zeros_like(u)
        return torch.stack([RingPDE.ungrid(transport(u, vk.to(u), vn.to(u), z, h)) for h in hs], 1)

    fns = {"v2 (learned)": predict_v2, "oracle camera flow (true motion, no learning)": predict_oracle}
    if args.v1_ckpt:
        from .v1_model import RingWorld, V1Config
        ck = torch.load(args.v1_ckpt, weights_only=False, map_location=dev)
        v1 = RingWorld(V1Config(**ck["cfg"])).to(dev); v1.load_state_dict(ck["state"]); v1.eval()

        def predict_v1(S0, S1, dt_in, hs, cam):
            steps = [max(1, int(round(h / dt_in))) for h in hs]
            ro = v1.rollout(S0, S1, dt_in, max(steps))
            return torch.stack([ro[:, s - 1] for s in steps], 1)

        fns["v1 (latent next-frame, same training)"] = predict_v1
    sets = [("input 1/60 s (never seen)", va60, 1), ("input 1/30 s (never seen)", va, 1),
            ("input 1/15 s (trained)", va, 2), ("input 2/15 s (never seen)", va, 4)]

    @torch.no_grad()
    def understanding(d):
        model.eval()
        cams, gts, aucs, base, heads = [], [], [], [], []
        for i in range(0, d["S"].shape[0], 32):
            x = d["S"][i:i + 32].float().to(dev)
            for t in (2, 6, 10):
                vk, vn, qq = model.fields(RingPDE.grid(x[:, t]), RingPDE.grid(x[:, t - 2]), 2 * d["dt"])
                ch = read_camera(rf, vk.cpu(), vn.cpu())
                if cfg.global_cam or cfg.cam_lsq:
                    heads.append(model.camera().cpu())
                cams.append(ch); gts.append(d["cam"][i:i + 32, t])
                aucs.append(object_auc(rf, vk.cpu(), vn.cpu(), ch, d["pos"][i:i + 32, t]))
                diff = (x[:, t] - x[:, t - 2]).abs().sum(-1).cpu()
                base.append(auc_from_score(rf, diff, d["pos"][i:i + 32, t]))
        cams, gts = torch.cat(cams).numpy(), torch.cat(gts).numpy()
        model.train()
        extra = {}
        if heads:
            hd = torch.cat(heads).numpy()
            extra = {"camera_R2_from_camera_head [s,omega,bx,by]": r2(hd, gts),
                     "camera_head_mean_abs_err [s,omega,bx,by]": np.abs(hd - gts).mean(0).tolist()}
        return {**extra,"camera_R2_direct_readout [s,omega,bx,by]": r2(cams, gts),
                "camera_mean_abs_err [s,omega,bx,by]": np.abs(cams - gts).mean(0).tolist(),
                "camera_mean_abs_truth [s,omega,bx,by]": np.abs(gts).mean(0).tolist(),
                "object_AUC_flow_minus_camera": float(np.nanmean(aucs)),
                "object_AUC_baseline_raw_frame_difference": float(np.nanmean(base))}

    model.train()
    run, t1 = None, time.time()
    if args.eval_only:
        model.load_state_dict(torch.load(os.path.join(args.out, "ringpde.pt"), map_location=dev)["state"])
    for step in range(1, 0 if args.eval_only else args.steps + 1):
        idx = torch.randint(0, N, (args.bs,), generator=g)
        a = torch.randint(0, T - (args.n_frames - 1) * STRIDE, (args.bs,), generator=g)
        fr = a[:, None] + STRIDE * torch.arange(args.n_frames)[None]
        x = S[idx[:, None], fr].float().to(dev)
        dt = torch.full((args.bs,), STRIDE / 30, device=dev)
        loss, parts = model.loss(x, dt, lam_q=args.lam_q, lam_cons=args.lam_cons, lam_loc=args.lam_loc, lam_res=args.lam_res)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = parts if run is None else {k: 0.98 * run[k] + 0.02 * v for k, v in parts.items()}
        if step % 100 == 0:
            say(f"step {step}/{args.steps} " + " ".join(f"{k} {v:.5f}" for k, v in run.items()) + f" ({time.time() - t1:.0f}s)")
        if step % args.eval_every == 0 and step < args.steps:
            model.eval()
            r = eval_rates({"v2": predict_v2}, sets, dev)
            model.train()
            say(f"eval @ {step}: " + " | ".join(f"{k}: v2/copy {v['v2']['final_mse_vs_copy']:.3f}" for k, v in r.items()))
    if not args.eval_only:
        torch.save({"cfg": cfg.to_dict(), "state": model.state_dict()}, os.path.join(args.out, "ringpde.pt"))
    model.eval()
    m = {"frame_rate_test": eval_rates(fns, sets, dev), "understanding": understanding(va), "args": vars(args),
         "train_seconds": time.time() - t1}
    json.dump(m, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    say(json.dumps({k: v for k, v in m.items() if k != "args"}, indent=1))
    # picture: truth / v2 / copy at 0.267 s for 1/60 input, and the learned flow + source for one clip
    with torch.no_grad():
        x = va60["S"][:6].float().to(dev)
        p = model.predict(x[:, 0], x[:, 1], 1 / 60, [H_PHYS])[:, 0]
        tgt = x[:, 1 + int(round(H_PHYS * 60))]
        save_grid([rf.render_samples(a.cpu()) for a in (tgt, p, x[:, 1])], os.path.join(args.out, "predict_from_60fps.png"))
    say(f"done -> {args.out}")


if __name__ == "__main__":
    main()
