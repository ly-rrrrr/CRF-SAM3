"""v2: learn the equation of motion of the video on the ring field.

The ring field u(k, n, t): ring k (log radius), angle sample n, time t, read by the fixed retina.
Ring k is blurred in proportion to its radius, so zooming the camera by a factor q moves ring k's content onto
ring k+1 together with the right amount of blur, and rotating moves angle sample n onto n+1. On this grid, both
camera motions are uniform flows along the grid axes: the future is the present, shifted.

So we model the video as a transport equation with a source:

    du/dt = - v_k * du/dk - v_n * du/dn + q

- (v_k, v_n): flow speed along the ring index and along the angle index (cells per second), one value per cell.
- q: the source, what flow cannot explain (things appearing, occlusion, light), per cell and color.

Understanding = the network G that reads (v_k, v_n, q) from the present frame and its rate of change.
Generation   = integrating the equation forward for ANY time step dt (semi-Lagrangian: each cell looks back
               along the flow and takes the value found there, then adds q*dt). The frame rate never enters G's
               output, only the integrator, which is exact for uniform flows at any dt.
The content of the frame (the hard term) cancels: for camera motion we never need to know what the frame looks
like, only how far it shifts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class V2Config:
    K: int = 24
    Nt: int = 64
    C: int = 3
    width: int = 32        # U-Net base channels
    v_scale: float = 20.0  # output units: cells per second = v_scale * raw
    q_scale: float = 2.0   # source units: per second = q_scale * raw
    dt_ref: float = 1 / 15
    global_cam: bool = False  # v2.1: flow = camera flow (4 numbers per frame, fixed geometry) + local flow
    cam_lsq: bool = False     # v2.3: the 4 camera numbers are SOLVED from the equation (robust least squares), not learned
    log_q: float = math.log((0.5 * math.hypot(128, 128)) ** (1 / 23))  # ring spacing of the default retina

    def to_dict(self):
        return asdict(self)


# ---------------- calculus on the ring grid (fixed, exact) ----------------

def d_k(u):
    """du/dk, central difference, edges one-sided. u [B,C,K,N]"""
    up = torch.cat([u[:, :, :1], u, u[:, :, -1:]], 2)
    return 0.5 * (up[:, :, 2:] - up[:, :, :-2])


def d_n(u):
    """du/dn, central difference, periodic in angle."""
    return 0.5 * (torch.roll(u, -1, 3) - torch.roll(u, 1, 3))


def _sample(f, k, n):
    """bilinear sample of f [B,C,K,N] at fractional (k, n) [B,K,N]; periodic in n, clamped in k."""
    B, C, K, N = f.shape
    n = torch.remainder(n, N)
    fp = torch.cat([f, f[..., :1]], 3)  # column N = column 0
    gx = 2 * n / N - 1
    gy = 2 * k.clamp(0, K - 1) / (K - 1) - 1
    return F.grid_sample(fp, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)


def transport(u, vk, vn, q, dt, max_cfl: float = 0.5):
    """Integrate du/dt = -vk du/dk - vn du/dn + q over dt (seconds) with the fields held fixed.

    Semi-Lagrangian: trace each cell's path back through the flow for time dt (in small sub-steps, so the
    path can bend where the flow changes), then read u ONCE at the departure point and add q*dt.
    Reading once means no extra blur from repeated interpolation, and a uniform flow is reproduced exactly at any dt.
    u [B,C,K,N], vk/vn [B,1,K,N] cells/s, q [B,C,K,N] per s, dt float or [B].
    """
    B, C, K, N = u.shape
    dt = torch.as_tensor(dt, dtype=u.dtype, device=u.device).reshape(-1).expand(B).reshape(B, 1, 1)
    vmax = torch.maximum(vk.detach().abs().amax((1, 2, 3)), vn.detach().abs().amax((1, 2, 3)))
    n_sub = int(torch.clamp(torch.ceil((vmax * dt.view(-1)).max() / max_cfl), 1, 64).item())
    h = dt / n_sub
    k = torch.arange(K, device=u.device, dtype=u.dtype).view(1, K, 1).expand(B, K, N)
    n = torch.arange(N, device=u.device, dtype=u.dtype).view(1, 1, N).expand(B, K, N)
    V = torch.cat([vk, vn], 1)
    for _ in range(n_sub):
        v = _sample(V, k, n)
        k = (k - v[:, 0] * h).clamp(0, K - 1)
        n = n - v[:, 1] * h
    return _sample(u, k, n) + q * dt.view(B, 1, 1, 1)


# ---------------- the understanding network G ----------------

class CircConv(nn.Module):
    """3x3 conv on the ring grid: periodic in angle, edge-replicated in ring index."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=0)

    def forward(self, x):
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1, self.c2 = CircConv(cin, cout, stride), CircConv(cout, cout)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)

    def forward(self, x):
        return F.silu(self.n2(self.c2(F.silu(self.n1(self.c1(x))))))


class FieldNet(nn.Module):
    """(u, rate, du/dk, du/dn, ring index) -> (v_k, v_n, q). A small U-Net on the 24 x 64 ring grid.

    Convolution on this grid is shared across rings and angles, so a zoom (shift in k) or rotation (shift in n)
    of the input gives the same shift of the output: the equivariance the fixed center buys us.
    """

    def __init__(self, cfg: V2Config):
        super().__init__()
        c, w = cfg, cfg.width
        cin = 4 * c.C + 1
        self.e1 = Block(cin, w)
        self.e2 = Block(w, 2 * w, stride=2)
        self.e3 = Block(2 * w, 4 * w, stride=2)
        self.d2 = Block(4 * w + 2 * w, 2 * w)
        self.d1 = Block(2 * w + w, w)
        self.out = nn.Conv2d(w, 2 + c.C, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        # whole-frame camera head: (zoom s 1/s, rotation w rad/s, pan bx, by px/s)
        self.cam = nn.Sequential(nn.Linear(4 * w, 4 * w), nn.SiLU(), nn.Linear(4 * w, 4))
        nn.init.zeros_(self.cam[2].weight); nn.init.zeros_(self.cam[2].bias)
        self.last_cam = None

    def forward(self, x):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        self.last_cam = self.cam(h3.mean((2, 3))) * torch.tensor([0.3, 0.3, 10.0, 10.0], device=x.device)
        u2 = F.interpolate(h3, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.d2(torch.cat([u2, h2], 1))
        u1 = F.interpolate(h, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.d1(torch.cat([u1, h1], 1)))


class RingPDE(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.cfg = cfg
        self.G = FieldNet(cfg)
        self.register_buffer("kcoord", (torch.arange(cfg.K, dtype=torch.float32) / (cfg.K - 1)).view(1, 1, cfg.K, 1))
        r = torch.exp(cfg.log_q * torch.arange(cfg.K, dtype=torch.float32))  # ring radii in px (r0 = 1)
        self.register_buffer("r", r.view(1, 1, cfg.K, 1))
        self.register_buffer("phi", (2 * math.pi * torch.arange(cfg.Nt, dtype=torch.float32) / cfg.Nt).view(1, 1, 1, cfg.Nt))
        self.local = None
        self.last_cam = None

    @torch.no_grad()
    def solve_camera(self, u, u_prev, dt, k_lo=4, k_hi=20, iters=6):
        """v2.3: solve rate + v_cam(s,w,bx,by) . grad u = 0 for the 4 camera numbers, rings k_lo..k_hi,
        Tukey-weighted so moving objects count as outliers. Exact elimination of the global unknowns."""
        B = u.shape[0]
        dt = torch.as_tensor(dt, dtype=u.dtype, device=u.device).reshape(-1, 1, 1, 1)
        um = 0.5 * (u + u_prev)
        rate = (u - u_prev) / dt
        gk, gn = d_k(um), d_n(um)
        eye = torch.eye(4, device=u.device, dtype=u.dtype)
        cols = []
        for i in range(4):
            vk, vn = self.camera_flow(eye[i:i + 1].expand(B, 4))
            cols.append((vk * gk + vn * gn)[:, :, k_lo:k_hi + 1].flatten(1))
        A = torch.stack(cols, -1)
        b = -rate[:, :, k_lo:k_hi + 1].flatten(1)
        w = torch.ones_like(b)
        reg = 1e-6 * eye
        for _ in range(iters):
            Aw = A * w[..., None]
            x = torch.linalg.solve(Aw.transpose(1, 2) @ A + reg, (Aw.transpose(1, 2) @ b[..., None]))[..., 0]
            res = b - (A @ x[..., None])[..., 0]
            c = 4.685 * 1.4826 * res.abs().median(1, keepdim=True).values + 1e-8
            w = (1 - (res / c) ** 2).clamp(min=0) ** 2
        return x

    def camera_flow(self, cam):
        """(s, w, bx, by) [B,4] -> flow fields along ring index and angle index [B,1,K,N], cells/s. Fixed geometry."""
        s, w, bx, by = (cam[:, i].view(-1, 1, 1, 1) for i in range(4))
        vr = s * self.r + bx * torch.cos(self.phi) + by * torch.sin(self.phi)
        vt = w * self.r - bx * torch.sin(self.phi) + by * torch.cos(self.phi)
        return vr / (self.r * self.cfg.log_q), vt / self.r * self.cfg.Nt / (2 * math.pi)

    @staticmethod
    def grid(S):
        """ring samples [B,K,N,C] -> field [B,C,K,N]"""
        return S.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def ungrid(u):
        return u.permute(0, 2, 3, 1).contiguous()

    def fields(self, u, u_prev, dt):
        """understanding: present frame + its rate of change -> (v_k, v_n, q), all per second."""
        c = self.cfg
        dt = torch.as_tensor(dt, dtype=u.dtype, device=u.device).reshape(-1, 1, 1, 1)
        rate = (u - u_prev) * (c.dt_ref / dt)
        x = torch.cat([u - 0.5, rate, d_k(u), d_n(u), self.kcoord.expand(u.shape[0], 1, -1, u.shape[3])], 1)
        o = self.G(x)
        vk, vn, q = o[:, :1] * c.v_scale, o[:, 1:2] * c.v_scale, o[:, 2:] * c.q_scale
        self.local = (vk, vn)
        if c.cam_lsq:
            self.last_cam = self.solve_camera(u, u_prev, dt)
            ck, cn = self.camera_flow(self.last_cam)
            vk, vn = vk + ck, vn + cn
        elif c.global_cam:
            self.last_cam = self.G.last_cam
            ck, cn = self.camera_flow(self.G.last_cam)
            vk, vn = vk + ck, vn + cn
        return vk, vn, q

    def camera(self):
        """the whole-frame camera motion from the last fields() call (v2.1 only)."""
        return self.last_cam

    def predict(self, S_prev, S_now, dt_in, horizons):
        """Given two frames dt_in apart, predict ring samples at each future horizon (seconds after S_now)."""
        u, up = self.grid(S_now), self.grid(S_prev)
        vk, vn, q = self.fields(u, up, dt_in)
        return torch.stack([self.ungrid(transport(u, vk, vn, q, h)) for h in horizons], 1)

    def loss(self, S, dt, lam_q=0.005, lam_tv=1e-4, lam_cons=0.1, lam_loc=0.0, lam_res=0.0):
        """S [B,F,K,N,C]: frames f0..f(F-1) at interval dt [B]. Fields from (f0,f1) must predict f2..f(F-1);
        fields read at f2 from (f1,f2) and from (f0,f2) must agree (the derivative cannot depend on the interval).
        lam_loc > 0 (v2.1) makes local flow cost more than the 4 whole-frame camera numbers.
        lam_res > 0 (v2.2) also asks the equation itself to hold at the present moment (derivative level, no integration):
        (u1 - u0)/dt = -v_k du/dk - v_n du/dn + q, with the spatial derivatives taken at the midpoint frame."""
        n_f = S.shape[1]
        u = [self.grid(S[:, i]) for i in range(n_f)]
        vk, vn, q = self.fields(u[1], u[0], dt)
        lk, ln = self.local
        loc = (lk.abs().mean() + ln.abs().mean()) / self.cfg.v_scale
        dtv = torch.as_tensor(dt, dtype=u[1].dtype, device=u[1].device).reshape(-1, 1, 1, 1)
        um = 0.5 * (u[0] + u[1])
        resid = (u[1] - u[0]) / dtv + vk * d_k(um) + vn * d_n(um) - q
        res = (resid * self.cfg.dt_ref).pow(2).mean()
        pred = 0.0
        for j in range(1, n_f - 1):
            pred = pred + F.mse_loss(transport(u[1], vk, vn, q, dt * j), u[1 + j])
        pred = pred / (n_f - 2)
        a = self.fields(u[2], u[1], dt)
        b = self.fields(u[2], u[0], 2 * dt)
        sc = 1.0 / self.cfg.v_scale
        cons = sum(F.mse_loss(x * sc, y * sc) for x, y in zip(a, b)) / 3
        src = q.abs().mean() / self.cfg.q_scale
        tv = ((vk[..., 1:, :] - vk[..., :-1, :]).abs().mean() + (vn[..., 1:] - vn[..., :-1]).abs().mean()) / self.cfg.v_scale
        total = pred + lam_cons * cons + lam_q * src + lam_tv * tv + lam_loc * loc + lam_res * res
        return total, {"pred": pred.item(), "cons": cons.item(), "src": src.item(), "tv": tv.item(), "loc": loc.item(), "res": res.item()}
