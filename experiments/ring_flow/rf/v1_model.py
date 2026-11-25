"""v1: everything after the fixed retina is learned.

Fixed (no parameters): the retina. Center at the frame center, K log-spaced rings, Nt samples per ring,
ring k reads the frame blurred by sigma_k (rf.ring.RingFilter.samples).

Learned (four networks, trained together, self-supervised on raw video):
- Encoder E:    ring samples of ONE frame -> state h, one token per polar cell (ring k, angle sector p).
- Derivative G: (h_t, h_t - h_prev) -> a small motion code: m (a few numbers per cell) + g (a few numbers for the
                whole frame). This is "understanding = derivative": the only thing passed from the past to the
                future. There is no memory beyond the previous frame.
- Integrator V: (h_t, m, g) -> dh/dt. One Euler step: h_next = h_t + dt * dh/dt. Changing the frame rate only
                changes dt. This is "generation = integration".
- Decoder D:    state h -> ring samples, so every prediction is checked against what the retina really sees.

The bottleneck (m, g) is the point: the predictor never sees the raw velocity, so whatever motion the future
needs must be squeezed through m and g. After training we probe them: did camera zoom / rotation / pan and the
moving objects show up there by themselves?
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class V1Config:
    K: int = 24
    Nt: int = 64
    C: int = 3
    P: int = 8            # angle sectors per ring -> K*P tokens (polar cells)
    d: int = 128          # token width
    enc_layers: int = 4
    dec_layers: int = 2
    der_layers: int = 2
    int_layers: int = 3
    heads: int = 4
    dm: int = 4           # motion code per cell
    dg: int = 8           # motion code for the whole frame
    dt_ref: float = 1 / 30

    def to_dict(self):
        return asdict(self)


def _tf(d, layers, heads):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class PolarPos(nn.Module):
    """Position of a polar cell = which ring (learned) + which angle (fixed sin/cos, shared by all rings)."""

    def __init__(self, K, P, d):
        super().__init__()
        self.ring = nn.Parameter(torch.randn(K, 1, d) * 0.02)
        ang = (torch.arange(P) + 0.5) / P * 2 * math.pi
        feats = torch.cat([torch.stack([torch.cos(h * ang), torch.sin(h * ang)], -1) for h in range(1, 5)], -1)  # [P,8]
        self.register_buffer("ang", feats)
        self.lin = nn.Linear(8, d)

    def forward(self):
        return (self.ring + self.lin(self.ang)[None]).reshape(-1, self.ring.shape[-1])  # [K*P,d]


class RingWorld(nn.Module):
    def __init__(self, cfg: V1Config):
        super().__init__()
        self.cfg = c = cfg
        assert c.Nt % c.P == 0
        self.cell = (c.Nt // c.P) * c.C
        self.N = c.K * c.P
        # encoder / decoder
        self.pos_e, self.pos_d = PolarPos(c.K, c.P, c.d), PolarPos(c.K, c.P, c.d)
        self.inp = nn.Linear(self.cell, c.d)
        self.enc = _tf(c.d, c.enc_layers, c.heads)
        self.enc_norm = nn.LayerNorm(c.d)
        self.dec = _tf(c.d, c.dec_layers, c.heads)
        self.out = nn.Sequential(nn.LayerNorm(c.d), nn.Linear(c.d, self.cell))
        # derivative (understanding)
        self.pos_g = PolarPos(c.K, c.P, c.d)
        self.g_in = nn.Linear(2 * c.d, c.d)
        self.g_tok = nn.Parameter(torch.randn(1, 1, c.d) * 0.02)
        self.der = _tf(c.d, c.der_layers, c.heads)
        self.m_out = nn.Sequential(nn.LayerNorm(c.d), nn.Linear(c.d, c.dm))
        self.gl_out = nn.Sequential(nn.LayerNorm(c.d), nn.Linear(c.d, c.dg))
        # integrator (generation)
        self.pos_v = PolarPos(c.K, c.P, c.d)
        self.v_in = nn.Linear(c.d + c.dm + c.dg, c.d)
        self.vel = _tf(c.d, c.int_layers, c.heads)
        self.v_out = nn.Sequential(nn.LayerNorm(c.d), nn.Linear(c.d, c.d))
        nn.init.zeros_(self.v_out[1].weight); nn.init.zeros_(self.v_out[1].bias)

    # ring samples [B,K,Nt,C] <-> cells [B,K*P,cell]
    def to_cells(self, S):
        B = S.shape[0]; c = self.cfg
        return S.reshape(B, c.K, c.P, c.Nt // c.P, c.C).reshape(B, self.N, self.cell)

    def from_cells(self, X):
        B = X.shape[0]; c = self.cfg
        return X.reshape(B, c.K, c.P, c.Nt // c.P, c.C).reshape(B, c.K, c.Nt, c.C)

    def encode(self, S):
        return self.enc_norm(self.enc(self.inp(self.to_cells(S) - 0.5) + self.pos_e()))

    def decode(self, h):
        return self.from_cells(self.out(self.dec(h + self.pos_d())) + 0.5)

    def derivative(self, h, h_prev, dt):
        """understanding: (state, rate of change) -> motion code (m per cell, g per frame)."""
        rate = (h - h_prev) * (self.cfg.dt_ref / dt)
        x = self.g_in(torch.cat([h, rate], -1)) + self.pos_g()
        x = self.der(torch.cat([self.g_tok.expand(x.shape[0], -1, -1), x], 1))
        return self.m_out(x[:, 1:]), self.gl_out(x[:, 0])

    def step(self, h, m, g, dt):
        """generation: one Euler step of dh/dt = V(h, m, g)."""
        x = self.v_in(torch.cat([h, m, g[:, None].expand(-1, h.shape[1], -1)], -1)) + self.pos_v()
        return h + (dt / self.cfg.dt_ref) * self.v_out(self.vel(x))

    def rollout(self, S0, S1, dt, steps):
        """Given two frames, predict `steps` future frames with no other memory. Returns ring samples [B,steps,K,Nt,C]."""
        hp, h = self.encode(S0), self.encode(S1)
        out = []
        for _ in range(steps):
            m, g = self.derivative(h, hp, dt)
            hp, h = h, self.step(h, m, g, dt)
            out.append(self.decode(h))
        return torch.stack(out, 1)

    def loss(self, S, dt):
        """S [B,4,K,Nt,C]: frames t-1, t, t+1, t+2 at interval dt [B]. Returns total loss and parts."""
        B = S.shape[0]
        h = self.encode(S.flatten(0, 1)).view(B, 4, self.N, -1)
        rec = F.mse_loss(self.decode(h.flatten(0, 1)), S.flatten(0, 1))
        dtv = dt.view(B, 1, 1)
        m, g = self.derivative(h[:, 1], h[:, 0], dtv)
        h2 = self.step(h[:, 1], m, g, dtv)
        m2, g2 = self.derivative(h2, h[:, 1], dtv)
        h3 = self.step(h2, m2, g2, dtv)
        p1 = F.mse_loss(self.decode(h2), S[:, 2])
        p2 = F.mse_loss(self.decode(h3), S[:, 3])
        lat = F.mse_loss(h2, h[:, 2].detach()) + F.mse_loss(h3, h[:, 3].detach())
        total = rec + p1 + p2 + 0.1 * lat
        return total, {"rec": rec.item(), "pred1": p1.item(), "pred2": p2.item(), "lat": lat.item()}
