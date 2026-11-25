"""The calculus engine: flow matching on two axes.

- RadialFlow (within a frame): generates ring k from rings < k, from the center outward. Ring k sees the frame after
  heat-diffusion time sigma_k^2/2, so going outward runs with the diffusion and generating detail runs against it.
- TemporalFlow (between frames): the state is only (Z, dZ/dt), with no memory. It learns the distribution of the
  acceleration a, in per-unit-time quantities, so changing the frame rate only changes the integration step.
Both are small transformers whose tokens are the K rings.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def z_to_real(Z: torch.Tensor) -> torch.Tensor:
    """complex [B,K,M+1,C] -> real [B,K,(M+1)*C*2]"""
    B, K = Z.shape[:2]
    return torch.view_as_real(Z).reshape(B, K, -1)


def real_to_z(x: torch.Tensor, M1: int, C: int) -> torch.Tensor:
    B, K = x.shape[:2]
    return torch.view_as_complex(x.reshape(B, K, M1, C, 2).contiguous())


class Normalizer(nn.Module):
    def __init__(self, dim_shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim_shape))
        self.register_buffer("std", torch.ones(dim_shape))

    def fit(self, x: torch.Tensor):
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp(min=1e-4))
        return self

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inv(self, x):
        return x * self.std + self.mean


class TauEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(16) * 4.0, requires_grad=False)
        self.lin = nn.Sequential(nn.Linear(32, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, tau):  # tau [B] or [B,K]
        a = tau[..., None] * self.freqs * 2 * math.pi
        return self.lin(torch.cat([a.sin(), a.cos()], -1))


class RingNet(nn.Module):
    def __init__(self, K, d_in, d_out, hidden=256, layers=4, heads=4, causal=False):
        super().__init__()
        self.K, self.causal = K, causal
        self.inp = nn.Linear(d_in, hidden)
        self.ring = nn.Parameter(torch.zeros(1, K, hidden))
        nn.init.normal_(self.ring, std=0.02)
        self.tau = TauEmbed(hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d_out))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)

    def forward(self, x, tau):
        h = self.inp(x) + self.ring + (self.tau(tau) if tau.dim() == 2 else self.tau(tau)[:, None])
        mask = None
        if self.causal:
            mask = torch.triu(torch.full((self.K, self.K), float("-inf"), device=x.device), 1)
        return self.out(self.enc(h, mask=mask))


class TemporalFlow(nn.Module):
    """u_t(x_tau, tau | Z, Zdot) -> velocity towards the acceleration a (all normalized)."""

    def __init__(self, K, D, **kw):
        super().__init__()
        self.D = D
        self.nz, self.nv, self.na = Normalizer((K, D)), Normalizer((K, D)), Normalizer((K, D))
        self.net = RingNet(K, 3 * D, D, **kw)

    def loss(self, z, v, a):
        zn, vn, an = self.nz(z), self.nv(v), self.na(a)
        eps = torch.randn_like(an)
        tau = torch.rand(an.shape[0], device=an.device)
        xt = (1 - tau)[:, None, None] * eps + tau[:, None, None] * an
        u = self.net(torch.cat([zn, vn, xt], -1), tau)
        return ((u - (an - eps)) ** 2).mean()

    @torch.no_grad()
    def sample(self, z, v, steps=16):
        zn, vn = self.nz(z), self.nv(v)
        x = torch.randn_like(zn)
        for i in range(steps):
            tau = torch.full((z.shape[0],), i / steps, device=z.device)
            x = x + self.net(torch.cat([zn, vn, x], -1), tau) / steps
        return self.na.inv(x)


class RadialFlow(nn.Module):
    """Causal over rings: token k sees clean rings < k (shifted input) plus its own noisy x_tau."""

    def __init__(self, K, D, **kw):
        super().__init__()
        self.K, self.D = K, D
        self.nz = Normalizer((K, D))
        self.net = RingNet(K, 2 * D, D, causal=True, **kw)

    def _shift(self, zn):
        return torch.cat([torch.zeros_like(zn[:, :1]), zn[:, :-1]], 1)

    def loss(self, z):
        zn = self.nz(z)
        eps = torch.randn_like(zn)
        tau = torch.rand(zn.shape[0], self.K, device=zn.device)  # independent per ring
        xt = (1 - tau)[..., None] * eps + tau[..., None] * zn
        u = self.net(torch.cat([self._shift(zn), xt], -1), tau)
        return ((u - (zn - eps)) ** 2).mean()

    @torch.no_grad()
    def sample(self, n, steps=16, prefix=None, device="cpu"):
        """Generate rings outward. prefix: real [n,k0,D] of known inner rings (un-normalized)."""
        zn = torch.zeros(n, self.K, self.D, device=device)
        k0 = 0
        if prefix is not None:
            k0 = prefix.shape[1]
            zn[:, :k0] = (prefix - self.nz.mean[:k0]) / self.nz.std[:k0]
        for k in range(k0, self.K):
            x = torch.randn(n, self.D, device=device)
            for i in range(steps):
                # same distribution as training: finished rings = clean (tau=1), current ring = x at tau,
                # later rings are never seen by ring k (causal), so they can stay as pure noise (tau=0)
                tau = torch.zeros(n, self.K, device=device)
                tau[:, :k] = 1.0
                tau[:, k] = i / steps
                inp = torch.randn(n, self.K, self.D, device=device)
                inp[:, :k] = zn[:, :k]
                inp[:, k] = x
                u = self.net(torch.cat([self._shift(zn), inp], -1), tau)[:, k]
                x = x + u / steps
            zn[:, k] = x
        return self.nz.inv(zn)
