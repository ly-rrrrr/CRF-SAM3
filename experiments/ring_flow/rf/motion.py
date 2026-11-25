"""Understanding = the derivative field. Reads camera and local motion from ring derivatives, with no training.

At every ring sample x (relative to the center) we use brightness constancy on the ring-filtered frames:
    dI/dt + (s x + omega J x + b) . grad I = 0
and solve a 4-parameter weighted least squares for (s, omega, bx, by): zoom rate, rotation rate, pan.
Ring k reads blur level k, so outer rings only see coarse motion ("attention disperses outward").
Every ring has the same number of samples, so the center weighs most.
The robust (IRLS / Tukey) residual, arranged as a [K, Nt] map, is the local motion: something moving on its own.
Answers spanning time come from integrating these derivatives on demand (integrate()).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .ring import RingFilter


def _grad_xy(x: torch.Tensor):
    gy, gx = torch.gradient(x, dim=(-2, -1))
    return gx, gy


@torch.no_grad()
def ring_derivatives(rf: RingFilter, I0: torch.Tensor, I1: torch.Tensor, dt: float):
    """I0, I1 [C,H,W] -> It, gx, gy sampled on rings, each [K,Nt,C]."""
    B0 = rf.blur_stack(I0[None])
    B1 = rf.blur_stack(I1[None])
    Bm = 0.5 * (B0 + B1)
    gx, gy = _grad_xy(Bm)
    It = (B1 - B0) / dt
    return rf.sample_stack(It)[0], rf.sample_stack(gx)[0], rf.sample_stack(gy)[0]


@torch.no_grad()
def estimate_motion(rf: RingFilter, I0: torch.Tensor, I1: torch.Tensor, dt: float, irls: int = 6):
    """Returns dict: params=(s, omega, bx, by), resid=[K,Nt] local-motion map (px/s, approx), weights."""
    It, gx, gy = ring_derivatives(rf, I0, I1, dt)
    xr = (rf.px - rf.cx)[..., None].expand_as(gx).double()
    yr = (rf.py - rf.cy)[..., None].expand_as(gx).double()
    gx, gy, It = gx.double(), gy.double(), It.double()
    A = torch.stack([gx * xr + gy * yr, -gx * yr + gy * xr, gx, gy], dim=-1).reshape(-1, 4)
    y = (-It).reshape(-1)
    valid = rf.valid[..., None].expand_as(gx).reshape(-1).double()
    w = valid.clone()
    p = torch.zeros(4, dtype=torch.float64)
    for it in range(irls + 1):
        Aw = A * w[:, None]
        p = torch.linalg.solve(Aw.T @ A + 1e-9 * torch.eye(4, dtype=torch.float64), Aw.T @ y)
        r = (A @ p - y)
        if it < irls:
            # Tukey biweight: samples on independently moving things get weight 0
            rv = r[valid > 0].abs()
            c = 4.685 * 1.4826 * rv.median().clamp(min=1e-9)
            u = (r.abs() / c).clamp(max=1.0)
            w = valid * (1 - u ** 2) ** 2
    r = r.reshape(gx.shape)  # [K,Nt,C]
    gmag = torch.sqrt((gx ** 2 + gy ** 2).sum(-1)).clamp(min=1e-3)
    resid = torch.sqrt((r ** 2).sum(-1)) / gmag  # normal-flow speed not explained by the camera
    resid = resid * rf.valid.double()
    return {"params": p.float().numpy(), "resid": resid.float(), "weights": w.reshape(gx.shape).float()}


def resid_peak(rf: RingFilter, resid: torch.Tensor, smooth: int = 1):
    """Locate the strongest local motion: returns (x, y) pixel position, ring index, angle, value."""
    R = resid.clone()
    if smooth:
        R = (R + R.roll(1, 1) + R.roll(-1, 1)) / 3
    k, n = divmod(int(torch.argmax(R)), R.shape[1])
    return float(rf.px[k, n]), float(rf.py[k, n]), k, float(rf.phi[n]), float(R[k, n])


def integrate(values: np.ndarray, dt: float) -> np.ndarray:
    """Cumulative trapezoid of a sampled derivative: on-demand integration done by the question, not by the model."""
    v = np.asarray(values, float)
    out = np.zeros_like(v)
    out[1:] = np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, axis=0)
    return out


def count_sign_changes(x: np.ndarray, thresh: float = 0.0, from_pos: bool = True) -> int:
    """Count + to - (or - to +) crossings; e.g. 'how many times did it bounce' on a vertical velocity."""
    s = np.sign(np.where(np.abs(x) < thresh, 0, x))
    s = s[s != 0]
    if from_pos:
        return int(np.sum((s[:-1] > 0) & (s[1:] < 0)))
    return int(np.sum((s[:-1] < 0) & (s[1:] > 0)))
