"""A synthetic 2D world with continuous time, so it can be rendered at any frame rate with exactly known ground truth.

- Background: smooth colored texture (sum of sinusoids plus soft blobs), static in world coordinates.
- Objects: circles, squares and triangles under gravity. They bounce off the floor and the walls (continuous-time physics, fine substeps).
- Camera: zoom about the frame center, rotation and pan, all at constant rates. The camera's instantaneous image-motion
  parameters (s, omega, bx, by) are known analytically.

Image coords: x right, y down, pixels. The world is drawn in world coords; the camera maps world to image:
    x_img = c + exp(S(t)) R(Omega(t)) (x_w - c) + B(t),   S = s*t, Omega = w*t, B = b*t
The image velocity of a static world point is v(x) = s (x - c - B) + w J (x - c - B) + b, with J(x, y) = (-y, x).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

SHAPES = ("circle", "square", "triangle")


@dataclass
class Obj:
    shape: str
    radius: float
    color: tuple
    pos: tuple
    vel: tuple


@dataclass
class World:
    H: int = 128
    W: int = 128
    seed: int = 0
    n_obj: int = 2
    gravity: float = 400.0          # px / s^2 (world units)
    restitution: float = 0.85
    cam_s: float = 0.0              # zoom rate (1/s)
    cam_w: float = 0.0              # rotation rate (rad/s)
    cam_b: tuple = (0.0, 0.0)       # pan rate (px/s)
    T_max: float = 4.0
    h: float = 1e-3                 # physics substep
    objs: list = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self._rng = rng
        # background texture
        nw = 6
        self.bg_base = rng.uniform(0.25, 0.6, size=3)
        self.bg_k = rng.normal(size=(nw, 2)) * (2 * math.pi / rng.uniform(18, 70, size=(nw, 1)))
        self.bg_ph = rng.uniform(0, 2 * math.pi, size=nw)
        self.bg_amp = rng.uniform(0.03, 0.09, size=(nw, 3))
        nb = 10
        self.blob_p = rng.uniform(-0.3, 1.3, size=(nb, 2)) * np.array([self.W, self.H])
        self.blob_r = rng.uniform(4, 14, size=nb)
        self.blob_c = rng.uniform(-0.2, 0.2, size=(nb, 3))
        if not self.objs:
            for _ in range(self.n_obj):
                R = float(rng.uniform(6, 13))
                self.objs.append(Obj(
                    shape=str(rng.choice(SHAPES)), radius=R,
                    color=tuple(rng.uniform(0.05, 1.0, size=3)),
                    pos=(float(rng.uniform(R + 4, self.W - R - 4)), float(rng.uniform(R + 4, self.H * 0.6))),
                    vel=(float(rng.uniform(-60, 60)), float(rng.uniform(-80, 20))),
                ))
        self._simulate()

    # ---------------- physics ----------------
    def _simulate(self):
        n = int(math.ceil(self.T_max / self.h)) + 2
        self.traj = np.zeros((len(self.objs), n, 2))
        for i, o in enumerate(self.objs):
            p = np.array(o.pos, float)
            v = np.array(o.vel, float)
            R = o.radius
            for j in range(n):
                self.traj[i, j] = p
                v[1] += self.gravity * self.h
                p = p + v * self.h
                if p[1] > self.H - 1 - R and v[1] > 0:
                    p[1] = self.H - 1 - R
                    v[1] = -self.restitution * v[1]
                    if abs(v[1]) < 15:  # come to rest
                        v[1] = 0.0
                for d, lim in ((0, self.W - 1),):
                    if p[d] < R and v[d] < 0:
                        p[d] = R; v[d] = -v[d]
                    if p[d] > lim - R and v[d] > 0:
                        p[d] = lim - R; v[d] = -v[d]

    def obj_pos(self, t: float) -> np.ndarray:
        """world positions [n_obj,2] at time t (linear interp of fine trajectory)."""
        if len(self.objs) == 0:
            return np.zeros((0, 2))
        f = t / self.h
        j = int(math.floor(f)); a = f - j
        j = min(max(j, 0), self.traj.shape[1] - 2)
        return (1 - a) * self.traj[:, j] + a * self.traj[:, j + 1]

    # ---------------- camera ----------------
    @property
    def c(self):
        return np.array([(self.W - 1) / 2.0, (self.H - 1) / 2.0])

    def cam(self, t):
        return self.cam_s * t, self.cam_w * t, np.array(self.cam_b, float) * t

    def world_to_img(self, xw: np.ndarray, t: float) -> np.ndarray:
        S, Om, B = self.cam(t)
        R = np.array([[math.cos(Om), -math.sin(Om)], [math.sin(Om), math.cos(Om)]])
        return self.c + math.exp(S) * (xw - self.c) @ R.T + B

    def gt_camera(self, t: float):
        """Instantaneous (s, omega, bx, by) in the estimator's model v = s x' + omega J x' + b, x' = x - c."""
        _, _, B = self.cam(t)
        s, w = self.cam_s, self.cam_w
        JB = np.array([-B[1], B[0]])
        b = np.array(self.cam_b, float) - s * B - w * JB
        return np.array([s, w, b[0], b[1]])

    # ---------------- rendering ----------------
    def _bg(self, X, Y):
        arg = X[..., None] * self.bg_k[:, 0] + Y[..., None] * self.bg_k[:, 1] + self.bg_ph  # [...,nw]
        col = self.bg_base + np.sin(arg) @ self.bg_amp  # [...,3]
        for p, r, cc in zip(self.blob_p, self.blob_r, self.blob_c):
            d = np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2)
            a = 1.0 / (1.0 + np.exp((d - r) / 1.5))
            col = col + a[..., None] * cc
        return col

    @staticmethod
    def _sd(shape, X, Y, p, R):
        dx, dy = X - p[0], Y - p[1]
        if shape == "circle":
            return np.sqrt(dx ** 2 + dy ** 2) - R
        if shape == "square":
            return np.maximum(np.abs(dx), np.abs(dy)) - 0.85 * R
        # equilateral triangle pointing up
        n = [(0.0, 1.0), (math.sqrt(3) / 2, -0.5), (-math.sqrt(3) / 2, -0.5)]
        return np.max(np.stack([dx * a + dy * b for a, b in n]), axis=0) - 0.55 * R

    def frame(self, t: float, ss: int = 2) -> np.ndarray:
        """Render frame at time t -> [3,H,W] float in [0,1] (supersampled ss x ss)."""
        S, Om, B = self.cam(t)
        o = (np.arange(ss) + 0.5) / ss - 0.5
        ys, xs = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        X = (xs[..., None, None] + o[None, None, None, :])  # [H,W,1,ss]
        Y = (ys[..., None, None] + o[None, None, :, None])  # [H,W,ss,1]
        X, Y = np.broadcast_arrays(X, Y)
        # image -> world
        u, v = X - self.c[0] - B[0], Y - self.c[1] - B[1]
        e = math.exp(-S); co, si = math.cos(-Om), math.sin(-Om)
        Xw = self.c[0] + e * (co * u - si * v)
        Yw = self.c[1] + e * (si * u + co * v)
        col = self._bg(Xw, Yw)
        px_world = e  # size of one image pixel in world units
        P = self.obj_pos(t)
        for i, ob in enumerate(self.objs):
            sd = self._sd(ob.shape, Xw, Yw, P[i], ob.radius)
            a = np.clip(0.5 - sd / px_world, 0, 1)[..., None]
            col = col * (1 - a) + a * np.array(ob.color)
        col = col.mean(axis=(2, 3))
        return np.clip(col, 0, 1).transpose(2, 0, 1).astype(np.float32)

    def video(self, times) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.frame(float(t)) for t in times]))

    def obj_img_pos(self, t: float) -> np.ndarray:
        P = self.obj_pos(t)
        return np.stack([self.world_to_img(p, t) for p in P]) if len(P) else np.zeros((0, 2))


def sample_world(seed: int, mode: str = "mixed", n_obj: int | None = None, H: int = 128, W: int = 128, T_max: float = 4.0) -> World:
    rng = np.random.default_rng(10_000 + seed)
    s = w = 0.0
    b = (0.0, 0.0)
    if mode in ("zoom", "mixed"):
        s = float(rng.uniform(-0.25, 0.25))
    if mode in ("rotate", "mixed"):
        w = float(rng.uniform(-0.4, 0.4))
    if mode in ("pan", "mixed"):
        b = (float(rng.uniform(-12, 12)), float(rng.uniform(-12, 12)))
    if n_obj is None:
        n_obj = int(rng.integers(1, 4))
    return World(H=H, W=W, seed=seed, n_obj=n_obj, cam_s=s, cam_w=w, cam_b=b, T_max=T_max)
