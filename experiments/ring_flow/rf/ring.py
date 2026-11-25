"""Fixed-center ring filter, renderer and detail splats.

Everything here is fixed math with no learnable parameters.

- RingFilter.encode: frame -> ring samples S [B,K,Nt,C] and ring coefficients Z [B,K,M+1,C] (complex).
  Ring k has radius r_k = r0 * q**k around the frame center. Each sample averages a Gaussian
  neighbourhood whose width grows with r_k. Gaussian blur is the heat equation, so ring k sees the
  frame after a diffusion time sigma_k^2 / 2.
- RingFilter.render: Z -> frame, by inverse polar interpolation (inner rings sharp, outer rings blurry).
- fit_splats / render_splats: add back the detail the rings lost, using a few small Gaussians.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


@dataclass
class RingConfig:
    H: int = 128
    W: int = 128
    K: int = 24            # number of rings
    Nt: int = 64           # samples per ring
    M: int = 16            # angular harmonics kept (m = 0..M)
    r0: float = 1.0        # radius of the innermost ring (px)
    rmax: float | None = None  # radius of the outermost ring; default: half the diagonal
    sigma_scale: float = 0.5   # blur width = sigma_scale * ring spacing
    sigma_min: float = 0.5

    def __post_init__(self):
        if self.rmax is None:
            self.rmax = 0.5 * math.hypot(self.H, self.W)

    @property
    def q(self) -> float:
        return (self.rmax / self.r0) ** (1.0 / (self.K - 1))

    def radii(self) -> torch.Tensor:
        return self.r0 * self.q ** torch.arange(self.K, dtype=torch.float32)

    def sigmas(self) -> torch.Tensor:
        r = self.radii()
        spacing = torch.maximum(r * (self.q - 1.0), r * 2 * math.pi / self.Nt)
        return torch.clamp(self.sigma_scale * spacing, min=self.sigma_min)

    def to_dict(self):
        return asdict(self)


def gaussian_blur_stack(img: torch.Tensor, sigmas: torch.Tensor, pad: int | None = None) -> torch.Tensor:
    """img [B,C,H,W] -> [B,K,C,H,W], blurred with each sigma (exact Gaussian via FFT, reflect padding)."""
    B, C, H, W = img.shape
    if pad is None:
        pad = int(min(max(3 * float(sigmas.max()), 4), min(H, W) - 1))
    x = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    Hp, Wp = x.shape[-2:]
    X = torch.fft.rfft2(x)  # [B,C,Hp,Wp//2+1]
    fy = torch.fft.fftfreq(Hp, device=img.device)
    fx = torch.fft.rfftfreq(Wp, device=img.device)
    f2 = fy[:, None] ** 2 + fx[None, :] ** 2  # cycles/px squared
    s = sigmas.to(img.device).view(-1, 1, 1)
    Hk = torch.exp(-2 * (math.pi ** 2) * (s ** 2) * f2)  # [K,Hp,Wf]
    Y = X[:, None] * Hk[None, :, None]  # [B,K,C,Hp,Wf]
    y = torch.fft.irfft2(Y, s=(Hp, Wp))
    return y[..., pad:pad + H, pad:pad + W]


class RingFilter:
    def __init__(self, cfg: RingConfig | None = None, center: tuple[float, float] | None = None, device="cpu"):
        self.cfg = cfg or RingConfig()
        c = self.cfg
        self.device = torch.device(device)
        # center in pixel coordinates (x, y); default = frame center
        self.cx, self.cy = center if center is not None else ((c.W - 1) / 2.0, (c.H - 1) / 2.0)
        self.r = c.radii().to(self.device)
        self.sig = c.sigmas().to(self.device)
        self.phi = (2 * math.pi * torch.arange(c.Nt, dtype=torch.float32) / c.Nt).to(self.device)
        # sample positions [K,Nt] in pixel coords
        self.px = self.cx + self.r[:, None] * torch.cos(self.phi)[None, :]
        self.py = self.cy + self.r[:, None] * torch.sin(self.phi)[None, :]
        self.valid = ((self.px >= 0) & (self.px <= c.W - 1) & (self.py >= 0) & (self.py <= c.H - 1)).float()

    # ---------- helpers ----------
    def _norm_grid(self, px, py):
        c = self.cfg
        gx = 2 * px / (c.W - 1) - 1
        gy = 2 * py / (c.H - 1) - 1
        return torch.stack([gx, gy], dim=-1)

    def blur_stack(self, img):
        return gaussian_blur_stack(img.to(self.device), self.sig)

    def sample_stack(self, stack: torch.Tensor) -> torch.Tensor:
        """stack [B,K,C,H,W] -> ring samples [B,K,Nt,C]; ring k reads blur level k."""
        B, K, C, H, W = stack.shape
        grid = self._norm_grid(self.px, self.py)  # [K,Nt,2]
        out = F.grid_sample(stack.reshape(B * K, C, H, W), grid[None].expand(B, K, -1, -1).reshape(B * K, 1, -1, 2),
                            mode="bilinear", padding_mode="border", align_corners=True)  # [B*K,C,1,Nt]
        return out.view(B, K, C, -1).permute(0, 1, 3, 2).contiguous()

    # ---------- encode / decode ----------
    def samples(self, img: torch.Tensor) -> torch.Tensor:
        return self.sample_stack(self.blur_stack(img))

    def coeffs_from_samples(self, S: torch.Tensor) -> torch.Tensor:
        Z = torch.fft.rfft(S, dim=2) / self.cfg.Nt  # [B,K,Nt//2+1,C]
        return Z[:, :, : self.cfg.M + 1]

    def samples_from_coeffs(self, Z: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        full = torch.zeros(Z.shape[0], c.K, c.Nt // 2 + 1, Z.shape[-1], dtype=Z.dtype, device=Z.device)
        full[:, :, : c.M + 1] = Z
        return torch.fft.irfft(full * c.Nt, n=c.Nt, dim=2)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """img [B,C,H,W] -> Z [B,K,M+1,C] complex."""
        return self.coeffs_from_samples(self.samples(img))

    def render_samples(self, S: torch.Tensor) -> torch.Tensor:
        """Inverse polar interpolation: ring samples [B,K,Nt,C] -> image [B,C,H,W]."""
        c = self.cfg
        B, K, Nt, C = S.shape
        ys, xs = torch.meshgrid(torch.arange(c.H, device=S.device, dtype=torch.float32),
                                torch.arange(c.W, device=S.device, dtype=torch.float32), indexing="ij")
        dx, dy = xs - self.cx, ys - self.cy
        rr = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1e-6)
        kk = (torch.log(rr / c.r0) / math.log(c.q)).clamp(0, K - 1)  # continuous ring index
        ph = torch.atan2(dy, dx) % (2 * math.pi)
        tt = ph / (2 * math.pi) * Nt  # continuous angle index in [0,Nt)
        # wrap: append first column so that index Nt maps to column 0
        Sw = torch.cat([S, S[:, :, :1]], dim=2)  # [B,K,Nt+1,C]
        img = Sw.permute(0, 3, 1, 2)  # [B,C,K,Nt+1]
        gx = 2 * tt / Nt - 1
        gy = 2 * kk / (K - 1) - 1
        grid = torch.stack([gx, gy], dim=-1)[None].expand(B, -1, -1, -1)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def render(self, Z: torch.Tensor) -> torch.Tensor:
        return self.render_samples(self.samples_from_coeffs(Z))

    def n_floats(self, C: int = 3) -> int:
        # m=0 coefficient is real; every other harmonic is complex
        return self.cfg.K * (1 + 2 * self.cfg.M) * C


# ---------------- detail splats ----------------

def _gauss_basis(centers: torch.Tensor, sigma, H: int, W: int) -> torch.Tensor:
    """sigma: float or per-splat tensor [n]"""
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=centers.device),
                            torch.arange(W, dtype=torch.float32, device=centers.device), indexing="ij")
    d2 = (xs.reshape(-1, 1) - centers[None, :, 0]) ** 2 + (ys.reshape(-1, 1) - centers[None, :, 1]) ** 2
    sig = torch.as_tensor(sigma, dtype=torch.float32, device=centers.device)
    return torch.exp(-d2 / (2 * sig ** 2))  # [H*W, n]


def splat_sigma(rf: "RingFilter", centers: torch.Tensor, frac: float = 0.35, lo: float = 1.0, hi: float = 3.0):
    """Splat width follows the ring blur at its radius: outer splats restore coarser lost detail."""
    c = rf.cfg
    r = torch.sqrt((centers[:, 0] - rf.cx) ** 2 + (centers[:, 1] - rf.cy) ** 2).clamp(min=1e-6)
    k = (torch.log(r / c.r0) / math.log(c.q)).clamp(0, c.K - 1)
    k0 = k.floor().long().clamp(max=c.K - 2); a = k - k0
    sig_ring = (1 - a) * rf.sig.cpu()[k0] + a * rf.sig.cpu()[k0 + 1]
    return (frac * sig_ring).clamp(lo, hi)


def fit_splats(residual: torch.Tensor, n: int, sigma=1.0, min_dist: float = 2.0, rf: "RingFilter | None" = None):
    """Pick n splat centers greedily where |residual| (lightly blurred) is largest, then solve the colors by least squares.

    residual [C,H,W] -> centers [n,2] (x,y), colors [n,C]
    """
    C, H, W = residual.shape
    if n <= 0:
        return torch.zeros(0, 2), torch.zeros(0, C), torch.zeros(0)
    e = gaussian_blur_stack(residual[None], torch.tensor([sigma]))[0, 0].abs().sum(0)  # [H,W]
    e = e.clone()
    centers = []
    rad = int(math.ceil(min_dist))
    for _ in range(n):
        idx = int(torch.argmax(e))
        y, x = divmod(idx, W)
        if e[y, x] <= 0:
            break
        centers.append((float(x), float(y)))
        e[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = 0
    centers = torch.tensor(centers, dtype=torch.float32, device=residual.device)
    if rf is not None:
        sigma = splat_sigma(rf, centers)
    A = _gauss_basis(centers, sigma, H, W)  # [HW,n]
    b = residual.reshape(C, -1).T  # [HW,C]
    colors = torch.linalg.lstsq(A, b).solution  # [n,C]
    return centers, colors, torch.as_tensor(sigma, dtype=torch.float32).expand(len(centers)).clone()


def render_splats(centers: torch.Tensor, colors: torch.Tensor, H: int, W: int, sigma=1.0) -> torch.Tensor:
    if centers.numel() == 0:
        return torch.zeros(colors.shape[-1] if colors.dim() == 2 else 3, H, W)
    A = _gauss_basis(centers, sigma, H, W)
    return (A @ colors).T.reshape(-1, H, W)


def psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    d2 = (a - b) ** 2
    if mask is not None:
        m = mask.to(d2.dtype).expand_as(d2)
        mse = (d2 * m).sum() / m.sum().clamp(min=1)
    else:
        mse = d2.mean()
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-12)))
