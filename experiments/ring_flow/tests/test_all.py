import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from rf.ring import RingConfig, RingFilter, fit_splats, render_splats, psnr
from rf.world import World, sample_world
from rf.motion import estimate_motion, integrate, count_sign_changes
from rf.model import TemporalFlow, RadialFlow, z_to_real, real_to_z
from rf.data import temporal_triples

RF = RingFilter(RingConfig())


def test_constant_image_roundtrip():
    img = torch.full((1, 3, 128, 128), 0.4)
    Z = RF.encode(img)
    assert Z.shape == (1, 24, 17, 3)
    assert torch.allclose(Z[:, :, 1:].abs(), torch.zeros(1), atol=1e-5)
    R = RF.render(Z)
    assert torch.allclose(R, img, atol=1e-4)


def test_rotation_is_phase_shift():
    w = sample_world(7, "static", n_obj=2)
    img = w.video([0.2])
    rot = torch.rot90(img, 1, dims=(2, 3))  # 90 degrees about the center
    Z0, Z1 = RF.encode(img), RF.encode(rot)
    # magnitudes of every harmonic are rotation invariant
    rel = (Z0.abs() - Z1.abs()).abs().mean() / Z0.abs().mean()
    assert rel < 0.02, rel


def test_splats_reduce_error():
    img = sample_world(3, "static", n_obj=3).video([0.3])[0]
    R = RF.render(RF.encode(img[None]))[0]
    c, col, sg = fit_splats(img - R, 100, rf=RF)
    assert psnr(R + render_splats(c, col, 128, 128, sg), img) > psnr(R, img) + 0.5


def test_world_gt_camera_matches_finite_difference():
    w = World(seed=1, n_obj=0, cam_s=0.2, cam_w=-0.3, cam_b=(5.0, -3.0))
    t, h = 1.0, 1e-5
    xw = np.array([[30.0, 90.0], [100.0, 20.0], [70.0, 70.0]])
    x0 = np.stack([w.world_to_img(p, t) for p in xw]); x1 = np.stack([w.world_to_img(p, t + h) for p in xw])
    v = (x1 - x0) / h
    s, om, bx, by = w.gt_camera(t)
    xr = x0 - w.c
    vm = s * xr + om * np.stack([-xr[:, 1], xr[:, 0]], 1) + np.array([bx, by])
    assert np.abs(v - vm).max() < 1e-3


def test_camera_motion_recovered():
    for mode in ["zoom", "rotate", "pan", "mixed"]:
        w = sample_world(11, mode, n_obj=0, T_max=1.5)
        dt = 1 / 30
        v = w.video([1.0, 1.0 + dt])
        p = estimate_motion(RF, v[0], v[1], dt)["params"]
        gt = w.gt_camera(1.0 + dt / 2)
        assert np.allclose(p[:2], gt[:2], atol=0.01), (mode, p, gt)
        assert np.allclose(p[2:], gt[2:], atol=0.4), (mode, p, gt)


def test_on_demand_integration():
    t = np.linspace(0, 1, 101)
    assert abs(integrate(np.ones_like(t), t[1] - t[0])[-1] - 1.0) < 1e-9
    vy = np.sin(2 * math.pi * 2 * t)  # two + to - crossings
    assert count_sign_changes(vy) == 2


def test_triples_reconstruct_exactly():
    Z = torch.randn(2, 6, 4, 3, 2, dtype=torch.complex64)
    dt = 0.1
    z, v, a = temporal_triples(Z, dt)
    z, v, a = [x.view(2, 4, *x.shape[1:]) for x in (z, v, a)]
    nxt = z + (v + a * dt) * dt  # the integration step used at rollout
    assert torch.allclose(nxt, Z[:, 2:], atol=1e-5)


def test_flows_run():
    K, D = 24, 17 * 3 * 2
    tf, rf = TemporalFlow(K, D, hidden=32, layers=1), RadialFlow(K, D, hidden=32, layers=1)
    x = torch.randn(4, K, D)
    assert torch.isfinite(tf.loss(x, x, x)) and torch.isfinite(rf.loss(x))
    assert tf.sample(x, x, steps=2).shape == (4, K, D)
    assert rf.sample(2, steps=2, prefix=x[:2, :20]).shape == (2, K, D)
    Z = real_to_z(x, 17, 3)
    assert torch.allclose(z_to_real(Z), x)


# ---------------- v1 (learned) ----------------

def test_v1_cells_roundtrip_and_shapes():
    from rf.v1_model import RingWorld, V1Config
    cfg = V1Config(d=32, enc_layers=1, dec_layers=1, der_layers=1, int_layers=1)
    net = RingWorld(cfg)
    S = torch.rand(2, cfg.K, cfg.Nt, cfg.C)
    assert torch.equal(net.from_cells(net.to_cells(S)), S)
    h = net.encode(S)
    assert h.shape == (2, cfg.K * cfg.P, cfg.d)
    m, g = net.derivative(h, h, 1 / 30)
    assert m.shape == (2, cfg.K * cfg.P, cfg.dm) and g.shape == (2, cfg.dg)
    # integrator starts as identity (zero-initialized output), so an untrained rollout = copy of the last state
    assert torch.allclose(net.step(h, m, g, 1 / 30), h)
    assert net.rollout(S, S, 1 / 30, 3).shape == (2, 3, cfg.K, cfg.Nt, cfg.C)


def test_v1_loss_decreases():
    from rf.v1_model import RingWorld, V1Config
    torch.manual_seed(0)
    cfg = V1Config(d=32, enc_layers=1, dec_layers=1, der_layers=1, int_layers=1)
    net = RingWorld(cfg)
    S = torch.rand(4, 4, cfg.K, cfg.Nt, cfg.C)
    dt = torch.full((4,), 1 / 30)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    first = None
    for _ in range(30):
        loss, _ = net.loss(S, dt)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first


# ---------------- v2 (equation of motion on the ring field) ----------------

def test_v2_transport_exact_for_uniform_flow_at_any_dt():
    from rf.v2_model import transport
    u = torch.rand(2, 3, 24, 64)
    z = torch.zeros(2, 1, 24, 64)
    # rotation by 3 angle cells: one big step == many small ones == exact roll
    for dt in (1.5, 0.75):
        out = transport(u, z, z + 3.0 / dt, torch.zeros_like(u), dt)
        assert torch.allclose(out, torch.roll(u, 3, 3), atol=1e-5)
    # zoom by 2 ring cells
    out = transport(u, z + 1.0, z, torch.zeros_like(u), 2.0)
    assert torch.allclose(out[:, :, 2:], u[:, :, :-2], atol=1e-5)
    # semigroup: two half steps == one full step (non-uniform flow, within interpolation error)
    vk = torch.randn(2, 1, 24, 64).abs() * 0.3
    one = transport(u, vk, z, torch.zeros_like(u), 1.0)
    two = transport(transport(u, vk, z, torch.zeros_like(u), 0.5), vk, z, torch.zeros_like(u), 0.5)
    assert (one - two).abs().mean() < 0.05


def test_v2_true_camera_flow_explains_camera_motion():
    """No learning: shifting the ring field by the true zoom/rotation predicts the next frame far better than copying."""
    import math
    from rf.v2_model import transport, RingPDE
    rf = RingFilter(RingConfig())
    w = sample_world(7, "rotate", n_obj=0)
    S = rf.samples(w.video([0.5, 0.5 + 0.1]))
    s, om, _, _ = w.gt_camera(0.5)
    u0, u1 = RingPDE.grid(S[:1]), RingPDE.grid(S[1:2])
    z = torch.zeros(1, 1, 24, 64)
    p = transport(u0, z + s / math.log(rf.cfg.q), z + om * 64 / (2 * math.pi), torch.zeros_like(u0), 0.1)
    assert ((p - u1) ** 2).mean() < 0.2 * ((u0 - u1) ** 2).mean()


def test_v2_loss_runs_and_decreases():
    from rf.v2_model import RingPDE, V2Config
    torch.manual_seed(0)
    net = RingPDE(V2Config(width=8))
    S = torch.rand(2, 5, 24, 64, 3)
    dt = torch.full((2,), 1 / 15)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    first = None
    for _ in range(15):
        loss, _ = net.loss(S, dt)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first
