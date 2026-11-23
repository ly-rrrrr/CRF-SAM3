"""Capture the attention that chosen query rows pay to chosen key columns in
every Qwen3-VL *text* layer, without changing the attention backend.

Mechanism: the text attention calls the module-level ``apply_rotary_pos_emb``
right after building q/k (the vision tower uses ``apply_rotary_pos_emb_vision``,
a different function). We wrap that function, take the post-RoPE q/k it
returns and compute softmax(q_rows · k^T / sqrt(d)) for the requested rows
only (causal mask applied by hand). Cost: H x n_rows x L per layer.

Works with sdpa / eager / flash because the real attention is untouched.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence

import torch

_STATE: Dict = {"active": False, "rows": None, "cols": None, "maps": []}
_PATCHED = {"done": False}


def _row_attention(q: torch.Tensor, k: torch.Tensor, rows: Sequence[int], cols: torch.Tensor) -> torch.Tensor:
    """q: (B,H,L,d) k: (B,Hk,L,d) post-RoPE. Returns (n_rows, n_cols) head-mean attention."""
    qr = q[0, :, list(rows), :].float()  # H,n,d
    kk = k[0].float()  # Hk,L,d
    rep = qr.shape[0] // kk.shape[0]
    if rep > 1:
        kk = kk.repeat_interleave(rep, dim=0)
    s = torch.matmul(qr, kk.transpose(-1, -2)) * (qr.shape[-1] ** -0.5)  # H,n,L
    L = kk.shape[1]
    r = torch.tensor(list(rows), device=s.device)
    future = torch.arange(L, device=s.device)[None, :] > r[:, None]
    s = s.masked_fill(future[None], float("-inf"))
    a = torch.softmax(s, dim=-1)
    return a[:, :, cols.to(a.device)].mean(0).cpu()


def install(modeling_module=None) -> None:
    """Wrap apply_rotary_pos_emb in the Qwen3-VL modeling module (idempotent)."""
    if _PATCHED["done"]:
        return
    if modeling_module is None:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling_module
    orig = modeling_module.apply_rotary_pos_emb

    def wrapped(q, k, cos, sin, *a, **kw):
        q2, k2 = orig(q, k, cos, sin, *a, **kw)
        if _STATE["active"] and q2.dim() == 4 and q2.shape[2] > max(_STATE["rows"]):
            _STATE["maps"].append(_row_attention(q2, k2, _STATE["rows"], _STATE["cols"]))
        return q2, k2

    modeling_module.apply_rotary_pos_emb = wrapped
    _PATCHED["done"] = True


@contextlib.contextmanager
def capture(rows: Sequence[int], cols: torch.Tensor):
    """Inside the block, every text layer appends a (n_rows, n_cols) map."""
    _STATE.update(active=True, rows=list(rows), cols=cols, maps=[])
    try:
        yield _STATE["maps"]
    finally:
        _STATE["active"] = False


def layer_maps(model, model_inputs: Dict, rows: Sequence[int], cols: torch.Tensor) -> torch.Tensor:
    """One teacher-forced forward; returns (num_layers, n_rows, n_cols)."""
    install()
    with torch.inference_mode(), capture(rows, cols) as maps:
        model(**model_inputs, use_cache=False)
    if not maps:
        raise RuntimeError("no attention captured: apply_rotary_pos_emb was not called by the text model")
    return torch.stack(maps, dim=0)
