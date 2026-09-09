"""Prototype residual adapter for SAM3 text features.

This module performs inference-time semantic calibration in SAM3's native text
feature space. It does not change SAM3 weights and does not rewrite category
strings. The adapter is attached to ``SAM3VLBackbone.text_feature_adapter``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM3TextPrototypeResidualAdapter(nn.Module):
    """Move SAM3 text memory slightly toward nearby visual noun prototypes."""

    def __init__(
        self,
        bank_path: str,
        metadata_path: str | None = None,
        top_k: int = 5,
        alpha: float = 0.2,
        temperature: float = 0.1,
        exclude_self: bool = True,
        min_similarity: float | None = None,
        log_first_n: int = 0,
        skip_captions: Iterable[str] | None = None,
        selection: str = "topk",
        candidate_pool_size: int = 32,
        diversity_lambda: float = 0.55,
        fusion_mode: str = "pooled_residual",
        weight_mode: str = "softmax",
    ) -> None:
        super().__init__()
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        pooled = F.normalize(bank["pooled_features"].float(), dim=-1)
        self.register_buffer("prototype_pooled", pooled, persistent=False)
        token_features = bank.get("token_features")
        if token_features is not None:
            self.register_buffer("prototype_tokens", token_features.float(), persistent=False)
        else:
            self.prototype_tokens = None

        metadata_file = metadata_path or bank.get("metadata_path")
        self.metadata: list[dict[str, Any]] = []
        if metadata_file:
            meta_path = Path(metadata_file)
            if not meta_path.is_absolute():
                meta_path = Path(bank_path).resolve().parent / meta_path.name
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.exclude_self = bool(exclude_self)
        self.min_similarity = None if min_similarity is None else float(min_similarity)
        self.log_first_n = int(log_first_n)
        self.skip_captions = {self._norm_text(x) for x in (skip_captions or ["visual"])}
        self.selection = str(selection or "topk").lower()
        self.candidate_pool_size = int(candidate_pool_size)
        self.diversity_lambda = float(diversity_lambda)
        self.fusion_mode = str(fusion_mode or "pooled_residual").lower()
        self.weight_mode = str(weight_mode or "softmax").lower()
        self._logged = 0

    @staticmethod
    def _norm_text(text: str) -> str:
        return " ".join(str(text).replace("_", " ").lower().split())

    def _self_mask(self, caption: str, scores: torch.Tensor) -> torch.Tensor:
        if not self.exclude_self or not self.metadata:
            return scores
        caption_key = self._norm_text(caption)
        if not caption_key:
            return scores
        masked = scores.clone()
        for idx, item in enumerate(self.metadata):
            prompt = self._norm_text(item.get("prompt_text", ""))
            category = self._norm_text(item.get("category_prompt_text", ""))
            if prompt == caption_key or category == caption_key:
                masked[idx] = -torch.inf
        return masked


    def _select_indices(self, scores: torch.Tensor, proto: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.selection not in {"mmr", "diverse", "diversity"}:
            return torch.topk(scores, k=k)

        finite_count = int(torch.isfinite(scores).sum().item())
        pool_k = min(max(self.candidate_pool_size, k), finite_count)
        pool_values, pool_indices = torch.topk(scores, k=pool_k)
        selected_local: list[int] = []
        selected_indices: list[int] = []
        pool_proto = F.normalize(proto[pool_indices], dim=-1)
        for _ in range(min(k, pool_k)):
            if not selected_local:
                local_idx = 0
            else:
                selected_proto = pool_proto[selected_local]
                redundancy = pool_proto @ selected_proto.transpose(0, 1)
                max_redundancy = redundancy.max(dim=1).values
                mmr = self.diversity_lambda * pool_values - (1.0 - self.diversity_lambda) * max_redundancy
                if selected_local:
                    mmr[torch.tensor(selected_local, device=mmr.device)] = -torch.inf
                local_idx = int(torch.argmax(mmr).item())
            selected_local.append(local_idx)
            selected_indices.append(int(pool_indices[local_idx].item()))
        selected = torch.tensor(selected_indices, device=scores.device, dtype=torch.long)
        return scores[selected], selected

    def _compute_weights(self, values: torch.Tensor) -> torch.Tensor:
        if self.weight_mode == "uniform":
            return torch.full_like(values, 1.0 / max(values.numel(), 1))
        if self.weight_mode == "rank":
            ranks = torch.arange(values.numel(), device=values.device, dtype=values.dtype) + 1.0
            weights = 1.0 / ranks
            return weights / weights.sum().clamp_min(1e-6)
        return torch.softmax(values / max(self.temperature, 1e-6), dim=0)

    def _log_neighbors(self, caption: str, values: torch.Tensor, indices: torch.Tensor) -> None:
        if self._logged >= self.log_first_n or not self.metadata:
            return
        rows = []
        for score, idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            item = self.metadata[int(idx)]
            rows.append(f"{item.get('prompt_text')}:{score:.3f}")
        print(
            f"[SAM3TextPrototypeAdapter] caption={caption!r} "
            f"alpha={self.alpha} top_k={self.top_k} neighbors=" + ", ".join(rows),
            flush=True,
        )
        self._logged += 1

    def forward(
        self,
        text_memory: torch.Tensor,
        text_attention_mask: torch.Tensor,
        captions: Iterable[str],
    ) -> torch.Tensor:
        if self.alpha == 0 or self.top_k <= 0:
            return text_memory
        if text_memory.ndim != 3:
            raise ValueError(f"Expected text_memory [L, B, C], got {tuple(text_memory.shape)}")

        orig_dtype = text_memory.dtype
        work = text_memory.float()
        proto = self.prototype_pooled.to(device=work.device, dtype=work.dtype)
        out = work.clone()
        captions = list(captions)

        valid_mask = (~text_attention_mask.bool()).to(device=work.device)
        for b, caption in enumerate(captions):
            if self._norm_text(caption) in self.skip_captions:
                continue
            valid = valid_mask[b]
            if not bool(valid.any()):
                continue
            tokens = work[:, b, :]
            pooled = tokens[valid].mean(dim=0)
            pooled = F.normalize(pooled, dim=-1)
            scores = proto @ pooled
            scores = self._self_mask(caption, scores)
            if self.min_similarity is not None:
                scores = scores.masked_fill(scores < self.min_similarity, -torch.inf)
            if not torch.isfinite(scores).any():
                continue
            k = min(self.top_k, int(torch.isfinite(scores).sum().item()))
            values, indices = self._select_indices(scores, proto, k)
            weights = self._compute_weights(values)
            centroid = (proto[indices] * weights[:, None]).sum(dim=0)
            if self.fusion_mode in {"token_mixture", "token_mix", "prototype_mixture"} and self.prototype_tokens is not None:
                proto_tokens = self.prototype_tokens.to(device=work.device, dtype=work.dtype)[indices]
                token_centroid = (proto_tokens * weights[:, None, None]).sum(dim=0)
                mixed = (1.0 - self.alpha) * tokens + self.alpha * token_centroid[: tokens.shape[0]]
                out[valid, b, :] = mixed[valid]
            else:
                delta = centroid - pooled
                out[valid, b, :] = tokens[valid] + self.alpha * delta[None, :]
            self._log_neighbors(caption, values, indices)

        return out.to(dtype=orig_dtype)


def install_sam3_text_prototype_adapter(predictor: Any, cfg: dict[str, Any]) -> None:
    """Attach prototype residual adapter to a SAM3 video predictor."""

    if not cfg or not cfg.get("enabled", True):
        return
    bank_path = cfg.get("bank_path")
    if not bank_path:
        raise ValueError("text prototype adapter requires bank_path")
    adapter = SAM3TextPrototypeResidualAdapter(
        bank_path=bank_path,
        metadata_path=cfg.get("metadata_path"),
        top_k=cfg.get("top_k", 5),
        alpha=cfg.get("alpha", 0.2),
        temperature=cfg.get("temperature", 0.1),
        exclude_self=cfg.get("exclude_self", True),
        min_similarity=cfg.get("min_similarity"),
        log_first_n=cfg.get("log_first_n", 0),
        skip_captions=cfg.get("skip_captions", ["visual"]),
        selection=cfg.get("selection", "topk"),
        candidate_pool_size=cfg.get("candidate_pool_size", 32),
        diversity_lambda=cfg.get("diversity_lambda", 0.55),
        fusion_mode=cfg.get("fusion_mode", "pooled_residual"),
        weight_mode=cfg.get("weight_mode", "softmax"),
    )
    model = getattr(predictor, "model", None)
    detector = getattr(model, "detector", None)
    backbone = getattr(detector, "backbone", None)
    if backbone is None or not hasattr(backbone, "text_feature_adapter"):
        raise RuntimeError("SAM3 predictor backbone does not expose text_feature_adapter")
    device = next(backbone.parameters()).device
    adapter.to(device=device)
    adapter.eval()
    backbone.text_feature_adapter = adapter
    print(
        "[SAM3TextPrototypeAdapter] installed "
        f"bank={bank_path} top_k={adapter.top_k} alpha={adapter.alpha} "
        f"temperature={adapter.temperature} exclude_self={adapter.exclude_self} "
        f"selection={adapter.selection} fusion={adapter.fusion_mode} weights={adapter.weight_mode}",
        flush=True,
    )
