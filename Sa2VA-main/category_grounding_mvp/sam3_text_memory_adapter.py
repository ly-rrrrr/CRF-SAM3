"""Prototype-augmented text memory adapter for SAM3.

This adapter keeps the original category text tokens and appends a small number
of SAM3-native prototype memory tokens selected from a visual-category prototype
bank. It is inference-only and does not rewrite the input prompt string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_text(text: Any) -> str:
    return " ".join(str(text or "").replace("_", " ").lower().split())


class SAM3TextMemoryPrototypeAdapter(nn.Module):
    """Append SAM3-native prototype tokens to language_features."""

    def __init__(
        self,
        bank_path: str,
        metadata_path: str | None = None,
        top_k: int = 4,
        selection: str = "mmr",
        candidate_pool_size: int = 24,
        diversity_lambda: float = 0.65,
        exclude_self: bool = True,
        min_similarity: float | None = None,
        token_mode: str = "pooled",
        log_first_n: int = 0,
        skip_captions: Iterable[str] | None = None,
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

        meta_file = metadata_path or bank.get("metadata_path")
        if not meta_file:
            raise ValueError("text memory adapter requires metadata_path or bank metadata_path")
        meta_path = Path(meta_file)
        if not meta_path.is_absolute():
            meta_path = Path(bank_path).resolve().parent / meta_path.name
        with meta_path.open("r", encoding="utf-8") as f:
            self.metadata: list[dict[str, Any]] = json.load(f)
        if len(self.metadata) != int(self.prototype_pooled.shape[0]):
            raise ValueError(
                f"metadata/prototype length mismatch: {len(self.metadata)} vs {self.prototype_pooled.shape[0]}"
            )

        self.top_k = int(top_k)
        self.selection = str(selection or "mmr").lower()
        self.candidate_pool_size = int(candidate_pool_size)
        self.diversity_lambda = float(diversity_lambda)
        self.exclude_self = bool(exclude_self)
        self.min_similarity = None if min_similarity is None else float(min_similarity)
        self.token_mode = str(token_mode or "pooled").lower()
        self.log_first_n = int(log_first_n)
        self.skip_captions = {_norm_text(x) for x in (skip_captions or ["visual"])}
        self._logged = 0

    def _self_mask(self, caption: str, scores: torch.Tensor) -> torch.Tensor:
        if not self.exclude_self:
            return scores
        key = _norm_text(caption)
        if not key:
            return scores
        masked = scores.clone()
        for idx, item in enumerate(self.metadata):
            keys = {
                _norm_text(item.get("prompt_text")),
                _norm_text(item.get("category_prompt_text")),
                _norm_text(item.get("category_name")),
            }
            if key in keys:
                masked[idx] = -torch.inf
        return masked

    def _select_indices(self, scores: torch.Tensor, proto: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.selection not in {"mmr", "diverse", "diversity"}:
            return torch.topk(scores, k=k)

        finite_count = int(torch.isfinite(scores).sum().item())
        pool_k = min(max(self.candidate_pool_size, k), finite_count)
        pool_values, pool_indices = torch.topk(scores, k=pool_k)
        pool_proto = F.normalize(proto[pool_indices], dim=-1)
        selected_local: list[int] = []
        selected_indices: list[int] = []
        for _ in range(min(k, pool_k)):
            if not selected_local:
                local_idx = 0
            else:
                selected_proto = pool_proto[selected_local]
                redundancy = (pool_proto @ selected_proto.T).max(dim=1).values
                mmr = self.diversity_lambda * pool_values - (1.0 - self.diversity_lambda) * redundancy
                mmr[torch.tensor(selected_local, device=mmr.device)] = -torch.inf
                local_idx = int(torch.argmax(mmr).item())
            selected_local.append(local_idx)
            selected_indices.append(int(pool_indices[local_idx].item()))
        indices = torch.tensor(selected_indices, device=scores.device, dtype=torch.long)
        return scores[indices], indices

    def _prototype_memory(self, indices: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.token_mode in {"full", "tokens", "token"} and self.prototype_tokens is not None:
            # [K, Lp, C] -> [K*Lp, C]. This is intentionally opt-in because it
            # increases text length substantially.
            return self.prototype_tokens.to(device=device, dtype=dtype)[indices].flatten(0, 1)
        return self.prototype_pooled.to(device=device, dtype=dtype)[indices]

    def _log_neighbors(self, caption: str, values: torch.Tensor, indices: torch.Tensor) -> None:
        if self._logged >= self.log_first_n:
            return
        rows = []
        for score, idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            item = self.metadata[int(idx)]
            rows.append(f"{item.get('prompt_text')}:{score:.3f}")
        print(
            f"[SAM3TextMemoryAdapter] caption={caption!r} appended={len(indices)} "
            f"mode={self.token_mode} neighbors=" + ", ".join(rows),
            flush=True,
        )
        self._logged += 1

    def forward(
        self,
        text_memory: torch.Tensor,
        text_attention_mask: torch.Tensor,
        captions: Iterable[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k <= 0:
            return text_memory, text_attention_mask
        if text_memory.ndim != 3:
            raise ValueError(f"Expected text_memory [L, B, C], got {tuple(text_memory.shape)}")

        orig_dtype = text_memory.dtype
        work = text_memory.float()
        proto = self.prototype_pooled.to(device=work.device, dtype=work.dtype)
        captions = list(captions)
        batch = int(work.shape[1])
        valid_mask = (~text_attention_mask.bool()).to(device=work.device)

        appended_per_batch: list[torch.Tensor] = []
        max_append = 0
        for b, caption in enumerate(captions):
            if _norm_text(caption) in self.skip_captions:
                memory = torch.empty((0, work.shape[-1]), device=work.device, dtype=work.dtype)
                appended_per_batch.append(memory)
                continue
            valid = valid_mask[b]
            if not bool(valid.any()):
                memory = torch.empty((0, work.shape[-1]), device=work.device, dtype=work.dtype)
                appended_per_batch.append(memory)
                continue
            pooled = F.normalize(work[:, b, :][valid].mean(dim=0), dim=-1)
            scores = self._self_mask(caption, proto @ pooled)
            if self.min_similarity is not None:
                scores = scores.masked_fill(scores < self.min_similarity, -torch.inf)
            if not torch.isfinite(scores).any():
                memory = torch.empty((0, work.shape[-1]), device=work.device, dtype=work.dtype)
                appended_per_batch.append(memory)
                continue
            k = min(self.top_k, int(torch.isfinite(scores).sum().item()))
            values, indices = self._select_indices(scores, proto, k)
            memory = self._prototype_memory(indices, dtype=work.dtype, device=work.device)
            appended_per_batch.append(memory)
            max_append = max(max_append, int(memory.shape[0]))
            self._log_neighbors(caption, values, indices)

        if max_append <= 0:
            return text_memory, text_attention_mask

        extra = torch.zeros((max_append, batch, work.shape[-1]), device=work.device, dtype=work.dtype)
        extra_mask = torch.ones((batch, max_append), device=text_attention_mask.device, dtype=text_attention_mask.dtype)
        for b, memory in enumerate(appended_per_batch):
            n = int(memory.shape[0])
            if n <= 0:
                continue
            extra[:n, b, :] = memory
            extra_mask[b, :n] = False

        out_memory = torch.cat([work, extra], dim=0).to(dtype=orig_dtype)
        out_mask = torch.cat([text_attention_mask, extra_mask], dim=1)
        return out_memory, out_mask


def install_sam3_text_memory_adapter(predictor: Any, cfg: dict[str, Any]) -> None:
    if not cfg or not cfg.get("enabled", True):
        return
    bank_path = cfg.get("bank_path")
    if not bank_path:
        raise ValueError("text memory adapter requires bank_path")
    adapter = SAM3TextMemoryPrototypeAdapter(
        bank_path=bank_path,
        metadata_path=cfg.get("metadata_path"),
        top_k=cfg.get("top_k", 4),
        selection=cfg.get("selection", "mmr"),
        candidate_pool_size=cfg.get("candidate_pool_size", 24),
        diversity_lambda=cfg.get("diversity_lambda", 0.65),
        exclude_self=cfg.get("exclude_self", True),
        min_similarity=cfg.get("min_similarity"),
        token_mode=cfg.get("token_mode", "pooled"),
        log_first_n=cfg.get("log_first_n", 0),
        skip_captions=cfg.get("skip_captions", ["visual"]),
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
        "[SAM3TextMemoryAdapter] installed "
        f"bank={bank_path} top_k={adapter.top_k} selection={adapter.selection} "
        f"token_mode={adapter.token_mode} exclude_self={adapter.exclude_self}",
        flush=True,
    )
