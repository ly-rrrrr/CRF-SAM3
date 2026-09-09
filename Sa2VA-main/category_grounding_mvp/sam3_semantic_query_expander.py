"""SAM3-native semantic query expansion for recall prompts.

The expander uses a precomputed SAM3 text prototype bank to enlarge the prompt
space when the primary category does not produce a useful SAM3 response. It
does not rewrite categories with hand-written aliases. It only proposes nearby
visual nouns in SAM3's own text-feature space.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _norm_text(text: Any) -> str:
    return " ".join(str(text or "").replace("_", " ").lower().split())


class SAM3SemanticQueryExpander:
    """Return diverse prototype prompts near a category in SAM3 text space."""

    def __init__(
        self,
        bank_path: str,
        metadata_path: str | None = None,
        top_k: int = 4,
        candidate_pool_size: int = 24,
        diversity_lambda: float = 0.65,
        exclude_self: bool = True,
        min_similarity: float | None = None,
        sam3_repo: str | None = None,
        checkpoint_path: str | None = None,
        device: str | torch.device = "cuda",
        log: bool = True,
    ) -> None:
        self.bank_path = str(bank_path)
        self.top_k = int(top_k)
        self.candidate_pool_size = int(candidate_pool_size)
        self.diversity_lambda = float(diversity_lambda)
        self.exclude_self = bool(exclude_self)
        self.min_similarity = None if min_similarity is None else float(min_similarity)
        self.sam3_repo = sam3_repo
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if str(device) == "cpu" or torch.cuda.is_available() else "cpu")
        self.log = bool(log)

        bank = torch.load(self.bank_path, map_location="cpu", weights_only=True)
        self.prototypes = F.normalize(bank["pooled_features"].float(), dim=-1)
        meta_file = metadata_path or bank.get("metadata_path")
        if not meta_file:
            raise ValueError("SAM3SemanticQueryExpander requires metadata_path or bank metadata_path")
        meta_path = Path(meta_file)
        if not meta_path.is_absolute():
            meta_path = Path(self.bank_path).resolve().parent / meta_path.name
        with meta_path.open("r", encoding="utf-8") as f:
            self.metadata: list[dict[str, Any]] = json.load(f)
        if len(self.metadata) != int(self.prototypes.shape[0]):
            raise ValueError(
                f"metadata/prototype length mismatch: {len(self.metadata)} vs {self.prototypes.shape[0]}"
            )

        self._exact_rows: dict[str, list[int]] = {}
        for idx, item in enumerate(self.metadata):
            for key in (
                item.get("prompt_text"),
                item.get("category_prompt_text"),
                item.get("category_name"),
            ):
                nkey = _norm_text(key)
                if nkey:
                    self._exact_rows.setdefault(nkey, []).append(idx)
        self._encoder = None

    def _load_encoder(self):
        if self._encoder is not None:
            return self._encoder
        if not self.sam3_repo or not self.checkpoint_path:
            raise ValueError(
                "Query is not an exact prototype. Provide sam3_repo and checkpoint_path "
                "to encode arbitrary query text."
            )
        from category_grounding_mvp.build_sam3_text_prototype_bank import build_text_encoder

        self._encoder = build_text_encoder(
            Path(self.sam3_repo),
            Path(self.checkpoint_path),
            self.device,
        )
        return self._encoder

    @torch.no_grad()
    def _encode_query(self, query: str) -> torch.Tensor:
        qkey = _norm_text(query)
        exact = self._exact_rows.get(qkey)
        if exact:
            return F.normalize(self.prototypes[exact].mean(dim=0), dim=-1)

        encoder = self._load_encoder()
        text_attention_mask, text_memory, _ = encoder([query], device=self.device)
        token_features = text_memory.transpose(0, 1).contiguous().float()
        valid_mask = (~text_attention_mask).to(token_features.dtype)
        pooled = (token_features * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        return F.normalize(pooled.cpu()[0], dim=-1)

    def _mask_self(self, query: str, scores: torch.Tensor) -> torch.Tensor:
        if not self.exclude_self:
            return scores
        qkey = _norm_text(query)
        masked = scores.clone()
        for idx, item in enumerate(self.metadata):
            keys = {
                _norm_text(item.get("prompt_text")),
                _norm_text(item.get("category_prompt_text")),
                _norm_text(item.get("category_name")),
            }
            if qkey in keys:
                masked[idx] = -torch.inf
        return masked

    def _select_mmr(self, scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        finite_count = int(torch.isfinite(scores).sum().item())
        pool_k = min(max(self.candidate_pool_size, k), finite_count)
        pool_values, pool_indices = torch.topk(scores, k=pool_k)
        pool_proto = F.normalize(self.prototypes[pool_indices], dim=-1)
        selected_local: list[int] = []
        selected_indices: list[int] = []
        for _ in range(min(k, pool_k)):
            if not selected_local:
                local_idx = 0
            else:
                selected_proto = pool_proto[selected_local]
                redundancy = (pool_proto @ selected_proto.T).max(dim=1).values
                mmr = self.diversity_lambda * pool_values - (1.0 - self.diversity_lambda) * redundancy
                mmr[torch.tensor(selected_local)] = -torch.inf
                local_idx = int(torch.argmax(mmr).item())
            selected_local.append(local_idx)
            selected_indices.append(int(pool_indices[local_idx].item()))
        indices = torch.tensor(selected_indices, dtype=torch.long)
        return scores[indices], indices

    def expand(self, query: str, existing: list[str] | None = None) -> list[str]:
        q = str(query or "").strip()
        if not q or self.top_k <= 0:
            return []
        existing_norm = {_norm_text(x) for x in (existing or []) if str(x).strip()}
        existing_norm.add(_norm_text(q))

        qvec = self._encode_query(q)
        scores = self.prototypes @ qvec
        scores = self._mask_self(q, scores)
        if self.min_similarity is not None:
            scores = scores.masked_fill(scores < self.min_similarity, -torch.inf)
        if not torch.isfinite(scores).any():
            return []

        k = min(self.top_k, int(torch.isfinite(scores).sum().item()))
        values, indices = self._select_mmr(scores, k)
        prompts: list[str] = []
        rows = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            item = self.metadata[int(idx)]
            prompt = str(item.get("prompt_text") or item.get("category_prompt_text") or "").strip()
            pkey = _norm_text(prompt)
            if not prompt or pkey in existing_norm:
                continue
            prompts.append(prompt)
            existing_norm.add(pkey)
            rows.append(f"{prompt}:{score:.3f}")
        if self.log:
            print(
                f"[SAM3SemanticQueryExpander] query={q!r} expanded={prompts} "
                f"neighbors={', '.join(rows)}",
                flush=True,
            )
        return prompts


def build_sam3_semantic_query_expander(cfg: dict[str, Any] | None) -> SAM3SemanticQueryExpander | None:
    if not cfg or not cfg.get("enabled", False):
        return None
    bank_path = cfg.get("bank_path")
    if not bank_path:
        raise ValueError("semantic_query_expander requires bank_path")
    return SAM3SemanticQueryExpander(
        bank_path=bank_path,
        metadata_path=cfg.get("metadata_path"),
        top_k=cfg.get("top_k", 4),
        candidate_pool_size=cfg.get("candidate_pool_size", 24),
        diversity_lambda=cfg.get("diversity_lambda", 0.65),
        exclude_self=cfg.get("exclude_self", True),
        min_similarity=cfg.get("min_similarity"),
        sam3_repo=cfg.get("sam3_repo"),
        checkpoint_path=cfg.get("checkpoint_path"),
        device=cfg.get("device", "cuda"),
        log=cfg.get("log", True),
    )
