#!/usr/bin/env python3
"""Query nearest visual noun prototypes in SAM3 text-feature space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from build_sam3_text_prototype_bank import build_text_encoder


@torch.no_grad()
def encode_query(encoder, text: str, device: torch.device) -> torch.Tensor:
    text_attention_mask, text_memory, _ = encoder([text], device=device)
    token_features = text_memory.transpose(0, 1).contiguous()
    valid_mask = (~text_attention_mask).to(token_features.dtype)
    pooled = (token_features * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    return F.normalize(pooled, dim=-1).cpu()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="+")
    parser.add_argument(
        "--bank",
        default="category_grounding_mvp/data/sam3_text_prototypes/lvis_v1/prototype_bank.pt",
    )
    parser.add_argument(
        "--meta",
        default="category_grounding_mvp/data/sam3_text_prototypes/lvis_v1/prototype_meta.json",
    )
    parser.add_argument(
        "--sam3-repo",
        default="/mnt/csip-500/share/kongjiawei/Sa2VA-main/SAM3-DMS",
    )
    parser.add_argument(
        "--checkpoint",
        default="/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--selection", choices=["topk", "mmr"], default="topk")
    parser.add_argument("--candidate-pool-size", type=int, default=32)
    parser.add_argument("--diversity-lambda", type=float, default=0.35)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    prototypes = F.normalize(bank["pooled_features"].float(), dim=-1)
    with Path(args.meta).open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if len(metadata) != prototypes.shape[0]:
        raise ValueError(f"metadata/prototype length mismatch: {len(metadata)} vs {prototypes.shape[0]}")

    encoder = build_text_encoder(Path(args.sam3_repo), Path(args.checkpoint), device)

    for query in args.queries:
        q = encode_query(encoder, query, device)
        scores = prototypes @ q
        if args.exclude_self:
            qkey = " ".join(query.replace("_", " ").lower().split())
            for i, item in enumerate(metadata):
                pkey = " ".join(str(item.get("prompt_text", "")).replace("_", " ").lower().split())
                ckey = " ".join(str(item.get("category_prompt_text", "")).replace("_", " ").lower().split())
                if pkey == qkey or ckey == qkey:
                    scores[i] = -torch.inf
        k = min(args.top_k, int(torch.isfinite(scores).sum().item()))
        if args.selection == "mmr":
            pool_k = min(max(args.candidate_pool_size, k), int(torch.isfinite(scores).sum().item()))
            pool_values, pool_indices = torch.topk(scores, k=pool_k)
            pool_proto = prototypes[pool_indices]
            selected_local = []
            selected_indices = []
            for _ in range(k):
                if not selected_local:
                    local_idx = 0
                else:
                    selected_proto = pool_proto[selected_local]
                    redundancy = (pool_proto @ selected_proto.T).max(dim=1).values
                    mmr = args.diversity_lambda * pool_values - (1.0 - args.diversity_lambda) * redundancy
                    mmr[torch.tensor(selected_local)] = -torch.inf
                    local_idx = int(torch.argmax(mmr).item())
                selected_local.append(local_idx)
                selected_indices.append(int(pool_indices[local_idx].item()))
            indices = torch.tensor(selected_indices, dtype=torch.long)
            values = scores[indices]
        else:
            values, indices = torch.topk(scores, k=k)
        print(f"\n[query] {query} selection={args.selection}")
        for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            item = metadata[idx]
            print(
                f"{rank:02d} score={score:.4f} "
                f"prompt={item['prompt_text']!r} "
                f"category={item['category_name']!r} "
                f"freq={item.get('frequency')}"
            )


if __name__ == "__main__":
    main()
