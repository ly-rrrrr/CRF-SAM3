#!/usr/bin/env python3
"""Encode visual category names with SAM3's own text encoder.

The resulting prototype bank is meant for text-space semantic calibration:
it stores SAM3-native text token features and pooled features for category
names and synonyms. It does not use external CLIP embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_vocab_records(paths: list[Path], include_synonyms: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                source_record = json.loads(line)
                variants = source_record.get("variants") or []
                if not include_synonyms:
                    variants = [
                        {
                            "raw": source_record.get("name", source_record.get("prompt_text")),
                            "prompt_text": source_record.get("prompt_text"),
                        }
                    ]
                for variant in variants:
                    prompt_text = str(variant.get("prompt_text") or "").strip()
                    if not prompt_text:
                        continue
                    key = (source_record.get("source"), source_record.get("category_id"), prompt_text.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        {
                            "source": source_record.get("source"),
                            "category_id": source_record.get("category_id"),
                            "category_name": source_record.get("name"),
                            "category_prompt_text": source_record.get("prompt_text"),
                            "variant_raw": variant.get("raw"),
                            "prompt_text": prompt_text,
                            "frequency": source_record.get("frequency"),
                            "synset": source_record.get("synset"),
                            "definition": source_record.get("definition"),
                        }
                    )
    return records


def build_text_encoder(sam3_repo: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(sam3_repo))
    from sam3.model.text_encoder_ve import VETextEncoder
    from sam3.model.tokenizer_ve import SimpleTokenizer

    bpe_path = sam3_repo / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    encoder = VETextEncoder(
        tokenizer=SimpleTokenizer(bpe_path=str(bpe_path)),
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
        use_act_checkpoint=False,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    prefix = "detector.backbone.language_backbone."
    text_state = {
        key.removeprefix(prefix): value
        for key, value in ckpt.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    missing, unexpected = encoder.load_state_dict(text_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected text encoder keys: {unexpected[:10]}")
    if missing:
        raise RuntimeError(f"Missing text encoder keys: {missing[:10]}")

    encoder.to(device)
    encoder.eval()
    return encoder


@torch.no_grad()
def encode_records(encoder, records: list[dict[str, Any]], batch_size: int, device: torch.device):
    pooled_chunks = []
    token_chunks = []
    mask_chunks = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [item["prompt_text"] for item in batch]
        text_attention_mask, text_memory, _ = encoder(texts, device=device)
        token_features = text_memory.transpose(0, 1).contiguous()  # [B, L, 256]
        valid_mask = (~text_attention_mask).to(token_features.dtype)  # [B, L]
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (token_features * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled = F.normalize(pooled, dim=-1)

        pooled_chunks.append(pooled.cpu())
        token_chunks.append(token_features.cpu())
        mask_chunks.append(text_attention_mask.cpu())
        print(f"[prototype] encoded {min(start + batch_size, len(records))}/{len(records)}", flush=True)

    return {
        "pooled_features": torch.cat(pooled_chunks, dim=0),
        "token_features": torch.cat(token_chunks, dim=0),
        "language_mask": torch.cat(mask_chunks, dim=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-jsonl",
        nargs="+",
        default=["category_grounding_mvp/data/visual_vocab/lvis_v1/categories.jsonl"],
    )
    parser.add_argument(
        "--sam3-repo",
        default="/mnt/csip-500/share/kongjiawei/Sa2VA-main/SAM3-DMS",
    )
    parser.add_argument(
        "--checkpoint",
        default="/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="category_grounding_mvp/data/sam3_text_prototypes/lvis_v1",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-synonyms", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    vocab_paths = [Path(p) for p in args.vocab_jsonl]
    records = load_vocab_records(vocab_paths, include_synonyms=not args.no_synonyms)
    if not records:
        raise ValueError("No prototype records found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = build_text_encoder(Path(args.sam3_repo), Path(args.checkpoint), device)
    tensors = encode_records(encoder, records, args.batch_size, device)

    metadata_path = output_dir / "prototype_meta.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    bank_path = output_dir / "prototype_bank.pt"
    torch.save(
        {
            **tensors,
            "metadata_path": str(metadata_path),
            "num_records": len(records),
            "feature_dim": int(tensors["pooled_features"].shape[-1]),
            "sequence_length": int(tensors["token_features"].shape[1]),
            "sources": [str(p) for p in vocab_paths],
        },
        bank_path,
    )

    summary = {
        "num_records": len(records),
        "feature_dim": int(tensors["pooled_features"].shape[-1]),
        "sequence_length": int(tensors["token_features"].shape[1]),
        "bank_path": str(bank_path),
        "metadata_path": str(metadata_path),
        "device": str(device),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
