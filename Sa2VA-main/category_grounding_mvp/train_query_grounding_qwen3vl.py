#!/usr/bin/env python3
"""Train mask-supervised expression-conditioned target queries for Qwen3-VL.

This MVP does not feed GT masks as images. Masks supervise token-level response
maps predicted from the original frames and the referring expression.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_category_grounding_qwen3vl import CategoryJsonlDataset, CategorySFTCollator  # noqa: E402


SYSTEM_PROMPT = """You are a referring-expression target category grounding module.

Given a referring expression and original frames from the same video, identify
the visual object category of the target referred to by the expression. Return
exactly one JSON object with key "category".
"""


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def token_mask_from_rle(
    rle: Dict[str, Any],
    grid_h: int,
    grid_w: int,
    spatial_merge_size: int,
    positive_threshold: float,
) -> torch.Tensor:
    mask = decode_rle(rle)
    if mask.size == 0:
        token_h = max(1, grid_h // spatial_merge_size)
        token_w = max(1, grid_w // spatial_merge_size)
        return torch.zeros(token_h * token_w, dtype=torch.float32)
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.resize((int(grid_w), int(grid_h)), Image.Resampling.NEAREST)
    arr = (np.asarray(image) > 0).astype(np.float32)
    token_h = int(grid_h) // int(spatial_merge_size)
    token_w = int(grid_w) // int(spatial_merge_size)
    arr = arr[: token_h * spatial_merge_size, : token_w * spatial_merge_size]
    pooled = arr.reshape(token_h, spatial_merge_size, token_w, spatial_merge_size).mean(axis=(1, 3))
    return torch.from_numpy((pooled >= positive_threshold).astype(np.float32)).flatten()


class QueryGroundingCollator(CategorySFTCollator):
    def __init__(
        self,
        processor: Any,
        system_prompt: str,
        image_token_id: int,
        spatial_merge_size: int,
        positive_threshold: float = 0.05,
    ) -> None:
        super().__init__(processor=processor, system_prompt=system_prompt, category_query_tokens=[])
        self.image_token_id = int(image_token_id)
        self.spatial_merge_size = int(spatial_merge_size)
        self.positive_threshold = float(positive_threshold)

    def _expression_text(self, record: Dict[str, Any]) -> str:
        return (
            "Referring expression:\n"
            f"{record['expression']}\n\n"
            "The frames are sampled from visible target-mask frames during training, "
            "but no mask image is provided to the model."
        )

    def _messages(self, record: Dict[str, Any], with_answer: bool) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": self._expression_text(record)}]
        for frame_path in record.get("frame_paths", []):
            content.append({"type": "image", "image": frame_path})
        content.append({"type": "text", "text": self._output_text()})
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        if with_answer:
            target = record.get("target_text") or json.dumps({"category": record["category"]}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return messages

    def _encode_one(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out = super()._encode_one(record)
        expr = self.processor.tokenizer(
            record["expression"],
            return_tensors="pt",
            add_special_tokens=True,
        )
        out["expression_input_ids"] = expr["input_ids"][0]
        out["expression_attention_mask"] = expr["attention_mask"][0]

        image_grid = out.get("image_grid_thw")
        if image_grid is None:
            raise ValueError(f"Expected image_grid_thw for sample {record.get('sample_id')}")
        image_grid = image_grid.detach().cpu()
        mask_rles = record.get("mask_rles", [])
        if len(mask_rles) != int(image_grid.shape[0]):
            raise ValueError(
                f"mask/frame count mismatch for {record.get('sample_id')}: "
                f"masks={len(mask_rles)} grids={image_grid.shape[0]}"
            )
        parts: List[torch.Tensor] = []
        for rle, grid in zip(mask_rles, image_grid):
            _, h, w = [int(x) for x in grid.tolist()]
            parts.append(
                token_mask_from_rle(
                    rle,
                    grid_h=h,
                    grid_w=w,
                    spatial_merge_size=self.spatial_merge_size,
                    positive_threshold=self.positive_threshold,
                )
            )
        out["mask_labels"] = torch.cat(parts, dim=0)
        image_positions = (out["input_ids"] == self.image_token_id).nonzero(as_tuple=False).flatten()
        if int(image_positions.numel()) != int(out["mask_labels"].numel()):
            raise ValueError(
                f"visual token/mask label count mismatch for {record.get('sample_id')}: "
                f"tokens={image_positions.numel()} labels={out['mask_labels'].numel()}"
            )
        out["sample_id"] = record.get("sample_id", "")
        return out

    def __call__(self, records: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = [self._encode_one(record) for record in records]
        batch: Dict[str, torch.Tensor] = {}

        batch["input_ids"] = self._pad_1d(
            [x["input_ids"] for x in encoded],
            self.pad_token_id,
        )
        batch["attention_mask"] = self._pad_1d(
            [x["attention_mask"] for x in encoded],
            0,
        )
        batch["labels"] = self._pad_1d(
            [x["labels"] for x in encoded],
            -100,
        )
        if any("mm_token_type_ids" in x for x in encoded):
            batch["mm_token_type_ids"] = self._pad_1d(
                [x.get("mm_token_type_ids", torch.zeros_like(x["input_ids"])) for x in encoded],
                0,
            )

        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            values = [x[key] for x in encoded if key in x]
            if values:
                batch[key] = torch.cat(values, dim=0)
        batch["expression_input_ids"] = pad_sequence(
            [x["expression_input_ids"] for x in encoded],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        batch["expression_attention_mask"] = pad_sequence(
            [x["expression_attention_mask"] for x in encoded],
            batch_first=True,
            padding_value=0,
        )
        batch["mask_labels"] = pad_sequence(
            [x["mask_labels"] for x in encoded],
            batch_first=True,
            padding_value=0.0,
        )
        batch["mask_label_valid"] = pad_sequence(
            [torch.ones_like(x["mask_labels"], dtype=torch.bool) for x in encoded],
            batch_first=True,
            padding_value=False,
        )
        return batch


class ExpressionConditioner(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        query_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.summary_token = nn.Parameter(torch.zeros(1, 1, query_dim))
        self.proj = nn.Linear(llm_hidden_size, query_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=num_heads,
            dim_feedforward=query_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(query_dim)
        nn.init.normal_(self.summary_token, std=0.02)

    def forward(self, token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(token_embeds)
        bsz = x.shape[0]
        summary = self.summary_token.expand(bsz, -1, -1).to(dtype=x.dtype, device=x.device)
        x = torch.cat([summary, x], dim=1)
        valid = torch.cat(
            [torch.ones((bsz, 1), dtype=torch.bool, device=x.device), attention_mask.to(torch.bool)],
            dim=1,
        )
        x = self.encoder(x, src_key_padding_mask=~valid)
        return self.norm(x), valid


class QueryDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.text_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.visual_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        text: torch.Tensor,
        text_valid: torch.Tensor,
        visual: torch.Tensor,
        visual_valid: torch.Tensor,
    ) -> torch.Tensor:
        q = q + self.self_attn(self.norm1(q), self.norm1(q), self.norm1(q), need_weights=False)[0]
        q = q + self.text_attn(
            self.norm2(q),
            text,
            text,
            key_padding_mask=~text_valid,
            need_weights=False,
        )[0]
        q = q + self.visual_attn(
            self.norm3(q),
            visual,
            visual,
            key_padding_mask=~visual_valid,
            need_weights=False,
        )[0]
        q = q + self.ffn(self.norm4(q))
        return q


class ExpressionConditionedQueryDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([QueryDecoderLayer(dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(
        self,
        text: torch.Tensor,
        text_valid: torch.Tensor,
        visual: torch.Tensor,
        visual_valid: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_embed.unsqueeze(0).expand(visual.shape[0], -1, -1).to(visual.device, visual.dtype)
        for layer in self.layers:
            q = layer(q, text, text_valid, visual, visual_valid)
        return q


class MaskResponseHead(nn.Module):
    def __init__(self, dim: int, mask_dim: int = 256, logit_scale: float | None = None) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, mask_dim)
        self.v_proj = nn.Linear(dim, mask_dim)
        # q/v are L2-normalized, so their dot product is already a cosine
        # similarity in [-1, 1]. Dividing it by sqrt(mask_dim) collapses mask
        # probabilities around 0.5. Keep the legacy value when no scale is
        # provided so old checkpoints remain behavior-compatible, while new
        # runs can use a temperature that yields useful mask logits.
        self.logit_scale = float(1.0 / math.sqrt(mask_dim) if logit_scale is None else logit_scale)

    def forward(self, queries: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.q_proj(queries), dim=-1)
        v = F.normalize(self.v_proj(visual), dim=-1)
        return torch.einsum("bkd,bnd->bkn", q, v) * self.logit_scale


class TargetTokenModulator(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int = 512, alpha: float = 0.2) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_size),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, visual: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        orig_dtype = visual.dtype
        adapter_dtype = next(self.adapter.parameters()).dtype
        visual_f = visual.to(dtype=adapter_dtype)
        response_f = response.to(dtype=adapter_dtype)
        out = visual_f + self.alpha.to(dtype=adapter_dtype) * response_f.unsqueeze(-1) * self.adapter(visual_f)
        return out.to(dtype=orig_dtype)


def masked_bce_dice_cost(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits = logits.float()
    target = target.float().unsqueeze(1).expand_as(logits)
    valid_f = valid.unsqueeze(1).to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp_min(1.0)
    probs = torch.sigmoid(logits) * valid_f
    target_f = target * valid_f
    inter = (probs * target_f).sum(dim=-1)
    denom = probs.sum(dim=-1) + target_f.sum(dim=-1)
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return bce + dice


class Qwen3VLQueryGroundingModel(nn.Module):
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
        num_queries: int = 8,
        query_dim: int | None = None,
        conditioner_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 8,
        mask_dim: int = 256,
        mask_logit_scale: float | None = None,
        modulator_bottleneck: int = 512,
        lambda_mask: float = 1.0,
        lambda_cat: float = 0.5,
    ) -> None:
        super().__init__()
        self.qwen = qwen
        for param in self.qwen.parameters():
            param.requires_grad = False
        self.config = qwen.config
        hidden = int(qwen.config.text_config.hidden_size)
        query_dim = int(query_dim or hidden)
        self.expression_conditioner = ExpressionConditioner(hidden, query_dim, conditioner_layers, num_heads)
        self.query_decoder = ExpressionConditionedQueryDecoder(num_queries, query_dim, decoder_layers, num_heads)
        self.visual_to_query = nn.Linear(hidden, query_dim) if query_dim != hidden else nn.Identity()
        self.mask_head = MaskResponseHead(query_dim, mask_dim, logit_scale=mask_logit_scale)
        self.modulator = TargetTokenModulator(hidden, modulator_bottleneck)
        self.lambda_mask = float(lambda_mask)
        self.lambda_cat = float(lambda_cat)
        self.last_metrics: Dict[str, float] = {}

    def _pad_visual(self, inputs_embeds: torch.Tensor, visual_pos_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        chunks = [inputs_embeds[b, visual_pos_masks[b]] for b in range(inputs_embeds.shape[0])]
        return (
            pad_sequence(chunks, batch_first=True, padding_value=0.0),
            pad_sequence(
                [torch.ones(x.shape[0], dtype=torch.bool, device=x.device) for x in chunks],
                batch_first=True,
                padding_value=False,
            ),
        )

    def _scatter_visual(self, inputs_embeds: torch.Tensor, visual_pos_masks: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        out = inputs_embeds.clone()
        for b in range(out.shape[0]):
            n = int(visual_pos_masks[b].sum().item())
            out[b, visual_pos_masks[b]] = visual[b, :n].to(dtype=out.dtype)
        return out

    def _flat_response(self, response: torch.Tensor, visual_valid: torch.Tensor) -> torch.Tensor:
        return torch.cat([response[b, visual_valid[b]] for b in range(response.shape[0])], dim=0)

    def _modulate_deepstack(
        self,
        deepstack_visual_embeds: List[torch.Tensor] | None,
        flat_response: torch.Tensor | None,
    ) -> List[torch.Tensor] | None:
        if deepstack_visual_embeds is None or flat_response is None:
            return deepstack_visual_embeds
        return [self.modulator(x, flat_response.to(x.device)) for x in deepstack_visual_embeds]

    def save_query_grounding(self, output_dir: str | Path, extra_config: Dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "expression_conditioner": self.expression_conditioner.state_dict(),
                "query_decoder": self.query_decoder.state_dict(),
                "visual_to_query": self.visual_to_query.state_dict(),
                "mask_head": self.mask_head.state_dict(),
                "modulator": self.modulator.state_dict(),
            },
            output / "query_grounding_modules.pt",
        )
        if extra_config:
            (output / "query_grounding_config.json").write_text(
                json.dumps(extra_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        expression_input_ids: torch.LongTensor | None = None,
        expression_attention_mask: torch.Tensor | None = None,
        mask_labels: torch.Tensor | None = None,
        mask_label_valid: torch.Tensor | None = None,
        cache_position: Any | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> Qwen3VLCausalLMOutputWithPast:
        kwargs.pop("num_items_in_batch", None)
        model = self.qwen.model
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        if pixel_values is not None:
            image_embeds_tuple, deepstack_image_embeds = model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds_tuple, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_embeds_tuple, deepstack_video_embeds = model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds_tuple, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask_1d = image_mask[..., 0]
            video_mask_1d = video_mask[..., 0]
            visual_pos_masks = image_mask_1d | video_mask_1d
            deepstack_visual_embeds = []
            image_mask_joint = image_mask_1d[visual_pos_masks]
            video_mask_joint = video_mask_1d[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                joint[image_mask_joint, :] = img_embed
                joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds
        if visual_pos_masks is None:
            raise ValueError("Query grounding requires visual inputs.")
        if expression_input_ids is None or expression_attention_mask is None:
            raise ValueError("Missing expression ids for expression conditioner.")
        if mask_labels is None or mask_label_valid is None:
            raise ValueError("Missing token-level mask labels.")

        visual_hidden, visual_valid = self._pad_visual(inputs_embeds, visual_pos_masks)
        train_dtype = next(self.query_decoder.parameters()).dtype
        mask_labels = mask_labels.to(visual_hidden.device, dtype=train_dtype)
        mask_label_valid = mask_label_valid.to(visual_hidden.device) & visual_valid
        expr_embeds = model.get_input_embeddings()(expression_input_ids.to(inputs_embeds.device)).to(dtype=train_dtype)
        text_features, text_valid = self.expression_conditioner(
            expr_embeds,
            expression_attention_mask.to(inputs_embeds.device),
        )
        visual_query_input = visual_hidden.to(dtype=train_dtype)
        visual_query = self.visual_to_query(visual_query_input)
        queries = self.query_decoder(text_features, text_valid, visual_query, visual_valid)
        response_logits = self.mask_head(queries, visual_query)
        response_logits = response_logits.masked_fill(~visual_valid.unsqueeze(1), -30.0)

        costs = masked_bce_dice_cost(response_logits, mask_labels, mask_label_valid)
        k_star = costs.detach().argmin(dim=1)
        bidx = torch.arange(response_logits.shape[0], device=response_logits.device)
        mask_loss = costs[bidx, k_star].mean()
        selected_response = torch.sigmoid(response_logits[bidx, k_star]) * visual_valid.to(response_logits.dtype)

        with torch.no_grad():
            valid_f = mask_label_valid.to(dtype=selected_response.dtype)
            target_f = mask_labels * valid_f
            pred_f = selected_response * valid_f
            inter = (pred_f * target_f).sum(dim=-1)
            denom = pred_f.sum(dim=-1) + target_f.sum(dim=-1)
            soft_dice = ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
            valid_count = valid_f.sum(dim=-1).clamp_min(1.0)
            self.last_metrics = {
                "loss_mask": float(mask_loss.detach().float().cpu()),
                "mask_soft_dice": float(soft_dice.detach().float().cpu()),
                "mask_target_pos_rate": float((target_f.sum(dim=-1) / valid_count).mean().detach().float().cpu()),
                "mask_pred_pos_rate": float(((pred_f > 0.5).to(pred_f.dtype).sum(dim=-1) / valid_count).mean().detach().float().cpu()),
                "query_selected_mean": float(k_star.detach().float().mean().cpu()),
                "tokens_visual_mean": float(visual_valid.sum(dim=-1).detach().float().mean().cpu()),
            }
        modulated_visual = self.modulator(visual_hidden, selected_response)
        inputs_embeds = self._scatter_visual(inputs_embeds, visual_pos_masks, modulated_visual)
        deepstack_visual_embeds = self._modulate_deepstack(
            deepstack_visual_embeds,
            self._flat_response(selected_response, visual_valid),
        )

        loss = self.lambda_mask * mask_loss
        if labels is None or self.lambda_cat <= 0:
            self.last_metrics["loss_cat"] = 0.0
            self.last_metrics["loss_total_components"] = float(loss.detach().float().cpu())
            empty_logits = inputs_embeds.new_empty((inputs_embeds.shape[0], 0, self.qwen.config.text_config.vocab_size))
            return Qwen3VLCausalLMOutputWithPast(
                loss=loss,
                logits=empty_logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                rope_deltas=getattr(model, "rope_deltas", None),
            )

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()
            position_ids, rope_deltas = model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            model.rope_deltas = rope_deltas
        outputs = model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.qwen.lm_head(hidden_states[:, slice_indices, :])
        cat_loss = None
        if labels is not None:
            cat_loss = self.qwen.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.qwen.config.text_config.vocab_size,
            )
        loss = loss + self.lambda_cat * cat_loss
        self.last_metrics["loss_cat"] = float(cat_loss.detach().float().cpu())
        self.last_metrics["loss_total_components"] = float(loss.detach().float().cpu())
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=model.rope_deltas,
        )


class QueryGroundingTrainer(Trainer):
    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        metrics = getattr(model, "last_metrics", None)
        if metrics:
            logs = {**logs, **metrics}
        if logs:
            printable = {k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()}
            print("[QueryGroundingMetrics] " + json.dumps(printable, ensure_ascii=False, sort_keys=True), flush=True)
        super().log(logs, start_time=start_time)

    def _save(self, output_dir: str | None = None, state_dict: Dict[str, torch.Tensor] | None = None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.model, "save_query_grounding"):
            self.model.save_query_grounding(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--query-dim", type=int, default=1024)
    parser.add_argument("--conditioner-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mask-dim", type=int, default=256)
    parser.add_argument(
        "--mask-logit-scale",
        type=float,
        default=10.0,
        help="Temperature applied to cosine mask logits; legacy checkpoints used 1/sqrt(mask_dim).",
    )
    parser.add_argument("--modulator-bottleneck", type=int, default=512)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--lambda-cat", type=float, default=0.5)
    parser.add_argument("--mask-positive-threshold", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device-map", default="", choices=["", "auto"], help="Use HF device_map for model parallel loading.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map=args.device_map or None,
        local_files_only=True,
    )
    qwen.config.use_cache = False
    if args.gradient_checkpointing:
        qwen.gradient_checkpointing_enable()
    model = Qwen3VLQueryGroundingModel(
        qwen=qwen,
        num_queries=args.num_queries,
        query_dim=args.query_dim or None,
        conditioner_layers=args.conditioner_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        mask_dim=args.mask_dim,
        mask_logit_scale=args.mask_logit_scale,
        modulator_bottleneck=args.modulator_bottleneck,
        lambda_mask=args.lambda_mask,
        lambda_cat=args.lambda_cat,
    )
    trainable_dtype = torch.bfloat16 if args.bf16 else torch.float32
    trainable_modules = (
        model.expression_conditioner,
        model.query_decoder,
        model.visual_to_query,
        model.mask_head,
        model.modulator,
    )
    if getattr(qwen, "hf_device_map", None) is not None:
        model.hf_device_map = qwen.hf_device_map
        first_device = next((param.device for param in qwen.parameters() if param.device.type == "cuda"), None)
        if first_device is not None:
            for module in trainable_modules:
                module.to(device=first_device, dtype=trainable_dtype)
    else:
        for module in trainable_modules:
            module.to(dtype=trainable_dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[QueryGrounding] trainable={trainable} total={total} ratio={trainable/total:.6f}", flush=True)

    train_dataset = CategoryJsonlDataset(args.train_jsonl, args.max_train_samples)
    eval_dataset = CategoryJsonlDataset(args.val_jsonl, args.max_val_samples) if args.val_jsonl else None
    image_token_id = int(qwen.config.image_token_id)
    spatial_merge_size = int(qwen.model.visual.spatial_merge_size)
    collator = QueryGroundingCollator(
        processor=processor,
        system_prompt=SYSTEM_PROMPT,
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
        positive_threshold=args.mask_positive_threshold,
    )
    run_summary = vars(args).copy()
    run_summary.update(
        {
            "mode": "mask_supervised_expression_conditioned_target_query",
            "train_records": len(train_dataset),
            "val_records": len(eval_dataset) if eval_dataset is not None else 0,
            "image_token_id": image_token_id,
            "spatial_merge_size": spatial_merge_size,
            "trainable_params": trainable,
            "total_params": total,
        }
    )
    print("[QueryGrounding] run summary:", json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = QueryGroundingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train()
    model.save_query_grounding(output_dir, extra_config={"run_summary": run_summary, "system_prompt": SYSTEM_PROMPT})
    processor.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
