#!/usr/bin/env python3
"""Train a mask-conditioned visual-token adapter on top of frozen Qwen3-VL.

The adapter is inserted after Qwen3-VL has converted images into visual token
embeddings and scattered them into the language-token sequence, but before the
language model consumes the sequence. This keeps the RGB image encoder intact and
makes the trainable component explicitly operate on visual tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_category_grounding_qwen3vl import CategoryJsonlDataset  # noqa: E402
from train_mask_guided_category_qwen3vl import MaskGuidedCollator, SYSTEM_PROMPT  # noqa: E402


class VisualTokenAdapterCollator(MaskGuidedCollator):
    def __init__(
        self,
        processor: Any,
        system_prompt: str,
        category_query_tokens: List[str],
        image_token_id: int,
        spatial_merge_size: int,
    ) -> None:
        super().__init__(processor, system_prompt, category_query_tokens)
        self.image_token_id = int(image_token_id)
        self.spatial_merge_size = int(spatial_merge_size)

    def _image_group_order(self, record: Dict[str, Any]) -> List[int]:
        groups = [0] * len(record.get("original_frame_paths") or record.get("frame_paths", []))
        if record.get("input_mode") == "mask_evidence":
            groups.extend([1] * len(record.get("highlight_frame_paths", [])))
            groups.extend([2] * len(record.get("crop_frame_paths", [])))
        return groups

    def _encode_one(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out = super()._encode_one(record)
        token_types = torch.full_like(out["input_ids"], fill_value=-1)
        if "image_grid_thw" not in out:
            out["visual_token_type_ids"] = token_types
            return out

        image_positions = (out["input_ids"] == self.image_token_id).nonzero(as_tuple=False).flatten()
        group_order = self._image_group_order(record)
        image_grid = out["image_grid_thw"].detach().cpu()
        token_counts = (
            image_grid.prod(dim=-1) // (self.spatial_merge_size * self.spatial_merge_size)
        ).tolist()
        if len(group_order) != len(token_counts):
            raise ValueError(
                f"image group count mismatch: groups={len(group_order)} grids={len(token_counts)} "
                f"pair_id={record.get('pair_id')} mode={record.get('input_mode')}"
            )
        if int(sum(token_counts)) != int(image_positions.numel()):
            raise ValueError(
                f"image token count mismatch: placeholders={image_positions.numel()} grids={sum(token_counts)} "
                f"pair_id={record.get('pair_id')} mode={record.get('input_mode')}"
            )
        offset = 0
        for group_id, count in zip(group_order, token_counts):
            positions = image_positions[offset : offset + int(count)]
            token_types[positions] = int(group_id)
            offset += int(count)
        out["visual_token_type_ids"] = token_types
        return out

    def __call__(self, records: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(records)
        encoded = [self._encode_one(record) for record in records]
        batch["visual_token_type_ids"] = pad_sequence(
            [x["visual_token_type_ids"] for x in encoded],
            batch_first=True,
            padding_value=-1,
        )
        return batch


class MaskConditionedVisualTokenAdapter(nn.Module):
    """Small residual adapter for visual-token positions.

    group ids:
      0 = original frame token
      1 = target-highlight token
      2 = target-crop token
    """

    def __init__(self, hidden_size: int, bottleneck: int = 512, dropout: float = 0.05) -> None:
        super().__init__()
        self.group_embed = nn.Embedding(3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size * 2, bottleneck)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def adapt_flat(self, visual_embeds: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        if visual_embeds.numel() == 0:
            return visual_embeds
        valid = group_ids >= 0
        if not bool(valid.any()):
            return visual_embeds
        x = visual_embeds
        groups = group_ids.clamp(min=0, max=2).to(x.device)
        x_grouped = x + self.group_embed(groups).to(dtype=x.dtype)
        target = (group_ids == 1) | (group_ids == 2)
        if bool(target.any()):
            context = x_grouped[target].mean(dim=0, keepdim=True)
        else:
            context = x_grouped[valid].mean(dim=0, keepdim=True)
        context = self.context_norm(context).expand_as(x_grouped)
        z = torch.cat([self.norm(x_grouped), context], dim=-1)
        delta = self.up(self.dropout(self.act(self.down(z))))
        gate = torch.sigmoid(self.gate(z))
        return x + self.scale.to(dtype=x.dtype) * gate * delta

    def forward(self, inputs_embeds: torch.Tensor, visual_token_type_ids: torch.Tensor | None) -> torch.Tensor:
        if visual_token_type_ids is None:
            return inputs_embeds
        visual_token_type_ids = visual_token_type_ids.to(inputs_embeds.device)
        out = inputs_embeds.clone()
        for b in range(out.shape[0]):
            mask = visual_token_type_ids[b] >= 0
            if not bool(mask.any()):
                continue
            out[b, mask] = self.adapt_flat(out[b, mask], visual_token_type_ids[b, mask])
        return out


class Qwen3VLWithVisualTokenAdapter(nn.Module):
    def __init__(self, qwen: Qwen3VLForConditionalGeneration, adapter: MaskConditionedVisualTokenAdapter) -> None:
        super().__init__()
        self.qwen = qwen
        self.visual_adapter = adapter
        for param in self.qwen.parameters():
            param.requires_grad = False
        self.config = qwen.config

    def _adapt_deepstack(self, deepstack_visual_embeds: List[torch.Tensor] | None, flat_group_ids: torch.Tensor | None):
        if deepstack_visual_embeds is None or flat_group_ids is None:
            return deepstack_visual_embeds
        return [self.visual_adapter.adapt_flat(x, flat_group_ids.to(x.device)) for x in deepstack_visual_embeds]

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
        visual_token_type_ids: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> Qwen3VLCausalLMOutputWithPast:
        model = self.qwen.model
        kwargs.pop("num_items_in_batch", None)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_outputs = model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = image_outputs.deepstack_features
            image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_video_embeds = video_outputs.deepstack_features
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
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        flat_group_ids = None
        if visual_pos_masks is not None and visual_token_type_ids is not None:
            visual_token_type_ids = visual_token_type_ids.to(inputs_embeds.device)
            flat_group_ids = visual_token_type_ids[visual_pos_masks]

        inputs_embeds = self.visual_adapter(inputs_embeds, visual_token_type_ids)
        deepstack_visual_embeds = self._adapt_deepstack(deepstack_visual_embeds, flat_group_ids)

        if position_ids is None:
            position_ids = model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.qwen.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.qwen.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.qwen.config.text_config.vocab_size,
            )
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def save_adapter(self, output_dir: str | Path, extra_config: Dict[str, Any] | None = None) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.visual_adapter.state_dict(), output / "visual_token_adapter.pt")
        if extra_config is not None:
            (output / "visual_token_adapter_config.json").write_text(
                json.dumps(extra_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


class VisualTokenAdapterTrainer(Trainer):
    def _save(self, output_dir: str | None = None, state_dict: Dict[str, torch.Tensor] | None = None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.model
        if hasattr(model, "save_adapter"):
            model.save_adapter(output_dir)
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
    parser.add_argument("--bottleneck", type=int, default=512)
    parser.add_argument("--adapter-dropout", type=float, default=0.05)
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map="auto",
    )
    qwen.config.use_cache = False
    if args.gradient_checkpointing:
        qwen.gradient_checkpointing_enable()

    hidden_size = int(qwen.config.text_config.hidden_size)
    adapter = MaskConditionedVisualTokenAdapter(
        hidden_size=hidden_size,
        bottleneck=args.bottleneck,
        dropout=args.adapter_dropout,
    ).to(next(qwen.parameters()).device)
    model = Qwen3VLWithVisualTokenAdapter(qwen=qwen, adapter=adapter)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[VisualTokenAdapter] trainable={trainable} total={total} ratio={trainable/total:.6f}", flush=True)

    train_dataset = CategoryJsonlDataset(args.train_jsonl, args.max_train_samples)
    eval_dataset = CategoryJsonlDataset(args.val_jsonl, args.max_val_samples) if args.val_jsonl else None
    image_token_id = int(qwen.config.image_token_id)
    spatial_merge_size = int(qwen.model.visual.spatial_merge_size)
    collator = VisualTokenAdapterCollator(
        processor=processor,
        system_prompt=SYSTEM_PROMPT,
        category_query_tokens=[],
        image_token_id=image_token_id,
        spatial_merge_size=spatial_merge_size,
    )

    run_summary = {
        "mode": "mask_conditioned_visual_token_adapter",
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "output_dir": str(output_dir),
        "train_records": len(train_dataset),
        "val_records": len(eval_dataset) if eval_dataset is not None else 0,
        "hidden_size": hidden_size,
        "bottleneck": args.bottleneck,
        "adapter_dropout": args.adapter_dropout,
        "image_token_id": image_token_id,
        "spatial_merge_size": spatial_merge_size,
        "trainable_params": trainable,
        "total_params": total,
        "bf16": bool(args.bf16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
    }
    print("[VisualTokenAdapter] run summary:", json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)

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
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = VisualTokenAdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train()
    model.save_adapter(output_dir, extra_config={"run_summary": run_summary, "system_prompt": SYSTEM_PROMPT})
    processor.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
