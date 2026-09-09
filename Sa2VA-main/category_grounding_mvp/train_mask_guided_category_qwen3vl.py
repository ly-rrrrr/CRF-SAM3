#!/usr/bin/env python3
"""Train Qwen3-VL category grounding with mask-guided privileged evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
from peft import PeftModel, get_peft_model

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_category_grounding_qwen3vl import (
    CategoryJsonlDataset,
    CategorySFTCollator,
    add_category_query_tokens,
    build_lora_config,
    freeze_vision_encoder,
    load_mvp_config,
    make_category_query_tokens,
)

SYSTEM_PROMPT = """You are a referring-expression visual category grounding module.

Given a referring expression and visual evidence from the same video, identify the
visual object category of the target referred to by the expression. Some training
examples include target-highlight evidence created from ground-truth masks. Treat
that evidence as a training-time visual attention cue for binding the expression
to the target object. Return exactly one JSON object with key "category".
"""


class MaskGuidedCollator(CategorySFTCollator):
    def _expression_text(self, record: Dict[str, Any]) -> str:
        mode = record.get("input_mode", "no_mask")
        text = (
            "Referring expression:\n"
            f"{record['expression']}\n\n"
            "The original frames are from the same video and provide visual context."
        )
        if mode == "mask_evidence":
            text += (
                "\nThis is a training-time evidence view. Target-highlighted frames and target crops "
                "mark the visual region referred to by the expression. Use them to bind the "
                "expression to the target object category."
            )
        else:
            text += "\nNo target mask is provided in this view. Infer the target category from the expression and frames."
        return text

    def _messages(self, record: Dict[str, Any], with_answer: bool) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": self._expression_text(record)}]
        content.append({"type": "text", "text": "Original frames:"})
        for frame_path in record.get("original_frame_paths") or record.get("frame_paths", []):
            content.append({"type": "image", "image": frame_path})
        if record.get("input_mode") == "mask_evidence":
            highlight_paths = record.get("highlight_frame_paths", [])
            crop_paths = record.get("crop_frame_paths", [])
            if highlight_paths:
                content.append({"type": "text", "text": "Training-only target-highlighted frames:"})
                for frame_path in highlight_paths:
                    content.append({"type": "image", "image": frame_path})
            if crop_paths:
                content.append({"type": "text", "text": "Training-only target crops:"})
                for frame_path in crop_paths:
                    content.append({"type": "image", "image": frame_path})
        content.append({"type": "text", "text": self._output_text()})
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        if with_answer:
            target = record.get("target_text") or json.dumps({"category": record["category"]}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": target})
        return messages


def resolve_processor_path(adapter_path: str, model_path: str) -> str:
    if not adapter_path:
        return model_path
    base = Path(adapter_path)
    for candidate in [base, *base.parents]:
        if (candidate / "tokenizer_config.json").exists() or (candidate / "processor_config.json").exists():
            return str(candidate)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter-path", default="", help="Existing PEFT adapter to continue training from.")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--enable-projector-lora", action="store_true")
    parser.add_argument("--num-category-query-tokens", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = load_mvp_config(args.adapter_path) if args.adapter_path else {}
    processor = AutoProcessor.from_pretrained(
        resolve_processor_path(args.adapter_path, args.model_path),
        trust_remote_code=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map="auto",
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.adapter_path:
        query_tokens = list(adapter_config.get("category_query_tokens", []))
        if args.num_category_query_tokens > 0:
            requested = make_category_query_tokens(args.num_category_query_tokens)
            if query_tokens and query_tokens != requested:
                raise ValueError(f"adapter query tokens {query_tokens} != requested {requested}")
            query_tokens = requested
        if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(processor.tokenizer))
        freeze_vision_encoder(model)
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        query_tokens = make_category_query_tokens(args.num_category_query_tokens)
        query_token_ids = add_category_query_tokens(processor, model, query_tokens)
        freeze_vision_encoder(model)
        model = get_peft_model(model, build_lora_config(args, query_token_ids))
    model.print_trainable_parameters()

    train_dataset = CategoryJsonlDataset(args.train_jsonl, args.max_train_samples)
    eval_dataset = CategoryJsonlDataset(args.val_jsonl, args.max_val_samples) if args.val_jsonl else None
    run_summary = {
        "mode": "mask_guided_paired_sft",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "train_records": len(train_dataset),
        "val_records": len(eval_dataset) if eval_dataset is not None else 0,
        "enable_projector_lora": bool(args.enable_projector_lora),
        "num_category_query_tokens": len(query_tokens),
        "learning_rate": args.learning_rate,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "bf16": bool(args.bf16),
    }
    print("[MaskGuidedCategoryMVP] run summary:", json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)

    collator = MaskGuidedCollator(
        processor=processor,
        system_prompt=SYSTEM_PROMPT,
        category_query_tokens=query_tokens,
    )
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    mvp_config = {
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "continued_from_adapter_path": args.adapter_path,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "run_summary": run_summary,
        "system_prompt": SYSTEM_PROMPT,
        "category_query_tokens": query_tokens,
        "enable_projector_lora": bool(args.enable_projector_lora),
        "training_note": "Paired no-mask and training-time mask-evidence views share the same category supervision. This script does not yet add an explicit hidden/logit consistency loss.",
    }
    (output_dir / "mvp_config.json").write_text(json.dumps(mvp_config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
