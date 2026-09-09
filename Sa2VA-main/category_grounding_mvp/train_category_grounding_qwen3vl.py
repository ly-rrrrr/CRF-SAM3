#!/usr/bin/env python3
"""Train a category-only Qwen3-VL adapter for visual category grounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


SYSTEM_PROMPT = """You are a referring-expression category grounding module.

Given a referring expression and frames from the same video, identify the visual
object category of the target referred to by the expression. Return exactly one
JSON object with key "category".
"""


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_mvp_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path) / "mvp_config.json"
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


class CategoryJsonlDataset(Dataset):
    def __init__(self, path: str | Path, max_samples: int = 0) -> None:
        self.records = read_jsonl(path)
        if max_samples and max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class CategorySFTCollator:
    def __init__(
        self,
        processor: Any,
        system_prompt: str,
        category_query_tokens: List[str],
    ) -> None:
        self.processor = processor
        self.system_prompt = system_prompt.strip()
        self.category_query_tokens = category_query_tokens
        self.pad_token_id = int(processor.tokenizer.pad_token_id)

    def _expression_text(self, record: Dict[str, Any]) -> str:
        return (
            "Referring expression:\n"
            f"{record['expression']}\n\n"
            "The frames are from the same video and provide visual context for the expression."
        )

    def _output_text(self) -> str:
        query = ""
        if self.category_query_tokens:
            query = "\nCategory query tokens: " + " ".join(self.category_query_tokens)
        return "Return exactly one JSON object with key \"category\" only." + query

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
            target = record.get("target_text")
            if not target:
                target = json.dumps({"category": record["category"]}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": target})
        return messages

    def _encode_one(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        full = self.processor.apply_chat_template(
            self._messages(record, with_answer=True),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            self._messages(record, with_answer=False),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        out: Dict[str, torch.Tensor] = {}
        for key, value in full.items():
            if not torch.is_tensor(value):
                continue
            if key in {"input_ids", "attention_mask", "mm_token_type_ids"} and value.dim() == 2:
                out[key] = value[0]
            else:
                out[key] = value

        prompt_len = int(prompt["input_ids"].shape[-1])
        labels = out["input_ids"].clone()
        labels[:prompt_len] = -100
        out["labels"] = labels
        return out

    @staticmethod
    def _pad_1d(values: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        return pad_sequence(values, batch_first=True, padding_value=pad_value)

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
        return batch


def make_category_query_tokens(n: int) -> List[str]:
    if n <= 0:
        return []
    return [f"<|cat_query_{i}|>" for i in range(n)]


def add_category_query_tokens(processor: Any, model: Any, tokens: List[str]) -> List[int]:
    if not tokens:
        return []
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    model.resize_token_embeddings(len(tokenizer))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if any(int(x) < 0 for x in token_ids):
        raise RuntimeError(f"Failed to add category query tokens: {tokens}")
    return [int(x) for x in token_ids]


def freeze_vision_encoder(model: Any) -> None:
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        return
    for param in visual.parameters():
        param.requires_grad = False


def build_lora_config(args: argparse.Namespace, trainable_token_ids: List[int]) -> LoraConfig:
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if args.enable_projector_lora:
        # Use exact module paths. Plain "linear_fc1/linear_fc2" would also match
        # every Qwen3-VL vision-block MLP and make visual-tower training far too
        # memory hungry. These modules are the vision-to-language merger/projector
        # layers only.
        targets.extend(
            [
                "model.visual.merger.linear_fc1",
                "model.visual.merger.linear_fc2",
                "model.visual.deepstack_merger_list.0.linear_fc1",
                "model.visual.deepstack_merger_list.0.linear_fc2",
                "model.visual.deepstack_merger_list.1.linear_fc1",
                "model.visual.deepstack_merger_list.1.linear_fc2",
                "model.visual.deepstack_merger_list.2.linear_fc1",
                "model.visual.deepstack_merger_list.2.linear_fc2",
            ]
        )

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        trainable_token_indices=trainable_token_ids or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter-path", default="", help="Existing PEFT adapter to continue training from.")
    parser.add_argument("--resume-from-checkpoint", default="", help="Optional Trainer checkpoint path for interrupted runs.")
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

    adapter_config = load_mvp_config(args.adapter_path) if args.adapter_path else {}
    processor_path = args.adapter_path if args.adapter_path else args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
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
            if query_tokens and requested != query_tokens:
                raise ValueError(
                    "--num-category-query-tokens does not match the existing adapter tokens. "
                    f"adapter={query_tokens}, requested={requested}"
                )
            query_tokens = requested
        if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(processor.tokenizer))
        freeze_vision_encoder(model)
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        query_tokens = make_category_query_tokens(args.num_category_query_tokens)
        query_token_ids = add_category_query_tokens(processor, model, query_tokens)
        freeze_vision_encoder(model)
        lora_config = build_lora_config(args, query_token_ids)
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = CategoryJsonlDataset(args.train_jsonl, args.max_train_samples)
    eval_dataset = None
    if args.val_jsonl:
        eval_dataset = CategoryJsonlDataset(args.val_jsonl, args.max_val_samples)

    run_summary = {
        "mode": "continue_adapter" if args.adapter_path else "new_adapter",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "output_dir": str(output_dir),
        "train_records": len(train_dataset),
        "val_records": len(eval_dataset) if eval_dataset is not None else 0,
        "enable_projector_lora": bool(args.enable_projector_lora),
        "num_category_query_tokens": len(query_tokens),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "bf16": bool(args.bf16),
    }
    print("[CategoryGroundingMVP] run summary:", json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)

    collator = CategorySFTCollator(
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
        "lora_target_note": "LLM LoRA plus optional exact Qwen3-VL vision merger/projector LoRA.",
    }
    (output_dir / "mvp_config.json").write_text(
        json.dumps(mvp_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
