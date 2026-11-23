#!/usr/bin/env python3
"""Train Ground v2 (see README). Saves heads + LoRA adapter + processor to --output-dir.

  python -m gv2.train --model-path .../Qwen3-VL-4B-Instruct \
      --train-jsonl data/revos_v2/train.jsonl --val-jsonl data/revos_v2/val.jsonl \
      --output-dir runs/gv2_revos --bf16 --gradient-checkpointing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from .common import TGT_TOKEN
from .data import GroundV2Collator, JsonlDataset
from .model import GroundV2, attach_lora, base_model_of


class GV2Trainer(Trainer):
    """Trainer that (1) averages the loss parts over every micro-batch between two logs,
    (2) prints one ``[gv2] {json}`` line per log (gv2.plot_log draws curves from these lines),
    (3) flags a non-finite training loss so NanGuard can stop before anything bad is saved."""

    nonfinite = False
    nonfinite_step = -1

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # GroundV2 returns mean loss and ignores num_items_in_batch.
        self.model_accepts_loss_kwargs = False
        self.nonfinite = False
        self.nonfinite_step = -1

    def get_eval_dataloader(self, eval_dataset=None):
        import copy
        original = self.data_collator
        self.data_collator = copy.copy(original)
        self.data_collator.absent_prob = 0.0
        self.data_collator.max_frames = 0
        self.data_collator.stats = dict.fromkeys(original.stats, 0)
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original

    def _acc(self, train: bool) -> Dict[str, list]:
        name = "_acc_train" if train else "_acc_eval"
        if not hasattr(self, name):
            setattr(self, name, {})
        return getattr(self, name)

    def compute_loss(self, model, inputs, *a, **k):
        res = super().compute_loss(model, inputs, *a, **k)
        loss = res[0] if isinstance(res, tuple) else res
        m = model.module if hasattr(model, "module") else model
        acc = self._acc(m.training)
        for kk, v in getattr(m, "last_metrics", {}).items():
            acc.setdefault(kk, []).append(float(v))
        if m.training:
            bad = torch.tensor(0.0 if torch.isfinite(loss.detach()).all() else 1.0, device=loss.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)  # every rank stops together
            if bad.item() > 0 and not self.nonfinite:
                self.nonfinite, self.nonfinite_step = True, int(self.state.global_step) + 1
                print(f"[gv2] WARNING non-finite training loss at step {self.nonfinite_step}: "
                      f"{getattr(m, 'last_metrics', {})}", flush=True)
        return res

    def log(self, logs: Dict[str, float], *a, **k) -> None:
        train = "loss" in logs
        if train or "eval_loss" in logs:
            acc = self._acc(train)
            prefix = "" if train else "eval_"
            parts = {}
            for kk, v in sorted(acc.items()):
                pair = torch.tensor([sum(v), len(v)], device=self.args.device, dtype=torch.float64)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(pair)
                if pair[1] > 0:
                    parts[prefix + kk] = float(pair[0] / pair[1])
            logs = {**logs, **parts}
            acc.clear()
        logs = {**logs, "step": int(self.state.global_step)}
        if self.is_world_process_zero():
            print("[gv2] " + json.dumps({kk: (float(v) if isinstance(v, (int, float)) else v) for kk, v in logs.items()},
                                        ensure_ascii=False, sort_keys=True), flush=True)
        super().log({kk: v for kk, v in logs.items() if kk != "step"}, *a, **k)

    def _save(self, output_dir=None, state_dict=None) -> None:
        if self.nonfinite or not self.is_world_process_zero():
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir, extra={"run": getattr(self, "run_summary", {})})
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


class NanGuard(TrainerCallback):
    """Stops right after the optimizer step that saw a non-finite loss, and cancels the
    save/eval of that step, so every checkpoint on disk predates the blow-up."""

    def __init__(self, trainer: GV2Trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kw):
        if self.trainer.nonfinite:
            control.should_training_stop = True
            control.should_save = False
            control.should_evaluate = False
            control.should_log = True
        return control


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    # architecture
    p.add_argument("--feature-layer", type=int, default=12, help="read heatmap features from the output of this text layer (1-based)")
    p.add_argument("--query-dim", type=int, default=1024)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--mask-dim", type=int, default=256)
    p.add_argument("--logit-scale", type=float, default=10.0)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--summary-drop", type=float, default=0.3)
    p.add_argument("--no-summary", action="store_true", help="ablation rows B-D: no target-summary tokens")
    # LoRA
    p.add_argument("--no-lora", action="store_true", help="ablation row B: LLM frozen")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-layers", default="", help="e.g. '0-11,30-35' (0-based) to save VRAM; default all layers")
    # supervision
    p.add_argument("--lambda-mask", type=float, default=1.0)
    p.add_argument("--lambda-cat", type=float, default=0.5)
    p.add_argument("--neg-weight", type=float, default=3.0)
    p.add_argument("--absent-prob", type=float, default=0.2)
    p.add_argument("--positive-threshold", type=float, default=0.05)
    p.add_argument("--max-image-side", type=int, default=0, help="0 = original resolution (default)")
    p.add_argument("--max-frames", type=int, default=0, help="VRAM fallback: keep at most this many frames per sample")
    p.add_argument("--frame-cache", default="runs/_frame_cache")
    # optimisation
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping (HF default)")
    p.add_argument("--report-to", default="none", help="'tensorboard' to also write TensorBoard logs under output-dir/runs")
    return p.parse_args(argv)


def parse_layers(spec: str):
    out = []
    for part in [x for x in spec.split(",") if x.strip()]:
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out or None


def build(args, processor, qwen):
    """Shared by train() and the tests."""
    tok = processor.tokenizer
    if TGT_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [TGT_TOKEN]})
    if len(tok) > qwen.get_input_embeddings().weight.shape[0]:
        qwen.resize_token_embeddings(len(tok))
    tgt_id = tok.convert_tokens_to_ids(TGT_TOKEN)
    im_start = tok.convert_tokens_to_ids("<|im_start|>") if "<|im_start|>" in tok.get_vocab() else None
    for prm in qwen.parameters():
        prm.requires_grad = False
    if args.gradient_checkpointing:
        # non-reentrant: the heatmap computed inside a checkpointed layer must keep its graph
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not args.no_lora:
        qwen = attach_lora(qwen, args.lora_r, args.lora_alpha, args.lora_dropout, parse_layers(args.lora_layers))
    model = GroundV2(qwen, tgt_token_id=tgt_id, im_start_id=im_start, feature_layer=args.feature_layer,
                     query_dim=args.query_dim, decoder_layers=args.decoder_layers, num_heads=args.num_heads,
                     mask_dim=args.mask_dim, logit_scale=args.logit_scale, topk=args.topk,
                     summary_drop=args.summary_drop, lambda_mask=args.lambda_mask, lambda_cat=args.lambda_cat,
                     use_summary=not args.no_summary)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    dev = next(base_model_of(qwen).model.language_model.layers[args.feature_layer - 1].parameters()).device
    for m in model.head_modules():
        m.to(device=dev, dtype=dtype)
    model.query_bias.data = model.query_bias.data.to(dev, dtype)
    model.null_summary.data = model.null_summary.data.to(dev, dtype)
    merge = int(base_model_of(qwen).model.visual.spatial_merge_size)
    return model, merge


def main(argv=None) -> None:
    args = parse_args(argv)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16 if args.bf16 else torch.float32)
    if torch.cuda.is_available():
        qwen.to(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")  # torchrun: one full copy per GPU
    qwen.config.use_cache = False
    model, merge = build(args, processor, qwen)

    train_ds = JsonlDataset(args.train_jsonl, args.max_train_samples)
    val_ds = JsonlDataset(args.val_jsonl, args.max_val_samples) if args.val_jsonl else None
    image_token_id = int(base_model_of(model.qwen).config.image_token_id)
    collator = GroundV2Collator(processor, merge, image_token_id, args.positive_threshold, args.neg_weight,
                                args.absent_prob, args.max_image_side, args.frame_cache, train=True,
                                max_frames=args.max_frames)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run = {**vars(args), "train_records": len(train_ds), "val_records": len(val_ds) if val_ds else 0,
           "trainable_params": trainable}
    print("[gv2] run:", json.dumps(run, ensure_ascii=False, indent=2), flush=True)

    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps, learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio, logging_steps=args.logging_steps, logging_first_step=True,
        save_steps=args.save_steps, eval_steps=args.eval_steps, eval_strategy="steps" if val_ds else "no",
        save_strategy="steps", bf16=args.bf16, remove_unused_columns=False, report_to=args.report_to,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=0, seed=args.seed, prediction_loss_only=True,
        ddp_find_unused_parameters=True,  # null-summary steps leave the summary MLPs unused
    )
    trainer = GV2Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                         data_collator=collator, processing_class=processor)
    trainer.run_summary = run
    trainer.add_callback(NanGuard(trainer))
    trainer.train()
    if trainer.nonfinite:
        print(f"[gv2] STOPPED: non-finite loss at step {trainer.nonfinite_step}. The weights in memory are "
              f"damaged and were NOT saved; checkpoints already in {out} predate the problem. "
              "Retry with a lower --learning-rate (e.g. 5e-5) or a smaller --max-grad-norm.", flush=True)
        raise SystemExit(2)
    trainer._save(str(out))
    print("[gv2] collator stats:", json.dumps(collator.stats), flush=True)


if __name__ == "__main__":
    main()
