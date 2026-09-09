# Category Grounding MVP

This is a standalone MVP for category-only multimodal fine-tuning. It learns to
map a referring expression plus supporting video frames to the referred target category. It
does not modify the CRF-SAM3 main training pipeline.

Goal:

```text
referring expression + supporting video frames -> {"category": "..."}
```

The target category is the object category resolved by
`Ref-Youtube-VOS/train/meta_expressions.json -> obj_id -> meta.json`.

## 1. Build Ref-Youtube-VOS SFT Data

```bash
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main

/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/build_refytvos_category_sft.py \
  --output-dir category_grounding_mvp/data/refytvos_category_sft \
  --frames-per-sample 3 \
  --val-ratio 0.1 \
  --random-view-copies 1
```

This writes:

```text
category_grounding_mvp/data/refytvos_category_sft/train.jsonl
category_grounding_mvp/data/refytvos_category_sft/val.jsonl
category_grounding_mvp/data/refytvos_category_sft/summary.json
```

`random-view-copies` adds extra training examples with different frame samples and
the same category label. This is a cheap view-consistency approximation without a
separate KL loss.

## 2. Train Category-Grounding Adapter

Minimal LLM LoRA:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/train_category_grounding_qwen3vl.py \
  --train-jsonl category_grounding_mvp/data/refytvos_category_sft/train.jsonl \
  --val-jsonl category_grounding_mvp/data/refytvos_category_sft/val.jsonl \
  --output-dir category_grounding_mvp/outputs/qwen3vl_refytvos_category_lora \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --bf16
```

MVP with projector LoRA and trainable category-query tokens:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/train_category_grounding_qwen3vl.py \
  --train-jsonl category_grounding_mvp/data/refytvos_category_sft/train.jsonl \
  --val-jsonl category_grounding_mvp/data/refytvos_category_sft/val.jsonl \
  --output-dir category_grounding_mvp/outputs/qwen3vl_refytvos_category_query_projector_lora \
  --enable-projector-lora \
  --num-category-query-tokens 8 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --bf16
```

## 3. Continue From A Trained Adapter

Later ReVOS or other-domain category data should be trained as a continuation of
the Ref-Youtube-VOS category adapter, not as a fresh adapter from the base model:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/train_category_grounding_qwen3vl.py \
  --adapter-path category_grounding_mvp/outputs/qwen3vl_refytvos_category_query_lora \
  --train-jsonl path/to/next_domain_train.jsonl \
  --val-jsonl path/to/next_domain_val.jsonl \
  --output-dir category_grounding_mvp/outputs/qwen3vl_refytvos_then_next_domain_category_lora \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 5e-5 \
  --gradient-checkpointing \
  --bf16
```

Use `--resume-from-checkpoint` only to resume an interrupted Trainer checkpoint
from the same run. Use `--adapter-path` for staged training from a previously
saved PEFT adapter.

## 4. Evaluate Category Generation

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/eval_category_grounding_qwen3vl.py \
  --jsonl category_grounding_mvp/data/refytvos_category_sft/val.jsonl \
  --adapter-path category_grounding_mvp/outputs/qwen3vl_refytvos_category_query_projector_lora \
  --output-json category_grounding_mvp/outputs/refytvos_val_category_eval.json \
  --max-samples 200
```

## Notes

- Vision encoder parameters are frozen.
- Images are still fed through Qwen3-VL; freezing does not mean text-only.
- The loss is only applied to assistant output tokens, not to the prompt or image
  placeholder tokens.
- Category-query tokens are added as trainable special tokens and placed in the
  user prompt. They act as a light task-specific grounding workspace without
  changing Qwen3-VL source code.
- Projector LoRA targets exact Qwen3-VL vision merger/projector modules
  (`model.visual.merger` and `model.visual.deepstack_merger_list.*`), while
  keeping the full vision encoder frozen.

## Mask-Guided Visual Category Grounding MVP

This branch addresses the failure mode where category-only SFT learns dataset category priors but may ignore the true referent in the expression. The key idea is to use masks as training-time privileged visual evidence, not as an inference-time requirement.

Each expression produces paired records with the same `pair_id` and `category`:

- `input_mode=no_mask`: expression + original frames -> category
- `input_mode=mask_evidence`: expression + original frames + target-highlighted frames + target crops -> category

The mask-evidence view teaches expression-target visual binding. The no-mask view keeps the inference setting aligned with deployment.

Build a small smoke dataset:

```bash
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/build_refytvos_mask_guided_sft.py \
  --output-dir category_grounding_mvp/data/refytvos_mask_guided_smoke \
  --frames-per-sample 2 \
  --max-samples 20 \
  --val-ratio 0.2
```

Build the full RefYTROS mask-guided dataset:

```bash
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/build_refytvos_mask_guided_sft.py \
  --output-dir category_grounding_mvp/data/refytvos_mask_guided_sft \
  --frames-per-sample 3 \
  --val-ratio 0.1
```

Train from scratch instead of continuing the category-only adapter, to avoid carrying over the closed-category prior from the earlier experiment:

```bash
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
CUDA_VISIBLE_DEVICES=0 /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/train_mask_guided_category_qwen3vl.py \
  --train-jsonl category_grounding_mvp/data/refytvos_mask_guided_sft/train.jsonl \
  --val-jsonl category_grounding_mvp/data/refytvos_mask_guided_sft/val.jsonl \
  --output-dir category_grounding_mvp/outputs/qwen3vl_refytvos_mask_guided_from_scratch \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 5e-5 \
  --warmup-steps 100 \
  --logging-steps 5 \
  --eval-steps 500 \
  --save-steps 500 \
  --gradient-checkpointing \
  --bf16
```

The builder samples frames from `mask_dict` by first selecting frames whose target mask area is at least `--min-area`, then sampling evenly over those non-empty mask frames. Each record stores `mask_frame_indices` and `mask_areas` for audit.

This MVP currently uses paired SFT supervision. It does not yet add explicit hidden-state or logits consistency between the two views, but the data format already preserves the `pair_id` needed for that.

## Mask-Conditioned Visual Token Adapter

`train_visual_token_adapter_qwen3vl.py` implements the advanced adapter route. It freezes Qwen3-VL and trains a small adapter on visual-token positions only.

Architecture:

```text
images -> frozen Qwen3-VL visual encoder/projector -> visual token embeddings
                                                     + visual token group ids
                                                     + target evidence context
                                                     -> MaskConditionedVisualTokenAdapter
                                                     -> frozen Qwen3-VL language model
                                                     -> category JSON loss
```

Visual token groups are assigned by the collator:

- `0`: original-frame visual tokens
- `1`: target-highlight visual tokens
- `2`: target-crop visual tokens

For `no_mask` records, only group `0` is present. For `mask_evidence` records, groups `0/1/2` are all present. The adapter uses highlight/crop tokens as target-evidence context when available, but the same adapter also processes no-mask samples, so inference can use original frames only.

This is different from the paired-SFT script:

- `train_mask_guided_category_qwen3vl.py`: no new visual-token adapter; uses Qwen3-VL + LoRA-style training path.
- `train_visual_token_adapter_qwen3vl.py`: actual visual-token adapter; freezes Qwen3-VL and trains only the adapter.

Train from scratch with the visual-token adapter:

```bash
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
CUDA_VISIBLE_DEVICES=0 /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/train_visual_token_adapter_qwen3vl.py \
  --train-jsonl category_grounding_mvp/data/refytvos_mask_guided_sft/train.jsonl \
  --val-jsonl category_grounding_mvp/data/refytvos_mask_guided_sft/val.jsonl \
  --output-dir category_grounding_mvp/outputs/qwen3vl_refytvos_visual_token_adapter \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --warmup-steps 100 \
  --logging-steps 5 \
  --eval-steps 500 \
  --save-steps 500 \
  --gradient-checkpointing \
  --bf16
```

A short smoke run can be done with:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/train_visual_token_adapter_qwen3vl.py \
  --train-jsonl category_grounding_mvp/data/refytvos_mask_guided_smoke/train.jsonl \
  --val-jsonl category_grounding_mvp/data/refytvos_mask_guided_smoke/val.jsonl \
  --output-dir /tmp/qwen3vl_visual_token_adapter_smoke \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --bf16
```

## Mask-Supervised Expression-Conditioned Target Query Grounding

This is the current research route. It replaces the category-only and
mask-evidence input routes with a target-query grounding framework:

```text
original frames -> frozen Qwen3-VL vision encoder -> visual tokens V
expression -> Qwen token embedding -> lightweight bidirectional conditioner -> T
learnable target queries + T + V -> token-level response maps
GT masks supervise response maps only during training
matched response map softly modulates V -> Vtarget
Vtarget + expression -> frozen Qwen3-VL decoder -> {"category": "..."}
```

Important constraints:

- GT masks are never converted into highlight/crop input images.
- Inference uses only original frames and the referring expression.
- Target queries are category-agnostic slots, not fixed category prototypes.
- Visual tokens are softly modulated; their count, order, and grid alignment are preserved.

Build a smoke dataset:

```bash
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/build_refytvos_query_grounding.py \
  --output-dir category_grounding_mvp/data/refytvos_query_grounding_smoke \
  --frames-per-sample 2 \
  --max-samples 8 \
  --val-ratio 0.25
```

Build the full RefYTROS query-grounding dataset:

```bash
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/build_refytvos_query_grounding.py \
  --output-dir category_grounding_mvp/data/refytvos_query_grounding \
  --frames-per-sample 3 \
  --val-ratio 0.1
```

One-step smoke training:

```bash
cd /mnt/csip-500/share/kongjiawei/Sa2VA-main
CUDA_VISIBLE_DEVICES=3 /mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python \
  category_grounding_mvp/train_query_grounding_qwen3vl.py \
  --model-path /mnt/csip-200/share/kongjiawei/Sa2VA-main/Qwen3-VL-4B-Instruct \
  --train-jsonl category_grounding_mvp/data/refytvos_query_grounding_smoke/train.jsonl \
  --output-dir /tmp/query_grounding_smoke \
  --max-train-samples 1 \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --save-steps 1 \
  --bf16
```

Full training should start conservatively with the frozen Qwen3-VL backbone and
only the new modules trainable. The current defaults train about 75.6M parameters
with `query_dim=1024`.

