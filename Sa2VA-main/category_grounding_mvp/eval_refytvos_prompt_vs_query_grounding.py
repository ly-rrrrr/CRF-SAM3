#!/usr/bin/env python3
"""Compare main category prompt vs query-grounding category prediction.

This script evaluates a small Ref-Youtube-VOS/RefYTROS validation JSONL slice.
It writes two eval JSON files compatible with category_eval_viewer.py:
one for the main prompt baseline, one for the query-grounding variant.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_category_grounding_qwen3vl import read_jsonl  # noqa: E402
from train_query_grounding_qwen3vl import (  # noqa: E402
    Qwen3VLQueryGroundingModel,
    QueryGroundingCollator,
    SYSTEM_PROMPT as QUERY_SYSTEM_PROMPT,
)


def norm_label(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = re.sub(r",(\s*[}\]])", r"\1", text[start : i + 1])
                try:
                    value = json.loads(block)
                    return value if isinstance(value, dict) else {}
                except Exception:
                    return {}
    return {}


def resolve_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def load_query_config(path: str | Path) -> Dict[str, Any]:
    base = Path(path)
    config_path = base / "query_grounding_config.json"
    if not config_path.exists() and base.name.startswith("checkpoint-"):
        config_path = base.parent / "query_grounding_config.json"
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload.get("run_summary", payload)


def build_prompt_messages(record: Dict[str, Any], system_prompt: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Referring expression:\n"
                f"{record['expression']}\n\n"
                "Several frames from the same video are provided as visual context."
            ),
        }
    ]
    for frame_path in record.get("frame_paths", []):
        content.append({"type": "image", "image": frame_path})
    content.append({"type": "text", "text": "Return one JSON object only."})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
        {"role": "user", "content": content},
    ]


def build_query_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Referring expression:\n"
                f"{record['expression']}\n\n"
                "The frames are from the same video and provide visual context for the expression."
            ),
        }
    ]
    for frame_path in record.get("frame_paths", []):
        content.append({"type": "image", "image": frame_path})
    content.append({"type": "text", "text": 'Return exactly one JSON object with key "category" only.'})
    return [
        {"role": "system", "content": [{"type": "text", "text": QUERY_SYSTEM_PROMPT.strip()}]},
        {"role": "user", "content": content},
    ]


def parse_category(response: str) -> str:
    payload = first_json(response)
    pred = str(payload.get("category", "")).strip()
    if pred:
        return pred
    match = re.search(r"\"category\"\s*:\s*\"([^\"]+)\"", response)
    if match:
        return match.group(1).strip()
    # Query-grounding checkpoints occasionally emit the intended short
    # category directly instead of wrapping it in JSON. Accept only a concise
    # noun phrase here; do not treat free-form explanations as categories.
    candidate = str(response).strip().strip(chr(96)).strip().strip("\"'")
    if re.fullmatch(r"[\w][\w /'-]{0,79}", candidate, flags=re.UNICODE):
        return candidate.strip()
    return ""


@torch.no_grad()
def generate_with_model(
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"do_sample_frames": False},
    ).to(resolve_device(model))
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], output_ids)]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def pad_visual_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if mask.shape[1] == target_len:
        return mask
    return F.pad(mask, (0, target_len - mask.shape[1]), value=False)


@torch.no_grad()
def build_query_context(
    qg: Qwen3VLQueryGroundingModel,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor] | None, Dict[str, float]]:
    qwen_model = qg.qwen.model
    input_ids = batch["input_ids"].to(resolve_device(qg.qwen))
    inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    pixel_values_videos = batch.get("pixel_values_videos")
    video_grid_thw = batch.get("video_grid_thw")
    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None
    if pixel_values is not None:
        pixel_values = pixel_values.to(inputs_embeds.device)
        image_grid_thw = image_grid_thw.to(inputs_embeds.device)
        image_embeds_tuple, deepstack_image_embeds = qwen_model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds_tuple, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(inputs_embeds.device)
        video_grid_thw = video_grid_thw.to(inputs_embeds.device)
        video_embeds_tuple, deepstack_video_embeds = qwen_model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds_tuple, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
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
    else:
        raise ValueError("Query grounding requires visual inputs.")

    visual_hidden, visual_valid = qg._pad_visual(inputs_embeds, visual_pos_masks)
    train_dtype = next(qg.query_decoder.parameters()).dtype
    expr_ids = batch["expression_input_ids"].to(inputs_embeds.device)
    expr_mask = batch["expression_attention_mask"].to(inputs_embeds.device)
    expr_embeds = qwen_model.get_input_embeddings()(expr_ids).to(dtype=train_dtype)
    text_features, text_valid = qg.expression_conditioner(expr_embeds, expr_mask)
    visual_query = qg.visual_to_query(visual_hidden.to(dtype=train_dtype))
    queries = qg.query_decoder(text_features, text_valid, visual_query, visual_valid)
    response_logits = qg.mask_head(queries, visual_query).masked_fill(~visual_valid.unsqueeze(1), -30.0)
    response = torch.sigmoid(response_logits).amax(dim=1) * visual_valid.to(response_logits.dtype)
    modulated_visual = qg.modulator(visual_hidden, response)
    deepstack_visual_embeds = qg._modulate_deepstack(
        deepstack_visual_embeds,
        qg._flat_response(response, visual_valid),
    )
    valid_f = visual_valid.to(dtype=response.dtype)
    stats = {
        "query_response_mean": float((response * valid_f).sum().float().cpu() / valid_f.sum().clamp_min(1).float().cpu()),
        "query_response_max": float(response.max().float().cpu()),
        "query_response_pos_rate": float(((response > 0.5).to(response.dtype) * valid_f).sum().float().cpu() / valid_f.sum().clamp_min(1).float().cpu()),
        "tokens_visual": float(visual_valid.sum().float().cpu()),
    }
    return visual_pos_masks, modulated_visual, deepstack_visual_embeds, stats


@torch.no_grad()
def generate_with_query_grounding(
    qg: Qwen3VLQueryGroundingModel,
    processor: Any,
    collator: QueryGroundingCollator,
    record: Dict[str, Any],
    max_new_tokens: int,
) -> Tuple[str, Dict[str, float]]:
    encoded = collator._encode_one(record)
    batch: Dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0) if value.dim() == 1 else value
    prompt = processor.apply_chat_template(
        build_query_messages(record),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"do_sample_frames": False},
    ).to(resolve_device(qg.qwen))
    # Reuse prompt ids from the generation template. Visual features and grids
    # come from the collator so token-level response maps stay aligned.
    batch["input_ids"] = prompt["input_ids"]
    batch["attention_mask"] = prompt["attention_mask"]
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
        if key in prompt and torch.is_tensor(prompt[key]):
            batch[key] = prompt[key]
    visual_pos_masks, modulated_visual, deepstack_visual_embeds, stats = build_query_context(qg, batch)
    qwen_model = qg.qwen.model
    generated: List[int] = []
    eos_ids = set()
    eos = processor.tokenizer.eos_token_id
    if isinstance(eos, int):
        eos_ids.add(eos)
    input_ids = prompt["input_ids"].to(resolve_device(qg.qwen))
    attention = prompt["attention_mask"].to(input_ids.device)
    image_grid_thw = batch.get("image_grid_thw")
    video_grid_thw = batch.get("video_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(input_ids.device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(input_ids.device)
    for _ in range(max_new_tokens):
        if generated:
            gen = torch.tensor([generated], dtype=input_ids.dtype, device=input_ids.device)
            cur_ids = torch.cat([input_ids, gen], dim=1)
            cur_attention = torch.cat([attention, torch.ones_like(gen)], dim=1)
        else:
            cur_ids = input_ids
            cur_attention = attention
        cur_embeds = qwen_model.get_input_embeddings()(cur_ids)
        cur_visual_mask = pad_visual_mask(visual_pos_masks.to(cur_ids.device), cur_ids.shape[1])
        cur_embeds = qg._scatter_visual(cur_embeds, cur_visual_mask, modulated_visual)
        position_ids, rope_deltas = qwen_model.get_rope_index(
            cur_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=cur_attention,
        )
        qwen_model.rope_deltas = rope_deltas
        outputs = qwen_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=cur_attention,
            inputs_embeds=cur_embeds,
            visual_pos_masks=cur_visual_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = qg.qwen.lm_head(outputs.last_hidden_state[:, -1, :])
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        if next_id in eos_ids:
            break
        piece = processor.tokenizer.decode(generated, skip_special_tokens=True)
        if "}" in piece:
            break
    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return text, stats


def write_eval(path: Path, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--jsonl", default="category_grounding_mvp/data/refytvos_query_grounding/val.jsonl")
    parser.add_argument("--query-module-dir", required=True)
    parser.add_argument("--main-prompt", default="configs/category_num_targets_inference_merged.txt")
    parser.add_argument("--output-dir", default="category_grounding_mvp/outputs/eval_query_grounding_compare")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device-map", default="auto", choices=["", "auto"])
    args = parser.parse_args()

    records = read_jsonl(args.jsonl)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map=args.device_map or None,
        local_files_only=True,
    )
    qwen.eval()
    query_cfg = load_query_config(args.query_module_dir)
    qg = Qwen3VLQueryGroundingModel(
        qwen=qwen,
        num_queries=int(query_cfg.get("num_queries", 8)),
        query_dim=int(query_cfg.get("query_dim", 1024)),
        conditioner_layers=int(query_cfg.get("conditioner_layers", 2)),
        decoder_layers=int(query_cfg.get("decoder_layers", 2)),
        num_heads=int(query_cfg.get("num_heads", 8)),
        mask_dim=int(query_cfg.get("mask_dim", 256)),
        mask_logit_scale=query_cfg.get("mask_logit_scale"),
        modulator_bottleneck=int(query_cfg.get("modulator_bottleneck", 512)),
        lambda_mask=0.0,
        lambda_cat=0.0,
    )
    state_path = Path(args.query_module_dir) / "query_grounding_modules.pt"
    state = torch.load(state_path, map_location="cpu")
    qg.expression_conditioner.load_state_dict(state["expression_conditioner"])
    qg.query_decoder.load_state_dict(state["query_decoder"])
    qg.visual_to_query.load_state_dict(state["visual_to_query"])
    qg.mask_head.load_state_dict(state["mask_head"])
    qg.modulator.load_state_dict(state["modulator"])
    train_dtype = torch.bfloat16 if args.bf16 else torch.float32
    first_device = next((p.device for p in qwen.parameters() if p.device.type == "cuda"), resolve_device(qwen))
    for module in (qg.expression_conditioner, qg.query_decoder, qg.visual_to_query, qg.mask_head, qg.modulator):
        module.to(device=first_device, dtype=train_dtype)
        module.eval()
    qg.eval()
    collator = QueryGroundingCollator(
        processor=processor,
        system_prompt=QUERY_SYSTEM_PROMPT,
        image_token_id=int(qwen.config.image_token_id),
        spatial_merge_size=int(qwen.model.visual.spatial_merge_size),
        positive_threshold=float(query_cfg.get("mask_positive_threshold", 0.05)),
    )
    main_prompt = Path(args.main_prompt).read_text(encoding="utf-8")

    base_records: List[Dict[str, Any]] = []
    query_records: List[Dict[str, Any]] = []
    base_exact = 0
    query_exact = 0
    for idx, record in enumerate(records, 1):
        gt = str(record.get("category", "")).strip()
        base_raw = generate_with_model(
            qwen,
            processor,
            build_prompt_messages(record, main_prompt),
            args.max_new_tokens,
        )
        base_pred = parse_category(base_raw)
        query_raw, query_stats = generate_with_query_grounding(qg, processor, collator, record, args.max_new_tokens)
        query_pred = parse_category(query_raw)
        base_ok = norm_label(base_pred) == norm_label(gt)
        query_ok = norm_label(query_pred) == norm_label(gt)
        base_exact += int(base_ok)
        query_exact += int(query_ok)
        common = {
            "sample_id": record.get("sample_id", ""),
            "global_index": record.get("global_index"),
            "video_id": record.get("video_id", ""),
            "exp_id": record.get("exp_id", ""),
            "obj_id": record.get("obj_id", ""),
            "expression": record.get("expression", ""),
            "category": gt,
            "frame_paths": record.get("frame_paths", []),
            "frames": record.get("frames", []),
            "mask_frame_indices": record.get("mask_frame_indices", []),
        }
        base_records.append(
            {
                **common,
                "pred_category": base_pred,
                "raw_response": base_raw,
                "status": "ok" if base_pred else "error",
                "category_exact": base_ok,
            }
        )
        query_records.append(
            {
                **common,
                "pred_category": query_pred,
                "raw_response": query_raw,
                "status": "ok" if query_pred else "error",
                "category_exact": query_ok,
                "query_stats": query_stats,
            }
        )
        print(
            f"[{idx}/{len(records)}] {record.get('video_id')}/{record.get('exp_id')} "
            f"gt={gt} main={base_pred or '<empty>'}({base_ok}) "
            f"query={query_pred or '<empty>'}({query_ok})",
            flush=True,
        )

    out_dir = Path(args.output_dir)
    base_summary = {
        "method_name": "main_prompt",
        "total": len(records),
        "category_exact": base_exact,
        "category_exact_rate": base_exact / len(records) if records else 0.0,
        "jsonl": args.jsonl,
        "prompt": args.main_prompt,
    }
    query_summary = {
        "method_name": "query_grounding",
        "total": len(records),
        "category_exact": query_exact,
        "category_exact_rate": query_exact / len(records) if records else 0.0,
        "jsonl": args.jsonl,
        "query_module_dir": args.query_module_dir,
    }
    write_eval(out_dir / "refytvos_val_subset_main_prompt.json", base_summary, base_records)
    write_eval(out_dir / "refytvos_val_subset_query_grounding.json", query_summary, query_records)
    write_eval(
        out_dir / "summary.json",
        {
            "main_prompt": base_summary,
            "query_grounding": query_summary,
            "max_samples": args.max_samples,
        },
        [],
    )
    print(json.dumps({"main_prompt": base_summary, "query_grounding": query_summary}, ensure_ascii=False, indent=2))
    print(f"[output] {out_dir}")


if __name__ == "__main__":
    main()
