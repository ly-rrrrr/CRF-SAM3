#!/usr/bin/env python3
"""Evaluate a Qwen3-VL category-grounding adapter on JSONL records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from train_category_grounding_qwen3vl import SYSTEM_PROMPT, read_jsonl


def resolve_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def norm_label(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def build_messages(
    record: Dict[str, Any],
    system_prompt: str,
    query_tokens: List[str],
) -> List[Dict[str, Any]]:
    query = ""
    if query_tokens:
        query = "\nCategory query tokens: " + " ".join(query_tokens)
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
    content.append(
        {
            "type": "text",
            "text": "Return exactly one JSON object with key \"category\" only." + query,
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
        {"role": "user", "content": content},
    ]


def load_mvp_config(adapter_path: str) -> Dict[str, Any]:
    if not adapter_path:
        return {}
    base = Path(adapter_path)
    for candidate in [base, *base.parents]:
        path = candidate / "mvp_config.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def has_processor_files(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in (
            "processor_config.json",
            "preprocessor_config.json",
            "image_processor_config.json",
            "tokenizer_config.json",
        )
    )


def resolve_processor_path(processor_path: str, adapter_path: str, model_path: str) -> str:
    if processor_path:
        return processor_path
    if adapter_path:
        base = Path(adapter_path)
        for candidate in [base, *base.parents]:
            if has_processor_files(candidate):
                return str(candidate)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--processor-path", default="")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.jsonl)
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    mvp_config = load_mvp_config(args.adapter_path)
    processor_path = resolve_processor_path(
        args.processor_path,
        args.adapter_path,
        args.model_path,
    )
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map="auto",
    )
    if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(processor.tokenizer))
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    system_prompt = mvp_config.get("system_prompt", SYSTEM_PROMPT)
    query_tokens = list(mvp_config.get("category_query_tokens", []))

    out_records: List[Dict[str, Any]] = []
    exact = 0
    valid = 0
    for idx, record in enumerate(records, 1):
        messages = build_messages(record, system_prompt, query_tokens)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_sample_frames": False},
        ).to(resolve_device(model))
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated_ids = [
            out[len(inp) :] for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        payload = first_json(response)
        pred = str(payload.get("category", "")).strip()
        gt = str(record.get("category", "")).strip()
        ok = norm_label(pred) == norm_label(gt)
        valid += int(bool(pred))
        exact += int(ok)
        out = dict(record)
        out.update(
            {
                "pred_category": pred,
                "raw_response": response,
                "status": "ok" if pred else "error",
                "category_exact": ok,
            }
        )
        out_records.append(out)
        print(
            f"[{idx}/{len(records)}] {record.get('video_id')}/{record.get('exp_id')} "
            f"gt={gt} pred={pred or '<empty>'} exact={ok}",
            flush=True,
        )

    summary = {
        "total": len(records),
        "valid": valid,
        "category_exact": exact,
        "valid_rate": valid / len(records) if records else 0.0,
        "category_exact_rate": exact / len(records) if records else 0.0,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "jsonl": args.jsonl,
    }
    payload = {"summary": summary, "records": out_records}
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
