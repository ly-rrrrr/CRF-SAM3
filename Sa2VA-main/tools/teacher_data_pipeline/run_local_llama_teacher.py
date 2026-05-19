#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import torch
try:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor


DEFAULT_MODEL_PATH = "/mnt/csip-500/share/kongjiawei/Sa2VA-main/gemma-4-26B-A4B-it"


def force_eager_experts(model) -> None:
    if hasattr(model, "config"):
        setattr(model.config, "_experts_implementation", "eager")
    for module in model.modules():
        if hasattr(module, "config"):
            try:
                setattr(module.config, "_experts_implementation", "eager")
            except Exception:
                pass


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_done_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    for row in load_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id:
            done.add(str(sample_id))
    return done


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = strip_code_fences(text)
    if not text:
        raise ValueError("empty_response")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no_json_object_found")

    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("json_object_not_dict")
    return obj


def decode_model_text(processor: AutoProcessor, generated_ids, input_len: int) -> str:
    return processor.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()


def build_messages(system_prompt: str, user_prompt: str, image_paths: List[str], max_images: int) -> List[Dict[str, Any]]:
    selected = image_paths[:max_images] if max_images > 0 else image_paths
    user_content: List[Dict[str, Any]] = []
    for path in selected:
        user_content.append({"type": "image", "path": path})
    user_content.append({"type": "text", "text": user_prompt})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def infer_one(
    processor: AutoProcessor,
    model: AutoModelForMultimodalLM,
    request_row: Dict[str, Any],
    max_images: int,
    max_new_tokens: int,
) -> tuple[Dict[str, Any], str]:
    teacher_request = request_row["teacher_request"]
    image_paths = teacher_request.get("images", [])
    if not image_paths:
        raise ValueError("no_images")

    messages = build_messages(
        system_prompt=teacher_request["system_prompt"],
        user_prompt=teacher_request["user_prompt"],
        image_paths=image_paths,
        max_images=max_images,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    target_device = getattr(model, "device", None)
    if target_device is None or str(target_device) == "meta":
        first_param = next(model.parameters())
        target_device = first_param.device

    moved_inputs = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved_inputs[key] = value.to(target_device)
        else:
            moved_inputs[key] = value

    input_len = moved_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **moved_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    raw_text = decode_model_text(processor, outputs, input_len)
    parsed = extract_first_json_object(raw_text)
    return parsed, raw_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Gemma multimodal teacher inference over teacher requests.")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-images", type=int, default=0, help="0 means use all request images.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    requests_path = Path(args.requests)
    output_path = Path(args.output)
    errors_path = Path(args.errors)

    done_ids = load_done_ids(output_path)
    error_ids = load_done_ids(errors_path) if args.skip_errors else set()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    force_eager_experts(model)
    model.eval()

    submitted = 0
    written = 0
    failed = 0

    for request_row in load_jsonl(requests_path):
        sample_id = str(request_row["sample_id"])
        if sample_id in done_ids:
            continue
        if args.skip_errors and sample_id in error_ids:
            continue

        submitted += 1
        try:
            output, raw_text = infer_one(
                processor=processor,
                model=model,
                request_row=request_row,
                max_images=args.max_images,
                max_new_tokens=args.max_new_tokens,
            )
            append_jsonl(
                output_path,
                {
                    "sample_id": sample_id,
                    "output": output,
                    "raw_text": raw_text,
                },
            )
            written += 1
        except Exception as exc:
            failed += 1
            append_jsonl(
                errors_path,
                {
                    "sample_id": sample_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

        if args.progress_every > 0 and submitted % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "submitted": submitted,
                        "written": written,
                        "failed": failed,
                        "output": str(output_path),
                    },
                    ensure_ascii=False,
                )
            )

    print(
        json.dumps(
            {
                "requests": str(requests_path),
                "output": str(output_path),
                "errors": str(errors_path),
                "submitted_this_run": submitted,
                "written_this_run": written,
                "failed_this_run": failed,
                "model_path": args.model_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
