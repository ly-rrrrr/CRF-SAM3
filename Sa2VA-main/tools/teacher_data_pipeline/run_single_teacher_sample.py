#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.teacher_data_pipeline.build_teacher_requests import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PROMPT,
    FIXED_CATEGORY_STAGE_NUM_FRAMES,
    DatasetResolver,
    build_response_schema,
    load_prompt,
    make_user_prompt,
    sample_category_stage_frames,
    sorted_frames,
)
from tools.teacher_data_pipeline.pack_teacher_labels import validate_output
from tools.teacher_data_pipeline.run_local_llama_teacher import (
    DEFAULT_MODEL_PATH,
    force_eager_experts,
    infer_one,
)

import torch
try:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor


DEFAULT_COMPARE_JSON = (
    "/mnt/csip-500/share/kongjiawei/Sa2VA-main/"
    "crf_sam3_attn/offline_cache_refytros_gemma4_teacher_prompt_failures/compare_failure_subset_iou.json"
)


def load_compare_expression(compare_json: Path, video_id: str, exp_id: str) -> Optional[str]:
    with compare_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("video_id")) == str(video_id) and str(row.get("exp_id")) == str(exp_id):
            expression = row.get("expression")
            if isinstance(expression, str) and expression.strip():
                return expression.strip()
    return None


def build_single_request_row(
    data_root: Path,
    dataset: str,
    split: str,
    video_id: str,
    exp_id: str,
    expression: str,
    prompt_file: Path,
) -> Dict[str, Any]:
    resolver = DatasetResolver(data_root)
    dataset_root, _, canonical_name = resolver.resolve(dataset, split)
    frame_dir = resolver.find_frame_dir(canonical_name, dataset_root, split, video_id)
    frames = sorted_frames(frame_dir)
    if not frames:
        raise FileNotFoundError(f"no frames found under: {frame_dir}")

    selected_frames = sample_category_stage_frames(frames, FIXED_CATEGORY_STAGE_NUM_FRAMES)
    system_prompt = load_prompt(prompt_file)
    sample_id = f"{canonical_name}:{split}:{video_id}:{exp_id}"

    return {
        "sample_id": sample_id,
        "source": {
            "dataset": canonical_name,
            "split": split,
            "video_id": str(video_id),
            "exp_id": str(exp_id),
            "expression": expression,
            "frame_dir": str(frame_dir),
            "frame_names": [p.name for p in selected_frames],
            "frame_paths": [str(p) for p in selected_frames],
        },
        "teacher_request": {
            "task": "masked_referent_category_extraction",
            "system_prompt": system_prompt,
            "user_prompt": make_user_prompt(expression),
            "response_schema": build_response_schema(),
            "images": [str(p) for p in selected_frames],
        },
    }


def load_model_and_processor(model_path: str, trust_remote_code: bool):
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    force_eager_experts(model)
    model.eval()
    return processor, model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the teacher pipeline for one sample: build request, invoke local Gemma, validate, and emit the packed label."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--expression", default=None, help="If omitted, can be looked up from --compare-json.")
    parser.add_argument("--dataset", default="refytvos")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--compare-json", default=DEFAULT_COMPARE_JSON)
    parser.add_argument("--max-images", type=int, default=0, help="0 means use all selected images.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", default=None, help="Optional path to save the full result bundle as JSON.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    prompt_file = Path(args.prompt_file)
    compare_json = Path(args.compare_json)

    expression = args.expression.strip() if isinstance(args.expression, str) and args.expression.strip() else None
    if expression is None and compare_json.exists():
        expression = load_compare_expression(compare_json, args.video_id, args.exp_id)
    if expression is None:
        raise SystemExit("expression is required, or it must be resolvable from --compare-json")

    request_row = build_single_request_row(
        data_root=data_root,
        dataset=args.dataset,
        split=args.split,
        video_id=str(args.video_id),
        exp_id=str(args.exp_id),
        expression=expression,
        prompt_file=prompt_file,
    )

    print(
        json.dumps(
            {
                "stage": "request_ready",
                "sample_id": request_row["sample_id"],
                "images": request_row["teacher_request"]["images"],
                "expression": expression,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    processor, model = load_model_and_processor(args.model_path, args.trust_remote_code)
    output, raw_text = infer_one(
        processor=processor,
        model=model,
        request_row=request_row,
        max_images=args.max_images,
        max_new_tokens=args.max_new_tokens,
    )

    normalized_label, reject_reason = validate_output(output)
    result_bundle: Dict[str, Any] = {
        "sample_id": request_row["sample_id"],
        "request_row": request_row,
        "response_row": {
            "sample_id": request_row["sample_id"],
            "output": output,
            "raw_text": raw_text,
        },
        "packed_row": None,
        "validation": {
            "accepted": normalized_label is not None,
            "reject_reason": reject_reason,
        },
    }
    if normalized_label is not None:
        result_bundle["packed_row"] = {
            "sample_id": request_row["sample_id"],
            "source": request_row["source"],
            "teacher_label": normalized_label,
        }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        result_bundle["output_json"] = str(output_path)

    print(json.dumps(result_bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
