#!/usr/bin/env python3
"""Generate offline ReVOS category supervision with an OpenAI-compatible VLM API.

The red box is derived from the annotated target mask. This script is therefore
for constructing training supervision only; it must not be used in a valid/test
inference path, where the target location is unavailable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as cocomask

ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = ROOT / ".env"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "revos_train_qwen37plus_bbox_categories_3views.json"
DEFAULT_DEBUG_DIR = ROOT / "results" / "revos_train_qwen37plus_bbox_debug_3views"

DEFAULT_SYSTEM_PROMPT = """You identify the visual category of a localized referent across temporally ordered video frames.
You receive the complete referring expression and up to three original RGB frames in chronological order. A red bounding box marks the same target referent in every frame.
Interpret the expression and all localized visual evidence jointly. Return the ordinary English category noun or short noun phrase that best names the boxed object. The category should describe the boxed object itself, not a surrounding person, action, relation, or scene. Output only the requested JSON object."""

CATEGORY_SCHEMA = {
    "name": "localized_target_category",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "The ordinary English category noun or short noun phrase for the boxed target.",
            }
        },
        "required": ["category"],
        "additionalProperties": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expression-file", default="meta_expressions_train_.json")
    parser.add_argument("--mask-dict", default=None)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--debug-dir", default=None, help="Optional debug-image directory. Omit to save only the final JSON.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--api-base", default="https://api.inferera.com/v1")
    parser.add_argument("--api-key-env", default="INFERERA_API_KEY")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--max-image-edge", type=int, default=1280)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--json-object", action="store_true", help="Use JSON-object mode when the gateway lacks json_schema support.")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip(chr(34)).strip(chr(39))
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_category(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text.strip(" .,!?:;\"'")


def first_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
    return {}


def union_target_mask(mask_dict: Dict[str, Any], anno_ids: Iterable[Any], frame_index: int) -> np.ndarray | None:
    merged: np.ndarray | None = None
    for anno_id in anno_ids:
        frames = mask_dict.get(str(anno_id), [])
        if frame_index >= len(frames) or not frames[frame_index]:
            continue
        decoded = cocomask.decode(frames[frame_index])
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        current = decoded.astype(bool)
        merged = current if merged is None else (merged | current)
    return merged


def select_temporal_target_frames(
    video: Dict[str, Any],
    expression: Dict[str, Any],
    mask_dict: Dict[str, Any],
) -> List[Tuple[str, int, str, np.ndarray]]:
    visible: List[Tuple[int, str, np.ndarray]] = []
    for frame_index, frame_name in enumerate(video.get("frames", [])):
        mask = union_target_mask(mask_dict, expression.get("anno_id", []), frame_index)
        if mask is not None:
            visible.append((frame_index, str(frame_name), mask))
    if not visible:
        raise ValueError("Target has no visible annotation mask in this video.")

    # GT masks organize temporal visual evidence for offline supervision only.
    # Category inference remains the VLM task.
    def best_in(candidates: List[Tuple[int, str, np.ndarray]], excluded: set[int]) -> Tuple[int, str, np.ndarray] | None:
        candidates = [item for item in candidates if item[0] not in excluded]
        return max(candidates, key=lambda item: (int(item[2].sum()), -item[0]), default=None)

    count = len(visible)
    third = max(1, (count + 2) // 3)
    early_pool = visible[:third]
    late_pool = visible[-third:]

    selected: List[Tuple[str, int, str, np.ndarray]] = []
    excluded: set[int] = set()
    canonical = best_in(visible, excluded)
    if canonical is not None:
        selected.append(("canonical", *canonical))
        excluded.add(canonical[0])
    early = best_in(early_pool, excluded)
    if early is not None:
        selected.append(("early", *early))
        excluded.add(early[0])
    late = best_in(late_pool, excluded)
    if late is not None:
        selected.append(("late", *late))
        excluded.add(late[0])

    # A short visible interval may collapse temporal slots. Keep distinct real
    # observations rather than fabricate duplicate views.
    for candidate in sorted(visible, key=lambda item: (int(item[2].sum()), -item[0]), reverse=True):
        if len(selected) >= 3:
            break
        if candidate[0] not in excluded:
            selected.append(("additional", *candidate))
            excluded.add(candidate[0])

    selected = sorted(selected, key=lambda item: item[1])
    return [(f"temporal_{index}", frame_index, frame_name, mask)
            for index, (_, frame_index, frame_name, mask) in enumerate(selected, 1)]


def target_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty target mask cannot produce a bbox.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def annotate_original(image_path: Path, bbox: Tuple[int, int, int, int], max_edge: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(232, 54, 56), width=max(3, min(image.size) // 180))
    label = "bbox"
    font = ImageFont.load_default()
    label_box = draw.textbbox((0, 0), label, font=font)
    pad = 7
    width = label_box[2] - label_box[0] + 2 * pad
    height = label_box[3] - label_box[1] + 2 * pad
    # Anchor the label to the target box: above its top-right corner when space permits,
    # otherwise immediately inside the same corner.
    label_x = min(max(0, x2 - width), max(0, image.width - width))
    label_y = y1 - height - 3
    if label_y < 0:
        label_y = min(max(0, y1 + 3), max(0, image.height - height))
    label_xy = (label_x, label_y)
    draw.rounded_rectangle((label_xy[0], label_xy[1], label_xy[0] + width, label_xy[1] + height), radius=4, fill=(232, 54, 56))
    draw.text((label_xy[0] + pad, label_xy[1] + pad - 1), label, fill=(255, 255, 255), font=font)
    if max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def invoke_api(
    api_base: str,
    api_key: str,
    model: str,
    expression: str,
    image_urls: List[Tuple[str, str]],
    use_json_object: bool,
    timeout: int,
    retries: int,
) -> Tuple[str, Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Referring expression:\n{expression}"},
        {
            "type": "text",
            "text": "The following images show the same boxed referent in temporal order. Use all available views to infer one category.",
        },
    ]
    for role, image_url in image_urls:
        user_content.extend(
            [
                {"type": "text", "text": f"Temporal view: {role}."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": 96,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": (
            {"type": "json_object"}
            if use_json_object
            else {"type": "json_schema", "json_schema": CATEGORY_SCHEMA}
        ),
    }
    body = json.dumps(payload).encode("utf-8")
    endpoint = api_base.rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
            content = raw_payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            parsed = first_json(str(content))
            category = normalize_category(parsed.get("category"))
            if not category:
                raise ValueError(f"Schema response lacks a usable category: {content!r}")
            return category, {}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
            last_error = error
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"Inferera request failed after {retries + 1} attempts: {last_error}")


def load_existing(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return payload, {str(record.get("sample_id")): record for record in records if isinstance(record, dict)}


def is_retryable_api_error(record: Dict[str, Any]) -> bool:
    if record.get("status") != "error":
        return False
    if record.get("error_kind") == "api_request":
        return True
    return str(record.get("error", "")).startswith("Inferera request failed after ")


def save_output(path: Path, records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "summary": {
            "total": len(records),
            "ok": sum(record.get("status") == "ok" for record in records),
            "errors": sum(record.get("status") == "error" for record in records),
            "model": args.model,
            "api_base": args.api_base,
            "input": "up to three temporally ordered original RGB frames with GT-derived target bboxes",
            "usage": "offline supervision only; never use this bbox-dependent output for ReVOS validation inference",
        },
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Export {args.api_key_env} before running this script.")

    root = Path(args.data_root)
    expression_file = Path(args.expression_file)
    if not expression_file.is_absolute():
        expression_file = root / expression_file
    mask_path = Path(args.mask_dict) if args.mask_dict else root / "mask_dict.json"
    meta = json.loads(expression_file.read_text(encoding="utf-8"))
    masks = json.loads(mask_path.read_text(encoding="utf-8"))

    output_path = Path(args.output_json)
    _, existing = load_existing(output_path)
    records_by_id: Dict[str, Dict[str, Any]] = dict(existing)
    retryable_errors = {
        sample_id for sample_id, record in existing.items() if is_retryable_api_error(record)
    }
    completed = set(existing) - retryable_errors
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    pending: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    for video_id in sorted(meta["videos"]):
        video = meta["videos"][video_id]
        for exp_id, expression in sorted(video.get("expressions", {}).items(), key=lambda item: str(item[0])):
            sample_id = f"revos:{video_id}:{exp_id}"
            if sample_id not in completed:
                pending.append((sample_id, video_id, video, {"exp_id": str(exp_id), **expression}))
    if args.max_samples > 0:
        pending = pending[: args.max_samples]

    print(
        f"[api-category] pending={len(pending)} resumed={len(completed)} "
        f"retry_api_errors={len(retryable_errors)} model={args.model}",
        flush=True,
    )
    print(f"[api-category] system prompt:\n{DEFAULT_SYSTEM_PROMPT}", flush=True)
    print("[api-category] user content: text=Referring expression: <original expression>; images=up to 3 annotated RGB frames in chronological order", flush=True)
    for index, (sample_id, video_id, video, expression) in enumerate(pending, 1):
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "video_id": video_id,
            "exp_id": expression["exp_id"],
            "expression": expression.get("exp", ""),
            "anno_id": expression.get("anno_id", []),
            "type_id": expression.get("type_id"),
        }
        exp_id = str(expression["exp_id"])
        expression_text = record["expression"]
        try:
            temporal_views = select_temporal_target_frames(video, expression, masks)
            encoded_views: List[Tuple[str, str]] = []
            view_records: List[Dict[str, Any]] = []
            for role, frame_index, frame_name, mask in temporal_views:
                bbox = target_bbox(mask)
                image_path = root / video_id / f"{frame_name}.jpg"
                image = annotate_original(image_path, bbox, args.max_image_edge)
                view_record: Dict[str, Any] = {
                    "role": role,
                    "frame_index": frame_index,
                    "frame": frame_name,
                    "bbox_xyxy": list(bbox),
                    "mask_area": int(mask.sum()),
                }
                if debug_dir:
                    debug_path = debug_dir / video_id / f"{exp_id}_{role}_{frame_index:05d}.jpg"
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(debug_path, quality=92)
                    view_record["debug_image"] = str(debug_path)
                view_records.append(view_record)
                encoded_views.append((role, image_data_url(image)))

            category, _api_data = invoke_api(
                args.api_base,
                api_key,
                args.model,
                record["expression"],
                encoded_views,
                args.json_object,
                args.request_timeout,
                args.retries,
            )
            record.update(
                {
                    "pred_category": category,
                    "temporal_views": view_records,
                    "status": "ok",
                }
            )
            view_summary = ", ".join("{}:{}".format(view["role"], view["frame"]) for view in view_records)
            print(
                f"[{index}/{len(pending)}] {video_id}/{exp_id} views={view_summary}\n"
                f"  expression: {expression_text}\n"
                f"  category: {category}",
                flush=True,
            )
        except Exception as error:
            error_kind = "api_request" if str(error).startswith("Inferera request failed after ") else "data"
            record.update({"status": "error", "error_kind": error_kind, "error": str(error)})
            print(
                f"[{index}/{len(pending)}] {video_id}/{exp_id} ERROR {error}\n"
                f"  expression: {expression_text}",
                flush=True,
            )
        records_by_id[sample_id] = record
        if index % max(args.save_every, 1) == 0:
            save_output(output_path, list(records_by_id.values()), args)

    save_output(output_path, list(records_by_id.values()), args)
    print(f"[output] {output_path}", flush=True)


if __name__ == "__main__":
    main()
