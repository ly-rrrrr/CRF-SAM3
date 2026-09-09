#!/usr/bin/env python3
"""Apply a query-grounding model to ReVOS metadata.

It uses RGB frames only and writes the category override cache consumed by
CRFSAM3Pipeline.
"""

from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from predict_query_grounding_refytros_meta import load_query_model, parse_category, predict_one, sample_frames


def write_payload(output_json: str, records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "summary": {
            "total": len(records),
            "valid": sum(bool(record.get("pred_category")) for record in records),
            "data_root": args.data_root,
            "expression_file": args.expression_file,
            "query_module_dir": args.query_module_dir,
            "source_training_dataset": args.source_training_dataset,
            "target_dataset": "ReVOS valid",
        },
        "records": records,
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_frame_path(frame_path: Path, cache_root: Path | None, max_image_side: int) -> str:
    if max_image_side <= 0:
        return str(frame_path)
    with Image.open(frame_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(max_image_side) / float(max(width, height)))
        if scale >= 1.0:
            return str(frame_path)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        if cache_root is None:
            cache_root = Path("category_grounding_mvp/outputs/revos_query_grounding_frame_cache")
        rel = frame_path.as_posix().replace("/", "__")
        out_path = cache_root / f"side{max_image_side}" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, quality=95)
        return str(out_path)


def load_revos_records(
    data_root: str,
    expression_file: str,
    frames_per_sample: int,
    max_image_side: int = 448,
    frame_cache_root: str | None = None,
) -> List[Dict[str, Any]]:
    root = Path(data_root)
    meta_path = Path(expression_file)
    if not meta_path.is_absolute():
        meta_path = root / meta_path
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for video_id in sorted(payload["videos"]):
        video = payload["videos"][video_id]
        frames = [str(x) for x in video.get("frames", [])]
        picked = sample_frames(frames, frames_per_sample)
        raw_frame_paths = [root / video_id / f"{frame}.jpg" for frame in picked]
        missing = [str(path) for path in raw_frame_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing sampled ReVOS frame: {missing[0]}")
        cache_root = Path(frame_cache_root) if frame_cache_root else None
        frame_paths = [prepare_frame_path(path, cache_root, max_image_side) for path in raw_frame_paths]
        expressions = video.get("expressions", {})
        exp_ids = sorted(expressions, key=lambda x: int(x) if str(x).isdigit() else str(x))
        for exp_id in exp_ids:
            exp_info = expressions[exp_id]
            records.append(
                {
                    "sample_id": f"revos:{video_id}:{exp_id}",
                    "video_id": video_id,
                    "exp_id": str(exp_id),
                    "expression": str(exp_info.get("exp", "")),
                    "type_id": exp_info.get("type_id"),
                    "frames": picked,
                    "frame_paths": frame_paths,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--query-module-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expression-file", default="meta_expressions_valid_.json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--frames-per-sample", type=int, default=3)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--frame-cache-root", default="category_grounding_mvp/outputs/revos_query_grounding_frame_cache")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device-map", default="auto", choices=["", "auto"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source-training-dataset",
        default="query-grounding checkpoint",
        help="Provenance label written to the output metadata.",
    )
    args = parser.parse_args()

    records = load_revos_records(
        args.data_root,
        args.expression_file,
        args.frames_per_sample,
        max_image_side=args.max_image_side,
        frame_cache_root=args.frame_cache_root,
    )
    if args.max_samples > 0:
        records = records[: args.max_samples]

    output_records = []
    done_ids = set()
    output_path = Path(args.output_json)
    if args.resume and output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_records = list(existing.get("records", []))
        recovered_count = 0
        unresolved_count = 0
        for existing_record in existing_records:
            pred = str(existing_record.get("pred_category") or "").strip()
            if not pred:
                recovered = parse_category(str(existing_record.get("raw_response") or ""))
                if recovered:
                    existing_record = dict(existing_record)
                    existing_record.update(pred_category=recovered, status="recovered_plain_text")
                    pred = recovered
                    recovered_count += 1
            if pred:
                output_records.append(existing_record)
            else:
                unresolved_count += 1
        done_ids = {str(record.get("sample_id")) for record in output_records if record.get("sample_id")}
        print(f"[resume] valid={len(output_records)} recovered={recovered_count} unresolved={unresolved_count} from {output_path}", flush=True)

    pending_count = sum(str(record["sample_id"]) not in done_ids for record in records)
    if pending_count:
        processor, query_model = load_query_model(args)
    else:
        processor, query_model = None, None
    print(f"[resume] pending_inference={pending_count}", flush=True)

    for index, record in enumerate(records, 1):
        if str(record["sample_id"]) in done_ids:
            continue
        pred, raw, stats = predict_one(processor, query_model, record, args.max_new_tokens)
        output_record = dict(record)
        output_record.update(
            {
                "pred_category": pred,
                "raw_response": raw,
                "query_stats": stats,
                "status": "ok" if pred else "error",
            }
        )
        output_records.append(output_record)
        print(
            f"[{index}/{len(records)}] {record['video_id']}/{record['exp_id']} "
            f"type={record.get('type_id')} pred={pred or '<empty>'}",
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.save_every > 0 and len(output_records) % args.save_every == 0:
            write_payload(args.output_json, output_records, args)
            print(f"[checkpoint] saved={len(output_records)} -> {args.output_json}", flush=True)

    write_payload(args.output_json, output_records, args)
    summary = {
        "total": len(output_records),
        "valid": sum(bool(record.get("pred_category")) for record in output_records),
        "data_root": args.data_root,
        "expression_file": args.expression_file,
        "query_module_dir": args.query_module_dir,
        "source_training_dataset": args.source_training_dataset,
        "target_dataset": "ReVOS valid",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[output] {output_path}", flush=True)


if __name__ == "__main__":
    main()
