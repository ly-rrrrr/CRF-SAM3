#!/usr/bin/env python3
"""
Rerun only missing/incomplete expressions for CRF-SAM3 outputs.

Usage:
  python rerun_missing_expressions.py --config configs/crf_sam3_config.yaml --split valid
"""

import argparse
import json
import os
import traceback
from datetime import datetime

import yaml

from crf_sam3_demo import CRFSAM3Pipeline
from run_crf_sam3_dataset import (
    _infer_video_hw,
    clear_cuda_cache,
    identify_gt_objects,
    load_gt_masks,
    save_masks,
    setup_logging,
)


def _build_qframe_config(cfg: dict, num_tube_frames: int):
    visual_tube_cfg = cfg.get("visual_tube", {})
    concept_use_qframe = cfg.get("concept_extraction", {}).get("use_qframe", False)
    sampling_strategy = visual_tube_cfg.get("sampling_strategy", "uniform")
    if sampling_strategy != "qframe" and not concept_use_qframe:
        return None, concept_use_qframe, sampling_strategy

    clip_model_path = cfg.get("model", {}).get("clip_model_path")
    if not clip_model_path:
        if sampling_strategy == "qframe":
            sampling_strategy = "uniform"
        concept_use_qframe = False
        return None, concept_use_qframe, sampling_strategy

    qframe_cfg = dict(visual_tube_cfg.get("qframe", {}) or {})
    qframe_cfg.setdefault("clip_model_path", clip_model_path)
    qframe_cfg.setdefault("device", cfg.get("pipeline", {}).get("device", "cuda"))
    qframe_cfg.setdefault("num_frames", num_tube_frames)
    return qframe_cfg, concept_use_qframe, sampling_strategy


def _list_missing_or_incomplete(meta_videos: dict, pred_base: str):
    """
    Return tuples (video_id, exp_id, reason, expected_frames, current_frames).
    reason in {"video_missing", "expr_missing", "frame_incomplete"}.
    """
    missing = []
    for video_id, vinfo in meta_videos.items():
        frames = sorted([str(x) for x in vinfo.get("frames", [])])
        expected_n = len(frames)
        video_dir = os.path.join(pred_base, video_id)

        if not os.path.isdir(video_dir):
            for exp_id in vinfo["expressions"].keys():
                missing.append((video_id, str(exp_id), "video_missing", expected_n, 0))
            continue

        for exp_id in vinfo["expressions"].keys():
            exp_dir = os.path.join(video_dir, str(exp_id))
            if not os.path.isdir(exp_dir):
                missing.append((video_id, str(exp_id), "expr_missing", expected_n, 0))
                continue

            cur_n = len([f for f in os.listdir(exp_dir) if f.endswith(".png")])
            if cur_n != expected_n:
                missing.append((video_id, str(exp_id), "frame_incomplete", expected_n, cur_n))

    return missing


def main():
    parser = argparse.ArgumentParser(description="Rerun missing expressions only.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--split", type=str, choices=["train", "valid", "valid_u"], default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Override output root.")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root.")
    parser.add_argument("--mllm_path", type=str, default=None, help="Override MLLM path.")
    parser.add_argument("--dry_run", action="store_true", help="Only list missing items, no inference.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N missing expressions.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.split:
        cfg["dataset"]["split"] = args.split
    if args.output_dir:
        cfg["output"]["output_dir"] = args.output_dir
    if args.data_root:
        cfg["dataset"]["data_root"] = args.data_root
    if args.mllm_path:
        cfg["model"]["mllm_path"] = args.mllm_path

    logger = setup_logging(cfg)
    split = cfg["dataset"]["split"]
    data_root = cfg["dataset"]["data_root"]
    output_dir = cfg["output"]["output_dir"]

    meta_path = os.path.join(data_root, split, "meta_expressions.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta_videos = meta["videos"]

    pred_base = os.path.join(output_dir, "Annotations")
    os.makedirs(pred_base, exist_ok=True)
    missing = _list_missing_or_incomplete(meta_videos, pred_base)

    print("=" * 72)
    print(f"[missing-check] split={split}")
    print(f"[missing-check] output={output_dir}")
    print(f"[missing-check] total_missing_or_incomplete={len(missing)}")
    print("=" * 72)
    if missing:
        for item in missing[:50]:
            print(
                f"{item[0]}/{item[1]} | reason={item[2]} | "
                f"expected={item[3]} current={item[4]}"
            )
        if len(missing) > 50:
            print(f"... ({len(missing) - 50} more)")

    if args.dry_run or not missing:
        return

    # Build pipeline config
    model_cfg = cfg["model"]
    pipe_cfg = cfg["pipeline"]
    disc_cfg = cfg.get("discrimination", {})
    visual_tube_cfg = cfg.get("visual_tube", {})

    num_tube_frames = pipe_cfg["num_tube_frames"]
    qframe_cfg, concept_use_qframe, sampling_strategy = _build_qframe_config(cfg, num_tube_frames)
    prompt_type = disc_cfg.get("prompt_type", "cot")
    prompt_templates = {
        "cot": "configs/discrimination_prompt.txt",
        "three_level": "configs/three_level_ttrl_cot.txt",
        "yesno": "configs/discrimination_prompt_yesno.txt",
    }
    prompt_template_path = prompt_templates.get(prompt_type, "configs/discrimination_prompt.txt")
    discriminator_backend = disc_cfg.get("backend", "mllm")
    reranker_cfg = disc_cfg.get("reranker", {})
    if discriminator_backend == "embedding":
        reranker_cfg = disc_cfg.get("embedding", {})

    print("[pipeline] Initializing CRFSAM3Pipeline ...")
    pipeline = CRFSAM3Pipeline(
        mllm_path=model_cfg["mllm_path"],
        sam3_checkpoint=model_cfg["sam3_checkpoint"],
        num_tube_frames=num_tube_frames,
        model_type=model_cfg.get("model_type", "qwen3"),
        prompt_type=prompt_type,
        prompt_template_path=prompt_template_path,
        visualization_style=visual_tube_cfg.get("visualization_style", "mask"),
        sampling_strategy=sampling_strategy,
        qframe_config=qframe_cfg,
        concept_use_qframe=concept_use_qframe,
        discriminator_backend=discriminator_backend,
        reranker_cfg=reranker_cfg,
    )

    gt_masks_root = cfg["dataset"].get("gt_masks_root", "")
    save_scores = bool(cfg.get("output", {}).get("save_scores", True))
    report = {
        "timestamp": datetime.now().isoformat(),
        "split": split,
        "output_dir": output_dir,
        "total_missing_before": len(missing),
        "processed": [],
        "failed": [],
    }
    scores_log = []

    to_process = missing if args.limit is None else missing[: args.limit]
    for idx, (video_id, exp_id, reason, expected_n, current_n) in enumerate(to_process, 1):
        expr = meta_videos[video_id]["expressions"][str(exp_id)]["exp"]
        video_frames = sorted([str(x) for x in meta_videos[video_id]["frames"]])
        video_path = os.path.join(data_root, split, "JPEGImages", video_id)
        video_hw = _infer_video_hw(video_path, video_frames)
        print(
            f"\n[{idx}/{len(to_process)}] {video_id}/{exp_id} "
            f"reason={reason} expected={expected_n} current={current_n}"
        )
        try:
            masks_dict, visual_tubes, details = pipeline.predict(
                video_path, expr, return_details=True
            )
            save_info = save_masks(
                masks_dict=masks_dict,
                output_dir=output_dir,
                video_id=video_id,
                exp_id=exp_id,
                expected_frames=video_frames,
                frame_hw=video_hw,
            )

            if save_scores:
                gt_masks_dict = load_gt_masks(gt_masks_root, video_id, exp_id)
                all_masks_dict = details.get("all_masks", {})
                gt_obj_ids, iou_scores = identify_gt_objects(
                    all_masks_dict, details, gt_masks_dict, iou_threshold=0.1
                )
                candidates_scores = []
                for obj_id in details["all_candidates"]:
                    score_info = details["scores"].get(obj_id, {})
                    candidates_scores.append(
                        {
                            "obj_id": obj_id,
                            "score": score_info.get("score", 0.0),
                            "prob_yes": score_info.get("prob_yes", 0.0),
                            "prob_partial": score_info.get("prob_partial", 0.0),
                            "prob_no": score_info.get("prob_no", 0.0),
                            "prediction": score_info.get("prediction", "N/A"),
                            "is_selected": obj_id in details["selected_candidates"],
                            "is_gt": obj_id in gt_obj_ids,
                            "iou": iou_scores.get(obj_id, 0.0),
                        }
                    )
                scores_log.append(
                    {
                        "video_id": video_id,
                        "exp_id": exp_id,
                        "expression": expr,
                        "category": details["category"],
                        "num_targets": details["num_targets"],
                        "num_candidates": len(details["all_candidates"]),
                        "num_selected": len(details["selected_candidates"]),
                        "num_gt_identified": len(gt_obj_ids),
                        "gt_match": len(gt_obj_ids) == details["num_targets"],
                        "candidates": candidates_scores,
                        "selected_obj_ids": details["selected_candidates"],
                        "gt_obj_ids": gt_obj_ids,
                    }
                )

            report["processed"].append(
                {
                    "video_id": video_id,
                    "exp_id": exp_id,
                    "reason": reason,
                    "saved_frames": save_info["saved_frames"],
                    "expected_frames": save_info["expected_frames"],
                    "missing_pred_filled": save_info["missing_pred_filled"],
                }
            )
        except Exception as e:
            logger.error(f"Failed {video_id}/{exp_id}: {e}")
            traceback.print_exc()
            report["failed"].append(
                {"video_id": video_id, "exp_id": exp_id, "reason": reason, "error": str(e)}
            )
        finally:
            clear_cuda_cache(logger)

    # Re-check
    missing_after = _list_missing_or_incomplete(meta_videos, pred_base)
    report["total_missing_after"] = len(missing_after)
    report["remaining_missing_sample"] = missing_after[:50]

    report_path = os.path.join(output_dir, "missing_rerun_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] report saved: {report_path}")
    print(
        f"[done] processed={len(report['processed'])}, "
        f"failed={len(report['failed'])}, remaining_missing={len(missing_after)}"
    )

    if save_scores and scores_log:
        scores_path = os.path.join(output_dir, "scores_rerun_missing.json")
        with open(scores_path, "w") as f:
            json.dump(scores_log, f, indent=2)
        print(f"[done] rerun scores saved: {scores_path}")


if __name__ == "__main__":
    main()
