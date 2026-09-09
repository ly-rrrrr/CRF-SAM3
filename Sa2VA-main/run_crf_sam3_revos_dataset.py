"""
Run CRF-SAM3 on RE-VOS dataset.

Key differences from MeViS/DAVIS runners:
1) RE-VOS metadata is a single json (e.g. meta_expressions_valid_.json).
2) video_id already contains sub-paths (e.g. OVIS/train/00e92ab4).
3) Official local evaluator (tools/eval/eval_revos.py) expects results.json
   with per-frame RLE masks.

This runner supports:
- results.json export (for eval_revos.py)
- optional submit-style PNG export
"""

import os
import argparse
import json
import yaml
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime
import numpy as np

from crf_sam3_demo import CRFSAM3Pipeline
from run_crf_sam3_dataset import (
    setup_logging,
    load_config,
    clear_cuda_cache,
    _infer_video_hw,
    save_masks,
)

try:
    from pycocotools import mask as cocomask
except Exception:
    cocomask = None


def _is_exp_complete(exp_dir: str, frame_names: List[str]) -> bool:
    if not os.path.isdir(exp_dir):
        return False
    for fr in frame_names:
        if not os.path.exists(os.path.join(exp_dir, f"{fr}.png")):
            return False
    return True


def _resolve_prompt_template(prompt_type: str) -> str:
    prompt_templates = {
        "cot": "configs/discrimination_prompt.txt",
        "three_level": "configs/three_level_ttrl_cot.txt",
        "yesno": "configs/discrimination_prompt_yesno.txt",
        "yesno_images": "configs/discrimination_prompt_yesno_train_aligned_20260806.txt",
    }
    return prompt_templates.get(prompt_type, "configs/discrimination_prompt.txt")


def _resolve_expression_file(data_root: str, expression_file: str) -> str:
    if os.path.isabs(expression_file):
        return expression_file
    return os.path.join(data_root, expression_file)


def _compose_binary_mask(frame_masks: Dict[int, np.ndarray], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if not frame_masks:
        return mask
    for _, obj_mask in frame_masks.items():
        mask = np.maximum(mask, (np.asarray(obj_mask) > 0).astype(np.uint8))
    return mask


def _encode_mask_to_rle(mask: np.ndarray) -> Dict:
    if cocomask is None:
        raise RuntimeError("pycocotools is required for RE-VOS RLE export.")
    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _build_revos_rles(
    masks_dict: Dict[int, Dict[int, np.ndarray]],
    frame_names: List[str],
    h: int,
    w: int,
) -> List[Dict]:
    """
    Convert pipeline masks_dict into RE-VOS per-frame binary RLE list.
    """
    rles: List[Dict] = []
    for idx, frame_name in enumerate(frame_names):
        key = None
        candidate_keys = [idx]
        if str(frame_name).isdigit():
            candidate_keys.insert(0, int(frame_name))
        for k in candidate_keys:
            if k in masks_dict:
                key = k
                break
        frame_masks = masks_dict.get(key, {}) if key is not None else {}
        binary = _compose_binary_mask(frame_masks, h, w)
        rles.append(_encode_mask_to_rle(binary))
    return rles


def _has_complete_result(revos_results: Dict, video_id: str, exp_id: str, n_frames: int) -> bool:
    video_res = revos_results.get(video_id, {})
    exp_res = video_res.get(str(exp_id))
    if exp_res is None:
        return False
    pred_masks = exp_res.get("prediction_masks", [])
    return isinstance(pred_masks, list) and len(pred_masks) == n_frames


def main():
    parser = argparse.ArgumentParser(
        description="Run CRF-SAM3 on RE-VOS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_crf_sam3_revos_dataset.py --config configs/crf_sam3_config.yaml \\
    --data_root /path/to/revos \\
    --expression_file meta_expressions_valid_.json \\
    --output_dir crf_sam3_revos_valid_ft
""",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")

    # common overrides
    parser.add_argument("--mllm_path", type=str, help="Override MLLM path")
    parser.add_argument("--base_mllm_path", type=str, help="Override shared base MLLM path")
    parser.add_argument("--mllm_adapter_path", type=str, help="Override MLLM adapter path")
    parser.add_argument("--sam3_checkpoint", type=str, help="Override SAM3 checkpoint path")
    parser.add_argument("--data_root", type=str, help="Override RE-VOS data root")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--backend", type=str, choices=["mllm", "reranker", "embedding"], help="Override backend")
    parser.add_argument("--max_videos", type=int, help="Process at most N videos")
    parser.add_argument("--max_expressions", type=int, help="Process at most N expressions per video")
    parser.add_argument("--num_tube_frames", type=int, help="Override number of tube frames")
    parser.add_argument("--skip_existing", action="store_true", help="Skip expressions with complete outputs")

    # RE-VOS specific
    parser.add_argument(
        "--expression_file",
        type=str,
        default="meta_expressions_valid_.json",
        help="RE-VOS expression json (relative to data_root or absolute)",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root for video folders; default=data_root",
    )
    parser.add_argument(
        "--save_submit_png",
        action="store_true",
        help="Also save submit-style PNG masks under output_dir/Annotations",
    )
    parser.add_argument(
        "--revos_json_name",
        type=str,
        default="results.json",
        help="Filename of exported RE-VOS json predictions",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides.
    if args.mllm_path:
        config["model"]["mllm_path"] = args.mllm_path
    if args.base_mllm_path:
        config["model"]["base_mllm_path"] = args.base_mllm_path
    if args.mllm_adapter_path:
        config["model"]["mllm_adapter_path"] = args.mllm_adapter_path
    if args.sam3_checkpoint:
        config["model"]["sam3_checkpoint"] = args.sam3_checkpoint
    if args.data_root:
        config.setdefault("dataset", {})["data_root"] = args.data_root
    if args.output_dir:
        config.setdefault("output", {})["output_dir"] = args.output_dir
    if args.backend:
        config.setdefault("discrimination", {})["backend"] = args.backend
    if args.max_videos is not None:
        config.setdefault("dataset", {})["max_videos"] = args.max_videos
    if args.max_expressions is not None:
        config.setdefault("dataset", {})["max_expressions"] = args.max_expressions
    if args.num_tube_frames is not None:
        config.setdefault("pipeline", {})["num_tube_frames"] = args.num_tube_frames
    if args.skip_existing:
        config.setdefault("dataset", {})["skip_existing"] = True

    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("CRF-SAM3 RE-VOS Runner")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if cocomask is None:
        raise RuntimeError("pycocotools is required. Please install it in your environment.")

    model_cfg = config["model"]
    model_type = model_cfg.get("model_type", "qwen3")
    mllm_path = model_cfg.get("mllm_path")
    base_mllm_path = model_cfg.get("base_mllm_path")
    mllm_adapter_path = model_cfg.get("mllm_adapter_path")
    sam3_checkpoint = model_cfg["sam3_checkpoint"]

    data_cfg = config.get("dataset", {})
    data_root = data_cfg.get("data_root", "")
    image_root = args.image_root if args.image_root else data_root
    max_videos = data_cfg.get("max_videos", None)
    max_expressions = data_cfg.get("max_expressions", None)
    skip_existing = bool(data_cfg.get("skip_existing", False))

    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("output_dir", "crf_sam3_revos_outputs")
    save_visual_tubes = bool(output_cfg.get("save_visual_tubes", False))
    save_scores = bool(output_cfg.get("save_scores", True))

    pipeline_cfg = config.get("pipeline", {})
    num_tube_frames = int(pipeline_cfg.get("num_tube_frames", 8))

    discr_cfg = config.get("discrimination", {})
    backend = discr_cfg.get("backend", "mllm")
    prompt_type = discr_cfg.get("prompt_type", "yesno")
    prompt_template_path = _resolve_prompt_template(prompt_type)

    visual_tube_cfg = config.get("visual_tube", {})
    visualization_style = visual_tube_cfg.get("visualization_style", "mask")
    sampling_strategy = visual_tube_cfg.get("sampling_strategy", "uniform")
    visual_tube_temp_dir = visual_tube_cfg.get("temp_dir", os.path.join(output_dir, "visual_tubes_temp"))

    concept_cfg = config.get("concept_extraction", {})
    sam3_recall_config = config.get("sam3_recall", {}) or {}
    concept_use_qframe = bool(concept_cfg.get("use_qframe", False))

    qframe_config = None
    if sampling_strategy == "qframe" or concept_use_qframe:
        clip_model_path = config["model"].get("clip_model_path")
        if clip_model_path:
            qframe_config = visual_tube_cfg.get("qframe", {}) or {}
            qframe_config.setdefault("clip_model_path", clip_model_path)
            qframe_config.setdefault("device", pipeline_cfg.get("device", "cuda"))
            qframe_config.setdefault("num_frames", num_tube_frames)
        else:
            concept_use_qframe = False
            if sampling_strategy == "qframe":
                sampling_strategy = "uniform"

    expr_path = _resolve_expression_file(data_root, args.expression_file)
    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"Expression file not found: {expr_path}")
    with open(expr_path, "r") as f:
        meta = json.load(f)
    videos = meta["videos"]

    logger.info("\nConfiguration:")
    logger.info(f"  Model Type: {model_type}")
    if base_mllm_path:
        logger.info(f"  Base MLLM: {base_mllm_path}")
        logger.info(f"  MLLM Adapter: {mllm_adapter_path if mllm_adapter_path else '(disabled)'}")
    else:
        logger.info(f"  MLLM: {mllm_path}")
    logger.info(f"  SAM3: {sam3_checkpoint}")
    logger.info(f"  Prompt Type: {prompt_type}")
    logger.info(f"  Prompt Template: {prompt_template_path}")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  Visual Tube Frames: {num_tube_frames}")
    logger.info(f"  Sampling Strategy: {sampling_strategy}")
    logger.info(f"  Visualization Style: {visualization_style}")
    logger.info(f"  Data Root: {data_root}")
    logger.info(f"  Image Root: {image_root}")
    logger.info(f"  Expression File: {expr_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Save Submit PNG: {bool(args.save_submit_png)}")
    logger.info(f"  RE-VOS JSON: {args.revos_json_name}")
    logger.info(f"  Max Videos: {max_videos if max_videos is not None else 'All'}")
    logger.info(f"  Max Expressions: {max_expressions if max_expressions is not None else 'All'}")
    logger.info(f"  Skip Existing: {skip_existing}")
    logger.info(f"  Visual Tube Temp: {visual_tube_temp_dir}")
    logger.info(f"  Save Scores: {save_scores}")
    logger.info(f"Found {len(videos)} videos in expression file")

    logger.info("\nInitializing pipeline...")
    reranker_cfg = discr_cfg.get("reranker", {})
    embedding_cfg = discr_cfg.get("embedding", {})
    if backend == "embedding":
        reranker_cfg = embedding_cfg

    pipeline = CRFSAM3Pipeline(
        mllm_path=mllm_path,
        base_mllm_path=base_mllm_path,
        mllm_adapter_path=mllm_adapter_path,
        visual_tube_temp_dir=visual_tube_temp_dir,
        sam3_checkpoint=sam3_checkpoint,
        num_tube_frames=num_tube_frames,
        model_type=model_type,
        prompt_type=prompt_type,
        prompt_template_path=prompt_template_path,
        visualization_style=visualization_style,
        sampling_strategy=sampling_strategy,
        qframe_config=qframe_config,
        concept_use_qframe=concept_use_qframe,
        concept_config=concept_cfg,
        discriminator_backend=backend,
        reranker_cfg=reranker_cfg,
        sam3_recall_cfg=sam3_recall_config,
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    revos_json_path = os.path.join(output_dir, args.revos_json_name)
    revos_results: Dict[str, Dict[str, Dict]] = {}
    if skip_existing and os.path.exists(revos_json_path):
        try:
            with open(revos_json_path, "r") as f:
                revos_results = json.load(f)
            logger.info(f"Loaded existing RE-VOS predictions: {revos_json_path}")
        except Exception as e:
            logger.warning(f"Failed to load existing RE-VOS json ({e}), start fresh.")
            revos_results = {}

    processed_video_count = 0
    processed_exp_count = 0
    skipped_exp_count = 0
    total_missing_filled = 0
    frame_coverage_report: List[Dict] = []
    scores_log: List[Dict] = []
    scores_path = os.path.join(output_dir, "scores.json")
    if save_scores and skip_existing and os.path.exists(scores_path):
        try:
            with open(scores_path, "r") as f:
                existing_scores = json.load(f)
            if isinstance(existing_scores, list):
                scores_log = existing_scores
                logger.info(f"Loaded existing score records: {len(scores_log)}")
        except Exception as e:
            logger.warning(f"Failed to load existing scores ({e}), rebuilding new records only.")
    score_keys = {
        (str(record.get("video_id")), str(record.get("exp_id")))
        for record in scores_log
        if isinstance(record, dict)
    }

    video_iter = videos.items()
    video_pbar = tqdm(video_iter, desc="Videos", position=0, disable=False)

    for video_id, video_info in video_pbar:
        if max_videos is not None and processed_video_count >= max_videos:
            break

        video_path = os.path.join(image_root, video_id)
        if not os.path.exists(video_path):
            logger.error(f"Missing video frames directory: {video_path}")
            continue

        expressions = video_info.get("expressions", {})
        video_frames = [str(x) for x in video_info.get("frames", [])]
        if len(video_frames) == 0:
            logger.warning(f"No frames in meta for video {video_id}, skipping")
            continue

        try:
            video_hw = _infer_video_hw(video_path, video_frames)
        except Exception as e:
            logger.error(f"Failed to infer frame size for video {video_id}: {e}")
            continue

        exp_iter = expressions.items()
        exp_pbar = tqdm(exp_iter, desc=f"{video_id}", position=1, leave=False, disable=False)

        exp_count = 0
        for exp_id, exp_info in exp_pbar:
            if max_expressions is not None and exp_count >= max_expressions:
                break

            exp_id_str = str(exp_id)
            referring_expression = exp_info.get("exp", "")

            if skip_existing and _has_complete_result(revos_results, video_id, exp_id_str, len(video_frames)):
                if args.save_submit_png:
                    exp_out_dir = os.path.join(output_dir, "Annotations", video_id, exp_id_str)
                    if not _is_exp_complete(exp_out_dir, video_frames):
                        # keep running if png is requested but incomplete
                        pass
                    else:
                        skipped_exp_count += 1
                        exp_count += 1
                        continue
                else:
                    skipped_exp_count += 1
                    exp_count += 1
                    continue

            try:
                masks_dict, visual_tubes, discrimination_details = pipeline.predict(
                    video_path,
                    referring_expression,
                    return_details=True,
                )

                # Optional PNG export
                if args.save_submit_png:
                    save_info = save_masks(
                        masks_dict=masks_dict,
                        output_dir=output_dir,
                        video_id=video_id,
                        exp_id=exp_id_str,
                        expected_frames=video_frames,
                        frame_hw=video_hw,
                    )
                    total_missing_filled += int(save_info["missing_pred_filled"])
                    frame_coverage_report.append(
                        {
                            "video_id": video_id,
                            "exp_id": exp_id_str,
                            "expected_frames": int(save_info["expected_frames"]),
                            "saved_frames": int(save_info["saved_frames"]),
                            "missing_pred_filled": int(save_info["missing_pred_filled"]),
                        }
                    )

                # RE-VOS RLE export
                pred_rles = _build_revos_rles(
                    masks_dict=masks_dict,
                    frame_names=video_frames,
                    h=int(video_hw[0]),
                    w=int(video_hw[1]),
                )
                if video_id not in revos_results:
                    revos_results[video_id] = {}
                revos_results[video_id][exp_id_str] = {
                    "video_id": video_id,
                    "exp_id": exp_id_str,
                    "exp": referring_expression,
                    "frames": video_frames,
                    "prediction_masks": pred_rles,
                }

                if save_scores:
                    score_record = {
                        "video_id": video_id,
                        "exp_id": exp_id_str,
                        "expression": referring_expression,
                        "category": discrimination_details.get("category"),
                        "primary_category": discrimination_details.get("primary_category"),
                        "used_category_query": discrimination_details.get("used_category_query"),
                        "category_override_used": discrimination_details.get("category_override_used", False),
                        "selected_candidates": discrimination_details.get("selected_candidates", []),
                        "all_candidates": discrimination_details.get("all_candidates", []),
                        "scores": discrimination_details.get("scores", {}),
                    }
                    score_key = (str(video_id), exp_id_str)
                    if score_key not in score_keys:
                        scores_log.append(score_record)
                        score_keys.add(score_key)
                    detail_dir = os.path.join(output_dir, "expression_details", video_id)
                    os.makedirs(detail_dir, exist_ok=True)
                    with open(os.path.join(detail_dir, f"{exp_id_str}.json"), "w") as f:
                        json.dump(score_record, f, indent=2)

                if save_visual_tubes:
                    visual_tubes_dir = os.path.join(output_dir, "visual_tubes", video_id, exp_id_str)
                    pipeline.save_visual_tubes(visual_tubes, visual_tubes_dir)

                processed_exp_count += 1
                exp_count += 1

            except torch.OutOfMemoryError:
                logger.error(f"CUDA OOM at {video_id}/{exp_id_str}. Abort to keep state consistent.")
                raise
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at {video_id}/{exp_id_str}. Abort to keep state consistent.")
                    raise
                logger.error(f"Runtime error at {video_id}/{exp_id_str}: {e}")
            except Exception as e:
                logger.error(f"Error at {video_id}/{exp_id_str}: {e}")
            finally:
                clear_cuda_cache(logger)

        processed_video_count += 1

        # Save progressively per video to avoid losing long runs.
        with open(revos_json_path, "w") as f:
            json.dump(revos_results, f)
        if save_scores and scores_log:
            with open(scores_path, "w") as f:
                json.dump(scores_log, f, indent=2)

    with open(revos_json_path, "w") as f:
        json.dump(revos_results, f)

    saved_scores_path = None
    if save_scores and scores_log:
        with open(scores_path, "w") as f:
            json.dump(scores_log, f, indent=2)
        saved_scores_path = scores_path

    if frame_coverage_report:
        with open(os.path.join(output_dir, "frame_coverage.json"), "w") as f:
            json.dump(frame_coverage_report, f, indent=2)

    summary = {
        "videos_total_in_meta": len(videos),
        "videos_processed": processed_video_count,
        "expressions_processed": processed_exp_count,
        "expressions_skipped_existing": skipped_exp_count,
        "missing_frames_filled_as_background": total_missing_filled,
        "output_dir": output_dir,
        "revos_json_path": revos_json_path,
        "save_submit_png": bool(args.save_submit_png),
        "scores_path": saved_scores_path,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Done")
    logger.info(json.dumps(summary, indent=2))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
