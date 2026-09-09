"""
Run CRF-SAM3 on Ref-Youtube-ROS dataset (challenge-style metadata), with
optional per-video category-level SAM3 recall cache.

Expected default structure:
  <data_root>/<split>/JPEGImages/<video_id>/<frame>.jpg
  <data_root>/<split>/meta_expressions_challenge.json

Output (submit-style PNG masks):
  <output_dir>/Annotations/<video_id>/<exp_id>/<frame>.png
"""

import os
import argparse
import json
import yaml
import torch
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime

from crf_sam3_demo_cached import CRFSAM3Pipeline
from run_crf_sam3_dataset import (
    setup_logging,
    load_config,
    clear_cuda_cache,
    _infer_video_hw,
    save_masks,
)


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
    }
    return prompt_templates.get(prompt_type, "configs/discrimination_prompt.txt")


def _resolve_meta_path(data_root: str, split: str, meta_file: Optional[str]) -> str:
    if meta_file:
        if os.path.isabs(meta_file):
            return meta_file
        return os.path.join(data_root, split, meta_file)

    # Preferred for Ref-Youtube-ROS valid challenge set.
    candidates = [
        os.path.join(data_root, split, "meta_expressions_challenge.json"),
        os.path.join(data_root, split, "meta_expressions.json"),
        os.path.join(data_root, "meta_expressions", split, "meta_expressions.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Cannot locate metadata file. Tried:\n  - "
        + "\n  - ".join(candidates)
    )


class VideoCategoryRecallCache:
    """
    Wrap pipeline.sam3_recall.recall_all_instances with an in-memory cache.

    Cache key: (absolute_video_path, normalized_category).
    The cache is reset when moving to a new video to avoid extra memory growth.
    """

    def __init__(self, pipeline: CRFSAM3Pipeline, logger=None):
        self.pipeline = pipeline
        self.logger = logger
        self._orig_fn = None
        self._cache: Dict = {}
        self.current_video_path: Optional[str] = None
        self.video_hits = 0
        self.video_misses = 0
        self.total_hits = 0
        self.total_misses = 0

    @staticmethod
    def _norm_category(category: str) -> str:
        if category is None:
            return ""
        return str(category).strip().lower()

    @staticmethod
    def _norm_video_path(video_path: str) -> str:
        return os.path.abspath(video_path)

    def set_video(self, video_path: str):
        norm_path = self._norm_video_path(video_path)
        if self.current_video_path == norm_path:
            return
        self.current_video_path = norm_path
        self._cache.clear()
        self.video_hits = 0
        self.video_misses = 0

    def get_video_stats(self) -> Dict[str, int]:
        return {
            "hits": int(self.video_hits),
            "misses": int(self.video_misses),
            "cached_categories": len(self._cache),
        }

    def get_total_stats(self) -> Dict[str, int]:
        return {
            "hits": int(self.total_hits),
            "misses": int(self.total_misses),
        }

    def install(self):
        if self._orig_fn is not None:
            return
        self._orig_fn = self.pipeline.sam3_recall.recall_all_instances

        def _cached_recall(video_path: str, category: str):
            norm_video = self._norm_video_path(video_path)
            if self.current_video_path != norm_video:
                # Defensive reset if caller forgot set_video().
                self.set_video(video_path)
            norm_category = self._norm_category(category)
            key = (norm_video, norm_category)

            if key in self._cache:
                self.video_hits += 1
                self.total_hits += 1
                if self.logger:
                    self.logger.debug(
                        f"[SAM3-Cache] HIT video={os.path.basename(norm_video)} category={norm_category}"
                    )
                return self._cache[key]

            self.video_misses += 1
            self.total_misses += 1
            if self.logger:
                self.logger.info(
                    f"[SAM3-Cache] MISS -> run recall video={os.path.basename(norm_video)} category={norm_category}"
                )
            masks = self._orig_fn(video_path, category)
            self._cache[key] = masks
            return masks

        self.pipeline.sam3_recall.recall_all_instances = _cached_recall

    def restore(self):
        if self._orig_fn is None:
            return
        self.pipeline.sam3_recall.recall_all_instances = self._orig_fn
        self._orig_fn = None


def main():
    parser = argparse.ArgumentParser(
        description="Run CRF-SAM3 on Ref-Youtube-ROS dataset (cached variant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_crf_sam3_refytros_dataset_cached.py --config configs/crf_sam3_config.yaml \\
      --data_root /path/to/ref-youtube-ros --split valid \\
      --output_dir crf_sam3_refytros_valid_submit
""",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")

    # common overrides
    parser.add_argument("--mllm_path", type=str, help="Override MLLM path")
    parser.add_argument("--sam3_checkpoint", type=str, help="Override SAM3 checkpoint path")
    parser.add_argument("--data_root", type=str, help="Override dataset root")
    parser.add_argument("--split", type=str, choices=["valid", "test", "train"], help="Override split")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--backend", type=str, choices=["mllm", "reranker", "embedding"], help="Override backend")
    parser.add_argument("--max_videos", type=int, help="Process at most N videos")
    parser.add_argument("--max_expressions", type=int, help="Process at most N expressions per video")
    parser.add_argument("--num_tube_frames", type=int, help="Override number of tube frames")
    parser.add_argument("--skip_existing", action="store_true", help="Skip completed expressions")

    # Ref-Youtube-ROS-specific
    parser.add_argument(
        "--meta_file",
        type=str,
        default=None,
        help="Metadata file name or absolute path (default auto-detect)",
    )
    parser.add_argument(
        "--disable_category_recall_cache",
        action="store_true",
        help="Disable per-video category-level SAM3 recall cache",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides.
    if args.mllm_path:
        config["model"]["mllm_path"] = args.mllm_path
    if args.sam3_checkpoint:
        config["model"]["sam3_checkpoint"] = args.sam3_checkpoint
    if args.data_root:
        config.setdefault("dataset", {})["data_root"] = args.data_root
    if args.split:
        config.setdefault("dataset", {})["split"] = args.split
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
    logger.info("CRF-SAM3 Ref-Youtube-ROS Runner (Cached)")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model_type = config["model"].get("model_type", "qwen3")
    mllm_path = config["model"]["mllm_path"]
    sam3_checkpoint = config["model"]["sam3_checkpoint"]

    data_cfg = config.get("dataset", {})
    data_root = data_cfg.get("data_root", "")
    split = data_cfg.get("split", "valid")
    max_videos = data_cfg.get("max_videos", None)
    max_expressions = data_cfg.get("max_expressions", None)
    skip_existing = bool(data_cfg.get("skip_existing", False))

    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("output_dir", "crf_sam3_refytros_outputs")
    save_visual_tubes = bool(output_cfg.get("save_visual_tubes", False))

    pipeline_cfg = config.get("pipeline", {})
    num_tube_frames = int(pipeline_cfg.get("num_tube_frames", 8))

    discr_cfg = config.get("discrimination", {})
    backend = discr_cfg.get("backend", "mllm")
    prompt_type = discr_cfg.get("prompt_type", "yesno")
    prompt_template_path = _resolve_prompt_template(prompt_type)

    visual_tube_cfg = config.get("visual_tube", {})
    visualization_style = visual_tube_cfg.get("visualization_style", "mask")
    sampling_strategy = visual_tube_cfg.get("sampling_strategy", "uniform")

    concept_cfg = config.get("concept_extraction", {})
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

    meta_path = _resolve_meta_path(data_root, split, args.meta_file)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    videos = meta["videos"]

    logger.info("\nConfiguration:")
    logger.info(f"  Model Type: {model_type}")
    logger.info(f"  MLLM: {mllm_path}")
    logger.info(f"  SAM3: {sam3_checkpoint}")
    logger.info(f"  Prompt Type: {prompt_type}")
    logger.info(f"  Prompt Template: {prompt_template_path}")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  Visual Tube Frames: {num_tube_frames}")
    logger.info(f"  Sampling Strategy: {sampling_strategy}")
    logger.info(f"  Visualization Style: {visualization_style}")
    logger.info(f"  Data Root: {data_root}")
    logger.info(f"  Split: {split}")
    logger.info(f"  Meta: {meta_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info("  Output Format: submit (Annotations/video_id/exp_id/*.png)")
    logger.info(f"  Max Videos: {max_videos if max_videos is not None else 'All'}")
    logger.info(f"  Max Expressions: {max_expressions if max_expressions is not None else 'All'}")
    logger.info(f"  Skip Existing: {skip_existing}")
    logger.info(f"  Category Recall Cache: {not args.disable_category_recall_cache}")
    logger.info(f"Found {len(videos)} videos")

    logger.info("\nInitializing pipeline...")
    reranker_cfg = discr_cfg.get("reranker", {})
    embedding_cfg = discr_cfg.get("embedding", {})
    if backend == "embedding":
        reranker_cfg = embedding_cfg

    pipeline = CRFSAM3Pipeline(
        mllm_path=mllm_path,
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
    )
    recall_cache = None
    if not args.disable_category_recall_cache:
        recall_cache = VideoCategoryRecallCache(pipeline, logger=logger)
        recall_cache.install()
        logger.info("Enabled per-video category recall cache.")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    processed_video_count = 0
    processed_exp_count = 0
    skipped_exp_count = 0
    total_missing_filled = 0
    frame_coverage_report: List[Dict] = []

    video_iter = videos.items()
    video_pbar = tqdm(video_iter, desc="Videos", position=0, disable=False)

    for video_id, video_info in video_pbar:
        if max_videos is not None and processed_video_count >= max_videos:
            break

        video_path = os.path.join(data_root, split, "JPEGImages", video_id)
        if not os.path.exists(video_path):
            logger.error(f"Missing video frames directory: {video_path}")
            continue
        if recall_cache is not None:
            recall_cache.set_video(video_path)

        expressions = video_info.get("expressions", {})
        video_frames = sorted([str(x) for x in video_info.get("frames", [])])
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

            referring_expression = exp_info.get("exp", "")
            exp_out_dir = os.path.join(output_dir, "Annotations", video_id, str(exp_id))
            if skip_existing and _is_exp_complete(exp_out_dir, video_frames):
                skipped_exp_count += 1
                exp_count += 1
                continue

            try:
                masks_dict, visual_tubes, _ = pipeline.predict(
                    video_path,
                    referring_expression,
                    return_details=True,
                )

                save_info = save_masks(
                    masks_dict=masks_dict,
                    output_dir=output_dir,
                    video_id=video_id,
                    exp_id=exp_id,
                    expected_frames=video_frames,
                    frame_hw=video_hw,
                )

                total_missing_filled += int(save_info["missing_pred_filled"])
                frame_coverage_report.append(
                    {
                        "video_id": video_id,
                        "exp_id": str(exp_id),
                        "expected_frames": int(save_info["expected_frames"]),
                        "saved_frames": int(save_info["saved_frames"]),
                        "missing_pred_filled": int(save_info["missing_pred_filled"]),
                    }
                )

                if save_visual_tubes:
                    visual_tubes_dir = os.path.join(output_dir, "visual_tubes", video_id, str(exp_id))
                    pipeline.save_visual_tubes(visual_tubes, visual_tubes_dir)

                processed_exp_count += 1
                exp_count += 1

            except torch.OutOfMemoryError:
                logger.error(f"CUDA OOM at {video_id}/{exp_id}. Abort to keep state consistent.")
                raise
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at {video_id}/{exp_id}. Abort to keep state consistent.")
                    raise
                logger.error(f"Runtime error at {video_id}/{exp_id}: {e}")
            except Exception as e:
                logger.error(f"Error at {video_id}/{exp_id}: {e}")
            finally:
                clear_cuda_cache(logger)

        if recall_cache is not None:
            stats = recall_cache.get_video_stats()
            logger.info(
                f"[SAM3-Cache] video={video_id} hits={stats['hits']} misses={stats['misses']} "
                f"cached_categories={stats['cached_categories']}"
            )
        processed_video_count += 1

    frame_report_path = os.path.join(output_dir, "frame_coverage.json")
    with open(frame_report_path, "w") as f:
        json.dump(frame_coverage_report, f, indent=2)

    summary = {
        "videos_total_in_meta": len(videos),
        "videos_processed": processed_video_count,
        "expressions_processed": processed_exp_count,
        "expressions_skipped_existing": skipped_exp_count,
        "missing_frames_filled_as_background": total_missing_filled,
        "output_dir": output_dir,
        "frame_coverage_path": frame_report_path,
    }
    if recall_cache is not None:
        total_cache_stats = recall_cache.get_total_stats()
        summary["sam3_category_cache_hits"] = total_cache_stats["hits"]
        summary["sam3_category_cache_misses"] = total_cache_stats["misses"]
        recall_cache.restore()

    with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Done")
    logger.info(json.dumps(summary, indent=2))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
