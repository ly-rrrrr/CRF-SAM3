"""
Complete Dataset Evaluation for TTRL Hard Sample Mining

This script:
1. Runs CRF-SAM3 pipeline on the entire MeViS dataset
2. Records detailed discrimination scores for each candidate
3. Computes J&F metrics (if ground truth available)
4. Identifies hard samples (low J&F scores)
5. Saves results for TTRL data preparation

Output format:
{
    "video_id": str,
    "exp_id": int,
    "expression": str,
    "gt_obj_id": int (if available),
    "candidates": [
        {
            "obj_id": int,
            "prob_yes": float,
            "prob_partial": float,
            "prob_no": float,
            "weighted_score": float,
            "prediction": str,  # "Yes", "Partial", or "No"
        }
    ],
    "selected_obj_id": int,  # Top-1 candidate
    "j_score": float (if GT available),
    "f_score": float (if GT available),
    "jf_score": float (if GT available),
    "is_correct": bool (if GT available),
}
"""

import os
import json
import glob
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# Import CRF-SAM3 components
from crf_sam3_demo import (
    ConceptExtractor,
    SAM3ConceptRecall,
    VisualTubeBuilder,
    SemanticDiscriminator,
)


class MeViSEvaluator:
    """Complete evaluation on MeViS dataset with detailed logging"""

    def __init__(
        self,
        config_path: str,
        mevis_root: str,
        split: str = "valid_u",
        output_dir: str = "evaluation_results"
    ):
        """
        Args:
            config_path: Path to CRF-SAM3 config file
            mevis_root: Root directory of MeViS dataset
            split: Dataset split (train/valid_u)
            output_dir: Directory to save results
        """
        self.config_path = config_path
        self.mevis_root = Path(mevis_root)
        self.split = split
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print(f"[MeViSEvaluator] Initializing for split: {split}")
        print(f"[MeViSEvaluator] Output directory: {output_dir}")

        # Load MeViS metadata
        self.load_mevis_metadata()

        # Initialize CRF-SAM3 components
        self.init_components()

        # Results storage
        self.results = []

    def load_mevis_metadata(self):
        """Load MeViS metadata with ground truth annotations"""
        meta_path = self.mevis_root / self.split / "meta_expressions.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"MeViS metadata not found: {meta_path}")

        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        # Build video-expression pairs
        self.video_exp_pairs = []
        for video_id, video_data in self.metadata['videos'].items():
            for exp_data in video_data['expressions']:
                exp_id = exp_data['exp_id']
                expression = exp_data['exp']
                obj_id = exp_data.get('obj_id', None)

                self.video_exp_pairs.append({
                    'video_id': video_id,
                    'exp_id': exp_id,
                    'expression': expression,
                    'obj_id': obj_id
                })

        print(f"[MeViSEvaluator] Loaded {len(self.video_exp_pairs)} video-expression pairs")

    def init_components(self):
        """Initialize all CRF-SAM3 components"""
        # ConceptExtractor
        self.concept_extractor = ConceptExtractor(
            mllm_path=self.config['concept_extraction']['mllm_path'],
            model_type=self.config['concept_extraction']['model_type']
        )

        # SAM3ConceptRecall
        self.sam3_recall = SAM3ConceptRecall(
            sa2va_model_path=self.config['sam3_recall']['sa2va_model_path'],
            device=self.config['sam3_recall']['device']
        )

        # VisualTubeBuilder
        self.visual_tube_builder = VisualTubeBuilder(
            sampling_strategy=self.config['visual_tube']['sampling_strategy'],
            num_frames=self.config['visual_tube']['num_frames'],
            visualization_mode=self.config['visual_tube']['visualization_mode'],
            qframe_config=self.config.get('qframe', {}),
            save_tubes=False  # Don't save to disk during batch evaluation
        )

        # SemanticDiscriminator
        self.discriminator = SemanticDiscriminator(
            mllm_path=self.config['discrimination']['mllm_path'],
            model_type=self.config['discrimination']['model_type'],
            prompt_path=self.config['discrimination']['prompt_path'],
            temperature=self.config['discrimination'].get('temperature', 0.01),
            use_logits=self.config['discrimination'].get('use_logits', True)
        )

        print(f"[MeViSEvaluator] All components initialized")

    def compute_iou(
        self,
        pred_masks: np.ndarray,
        gt_masks: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute J (IoU), F (boundary F-measure), and J&F scores

        Args:
            pred_masks: Predicted masks [T, H, W]
            gt_masks: Ground truth masks [T, H, W]

        Returns:
            (j_score, f_score, jf_score)
        """
        # Ensure binary masks
        pred_masks = (pred_masks > 0).astype(np.uint8)
        gt_masks = (gt_masks > 0).astype(np.uint8)

        # Compute J (IoU) per frame
        j_scores = []
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            if union == 0:
                j = 1.0 if intersection == 0 else 0.0
            else:
                j = intersection / union

            j_scores.append(j)

        j_score = np.mean(j_scores)

        # Simplified F-score (boundary precision/recall)
        # For now, use J as proxy (full F computation requires boundary extraction)
        f_score = j_score  # Placeholder

        jf_score = (j_score + f_score) / 2.0

        return j_score, f_score, jf_score

    def load_ground_truth_masks(
        self,
        video_id: str,
        obj_id: int
    ) -> np.ndarray:
        """
        Load ground truth masks for a video-object pair

        Returns:
            masks: [T, H, W] numpy array
        """
        annotations_dir = self.mevis_root / self.split / "Annotations" / video_id

        if not annotations_dir.exists():
            return None

        mask_files = sorted(glob.glob(str(annotations_dir / "*.png")))
        if len(mask_files) == 0:
            return None

        masks = []
        for mask_file in mask_files:
            mask = np.array(Image.open(mask_file))
            # Extract specific object ID
            binary_mask = (mask == obj_id).astype(np.uint8)
            masks.append(binary_mask)

        return np.stack(masks, axis=0)

    def evaluate_single_sample(
        self,
        video_id: str,
        exp_id: int,
        expression: str,
        gt_obj_id: int = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single video-expression pair

        Returns:
            Result dictionary with detailed scores
        """
        # Get video frames directory
        video_frames_dir = self.mevis_root / self.split / "JPEGImages" / video_id

        if not video_frames_dir.exists():
            print(f"[WARN] Video frames not found: {video_frames_dir}")
            return None

        # Load frames
        frame_files = sorted(glob.glob(str(video_frames_dir / "*.jpg")))
        if len(frame_files) == 0:
            print(f"[WARN] No frames found in {video_frames_dir}")
            return None

        frames = [np.array(Image.open(f)) for f in frame_files]

        # Step 1: Extract category
        try:
            category, num_targets, category_reasoning = self.concept_extractor.extract_category(
                referring_expression=expression,
                video_path=str(video_frames_dir)
            )
        except Exception as e:
            print(f"[ERROR] Concept extraction failed for {video_id}/{exp_id}: {str(e)}")
            return None

        # Step 2: SAM3 concept recall
        try:
            sam3_output = self.sam3_recall.recall_objects(
                frames=frames,
                category=category,
                num_targets=num_targets
            )
        except Exception as e:
            print(f"[ERROR] SAM3 recall failed for {video_id}/{exp_id}: {str(e)}")
            return None

        if not sam3_output or len(sam3_output['object_ids']) == 0:
            print(f"[WARN] No candidates recalled for {video_id}/{exp_id}")
            return {
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": expression,
                "gt_obj_id": gt_obj_id,
                "category": category,
                "num_targets": num_targets,
                "sub_conditions": category_reasoning.get("sub_conditions", []),
                "final_rule_applied": category_reasoning.get("final_rule_applied", "N/A"),
                "candidates": [],
                "selected_obj_id": None,
                "is_correct": False,
                "error": "no_candidates"
            }

        # Step 3: Build Visual Tubes and discriminate each candidate
        candidates_results = []

        for obj_id_candidate in sam3_output['object_ids']:
            try:
                # Build visual tube
                visual_tube = self.visual_tube_builder.build_tube(
                    frames=frames,
                    masks=sam3_output['masks'],
                    object_id=obj_id_candidate,
                    referring_expression=expression
                )

                if visual_tube is None:
                    continue

                # Discriminate
                disc_result = self.discriminator.discriminate_single(
                    visual_tube=visual_tube,
                    referring_expression=expression
                )

                candidates_results.append({
                    "obj_id": obj_id_candidate,
                    "prob_yes": disc_result['prob_yes'],
                    "prob_partial": disc_result['prob_partial'],
                    "prob_no": disc_result['prob_no'],
                    "weighted_score": disc_result['score'],
                    "prediction": disc_result['prediction'],
                })

            except Exception as e:
                print(f"[ERROR] Discrimination failed for {video_id}/{exp_id}/obj_{obj_id_candidate}: {str(e)}")
                continue

        if len(candidates_results) == 0:
            return {
                "video_id": video_id,
                "exp_id": exp_id,
                "expression": expression,
                "gt_obj_id": gt_obj_id,
                "category": category,
                "num_targets": num_targets,
                "candidates": [],
                "selected_obj_id": None,
                "is_correct": False,
                "error": "discrimination_failed"
            }

        # Select top-1 candidate
        candidates_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        selected_obj_id = candidates_results[0]['obj_id']

        # Compute J&F if ground truth available
        j_score, f_score, jf_score = None, None, None
        is_correct = None

        if gt_obj_id is not None:
            # Load ground truth masks
            gt_masks = self.load_ground_truth_masks(video_id, gt_obj_id)

            if gt_masks is not None and selected_obj_id in sam3_output['object_ids']:
                # Extract predicted masks for selected object
                pred_masks = sam3_output['masks'] == selected_obj_id

                # Compute metrics
                j_score, f_score, jf_score = self.compute_iou(pred_masks, gt_masks)

                # Check correctness (top-1 accuracy)
                is_correct = (selected_obj_id == gt_obj_id)

        # Build result
        result = {
            "video_id": video_id,
            "exp_id": exp_id,
            "expression": expression,
            "gt_obj_id": gt_obj_id,
            "category": category,
            "num_targets": num_targets,
            "sub_conditions": category_reasoning.get("sub_conditions", []),
            "final_rule_applied": category_reasoning.get("final_rule_applied", "N/A"),
            "candidates": candidates_results,
            "selected_obj_id": selected_obj_id,
            "j_score": j_score,
            "f_score": f_score,
            "jf_score": jf_score,
            "is_correct": is_correct,
        }

        return result

    def run_evaluation(
        self,
        max_samples: int = None,
        save_interval: int = 50
    ):
        """
        Run evaluation on entire dataset

        Args:
            max_samples: Maximum number of samples to evaluate (for testing)
            save_interval: Save intermediate results every N samples
        """
        pairs_to_process = self.video_exp_pairs[:max_samples] if max_samples else self.video_exp_pairs

        print(f"[MeViSEvaluator] Evaluating {len(pairs_to_process)} samples...")

        for i, pair in enumerate(tqdm(pairs_to_process, desc="Evaluating")):
            try:
                result = self.evaluate_single_sample(
                    video_id=pair['video_id'],
                    exp_id=pair['exp_id'],
                    expression=pair['expression'],
                    gt_obj_id=pair.get('obj_id')
                )

                if result is not None:
                    self.results.append(result)

                # Save intermediate results
                if (i + 1) % save_interval == 0:
                    self._save_intermediate(i + 1)

            except Exception as e:
                print(f"[ERROR] Failed to evaluate {pair['video_id']}/{pair['exp_id']}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Save final results
        self._save_final()

        # Print summary statistics
        self._print_statistics()

        return self.results

    def _save_intermediate(self, num_processed: int):
        """Save intermediate results"""
        output_path = self.output_dir / f"evaluation_{self.split}_intermediate_{num_processed}.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"[INFO] Saved intermediate results: {output_path}")

    def _save_final(self):
        """Save final results"""
        output_path = self.output_dir / f"evaluation_{self.split}_full.json"
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"[MeViSEvaluator] Saved final results: {output_path}")

    def _print_statistics(self):
        """Print evaluation statistics"""
        print("\n" + "=" * 60)
        print("EVALUATION STATISTICS")
        print("=" * 60)

        total_samples = len(self.results)
        print(f"Total samples evaluated: {total_samples}")

        # Top-1 accuracy
        correct_count = sum(1 for r in self.results if r.get('is_correct') is True)
        total_with_gt = sum(1 for r in self.results if r.get('is_correct') is not None)

        if total_with_gt > 0:
            top1_accuracy = correct_count / total_with_gt * 100
            print(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({correct_count}/{total_with_gt})")

        # J&F scores
        jf_scores = [r['jf_score'] for r in self.results if r.get('jf_score') is not None]
        if len(jf_scores) > 0:
            mean_jf = np.mean(jf_scores) * 100
            print(f"Mean J&F Score: {mean_jf:.2f}%")

        # Hard samples (low J&F)
        hard_samples = [r for r in self.results if r.get('jf_score') is not None and r['jf_score'] < 0.5]
        print(f"Hard samples (J&F < 50%): {len(hard_samples)} ({len(hard_samples)/len(jf_scores)*100:.1f}%)")

        # No candidates recalled
        no_candidates = sum(1 for r in self.results if len(r.get('candidates', [])) == 0)
        print(f"Samples with no candidates: {no_candidates} ({no_candidates/total_samples*100:.1f}%)")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run complete MeViS evaluation for TTRL data preparation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/crf_sam3_config.yaml",
        help="Path to CRF-SAM3 config file"
    )
    parser.add_argument(
        "--mevis_root",
        type=str,
        default="/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/mevis",
        help="Root directory of MeViS dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid_u",
        choices=["train", "valid_u"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Save intermediate results every N samples"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = MeViSEvaluator(
        config_path=args.config,
        mevis_root=args.mevis_root,
        split=args.split,
        output_dir=args.output_dir
    )

    # Run evaluation
    evaluator.run_evaluation(
        max_samples=args.max_samples,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
