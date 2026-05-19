#!/usr/bin/env python3
"""
Simplified evaluation script for MAPP-SAM3
Directly reads PNG masks instead of RLE-encoded JSON
"""
import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask
import sys
sys.path.append('./third_parts/revos/utils')
from metircs import db_eval_iou, db_eval_boundary


def _resolve_pred_dir(pred_root, video_id, exp_id):
    cands = [
        os.path.join(pred_root, 'Annotations', video_id, str(exp_id)),
        os.path.join(pred_root, video_id, str(exp_id)),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return cands[0]


def evaluate_single_video(video_id, exp_id, pred_mask_dir, mask_dict, meta_expressions):
    """
    Evaluate a single video expression

    Args:
        video_id: Video ID (e.g., "e339e02b5b3c")
        exp_id: Expression ID (e.g., "0")
        pred_mask_dir: Directory containing predicted PNG masks
        mask_dict: Loaded mask_dict.json dict
        meta_expressions: Loaded meta_expressions.json dict

    Returns:
        dict: {'J': float, 'F': float, 'J&F': float}
    """
    # Load video metadata
    vid_info = meta_expressions['videos'][video_id]
    exp_info = vid_info['expressions'][str(exp_id)]
    anno_ids = exp_info['anno_id']

    num_frames = len(vid_info['frames'])

    # Get GT mask shape from first non-empty GT mask (use GT resolution as reference)
    gt_h, gt_w = None, None
    for frame_idx in range(num_frames):
        for anno_id in anno_ids:
            mask_rle = mask_dict[str(anno_id)][frame_idx]
            if mask_rle:
                first_gt_mask = cocomask.decode(mask_rle)
                gt_h, gt_w = first_gt_mask.shape
                break
        if gt_h is not None:
            break

    if gt_h is None:
        raise ValueError(f"No GT masks found for video {video_id}, exp {exp_id}")

    # Initialize arrays with GT resolution
    gt_masks = np.zeros((num_frames, gt_h, gt_w), dtype=np.uint8)
    pred_masks = np.zeros((num_frames, gt_h, gt_w), dtype=np.uint8)

    # Load GT masks from mask_dict (RLE format)
    for frame_idx in range(num_frames):
        for anno_id in anno_ids:
            mask_rle = mask_dict[str(anno_id)][frame_idx]
            if mask_rle:
                # Decode RLE to binary mask
                gt_mask = cocomask.decode(mask_rle)
                gt_masks[frame_idx] += gt_mask

    # Load predicted masks from PNG files and resize to GT resolution if needed
    for frame_idx in range(num_frames):
        pred_path = os.path.join(pred_mask_dir, f'{frame_idx:05d}.png')
        if os.path.exists(pred_path):
            pred_mask = np.array(Image.open(pred_path))
            # Convert to binary: any non-zero value is foreground
            pred_binary = (pred_mask > 0).astype(np.uint8)

            # Resize to GT resolution if needed
            if pred_binary.shape != (gt_h, gt_w):
                print(f"[Warning] Resizing prediction from {pred_binary.shape} to ({gt_h}, {gt_w}) for {video_id}/{exp_id} frame {frame_idx}")
                pred_img = Image.fromarray(pred_binary * 255)
                pred_img = pred_img.resize((gt_w, gt_h), Image.NEAREST)
                pred_binary = (np.array(pred_img) > 0).astype(np.uint8)

            pred_masks[frame_idx] = pred_binary

    # Compute metrics
    j = db_eval_iou(gt_masks, pred_masks).mean()
    f = db_eval_boundary(gt_masks, pred_masks).mean()

    return {
        'J': float(j),
        'F': float(f),
        'J&F': float((j + f) / 2)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate MAPP-SAM3 results on MeViS')
    parser.add_argument('--pred_root', type=str, default='./mapp_sam3_results',
                        help='Root directory of predicted masks')
    parser.add_argument('--mask_dict_path', type=str,
                        default='/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/mevis/valid_u/mask_dict.json',
                        help='Path to mask_dict.json')
    parser.add_argument('--meta_path', type=str,
                        default='/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/mevis/valid_u/meta_expressions.json',
                        help='Path to meta_expressions.json')
    parser.add_argument('--video_id', type=str, default=None,
                        help='Evaluate single video (optional)')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Evaluate single expression (optional, requires --video_id)')

    args = parser.parse_args()

    # Load metadata
    print("Loading metadata...")
    with open(args.meta_path, 'r') as f:
        meta_expressions = json.load(f)

    print("Loading GT masks...")
    with open(args.mask_dict_path, 'r') as f:
        mask_dict = json.load(f)

    # Single video evaluation
    if args.video_id is not None:
        if args.exp_id is None:
            # Evaluate all expressions for this video
            vid_info = meta_expressions['videos'][args.video_id]
            exp_ids = list(vid_info['expressions'].keys())
        else:
            exp_ids = [args.exp_id]

        print(f"Evaluating video {args.video_id}, expressions: {exp_ids}")

        results = {}
        for exp_id in exp_ids:
            pred_mask_dir = _resolve_pred_dir(args.pred_root, args.video_id, exp_id)

            if not os.path.exists(pred_mask_dir):
                print(f"Warning: {pred_mask_dir} not found, skipping")
                continue

            print(f"\nEvaluating {args.video_id}/{exp_id}...")
            result = evaluate_single_video(
                args.video_id, exp_id, pred_mask_dir, mask_dict, meta_expressions
            )

            results[f"{args.video_id}_{exp_id}"] = result
            print(f"  J: {result['J']*100:.2f}")
            print(f"  F: {result['F']*100:.2f}")
            print(f"  J&F: {result['J&F']*100:.2f}")

        # Overall statistics
        if len(results) > 0:
            avg_j = np.mean([r['J'] for r in results.values()])
            avg_f = np.mean([r['F'] for r in results.values()])
            print(f"\n{'='*50}")
            print(f"Overall Results:")
            print(f"  J: {avg_j*100:.2f}")
            print(f"  F: {avg_f*100:.2f}")
            print(f"  J&F: {(avg_j+avg_f)/2*100:.2f}")

    else:
        # Full dataset evaluation
        print("Evaluating full dataset...")
        all_results = {}
        video_results = {}  # Store results per video

        for video_id in meta_expressions['videos']:
            vid_info = meta_expressions['videos'][video_id]
            video_exp_results = []

            for exp_id in vid_info['expressions']:
                pred_mask_dir = _resolve_pred_dir(args.pred_root, video_id, exp_id)

                if not os.path.exists(pred_mask_dir):
                    continue

                result = evaluate_single_video(
                    video_id, exp_id, pred_mask_dir, mask_dict, meta_expressions
                )

                all_results[f"{video_id}_{exp_id}"] = result
                video_exp_results.append(result)

            # Compute video-level statistics (average over all expressions in this video)
            if len(video_exp_results) > 0:
                video_j = np.mean([r['J'] for r in video_exp_results])
                video_f = np.mean([r['F'] for r in video_exp_results])
                video_results[video_id] = {
                    'J': float(video_j),
                    'F': float(video_f),
                    'J&F': float((video_j + video_f) / 2),
                    'num_expressions': len(video_exp_results)
                }

        # Overall statistics (average over all expressions)
        avg_j = np.mean([r['J'] for r in all_results.values()])
        avg_f = np.mean([r['F'] for r in all_results.values()])

        # Video-level statistics (average over all videos)
        video_avg_j = np.mean([r['J'] for r in video_results.values()])
        video_avg_f = np.mean([r['F'] for r in video_results.values()])

        print(f"\n{'='*50}")
        print(f"Evaluated {len(all_results)} expressions across {len(video_results)} videos")
        print(f"\nExpression-level Results (average over all expressions):")
        print(f"  J: {avg_j*100:.2f}")
        print(f"  F: {avg_f*100:.2f}")
        print(f"  J&F: {(avg_j+avg_f)/2*100:.2f}")

        print(f"\nVideo-level Results (average over all videos):")
        print(f"  J: {video_avg_j*100:.2f}")
        print(f"  F: {video_avg_f*100:.2f}")
        print(f"  J&F: {(video_avg_j+video_avg_f)/2*100:.2f}")

        # Save results
        output_path = os.path.join(args.pred_root, 'evaluation_results.json')
        with open(output_path, 'w') as f:
            json.dump({
                'per_expression': all_results,
                'per_video': video_results,
                'overall_expression_level': {
                    'J': round(avg_j * 100, 2),
                    'F': round(avg_f * 100, 2),
                    'J&F': round((avg_j + avg_f) / 2 * 100, 2),
                    'num_expressions': len(all_results)
                },
                'overall_video_level': {
                    'J': round(video_avg_j * 100, 2),
                    'F': round(video_avg_f * 100, 2),
                    'J&F': round((video_avg_j + video_avg_f) / 2 * 100, 2),
                    'num_videos': len(video_results)
                }
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
