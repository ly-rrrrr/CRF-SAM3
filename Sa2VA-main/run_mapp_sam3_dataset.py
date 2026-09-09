#!/usr/bin/env python3
"""
Batch inference script for MAPP-SAM3 on MeViS valid_u dataset
Processes all videos and expressions, saving PNG masks
"""
import os
import json
import argparse
import time
from pathlib import Path
import sys
from tqdm import tqdm

# Import MAPP-SAM3 pipeline
from mapp_sam3_demo import MAPPSAM3Pipeline, load_video_frames


def main():
    parser = argparse.ArgumentParser(description='Run MAPP-SAM3 on MeViS valid_u dataset')
    parser.add_argument('--config', type=str, default='./mapp_sam3_config.yaml',
                        help='Path to config file')
    parser.add_argument('--meta_path', type=str,
                        default='/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/mevis/valid_u/meta_expressions.json',
                        help='Path to meta_expressions.json')
    parser.add_argument('--video_root', type=str,
                        default='/mnt/csip-200/share/kongjiawei/Sa2VA/data/video_datas/mevis/valid_u/JPEGImages',
                        help='Root directory of video frames')
    parser.add_argument('--output_root', type=str, default='./mapp_sam3_results',
                        help='Output directory for masks')
    parser.add_argument('--start_from', type=str, default=None,
                        help='Resume from a specific video_id (optional)')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process (for testing)')

    args = parser.parse_args()

    # Load metadata
    print("Loading MeViS metadata...")
    with open(args.meta_path, 'r') as f:
        meta_data = json.load(f)

    videos = meta_data['videos']
    total_videos = len(videos)
    total_expressions = sum(len(v['expressions']) for v in videos.values())

    print(f"Dataset: {total_videos} videos, {total_expressions} expressions")

    # Initialize pipeline
    print("\nInitializing MAPP-SAM3 Pipeline...")

    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract paths from config
    clip_model_path = config['models']['clip_model_path']
    mllm_path = config['models']['mllm_path']
    sam3_checkpoint = config['models']['sam3_checkpoint']
    device = config['models']['device']

    pipeline = MAPPSAM3Pipeline(
        clip_model_path=clip_model_path,
        mllm_path=mllm_path,
        sam3_checkpoint=sam3_checkpoint,
        device=device
    )
    print("Pipeline ready!\n")

    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)

    # Process videos
    processed_exp_count = 0  # Number of expressions processed
    skipped_count = 0
    failed_count = 0
    processed_video_count = 0  # Number of videos processed
    start_time = time.time()

    resume_mode = args.start_from is not None

    # Create progress bar for videos
    video_pbar = tqdm(videos.items(), total=len(videos), desc="Processing videos", unit="video")

    for video_idx, (video_id, vid_info) in enumerate(video_pbar):
        # Resume logic
        if resume_mode:
            if video_id == args.start_from:
                resume_mode = False
                print(f"Resuming from video {video_id}")
            else:
                continue

        # Max videos limit (check number of videos, not expressions)
        if args.max_videos is not None and processed_video_count >= args.max_videos:
            print(f"\nReached max_videos limit ({args.max_videos}), stopping.")
            break

        print(f"\n{'='*70}")
        print(f"Video {video_idx + 1}/{total_videos}: {video_id}")
        print(f"{'='*70}")

        # Load video frames
        video_path = os.path.join(args.video_root, video_id)

        try:
            video_frames = load_video_frames(video_path)
            num_frames = len(video_frames)
            print(f"Loaded {num_frames} frames")
        except Exception as e:
            print(f"ERROR: Failed to load video frames: {e}")
            failed_count += 1
            continue

        # Process each expression
        expressions = vid_info['expressions']

        # Update video progress bar
        video_pbar.set_postfix({'video': video_id, 'expressions': len(expressions)})

        # Create progress bar for expressions in this video
        exp_pbar = tqdm(expressions.items(), total=len(expressions),
                        desc=f"  Expressions in {video_id}", unit="exp", leave=False)

        for exp_idx, (exp_id, exp_info) in enumerate(exp_pbar):
            referring_expression = exp_info['exp']
            output_dir = os.path.join(args.output_root, video_id, exp_id)

            # Update expression progress bar
            exp_pbar.set_postfix({'exp_id': exp_id, 'text': referring_expression[:30] + '...' if len(referring_expression) > 30 else referring_expression})

            # Skip if already processed
            if os.path.exists(output_dir):
                num_existing = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
                if num_existing == num_frames:
                    skipped_count += 1
                    exp_pbar.set_postfix({'status': 'SKIP', 'exp_id': exp_id})
                    continue

            try:
                # Run MAPP-SAM3
                exp_start_time = time.time()

                masks = pipeline.predict(
                    video_frames=video_frames,
                    video_path=video_path,
                    referring_expression=referring_expression,
                    video_id=video_id
                )

                exp_time = time.time() - exp_start_time

                # Save masks
                os.makedirs(output_dir, exist_ok=True)
                pipeline.save_masks(masks, output_dir)

                processed_exp_count += 1
                exp_pbar.set_postfix({'status': 'OK', 'time': f'{exp_time:.1f}s', 'masks': len(masks)})

            except Exception as e:
                failed_count += 1
                exp_pbar.set_postfix({'status': 'FAIL', 'error': str(e)[:20]})
                tqdm.write(f"  ✗ ERROR in {video_id}/{exp_id}: {e}")
                continue

        # Mark this video as processed
        processed_video_count += 1

    # Summary
    total_time = time.time() - start_time
    video_pbar.close()

    print(f"\n{'='*70}")
    print(f"Batch Processing Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Videos processed: {processed_video_count}")
    print(f"Expressions processed: {processed_exp_count}")
    print(f"Expressions skipped: {skipped_count} (already existed)")
    print(f"Expressions failed: {failed_count}")
    if processed_exp_count > 0:
        avg_time = total_time / processed_exp_count
        print(f"Average time per expression: {avg_time:.1f}s")
        # Estimate total time for all expressions
        remaining_exps = total_expressions - processed_exp_count - skipped_count
        if remaining_exps > 0:
            estimated_time = remaining_exps * avg_time
            print(f"Estimated time for remaining {remaining_exps} expressions: {estimated_time/3600:.2f} hours")

    print(f"\nResults saved to: {args.output_root}")
    print(f"\nTo evaluate, run:")
    print(f"  python eval_mapp_sam3.py --pred_root {args.output_root}")


if __name__ == '__main__':
    main()
