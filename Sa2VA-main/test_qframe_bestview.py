"""
Test Script for Q-Frame and MLLM Best View Modules
Purpose: Validate the effectiveness of keyframe selection and best frame reasoning
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from pathlib import Path

# Add project paths
sys.path.append("/mnt/csip-200/share/kongjiawei/q-frame-master")
sys.path.append("/mnt/csip-200/share/kongjiawei/sam3-main")
sys.path.append("/mnt/csip-200/share/kongjiawei/Sa2VA-main")

# Import our modules
from mapp_sam3_demo import QFrameSelector, BestViewReasoner, load_mevis_sample


def visualize_qframe_results(
    video_frames: List[np.ndarray],
    selected_indices: List[int],
    referring_expression: str,
    output_dir: str,
    video_id: str
):
    """
    Visualize Q-Frame selection results

    Args:
        video_frames: All video frames
        selected_indices: Indices selected by Q-Frame
        referring_expression: The referring expression
        output_dir: Directory to save visualizations
        video_id: Video identifier
    """
    print(f"\n{'='*60}")
    print(f"Visualizing Q-Frame Results for {video_id}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Create a timeline visualization
    fig, ax = plt.subplots(figsize=(16, 2))
    total_frames = len(video_frames)

    # Plot all frames as gray bars
    for i in range(total_frames):
        color = 'red' if i in selected_indices else 'lightgray'
        ax.barh(0, 1, left=i, height=0.5, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlim(-1, total_frames)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_title(f'Q-Frame Selection Timeline\nExpression: "{referring_expression}"\nSelected: {len(selected_indices)}/{total_frames} frames',
                 fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Selected by Q-Frame'),
        Patch(facecolor='lightgray', label='Not Selected')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_id}_qframe_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved timeline: {video_id}_qframe_timeline.png")

    # 2. Create a grid of selected frames
    num_selected = len(selected_indices)
    cols = min(8, num_selected)
    rows = (num_selected + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(f'Q-Frame Selected Keyframes\n"{referring_expression}"',
                 fontsize=14, fontweight='bold')

    for idx, frame_idx in enumerate(selected_indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]

        frame = video_frames[frame_idx]
        ax.imshow(frame)
        ax.set_title(f'Frame {frame_idx}', fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(num_selected, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_id}_qframe_keyframes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved keyframes grid: {video_id}_qframe_keyframes.png")

    # 3. Analyze frame distribution
    print(f"\n{'='*60}")
    print("Q-Frame Selection Analysis:")
    print(f"{'='*60}")
    print(f"Total frames: {total_frames}")
    print(f"Selected frames: {num_selected}")
    print(f"Selection rate: {num_selected/total_frames*100:.1f}%")
    print(f"\nSelected indices: {selected_indices}")

    # Calculate gaps between selected frames
    if len(selected_indices) > 1:
        gaps = [selected_indices[i+1] - selected_indices[i] for i in range(len(selected_indices)-1)]
        print(f"\nGaps between selections:")
        print(f"  Min gap: {min(gaps)} frames")
        print(f"  Max gap: {max(gaps)} frames")
        print(f"  Avg gap: {np.mean(gaps):.1f} frames")
        print(f"  Std gap: {np.std(gaps):.1f} frames")

        # Analyze temporal distribution
        early_frames = sum(1 for idx in selected_indices if idx < total_frames / 3)
        mid_frames = sum(1 for idx in selected_indices if total_frames / 3 <= idx < 2 * total_frames / 3)
        late_frames = sum(1 for idx in selected_indices if idx >= 2 * total_frames / 3)

        print(f"\nTemporal distribution:")
        print(f"  Early (0-{total_frames//3}): {early_frames} frames ({early_frames/num_selected*100:.1f}%)")
        print(f"  Middle ({total_frames//3}-{2*total_frames//3}): {mid_frames} frames ({mid_frames/num_selected*100:.1f}%)")
        print(f"  Late ({2*total_frames//3}-{total_frames}): {late_frames} frames ({late_frames/num_selected*100:.1f}%)")


def visualize_bestview_results(
    video_frames: List[np.ndarray],
    candidate_indices: List[int],
    best_frames: Dict[int, Tuple[int, str]],
    referring_expression: str,
    output_dir: str,
    video_id: str
):
    """
    Visualize Best View Reasoning results

    Args:
        video_frames: All video frames
        candidate_indices: Candidate frame indices from Q-Frame
        best_frames: Dict mapping obj_id to (frame_idx, phrase)
        referring_expression: Original referring expression
        output_dir: Directory to save visualizations
        video_id: Video identifier
    """
    print(f"\n{'='*60}")
    print(f"Visualizing Best View Results for {video_id}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Create comparison visualization
    num_objs = len(best_frames)
    fig, axes = plt.subplots(num_objs, 3, figsize=(12, num_objs*3))

    if num_objs == 1:
        axes = [axes]

    fig.suptitle(f'Best View Reasoning Results\nOriginal: "{referring_expression}"',
                 fontsize=14, fontweight='bold')

    for obj_id, (best_frame_idx, simple_phrase) in best_frames.items():
        row = obj_id

        # Show first candidate frame
        axes[row][0].imshow(video_frames[candidate_indices[0]])
        axes[row][0].set_title(f'First Candidate\nFrame {candidate_indices[0]}', fontsize=10)
        axes[row][0].axis('off')

        # Show best frame
        axes[row][1].imshow(video_frames[best_frame_idx])
        axes[row][1].set_title(f'✨ Best Frame ✨\nFrame {best_frame_idx}',
                               fontsize=10, fontweight='bold', color='red')
        axes[row][1].axis('off')
        # Add red border
        for spine in axes[row][1].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)

        # Show last candidate frame
        axes[row][2].imshow(video_frames[candidate_indices[-1]])
        axes[row][2].set_title(f'Last Candidate\nFrame {candidate_indices[-1]}', fontsize=10)
        axes[row][2].axis('off')

        # Add text below the row
        fig.text(0.5, (num_objs - row - 0.5) / num_objs - 0.02,
                f'Object {obj_id}: "{simple_phrase}"',
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_id}_bestview_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved best view comparison: {video_id}_bestview_comparison.png")

    # 2. Show all candidate frames with best frame highlighted
    num_candidates = len(candidate_indices)
    cols = min(8, num_candidates)
    rows = (num_candidates + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle(f'All Candidates (Best Frames Highlighted)', fontsize=14, fontweight='bold')

    best_frame_indices = [frame_idx for _, (frame_idx, _) in best_frames.items()]

    for idx, frame_idx in enumerate(candidate_indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]

        frame = video_frames[frame_idx]
        ax.imshow(frame)

        is_best = frame_idx in best_frame_indices
        title = f'Frame {frame_idx}'
        if is_best:
            title = f'✨ {title} ✨'
            ax.set_title(title, fontsize=10, fontweight='bold', color='red')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        else:
            ax.set_title(title, fontsize=10)

        ax.axis('off')

    # Hide unused subplots
    for idx in range(num_candidates, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_id}_bestview_all_candidates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved all candidates: {video_id}_bestview_all_candidates.png")

    # 3. Print analysis
    print(f"\n{'='*60}")
    print("Best View Reasoning Analysis:")
    print(f"{'='*60}")
    print(f"Original expression: \"{referring_expression}\"")
    print(f"Number of objects: {num_objs}")
    print(f"Number of candidates: {num_candidates}")

    for obj_id, (best_frame_idx, simple_phrase) in best_frames.items():
        print(f"\nObject {obj_id}:")
        print(f"  Best frame: {best_frame_idx}")
        print(f"  Simplified phrase: \"{simple_phrase}\"")
        print(f"  Position in candidates: {candidate_indices.index(best_frame_idx) + 1}/{num_candidates}")

        # Calculate distance from start
        distance_from_start = best_frame_idx - candidate_indices[0]
        distance_from_end = candidate_indices[-1] - best_frame_idx
        print(f"  Distance from first candidate: {distance_from_start} frames")
        print(f"  Distance from last candidate: {distance_from_end} frames")


def compare_with_baseline(
    video_frames: List[np.ndarray],
    qframe_indices: List[int],
    best_frame_idx: int,
    referring_expression: str,
    output_dir: str,
    video_id: str
):
    """
    Compare Q-Frame + Best View with baseline strategies

    Args:
        video_frames: All video frames
        qframe_indices: Indices selected by Q-Frame
        best_frame_idx: Best frame selected by MLLM
        referring_expression: The referring expression
        output_dir: Directory to save comparison
        video_id: Video identifier
    """
    print(f"\n{'='*60}")
    print(f"Comparing with Baseline Strategies")
    print(f"{'='*60}")

    total_frames = len(video_frames)

    # Define baseline strategies
    baselines = {
        'First Frame': 0,
        'Middle Frame': total_frames // 2,
        'Last Frame': total_frames - 1,
        'Uniform Sample (First)': (np.linspace(0, total_frames - 1, 16, dtype=int))[0],
        'Q-Frame (First)': qframe_indices[0],
        'MAPP-SAM3 (Best)': best_frame_idx
    }

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    fig.suptitle(f'Strategy Comparison\n"{referring_expression}"', fontsize=14, fontweight='bold')

    for idx, (strategy, frame_idx) in enumerate(baselines.items()):
        ax = axes[idx]
        ax.imshow(video_frames[frame_idx])

        title = f'{strategy}\nFrame {frame_idx}'
        color = 'red' if strategy == 'MAPP-SAM3 (Best)' else 'black'
        weight = 'bold' if strategy == 'MAPP-SAM3 (Best)' else 'normal'

        ax.set_title(title, fontsize=11, color=color, fontweight=weight)
        ax.axis('off')

        if strategy == 'MAPP-SAM3 (Best)':
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{video_id}_strategy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved strategy comparison: {video_id}_strategy_comparison.png")

    # Print comparison table
    print(f"\n{'='*60}")
    print("Strategy Comparison Table:")
    print(f"{'='*60}")
    print(f"{'Strategy':<25} {'Frame Index':<15} {'Notes'}")
    print(f"{'-'*60}")

    for strategy, frame_idx in baselines.items():
        is_ours = '← Our Method' if strategy == 'MAPP-SAM3 (Best)' else ''
        print(f"{strategy:<25} {frame_idx:<15} {is_ours}")


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test Q-Frame and Best View')
    parser.add_argument('--config', type=str, default='mapp_sam3_config.yaml')
    # parser.add_argument('--video_id', type=str, default='d56a6ec78cfa')
    # parser.add_argument('--video_id', type=str, default='377b1c5f365c')
    parser.add_argument('--video_id', type=str, default='d32e4f23eb40')
    parser.add_argument('--exp_id', type=int, default=19)
    parser.add_argument('--output_dir', type=str, default='./qframe_bestview_test_results')
    parser.add_argument('--num_candidates', type=int, default=8, help='Number of Q-Frame candidates')
    args = parser.parse_args()

    print("="*60)
    print("Q-Frame & Best View Testing Script")
    print("="*60)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    clip_model_path = config['models']['clip_model_path']
    mllm_path = config['models']['mllm_path']
    sam3_checkpoint = config['models']['sam3_checkpoint']
    data_root = config['data']['mevis_root']
    device = config['models']['device']

    # Load video sample
    print(f"\n[1/4] Loading MeViS sample: {args.video_id}")
    frames, expression, frame_names = load_mevis_sample(data_root, args.video_id, args.exp_id)
    print(f"✓ Loaded {len(frames)} frames")
    print(f"✓ Expression: \"{expression}\"")

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.video_id, str(args.exp_id))
    os.makedirs(output_dir, exist_ok=True)

    # Test Q-Frame
    print(f"\n[2/4] Testing Q-Frame Module")
    print("-" * 60)
    qframe_selector = QFrameSelector(clip_model_path, device, apply_rebalancing=False)
    qframe_indices = qframe_selector.select_keyframes(
        frames, expression, num_candidates=args.num_candidates
    )

    # Visualize Q-Frame results
    visualize_qframe_results(
        frames, qframe_indices, expression, output_dir, args.video_id
    )

    # Test Best View Reasoning
    print(f"\n[3/4] Testing Best View Module")
    print("-" * 60)
    bestview_reasoner = BestViewReasoner(mllm_path, sam3_checkpoint, device)

    # Skip Presence Head for now
    filtered_indices = qframe_indices

    # Find best frames - MLLM will determine number of objects
    best_frames = bestview_reasoner.find_best_frames(
        frames, filtered_indices, expression
    )

    # Visualize Best View results
    visualize_bestview_results(
        frames, qframe_indices, best_frames, expression, output_dir, args.video_id
    )

    # Compare with baselines
    print(f"\n[4/4] Comparing with Baseline Strategies")
    print("-" * 60)
    best_frame_idx = best_frames[0][0]  # Get frame index for first object
    compare_with_baseline(
        frames, qframe_indices, best_frame_idx, expression, output_dir, args.video_id
    )

    # Save results to JSON
    results = {
        'video_id': args.video_id,
        'exp_id': args.exp_id,
        'expression': expression,
        'total_frames': len(frames),
        'qframe_selected': qframe_indices,
        'num_qframe_candidates': len(qframe_indices),
        'best_frames': {
            str(obj_id): {
                'frame_idx': frame_idx,
                'simple_phrase': phrase
            }
            for obj_id, (frame_idx, phrase) in best_frames.items()
        }
    }

    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {results_file}")

    # Final summary
    print(f"\n{'='*60}")
    print("Test Complete! Summary:")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated visualizations:")
    print(f"  1. {args.video_id}_qframe_timeline.png")
    print(f"  2. {args.video_id}_qframe_keyframes.png")
    print(f"  3. {args.video_id}_bestview_comparison.png")
    print(f"  4. {args.video_id}_bestview_all_candidates.png")
    print(f"  5. {args.video_id}_strategy_comparison.png")
    print(f"  6. results.json")
    print(f"\n🎉 You can now review these visualizations to evaluate Q-Frame and Best View!")


if __name__ == "__main__":
    main()
