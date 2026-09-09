import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import json
import os
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import numpy as np
except Exception:
    np = None

from tools._phase2_analysis_common import (
    build_runtime_matcher,
    discover_expression_dirs,
    load_analysis_context,
    load_expression_data,
    release_model,
    resolve_model_runs,
    run_candidate_forward,
    sample_expression_dirs,
    summarize_values,
)


def rank_of_obj(sorted_ids: List[int], target_ids: List[int]) -> int:
    target_set = set(target_ids)
    for idx, obj_id in enumerate(sorted_ids, start=1):
        if obj_id in target_set:
            return idx
    return len(sorted_ids) + 1


def _clean(values: List[float]) -> List[float]:
    return [float(v) for v in values if v is not None]


def _save_paper_figures(out_dir: str, summary: Dict) -> None:
    if plt is None or np is None:
        return
    before = summary['before']
    after = summary.get('after')
    if not after:
        return

    before_pos = _clean(before['positive_prob_scores'])
    before_neg = _clean(before['negative_prob_scores'])
    after_pos = _clean(after['positive_prob_scores'])
    after_neg = _clean(after['negative_prob_scores'])
    gap_before = _clean(before['gap_logit_values'])
    gap_after = _clean(after['gap_logit_values'])
    rank_before = _clean(before['best_pos_rank_values'])
    rank_after = _clean(after['best_pos_rank_values'])

    paired = summary.get('comparison', {})
    paired_examples = paired.get('paired_examples', [])
    rank_improved = sum(1 for row in paired_examples if row['rank_improvement'] > 0) / len(paired_examples) if paired_examples else 0.0
    gap_improved = sum(1 for row in paired_examples if (row['gap_logit_delta'] or 0.0) > 0) / len(paired_examples) if paired_examples else 0.0
    top1_before = float(before['stats']['top1_pos_rate'])
    top1_after = float(after['stats']['top1_pos_rate'])
    flip_to_pos = float(paired.get('flip_to_pos_rate', 0.0))

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    # Main paper figure: one panel for L_cls, one panel for L_rank.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    prob_data = [before_pos, before_neg, after_pos, after_neg]
    prob_pos = [1.0, 2.0, 4.0, 5.0]
    prob_colors = ['#4c78a8', '#c44e52', '#59a14f', '#f28e2b']
    prob_labels = ['Base\nPos', 'Base\nNeg', 'FT\nPos', 'FT\nNeg']
    bp = ax.boxplot(prob_data, positions=prob_pos, widths=0.58, patch_artist=True, showfliers=True)
    for patch, color in zip(bp['boxes'], prob_colors):
        patch.set(facecolor=color, edgecolor=color, alpha=0.8)
    for key in ['whiskers', 'caps', 'medians']:
        for artist in bp[key]:
            artist.set(color='#2f2f2f', linewidth=1.1)
    for flier, color in zip(bp['fliers'], prob_colors):
        flier.set(marker='o', markersize=2.4, markerfacecolor=color, markeredgecolor='none', alpha=0.2)
    ax.set_xticks(prob_pos)
    ax.set_xticklabels(prob_labels)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Yes probability')
    ax.set_title('L_cls: positive/negative separation')
    ax.grid(axis='y', alpha=0.16, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    gap_data = [gap_before, gap_after]
    gap_pos = [1.0, 2.0]
    gap_colors = ['#8f8f8f', '#1f9d8a']
    bp = ax.boxplot(gap_data, positions=gap_pos, widths=0.56, patch_artist=True, showfliers=True)
    for patch, color in zip(bp['boxes'], gap_colors):
        patch.set(facecolor=color, edgecolor=color, alpha=0.85)
    for key in ['whiskers', 'caps', 'medians']:
        for artist in bp[key]:
            artist.set(color='#2f2f2f', linewidth=1.1)
    for flier, color in zip(bp['fliers'], gap_colors):
        flier.set(marker='o', markersize=2.4, markerfacecolor=color, markeredgecolor='none', alpha=0.2)
    ax.axhline(0.0, color='#555555', linewidth=1.0, linestyle='--')
    ax.set_xticks(gap_pos)
    ax.set_xticklabels(['Base', 'Finetuned'])
    ax.set_ylabel('max positive logit - max negative logit')
    ax.set_title('L_rank: expression-level ranking gap')
    ax.grid(axis='y', alpha=0.16, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'paper_cls_rank_main.png'), dpi=260, bbox_inches='tight')
    plt.close(fig)

    # Optional support figure: rank placement and top-1 success.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    rank_vals = rank_before + rank_after
    rank_max = int(max(rank_vals)) if rank_vals else 1
    bins = np.arange(1, rank_max + 2)
    ax.hist(rank_before, bins=bins, histtype='step', linewidth=2.2, density=True, color='#4c78a8', label='Base')
    ax.hist(rank_after, bins=bins, histtype='step', linewidth=2.2, density=True, color='#59a14f', label='Finetuned')
    ax.set_xticks(range(1, min(rank_max, 10) + 1))
    ax.set_xlabel('Rank of the highest-scoring positive candidate')
    ax.set_ylabel('Normalized frequency')
    ax.set_title('Best positive candidate rank')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=0.16, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    labels = ['Top-1 pos\n(base)', 'Top-1 pos\n(FT)', 'Rank\nimproved', 'Flip-to-pos']
    values = [top1_before, top1_after, rank_improved, flip_to_pos]
    colors = ['#4c78a8', '#59a14f', '#f28e2b', '#8e63ce']
    bars = ax.bar(range(len(labels)), values, color=colors, width=0.64)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Fraction of expressions')
    ax.set_title('Ranking outcome summary')
    ax.grid(axis='y', alpha=0.16, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'paper_rank_support.png'), dpi=260, bbox_inches='tight')
    plt.close(fig)

def maybe_plot(out_path: str, summary: Dict) -> None:
    if plt is None or np is None:
        return
    before = summary['before']
    after = summary.get('after')

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    fig.patch.set_facecolor('white')

    before_pos = _clean(before['positive_prob_scores'])
    before_neg = _clean(before['negative_prob_scores'])
    after_pos = _clean(after['positive_prob_scores']) if after else []
    after_neg = _clean(after['negative_prob_scores']) if after else []

    ax = axes[0, 0]
    prob_data = [before_pos, before_neg]
    prob_pos = [1.0, 2.0]
    prob_colors = ['#2878b5', '#9c2f2f']
    prob_labels = ['Base\nPositive', 'Base\nNegative']
    if after:
        prob_data.extend([after_pos, after_neg])
        prob_pos.extend([4.0, 5.0])
        prob_colors.extend(['#4daf4a', '#f28e2b'])
        prob_labels.extend(['Finetuned\nPositive', 'Finetuned\nNegative'])
    bp = ax.boxplot(
        prob_data,
        positions=prob_pos,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(bp['boxes'], prob_colors):
        patch.set(facecolor=color, edgecolor=color, alpha=0.78)
    for key in ['whiskers', 'caps', 'medians']:
        for artist in bp[key]:
            artist.set(color='#2f2f2f', linewidth=1.2)
    for flier, color in zip(bp['fliers'], prob_colors):
        flier.set(marker='o', markersize=2.8, markerfacecolor=color, markeredgecolor='none', alpha=0.25)
    ax.set_xticks(prob_pos)
    ax.set_xticklabels(prob_labels)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Verifier yes probability')
    ax.set_title('A. Positive/negative candidate separation')
    ax.grid(axis='y', alpha=0.18, linewidth=0.8)

    ax = axes[0, 1]
    gap_before = _clean(before['gap_logit_values'])
    gap_after = _clean(after['gap_logit_values']) if after else []
    gap_data = [gap_before] + ([gap_after] if after else [])
    gap_pos = [1.0] + ([2.0] if after else [])
    gap_labels = ['Base', 'Finetuned'] if after else ['Base']
    gap_colors = ['#7a7a7a', '#1f9d8a'] if after else ['#7a7a7a']
    bp = ax.boxplot(
        gap_data,
        positions=gap_pos,
        widths=0.58,
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(bp['boxes'], gap_colors):
        patch.set(facecolor=color, edgecolor=color, alpha=0.82)
    for key in ['whiskers', 'caps', 'medians']:
        for artist in bp[key]:
            artist.set(color='#2f2f2f', linewidth=1.2)
    for flier, color in zip(bp['fliers'], gap_colors):
        flier.set(marker='o', markersize=2.8, markerfacecolor=color, markeredgecolor='none', alpha=0.25)
    ax.axhline(0.0, color='#444444', linewidth=1.0, linestyle='--')
    ax.set_xticks(gap_pos)
    ax.set_xticklabels(gap_labels)
    ax.set_ylabel('max positive logit - max negative logit')
    ax.set_title('B. Expression-level ranking gap')
    ax.grid(axis='y', alpha=0.18, linewidth=0.8)

    ax = axes[1, 0]
    rank_before = _clean(before['best_pos_rank_values'])
    rank_after = _clean(after['best_pos_rank_values']) if after else []
    rank_vals = rank_before + rank_after
    rank_max = int(max(rank_vals)) if rank_vals else 1
    bins = np.arange(1, rank_max + 2)
    ax.hist(rank_before, bins=bins, histtype='step', linewidth=2.2, density=True, color='#2878b5', label='Base')
    if after:
        ax.hist(rank_after, bins=bins, histtype='step', linewidth=2.2, density=True, color='#4daf4a', label='Finetuned')
    ax.set_xticks(range(1, min(rank_max, 10) + 1))
    ax.set_xlabel('Rank of the highest-scoring positive candidate')
    ax.set_ylabel('Normalized frequency')
    ax.set_title('C. Where the best positive candidate appears')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=0.18, linewidth=0.8)

    ax = axes[1, 1]
    top1_before = float(before['stats']['top1_pos_rate'])
    top1_after = float(after['stats']['top1_pos_rate']) if after else 0.0
    paired = summary.get('comparison', {})
    paired_examples = paired.get('paired_examples', [])
    rank_improved = 0.0
    gap_improved = 0.0
    flip_to_pos = float(paired.get('flip_to_pos_rate', 0.0)) if after else 0.0
    if paired_examples:
        rank_improved = sum(1 for row in paired_examples if row['rank_improvement'] > 0) / len(paired_examples)
        gap_improved = sum(1 for row in paired_examples if (row['gap_logit_delta'] or 0.0) > 0) / len(paired_examples)
    labels = ['Top-1 positive\n(base)', 'Top-1 positive\n(finetuned)', 'Rank improved', 'Gap improved']
    values = [top1_before, top1_after, rank_improved, gap_improved]
    colors = ['#2878b5', '#4daf4a', '#f28e2b', '#8e63ce']
    bars = ax.bar(range(len(labels)), values, color=colors, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Fraction of expressions')
    ax.set_title('D. Candidate-selection success rates')
    ax.grid(axis='y', alpha=0.18, linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    if after:
        ax.text(
            0.98,
            0.05,
            f'Flip-to-positive rate: {flip_to_pos:.2f}',
            ha='right',
            va='bottom',
            transform=ax.transAxes,
            fontsize=9,
            color='#333333',
        )

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        f'Phase-2 verifier analysis on {summary["dataset_name"]} / {summary["split"]} '
        f'({summary["num_candidate_sets"]} expression sets)',
        y=0.98,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze L_cls/L_rank support statistics from offline-cache candidate sets.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset-name', default=None)
    parser.add_argument('--cache-root', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--finetuned-model-path', default=None)
    parser.add_argument('--max-expressions', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-frames', type=int, default=None)
    parser.add_argument('--mask-style', default=None, choices=['mask', 'bbox'])
    parser.add_argument('--device', default=None)
    parser.add_argument('--dtype', default=None)
    parser.add_argument('--frame-mode', choices=['sampled', 'all'], default='sampled')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    context = load_analysis_context(
        config_path=args.config,
        repo_root=repo_root,
        dataset_name=args.dataset_name,
        cache_root=args.cache_root,
        split=args.split,
        num_frames=args.num_frames,
        mask_style=args.mask_style,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    exp_dirs = discover_expression_dirs(context.cache_root, context.split)

    samples = []
    for exp_dir in exp_dirs:
        sample = load_expression_data(exp_dir, context, frame_mode=args.frame_mode)
        if sample is None or not sample.pos_obj_ids or not sample.neg_obj_ids:
            continue
        samples.append(sample)

    samples = sample_expression_dirs(samples, args.max_expressions, args.seed)

    summary: Dict = {
        'config': os.path.abspath(args.config),
        'dataset_name': context.dataset_name,
        'cache_root': context.cache_root,
        'split': context.split,
        'num_candidate_sets': len(samples),
        'pos_iou': context.pos_iou,
        'neg_iou': context.neg_iou,
        'models': [],
    }

    detailed = {}
    for tag, model_path, adapter_path in resolve_model_runs(context.base_model_path, args.finetuned_model_path):
        matcher = None
        try:
            matcher = build_runtime_matcher(context, model_path, adapter_path, args.device, args.dtype)
            positive_prob_scores = []
            negative_prob_scores = []
            positive_logit_scores = []
            negative_logit_scores = []
            gap_prob_values = []
            gap_logit_values = []
            best_pos_rank_values = []
            top1_is_pos_values = []
            expression_rows = []

            for sample in samples:
                cand_rows = []
                for cand in sample.candidates:
                    out = run_candidate_forward(matcher, sample.expression, cand.frames, cand.masks)
                    if out['prob_score'] is None or out['logit_score'] is None:
                        continue
                    row = {
                        'obj_id': cand.obj_id,
                        'obj_iou': cand.obj_iou,
                        'label': 'pos' if cand.obj_id in sample.pos_obj_ids else ('neg' if cand.obj_id in sample.neg_obj_ids else 'mid'),
                        'prob_score': out['prob_score'],
                        'logit_score': out['logit_score'],
                    }
                    cand_rows.append(row)
                    if row['label'] == 'pos':
                        positive_prob_scores.append(row['prob_score'])
                        positive_logit_scores.append(row['logit_score'])
                    elif row['label'] == 'neg':
                        negative_prob_scores.append(row['prob_score'])
                        negative_logit_scores.append(row['logit_score'])
                if not cand_rows:
                    continue
                sorted_by_logit = sorted(cand_rows, key=lambda x: x['logit_score'], reverse=True)
                top1_is_pos = int(sorted_by_logit[0]['label'] == 'pos')
                best_pos_rank = rank_of_obj([x['obj_id'] for x in sorted_by_logit], sample.pos_obj_ids)
                pos_prob = [x['prob_score'] for x in cand_rows if x['label'] == 'pos']
                neg_prob = [x['prob_score'] for x in cand_rows if x['label'] == 'neg']
                pos_logit = [x['logit_score'] for x in cand_rows if x['label'] == 'pos']
                neg_logit = [x['logit_score'] for x in cand_rows if x['label'] == 'neg']
                gap_prob = max(pos_prob) - max(neg_prob) if pos_prob and neg_prob else None
                gap_logit = max(pos_logit) - max(neg_logit) if pos_logit and neg_logit else None
                if gap_prob is not None:
                    gap_prob_values.append(gap_prob)
                if gap_logit is not None:
                    gap_logit_values.append(gap_logit)
                best_pos_rank_values.append(best_pos_rank)
                top1_is_pos_values.append(top1_is_pos)
                expression_rows.append({
                    'video_id': sample.video_id,
                    'exp_id': sample.exp_id,
                    'expression': sample.expression,
                    'num_candidates': len(cand_rows),
                    'top1_is_pos': bool(top1_is_pos),
                    'best_pos_rank': best_pos_rank,
                    'gap_prob': gap_prob,
                    'gap_logit': gap_logit,
                    'candidates': cand_rows,
                })

            detailed[tag] = expression_rows
            summary[tag] = {
                'model_path': model_path,
                'adapter_path': adapter_path,
                'num_expressions': len(expression_rows),
                'positive_prob_scores': positive_prob_scores,
                'negative_prob_scores': negative_prob_scores,
                'positive_logit_scores': positive_logit_scores,
                'negative_logit_scores': negative_logit_scores,
                'gap_prob_values': gap_prob_values,
                'gap_logit_values': gap_logit_values,
                'best_pos_rank_values': best_pos_rank_values,
                'top1_is_pos_values': top1_is_pos_values,
                'stats': {
                    'positive_prob': summarize_values(positive_prob_scores),
                    'negative_prob': summarize_values(negative_prob_scores),
                    'positive_logit': summarize_values(positive_logit_scores),
                    'negative_logit': summarize_values(negative_logit_scores),
                    'gap_prob': summarize_values(gap_prob_values),
                    'gap_logit': summarize_values(gap_logit_values),
                    'best_pos_rank': summarize_values(best_pos_rank_values),
                    'top1_pos_rate': float(sum(top1_is_pos_values) / max(len(top1_is_pos_values), 1)),
                },
            }
            summary['models'].append(tag)
        finally:
            release_model(matcher)

    if 'before' in summary and 'after' in summary:
        before_rows = detailed.get('before', [])
        after_rows = detailed.get('after', [])
        paired = []
        for b, a in zip(before_rows, after_rows):
            if b['video_id'] != a['video_id'] or b['exp_id'] != a['exp_id']:
                continue
            paired.append({
                'video_id': b['video_id'],
                'exp_id': b['exp_id'],
                'gap_logit_delta': None if b['gap_logit'] is None or a['gap_logit'] is None else a['gap_logit'] - b['gap_logit'],
                'gap_prob_delta': None if b['gap_prob'] is None or a['gap_prob'] is None else a['gap_prob'] - b['gap_prob'],
                'rank_improvement': b['best_pos_rank'] - a['best_pos_rank'],
                'top1_flip_to_pos': int((not b['top1_is_pos']) and a['top1_is_pos']),
            })
        summary['comparison'] = {
            'num_paired': len(paired),
            'gap_logit_delta': summarize_values([x['gap_logit_delta'] for x in paired if x['gap_logit_delta'] is not None]),
            'gap_prob_delta': summarize_values([x['gap_prob_delta'] for x in paired if x['gap_prob_delta'] is not None]),
            'rank_improvement': summarize_values([x['rank_improvement'] for x in paired]),
            'flip_to_pos_rate': float(sum(x['top1_flip_to_pos'] for x in paired) / max(len(paired), 1)),
            'paired_examples': paired[:50],
        }

    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'details.json'), 'w', encoding='utf-8') as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    maybe_plot(os.path.join(args.output_dir, 'overview.png'), summary)
    _save_paper_figures(args.output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
