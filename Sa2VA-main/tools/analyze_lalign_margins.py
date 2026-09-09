import argparse
import json
import os
import sys
from typing import Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
    compute_align_margin_metrics,
    discover_expression_dirs,
    load_analysis_context,
    load_expression_data,
    release_model,
    resolve_model_runs,
    run_candidate_forward,
    sample_expression_dirs,
    summarize_values,
)


def parse_layers(text: str) -> List[int]:
    vals = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError('No layers parsed')
    return vals


def _clean(values: List[float]) -> List[float]:
    return [float(v) for v in values if v is not None]


def _style_boxplot(bp, colors: List[str]) -> None:
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, edgecolor=color, alpha=0.8)
    for key in ['whiskers', 'caps', 'medians']:
        for artist in bp[key]:
            artist.set(color='#2f2f2f', linewidth=1.1)
    for flier, color in zip(bp['fliers'], colors):
        flier.set(marker='o', markersize=2.6, markerfacecolor=color, markeredgecolor='none', alpha=0.2)


def _save_distribution_figure(out_path: str, summary: Dict, layer: int) -> None:
    if plt is None or np is None:
        return
    key = str(layer)
    before = summary['before'][key]
    after = summary.get('after', {}).get(key)
    if after is None:
        return

    b_pos = _clean(before['positive_margin_sim'])
    b_neg = _clean(before['negative_margin_sim'])
    a_pos = _clean(after['positive_margin_sim'])
    a_neg = _clean(after['negative_margin_sim'])
    b_auc_pos = _clean(before['positive_auc_sim'])
    b_auc_neg = _clean(before['negative_auc_sim'])
    a_auc_pos = _clean(after['positive_auc_sim'])
    a_auc_neg = _clean(after['negative_auc_sim'])

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    data = [b_pos, b_neg, a_pos, a_neg]
    positions = [1.0, 2.0, 4.0, 5.0]
    labels = ['Base\nPos', 'Base\nNeg', 'FT\nPos', 'FT\nNeg']
    colors = ['#4c78a8', '#c44e52', '#59a14f', '#f28e2b']
    bp = ax.boxplot(data, positions=positions, widths=0.58, patch_artist=True, showfliers=True)
    _style_boxplot(bp, colors)
    ax.axhline(0.0, color='#555555', linewidth=1.0, linestyle='--')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('mean_in_sim - mean_out_sim')
    ax.set_title(f'L_align at layer {layer}: margin_sim')
    ax.grid(axis='y', alpha=0.16, linewidth=0.8)

    ax = axes[1]
    data = [b_auc_pos, b_auc_neg, a_auc_pos, a_auc_neg]
    bp = ax.boxplot(data, positions=positions, widths=0.58, patch_artist=True, showfliers=True)
    _style_boxplot(bp, colors)
    ax.axhline(0.5, color='#555555', linewidth=1.0, linestyle='--')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('AUC[in-mask sim > out-mask sim]')
    ax.set_title(f'L_align at layer {layer}: token-pair win rate')
    ax.grid(axis='y', alpha=0.16, linewidth=0.8)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches='tight')
    plt.close(fig)


def _save_paired_figure(out_path: str, details: Dict, layer: int) -> None:
    if plt is None or np is None:
        return
    key = str(layer)
    before_rows = details[key].get('before', [])
    after_rows = details[key].get('after', [])
    pos_pairs = []
    neg_pairs = []
    for b_expr, a_expr in zip(before_rows, after_rows):
        if b_expr['video_id'] != a_expr['video_id'] or b_expr['exp_id'] != a_expr['exp_id']:
            continue
        b_map = {x['obj_id']: x for x in b_expr['candidates']}
        a_map = {x['obj_id']: x for x in a_expr['candidates']}
        for obj_id, b_row in b_map.items():
            a_row = a_map.get(obj_id)
            if a_row is None:
                continue
            pair = (float(b_row['mean_margin_sim']), float(a_row['mean_margin_sim']))
            if b_row['label'] == 'pos':
                pos_pairs.append(pair)
            elif b_row['label'] == 'neg':
                neg_pairs.append(pair)

    if not pos_pairs and not neg_pairs:
        return

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')

    def draw(ax, pairs, title, color):
        if pairs:
            arr = np.asarray(pairs, dtype=float)
            x = arr[:, 0]
            y = arr[:, 1]
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            pad = 0.05 * max(1e-6, hi - lo)
            lo -= pad
            hi += pad
            ax.scatter(x, y, s=22, alpha=0.55, c=color, edgecolors='none')
            ax.plot([lo, hi], [lo, hi], linestyle='--', color='#555555', linewidth=1.0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            rate = float((y > x).mean()) if 'positive' in title.lower() else float((y < x).mean())
            label = 'after > before rate' if 'positive' in title.lower() else 'after < before rate'
            ax.text(0.98, 0.05, f'{label}: {rate:.2f}', ha='right', va='bottom', transform=ax.transAxes, fontsize=9)
        ax.set_title(title)
        ax.grid(alpha=0.16, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Base margin_sim')

    draw(axes[0], pos_pairs, f'Positive candidates (layer {layer})', '#4c78a8')
    draw(axes[1], neg_pairs, f'Negative candidates (layer {layer})', '#c44e52')
    axes[0].set_ylabel('Finetuned margin_sim')
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches='tight')
    plt.close(fig)


def _best_layer(summary: Dict, layers: List[int]) -> int:
    if 'comparison' not in summary:
        return layers[0]
    best = None
    best_score = None
    for layer in layers:
        row = summary['comparison']['layers'][str(layer)]
        score = row['positive_delta_auc_sim']['mean'] - row['negative_delta_auc_sim']['mean']
        if best_score is None or score > best_score:
            best_score = score
            best = layer
    return best if best is not None else layers[0]


def _save_overview(out_path: str, summary: Dict, layers: List[int]) -> None:
    if plt is None or np is None or 'comparison' not in summary:
        return
    layer_labels = [str(x) for x in layers]
    pos_auc = [summary['comparison']['layers'][str(layer)]['positive_delta_auc_sim']['mean'] for layer in layers]
    neg_auc = [summary['comparison']['layers'][str(layer)]['negative_delta_auc_sim']['mean'] for layer in layers]
    pos_rate = [summary['comparison']['layers'][str(layer)]['positive_delta_auc_sim_gt0_rate'] for layer in layers]
    neg_rate = [summary['comparison']['layers'][str(layer)]['negative_delta_auc_sim_lt0_rate'] for layer in layers]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].plot(layer_labels, pos_auc, marker='o', color='#4c78a8', label='pos Δauc')
    axes[0].plot(layer_labels, neg_auc, marker='o', color='#c44e52', label='neg Δauc')
    axes[0].axhline(0.0, color='#555555', linestyle='--', linewidth=1.0)
    axes[0].set_title('Mean Δ AUC by layer')
    axes[0].set_ylabel('after - before')
    axes[0].legend(frameon=False)

    axes[1].plot(layer_labels, pos_rate, marker='o', color='#4c78a8', label='pos Δauc > 0')
    axes[1].plot(layer_labels, neg_rate, marker='o', color='#c44e52', label='neg Δauc < 0')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title('Directional consistency by layer')
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.16, linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze L_align as representation-distribution shift on candidate sets.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset-name', default=None)
    parser.add_argument('--cache-root', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--finetuned-model-path', required=True)
    parser.add_argument('--layers', default=None, help='Comma-separated layer list; default uses attn_loss_layer from config')
    parser.add_argument('--max-expressions', type=int, default=60)
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
    layers = parse_layers(args.layers) if args.layers else [int(context.model_cfg.get('attn_loss_layer', -1))]
    os.makedirs(args.output_dir, exist_ok=True)

    exp_dirs = discover_expression_dirs(context.cache_root, context.split)
    exp_dirs = sample_expression_dirs(exp_dirs, args.max_expressions, args.seed)
    samples = []
    for exp_dir in exp_dirs:
        sample = load_expression_data(exp_dir, context, frame_mode=args.frame_mode)
        if sample is None or (not sample.pos_obj_ids and not sample.neg_obj_ids):
            continue
        samples.append(sample)
    samples = sample_expression_dirs(samples, args.max_expressions, args.seed)

    summary: Dict = {
        'config': os.path.abspath(args.config),
        'dataset_name': context.dataset_name,
        'cache_root': context.cache_root,
        'split': context.split,
        'layers': layers,
        'num_candidate_sets': len(samples),
        'models': [],
    }
    details: Dict[str, Dict] = {}
    model_runs = resolve_model_runs(context.base_model_path, args.finetuned_model_path)

    for layer in layers:
        layer_key = str(layer)
        details[layer_key] = {}
        for tag, model_path, adapter_path in model_runs:
            matcher = None
            try:
                matcher = build_runtime_matcher(context, model_path, adapter_path, args.device, args.dtype, layer_override=layer)
                pos_margin_sim, neg_margin_sim = [], []
                pos_margin_prob, neg_margin_prob = [], []
                pos_auc_sim, neg_auc_sim = [], []
                pos_in_sim, neg_in_sim = [], []
                pos_out_sim, neg_out_sim = [], []
                expr_rows = []
                for sample in samples:
                    cand_rows = []
                    for cand in sample.candidates:
                        out = run_candidate_forward(matcher, sample.expression, cand.frames, cand.masks)
                        frame_metrics = compute_align_margin_metrics(
                            matcher,
                            out['inputs']['input_ids'],
                            out['prompt_len'],
                            out['capture'],
                            out['inputs'].get('image_grid_thw'),
                            cand.masks,
                        )
                        if not frame_metrics:
                            continue
                        mean_in_sim = sum(x['mean_in_sim'] for x in frame_metrics) / len(frame_metrics)
                        mean_out_sim = sum(x['mean_out_sim'] for x in frame_metrics) / len(frame_metrics)
                        mean_margin_sim = sum(x['margin_sim'] for x in frame_metrics) / len(frame_metrics)
                        mean_in_prob = sum(x['mean_in_prob'] for x in frame_metrics) / len(frame_metrics)
                        mean_out_prob = sum(x['mean_out_prob'] for x in frame_metrics) / len(frame_metrics)
                        mean_margin_prob = sum(x['margin_prob'] for x in frame_metrics) / len(frame_metrics)
                        auc_sim = sum(x['auc_sim'] for x in frame_metrics) / len(frame_metrics)
                        label = 'pos' if cand.obj_id in sample.pos_obj_ids else ('neg' if cand.obj_id in sample.neg_obj_ids else 'mid')
                        row = {
                            'obj_id': cand.obj_id,
                            'obj_iou': cand.obj_iou,
                            'label': label,
                            'mean_in_sim': mean_in_sim,
                            'mean_out_sim': mean_out_sim,
                            'mean_margin_sim': mean_margin_sim,
                            'mean_in_prob': mean_in_prob,
                            'mean_out_prob': mean_out_prob,
                            'mean_margin_prob': mean_margin_prob,
                            'mean_auc_sim': auc_sim,
                            'prob_score': out['prob_score'],
                            'logit_score': out['logit_score'],
                        }
                        cand_rows.append(row)
                        if label == 'pos':
                            pos_margin_sim.append(mean_margin_sim)
                            pos_margin_prob.append(mean_margin_prob)
                            pos_auc_sim.append(auc_sim)
                            pos_in_sim.append(mean_in_sim)
                            pos_out_sim.append(mean_out_sim)
                        elif label == 'neg':
                            neg_margin_sim.append(mean_margin_sim)
                            neg_margin_prob.append(mean_margin_prob)
                            neg_auc_sim.append(auc_sim)
                            neg_in_sim.append(mean_in_sim)
                            neg_out_sim.append(mean_out_sim)
                    if cand_rows:
                        expr_rows.append({
                            'video_id': sample.video_id,
                            'exp_id': sample.exp_id,
                            'expression': sample.expression,
                            'candidates': cand_rows,
                        })
                details[layer_key][tag] = expr_rows
                summary.setdefault(tag, {})[layer_key] = {
                    'model_path': model_path,
                    'adapter_path': adapter_path,
                    'positive_margin_sim': pos_margin_sim,
                    'negative_margin_sim': neg_margin_sim,
                    'positive_margin_prob': pos_margin_prob,
                    'negative_margin_prob': neg_margin_prob,
                    'positive_auc_sim': pos_auc_sim,
                    'negative_auc_sim': neg_auc_sim,
                    'positive_mean_in_sim': pos_in_sim,
                    'positive_mean_out_sim': pos_out_sim,
                    'negative_mean_in_sim': neg_in_sim,
                    'negative_mean_out_sim': neg_out_sim,
                    'stats': {
                        'positive_margin_sim': summarize_values(pos_margin_sim),
                        'negative_margin_sim': summarize_values(neg_margin_sim),
                        'positive_margin_prob': summarize_values(pos_margin_prob),
                        'negative_margin_prob': summarize_values(neg_margin_prob),
                        'positive_auc_sim': summarize_values(pos_auc_sim),
                        'negative_auc_sim': summarize_values(neg_auc_sim),
                    },
                }
                if tag not in summary['models']:
                    summary['models'].append(tag)
            finally:
                release_model(matcher)

    if 'before' in summary and 'after' in summary:
        comparison = {'layers': {}}
        for layer in layers:
            key = str(layer)
            before_rows = details[key].get('before', [])
            after_rows = details[key].get('after', [])
            pos_delta_sim, neg_delta_sim = [], []
            pos_delta_prob, neg_delta_prob = [], []
            pos_delta_auc, neg_delta_auc = [], []
            pos_gt0 = pos_auc_gt0 = 0
            neg_lt0 = neg_auc_lt0 = 0
            pos_count = neg_count = 0
            for b_expr, a_expr in zip(before_rows, after_rows):
                if b_expr['video_id'] != a_expr['video_id'] or b_expr['exp_id'] != a_expr['exp_id']:
                    continue
                b_map = {x['obj_id']: x for x in b_expr['candidates']}
                a_map = {x['obj_id']: x for x in a_expr['candidates']}
                for obj_id, b_row in b_map.items():
                    a_row = a_map.get(obj_id)
                    if a_row is None:
                        continue
                    delta_sim = a_row['mean_margin_sim'] - b_row['mean_margin_sim']
                    delta_prob = a_row['mean_margin_prob'] - b_row['mean_margin_prob']
                    delta_auc = a_row['mean_auc_sim'] - b_row['mean_auc_sim']
                    if b_row['label'] == 'pos':
                        pos_delta_sim.append(delta_sim)
                        pos_delta_prob.append(delta_prob)
                        pos_delta_auc.append(delta_auc)
                        pos_count += 1
                        if delta_sim > 0:
                            pos_gt0 += 1
                        if delta_auc > 0:
                            pos_auc_gt0 += 1
                    elif b_row['label'] == 'neg':
                        neg_delta_sim.append(delta_sim)
                        neg_delta_prob.append(delta_prob)
                        neg_delta_auc.append(delta_auc)
                        neg_count += 1
                        if delta_sim < 0:
                            neg_lt0 += 1
                        if delta_auc < 0:
                            neg_auc_lt0 += 1
            comparison['layers'][key] = {
                'positive_delta_margin_sim': summarize_values(pos_delta_sim),
                'negative_delta_margin_sim': summarize_values(neg_delta_sim),
                'positive_delta_margin_prob': summarize_values(pos_delta_prob),
                'negative_delta_margin_prob': summarize_values(neg_delta_prob),
                'positive_delta_auc_sim': summarize_values(pos_delta_auc),
                'negative_delta_auc_sim': summarize_values(neg_delta_auc),
                'positive_delta_margin_sim_gt0_rate': float(pos_gt0 / max(pos_count, 1)),
                'negative_delta_margin_sim_lt0_rate': float(neg_lt0 / max(neg_count, 1)),
                'positive_delta_auc_sim_gt0_rate': float(pos_auc_gt0 / max(pos_count, 1)),
                'negative_delta_auc_sim_lt0_rate': float(neg_auc_lt0 / max(neg_count, 1)),
            }
        summary['comparison'] = comparison

    best_layer = _best_layer(summary, layers)
    summary['recommended_layer'] = best_layer

    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'details.json'), 'w', encoding='utf-8') as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    _save_overview(os.path.join(args.output_dir, 'overview.png'), summary, layers)
    _save_distribution_figure(os.path.join(args.output_dir, 'paper_lalign_distribution.png'), summary, best_layer)
    _save_paired_figure(os.path.join(args.output_dir, 'paper_lalign_scatter.png'), details, best_layer)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
