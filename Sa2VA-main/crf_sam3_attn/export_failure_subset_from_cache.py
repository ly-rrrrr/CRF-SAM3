#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from statistics import mean

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from crf_sam3_attn.dataset import summarize_iou_map  # noqa: E402


def iter_meta_paths(split_root: Path):
    yield from sorted(split_root.glob('*/*/meta.json'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Export bad-sample subsets from offline cache using a chosen J-based criterion.')
    parser.add_argument('--cache-root', required=True, help='Offline cache root, e.g. crf_sam3_attn/offline_cache_refytros')
    parser.add_argument('--split', default='train')
    parser.add_argument('--metric', choices=['max_j', 'sum_j', 'mean_j'], default='max_j')
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--output-json', default='')
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = (_REPO_ROOT / cache_root).resolve()
    split_root = cache_root / args.split
    if not split_root.is_dir():
        raise FileNotFoundError(f'Offline split root not found: {split_root}')

    rows = []
    for meta_path in iter_meta_paths(split_root):
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        iou_map = meta.get('iou_map', {})
        summary = summarize_iou_map(iou_map)
        score = float(summary.get(args.metric, 0.0))
        row = {
            'video_id': str(meta.get('video_id')),
            'exp_id': int(meta.get('exp_id')),
            'expression': meta.get('expression', ''),
            'category': meta.get('category', ''),
            'sam3_prompt_text': meta.get('sam3_prompt_text', meta.get('category', '')),
            'metric': args.metric,
            'metric_value': score,
            'max_j': summary['max_j'],
            'sum_j': summary['sum_j'],
            'mean_j': summary['mean_j'],
            'num_candidates': summary['num_candidates'],
            'iou_map': iou_map,
            'meta_path': str(meta_path.resolve()),
        }
        if score < args.threshold:
            rows.append(row)

    rows.sort(key=lambda r: (float(r['metric_value']), str(r['video_id']), int(r['exp_id'])))

    summary = {
        'cache_root': str(cache_root),
        'split': args.split,
        'metric': args.metric,
        'threshold': float(args.threshold),
        'num_samples': len(rows),
        'mean_metric_value': float(mean([r['metric_value'] for r in rows])) if rows else 0.0,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = (_REPO_ROOT / out_path).resolve()
    else:
        thr = str(args.threshold).replace('.', 'p')
        out_path = cache_root / f'{args.metric}_lt_{thr}_samples.json'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'summary': summary, 'samples': rows}, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Wrote subset to {out_path}')


if __name__ == '__main__':
    main()
