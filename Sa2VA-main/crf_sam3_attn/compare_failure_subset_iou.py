#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def max_iou_from_meta(meta: dict) -> float:
    iou_map = meta.get('iou_map', {})
    if not isinstance(iou_map, dict) or not iou_map:
        return 0.0
    values = []
    for v in iou_map.values():
        try:
            values.append(float(v))
        except Exception:
            continue
    return max(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare failure-subset max_iou before/after prompt swap.')
    parser.add_argument('--failure-json', required=True)
    parser.add_argument('--new-cache-root', required=True)
    parser.add_argument('--output-json', default='')
    args = parser.parse_args()

    failure_json = Path(args.failure_json)
    new_cache_root = Path(args.new_cache_root)
    if not new_cache_root.is_absolute():
        new_cache_root = (Path(__file__).resolve().parents[1] / new_cache_root).resolve()

    payload = load_json(failure_json)
    samples = payload.get('samples', payload)
    if not isinstance(samples, list):
        raise TypeError('failure-json must be a list or an object with a samples list')

    rows = []
    missing_new = 0
    for item in samples:
        if not isinstance(item, dict):
            continue
        video_id = str(item['video_id'])
        exp_id = str(item['exp_id'])
        old_max = float(item.get('max_iou', 0.0))
        new_meta = new_cache_root / 'train' / video_id / exp_id / 'meta.json'
        if not new_meta.exists():
            missing_new += 1
            rows.append({
                'video_id': video_id,
                'exp_id': exp_id,
                'expression': item.get('expression', ''),
                'old_max_iou': old_max,
                'new_max_iou': None,
                'delta': None,
                'status': 'missing_new_meta',
            })
            continue
        new_meta_obj = load_json(new_meta)
        new_max = max_iou_from_meta(new_meta_obj)
        rows.append({
            'video_id': video_id,
            'exp_id': exp_id,
            'expression': item.get('expression', ''),
            'old_max_iou': old_max,
            'new_max_iou': new_max,
            'delta': new_max - old_max,
            'old_category': item.get('sam3_prompt_text', item.get('category')),
            'new_category': new_meta_obj.get('sam3_prompt_text', new_meta_obj.get('category')),
            'status': 'ok',
        })

    valid = [r for r in rows if r['status'] == 'ok']
    improved = [r for r in valid if r['delta'] > 0]
    improved_big = [r for r in valid if r['new_max_iou'] >= 0.5]
    worsened = [r for r in valid if r['delta'] < 0]

    summary = {
        'failure_json': str(failure_json),
        'new_cache_root': str(new_cache_root),
        'total_samples': len(rows),
        'valid_compared': len(valid),
        'missing_new_meta': missing_new,
        'mean_old_max_iou': mean([r['old_max_iou'] for r in valid]) if valid else 0.0,
        'mean_new_max_iou': mean([r['new_max_iou'] for r in valid]) if valid else 0.0,
        'mean_delta': mean([r['delta'] for r in valid]) if valid else 0.0,
        'improved_count': len(improved),
        'improved_to_ge_0_5_count': len(improved_big),
        'worsened_count': len(worsened),
        'top_improvements': sorted(improved, key=lambda r: r['delta'], reverse=True)[:50],
        'still_bad': sorted(valid, key=lambda r: r['new_max_iou'])[:200],
    }

    out_path = Path(args.output_json) if args.output_json else new_cache_root / 'compare_failure_subset_iou.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'rows': rows}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Wrote comparison to {out_path}')


if __name__ == '__main__':
    main()
