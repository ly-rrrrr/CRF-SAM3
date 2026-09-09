#!/usr/bin/env python3
"""Evaluate SAM3-DMS reverse reconditioning on cached ReVOS low-recall samples."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from pycocotools import mask as cocomask

from crf_sam3_demo import DMSConceptRecall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', default='crf_sam3_attn/offline_cache_revos_api_categories_recall24_keep7_20260717')
    parser.add_argument('--split', default='train')
    parser.add_argument('--meta-path', default='/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos/meta_expressions_train_.json')
    parser.add_argument('--mask-dict-path', default='/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos_prepared/train/mask_dict.json')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--output-json', default='crf_sam3_attn/cache_reports/revos_reverse_mem7_candidate_pilot_20260719.json')
    return parser.parse_args()


def decode_rle(value):
    if not value:
        return None
    if isinstance(value, (list, tuple)):
        merged = None
        for item in value:
            decoded = decode_rle(item)
            if decoded is not None:
                merged = decoded if merged is None else np.logical_or(merged, decoded)
        return None if merged is None else merged.astype(np.uint8)
    if isinstance(value, dict) and isinstance(value.get('counts'), str):
        value = dict(value)
        value['counts'] = value['counts'].encode('utf-8')
    decoded = cocomask.decode(value)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(np.uint8)


def load_gt(meta, mask_dict, video_id: str, exp_id: int):
    video = meta[video_id]
    expression = video['expressions'][str(exp_id)]
    ids = expression.get('anno_id') or expression.get('obj_id') or []
    if isinstance(ids, (int, str)):
        ids = [ids]
    gt = {}
    for frame_idx in range(len(video.get('frames', []))):
        merged = None
        for obj_id in ids:
            seq = mask_dict.get(str(obj_id), mask_dict.get(obj_id))
            if seq is None or frame_idx >= len(seq):
                continue
            decoded = decode_rle(seq[frame_idx])
            if decoded is not None:
                merged = decoded if merged is None else np.logical_or(merged, decoded)
        if merged is not None:
            gt[frame_idx] = merged.astype(np.uint8)
    return gt


def j_score(pred, gt):
    if pred.shape != gt.shape:
        raise ValueError(f'prediction/GT shape mismatch: {pred.shape} vs {gt.shape}')
    union = np.logical_or(pred > 0, gt > 0).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred > 0, gt > 0).sum() / union)


def score_candidates(masks_by_obj, score_sums, gt, keep_top_k):
    if not masks_by_obj:
        return {}, 0.0
    ranked = sorted(score_sums, key=lambda obj_id: (-float(score_sums[obj_id]), int(obj_id)))
    kept = ranked[:keep_top_k] if keep_top_k else ranked
    values = {}
    for obj_id in kept:
        total = 0.0
        frames = masks_by_obj.get(obj_id, {})
        for frame_idx, gt_mask in gt.items():
            pred = frames.get(frame_idx)
            if pred is None:
                pred = np.zeros_like(gt_mask, dtype=np.uint8)
            total += j_score(pred, gt_mask)
        values[int(obj_id)] = total / max(1, len(gt))
    return values, max(values.values(), default=0.0)


def main() -> None:
    args = parse_args()
    cache_split = Path(args.cache_root) / args.split
    rows = []
    for path in cache_split.rglob('meta.json'):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
            row['_path'] = str(path)
            row['_first_frame'] = min(
                (min(v) for v in row.get('frame_indices', {}).values() if v),
                default=-1,
            )
            if float(row.get('max_j', 0.0)) < args.threshold:
                rows.append(row)
        except Exception:
            continue
    rows.sort(key=lambda row: (-int(row['_first_frame']), float(row.get('max_j', 0.0)), str(row['video_id']), int(row['exp_id'])))
    rows = rows[:args.limit] if args.limit > 0 else rows

    meta = json.loads(Path(args.meta_path).read_text(encoding='utf-8'))['videos']
    mask_dict = json.loads(Path(args.mask_dict_path).read_text(encoding='utf-8'))
    gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip()]
    recall = DMSConceptRecall(
        dms_repo='SAM3-DMS',
        checkpoint_path='/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt',
        gpus_to_use=gpu_ids,
        async_loading_frames=True,
        offload_video_to_cpu=True,
        offload_output_to_cpu_for_eval=True,
        reverse_recondition=True,
    )

    results = []
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        video_id, exp_id = str(row['video_id']), int(row['exp_id'])
        gt = load_gt(meta, mask_dict, video_id, exp_id)
        print(f'[{index}/{len(rows)}] {video_id}/{exp_id} old_max_j={float(row.get("max_j", 0.0)):.4f} category={row.get("category", "")}', flush=True)
        try:
            forward_masks, all_masks, score_sums = defaultdict(dict), defaultdict(dict), defaultdict(float)
            for frame_idx, frame_masks, frame_scores in recall.recall_all_instances_stream(
                video_path=str(row['video_path']),
                category=str(row['category']),
                candidate_top_k=int(row.get('candidate_recall_top_k', 24)),
                candidate_keep_top_k=(
                    int(row['candidate_keep_top_k'])
                    if row.get('candidate_keep_top_k') is not None
                    else 7
                ),
            ):
                is_reverse = bool(frame_scores) and all(float(v) == 0.0 for v in frame_scores.values())
                for obj_id, mask in frame_masks.items():
                    obj_id = int(obj_id)
                    mask = (np.asarray(mask) > 0).astype(np.uint8)
                    all_masks[obj_id][int(frame_idx)] = mask
                    if not is_reverse:
                        forward_masks[obj_id][int(frame_idx)] = mask
                    score_sums[obj_id] += float(frame_scores[obj_id])
            keep_top_k = row.get('candidate_keep_top_k')
            keep_top_k = int(keep_top_k) if keep_top_k is not None else None
            forward_iou, forward_best = score_candidates(forward_masks, score_sums, gt, keep_top_k)
            reverse_iou, reverse_best = score_candidates(all_masks, score_sums, gt, keep_top_k)
            result = {
                'video_id': video_id, 'exp_id': exp_id, 'expression': row.get('expression', ''),
                'category': row.get('category', ''), 'old_cached_max_j': float(row.get('max_j', 0.0)),
                'rerun_forward_max_j': forward_best, 'reverse_max_j': reverse_best,
                'delta': reverse_best - forward_best, 'num_candidates': len(all_masks),
                'forward_iou_map': {str(k): float(v) for k, v in forward_iou.items()},
                'reverse_iou_map': {str(k): float(v) for k, v in reverse_iou.items()},
                'status': 'ok',
            }
        except Exception as exc:
            result = {'video_id': video_id, 'exp_id': exp_id, 'status': 'error', 'error': repr(exc)}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(result)
        ok = [x for x in results if x.get('status') == 'ok']
        summary = {
            'requested': len(rows), 'completed': len(results), 'ok': len(ok),
            'mean_forward_max_j': float(np.mean([x['rerun_forward_max_j'] for x in ok])) if ok else 0.0,
            'mean_reverse_max_j': float(np.mean([x['reverse_max_j'] for x in ok])) if ok else 0.0,
            'mean_delta': float(np.mean([x['delta'] for x in ok])) if ok else 0.0,
            'improved': int(sum(x['delta'] > 1e-8 for x in ok)),
            'worse': int(sum(x['delta'] < -1e-8 for x in ok)),
            'forward_zero': int(sum(x['rerun_forward_max_j'] <= 1e-12 for x in ok)),
            'reverse_zero': int(sum(x['reverse_max_j'] <= 1e-12 for x in ok)),
        }
        output.write_text(json.dumps({'summary': summary, 'samples': results}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('[summary]', json.dumps(summary), flush=True)
    print(f'[done] {output}', flush=True)


if __name__ == '__main__':
    main()
