import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List

KJ_ROOT = Path('/mnt/csip-500/share/kongjiawei')
PROJECT_ROOT = KJ_ROOT / 'Sa2VA-main'
DATA_ROOT = KJ_ROOT / 'Sa2VA' / 'data' / 'video_datas'
OUT_ROOT = PROJECT_ROOT / 'transfer_mirror_crf_sam3_attn_20260518'
SUBSET_ROOT = PROJECT_ROOT / 'transfer_subset_crf_sam3_attn_20260518'
OFFLINE_CACHE_TARGET_TOTAL = 200
OFFLINE_CACHE_COUNTS = {'mevis': 67, 'davis17': 67, 'refytros': 66}

FILTERED_DIRS = [
    (PROJECT_ROOT / 'crf_sam3_attn', lambda p: p.name in {'.git', '__pycache__'} or p.name.startswith('offline_cache_')),
    (PROJECT_ROOT / 'configs', lambda p: p.name == '__pycache__'),
    (PROJECT_ROOT / 'tools' / 'eval', lambda p: p.name == '__pycache__'),
    (PROJECT_ROOT / 'tools' / 'teacher_data_pipeline', lambda p: p.name == '__pycache__'),
]

SINGLE_FILES = [
    PROJECT_ROOT / 'crf_sam3_demo.py',
    PROJECT_ROOT / 'run_crf_sam3_dataset.py',
    PROJECT_ROOT / 'run_crf_sam3_davis_dataset.py',
    PROJECT_ROOT / 'run_crf_sam3_refytros_dataset.py',
    PROJECT_ROOT / 'eval_mapp_sam3.py',
    PROJECT_ROOT / 'pyproject.toml',
    PROJECT_ROOT / 'README_CRF_SAM3.md',
    PROJECT_ROOT / 'README_CRF_SAM3_DEMO.md',
    PROJECT_ROOT / 'joint_train_20260419.log',
    PROJECT_ROOT / 'logs' / 'mevis_valid_joint_train_poswin_predict.log',
    PROJECT_ROOT / 'logs' / 'mevis_valid_u_joint_train_poswin_predict.log',
    PROJECT_ROOT / 'logs' / 'mevis_valid_u_joint_train_poswin_eval.log',
    PROJECT_ROOT / 'logs' / 'davis17_valid_joint_train_poswin_predict.log',
    PROJECT_ROOT / 'logs' / 'davis17_valid_joint_train_poswin_eval.log',
    PROJECT_ROOT / 'logs' / 'refytros_valid_joint_train_poswin_predict.log',
    PROJECT_ROOT / 'logs' / 'refytros_valid_joint_train_poswin_eval.log',
    PROJECT_ROOT / 'crf_sam3_mevis_valid_u_joint_train_poswin_mergedprompt' / 'evaluation_results.json',
    PROJECT_ROOT / 'crf_sam3_mevis_valid_u_joint_train_poswin_mergedprompt' / 'scores.json',
    PROJECT_ROOT / 'crf_sam3_mevis_valid_u_joint_train_poswin_mergedprompt' / 'frame_coverage.json',
    PROJECT_ROOT / 'crf_sam3_mevis_valid_u_joint_train_poswin_mergedprompt' / 'config.yaml',
    PROJECT_ROOT / 'crf_sam3_davis17_valid_joint_train_poswin_mergedprompt' / 'evaluation_results_davis_png.json',
    PROJECT_ROOT / 'crf_sam3_davis17_valid_joint_train_poswin_mergedprompt' / 'scores.json',
    PROJECT_ROOT / 'crf_sam3_davis17_valid_joint_train_poswin_mergedprompt' / 'frame_coverage.json',
    PROJECT_ROOT / 'crf_sam3_davis17_valid_joint_train_poswin_mergedprompt' / 'config.yaml',
    PROJECT_ROOT / 'crf_sam3_davis17_valid_joint_train_poswin_mergedprompt' / 'run_summary.json',
    PROJECT_ROOT / 'crf_sam3_davis17_valid_joint_train_poswin_mergedprompt' / 'expression_category_info.json',
    PROJECT_ROOT / 'crf_sam3_refytros_valid_joint_train_poswin_mergedprompt' / 'evaluation_results_refytros_png.json',
    PROJECT_ROOT / 'crf_sam3_refytros_valid_joint_train_poswin_mergedprompt' / 'scores.json',
    PROJECT_ROOT / 'crf_sam3_refytros_valid_joint_train_poswin_mergedprompt' / 'frame_coverage.json',
    PROJECT_ROOT / 'crf_sam3_refytros_valid_joint_train_poswin_mergedprompt' / 'config.yaml',
    PROJECT_ROOT / 'crf_sam3_refytros_valid_joint_train_poswin_mergedprompt' / 'run_summary.json',
    DATA_ROOT / 'mevis' / 'train' / 'meta_expressions.json',
    DATA_ROOT / 'mevis' / 'train' / 'mask_dict.json',
    DATA_ROOT / 'mevis' / 'valid_u' / 'meta_expressions.json',
    DATA_ROOT / 'mevis' / 'valid_u' / 'mask_dict.json',
    DATA_ROOT / 'davis17' / 'meta_expressions' / 'train' / 'meta_expressions.json',
    DATA_ROOT / 'davis17' / 'meta_expressions' / 'valid' / 'meta_expressions.json',
    DATA_ROOT / 'davis17' / 'train' / 'mask_dict.pkl',
    DATA_ROOT / 'davis17' / 'valid' / 'mask_dict.pkl',
    DATA_ROOT / 'Ref-Youtube-VOS' / 'train' / 'meta_expressions.json',
    DATA_ROOT / 'Ref-Youtube-VOS' / 'train' / 'mask_dict.pkl',
    DATA_ROOT / 'Ref-Youtube-VOS' / 'valid' / 'meta_expressions_challenge.json',
]


def rel_under_root(src: Path) -> Path:
    return src.relative_to(KJ_ROOT)


def copy_file(src: Path):
    if not src.exists():
        return
    dst = OUT_ROOT / rel_under_root(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir_exact(src: Path):
    if not src.exists():
        return
    dst = OUT_ROOT / rel_under_root(src)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def copy_dir_filtered(src: Path, exclude: Callable[[Path], bool]):
    if not src.exists():
        return
    dst = OUT_ROOT / rel_under_root(src)
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(dirpath: str, names):
        base = Path(dirpath)
        ignored = []
        for name in names:
            p = base / name
            if exclude(p):
                ignored.append(name)
        return ignored

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=_ignore)


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def safe_copy_if_exists(src: Path):
    if src.exists():
        if src.is_dir():
            copy_dir_exact(src)
        else:
            copy_file(src)


def copy_eval_sample(entry: dict, subset_meta: dict):
    video_id = entry['video_id']
    exp_id = entry['exp_id']
    sampled_raw_frames = entry.get('sampled_raw_frames', [])
    sampled_gt_frames = entry.get('sampled_gt_frames', [])
    source_meta = subset_meta['source_meta']
    source_output_root = Path(subset_meta['source_output_root'])

    if 'mevis/valid_u' in source_meta:
        raw_frame_root = DATA_ROOT / 'mevis' / 'valid_u' / 'JPEGImages' / video_id
        raw_gt_root = DATA_ROOT / 'mevis' / 'valid_u' / 'Annotations' / video_id
    elif 'davis17/meta_expressions/valid' in source_meta:
        raw_frame_root = DATA_ROOT / 'davis17' / 'valid' / 'JPEGImages' / video_id
        raw_gt_root = DATA_ROOT / 'davis17' / 'valid' / 'Annotations' / video_id
    elif 'Ref-Youtube-VOS/valid' in source_meta:
        raw_frame_root = DATA_ROOT / 'Ref-Youtube-VOS' / 'valid' / 'JPEGImages' / video_id
        raw_gt_root = DATA_ROOT / 'Ref-Youtube-VOS' / 'valid' / 'Annotations' / video_id
    else:
        raise RuntimeError(source_meta)

    for name in sampled_raw_frames:
        copy_file(raw_frame_root / name)
    for name in sampled_gt_frames:
        copy_file(raw_gt_root / name)

    safe_copy_if_exists(source_output_root / 'Annotations' / video_id / exp_id)
    safe_copy_if_exists(source_output_root / 'visual_tubes' / video_id / exp_id)
    safe_copy_if_exists(source_output_root / 'expression_details' / video_id / f'{exp_id}.json')


def copy_train_sample(entry: dict, subset_meta: dict):
    video_id = entry['video_id']
    sampled_raw_frames = entry.get('sampled_raw_frames', [])
    sampled_gt_frames = entry.get('sampled_gt_frames', [])
    source_meta = subset_meta['source_meta']

    if 'mevis/train' in source_meta:
        raw_frame_root = DATA_ROOT / 'mevis' / 'train' / 'JPEGImages' / video_id
        raw_gt_root = DATA_ROOT / 'mevis' / 'train' / 'Annotations' / video_id
    elif 'davis17/meta_expressions/train' in source_meta:
        raw_frame_root = DATA_ROOT / 'davis17' / 'train' / 'JPEGImages' / video_id
        raw_gt_root = DATA_ROOT / 'davis17' / 'train' / 'Annotations' / video_id
    elif 'Ref-Youtube-VOS/train' in source_meta:
        raw_frame_root = DATA_ROOT / 'Ref-Youtube-VOS' / 'train' / 'JPEGImages' / video_id
        raw_gt_root = DATA_ROOT / 'Ref-Youtube-VOS' / 'train' / 'Annotations' / video_id
    else:
        raise RuntimeError(source_meta)

    for name in sampled_raw_frames:
        copy_file(raw_frame_root / name)
    for name in sampled_gt_frames:
        copy_file(raw_gt_root / name)


def spread_select(items: List[dict], target_count: int) -> List[dict]:
    by_video: Dict[str, List[dict]] = {}
    for item in items:
        by_video.setdefault(item['video_id'], []).append(item)
    for lst in by_video.values():
        lst.sort(key=lambda x: str(x['exp_id']))
    order = sorted(by_video.keys())
    selected = []
    round_idx = 0
    while len(selected) < target_count:
        added = False
        for vid in order:
            lst = by_video[vid]
            if round_idx < len(lst):
                selected.append(lst[round_idx])
                added = True
                if len(selected) >= target_count:
                    break
        if not added:
            break
        round_idx += 1
    return selected


def collect_offline_cache_items(root: Path, dataset: str) -> List[dict]:
    items = []
    for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for exp_dir in sorted((p for p in video_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            meta_path = exp_dir / 'meta.json'
            meta = load_json(meta_path) if meta_path.exists() else {}
            items.append({
                'dataset': dataset,
                'video_id': str(meta.get('video_id', video_dir.name)),
                'exp_id': str(meta.get('exp_id', exp_dir.name)),
                'expression': meta.get('expression', ''),
                'category': meta.get('category'),
                'sam3_prompt_text': meta.get('sam3_prompt_text'),
                'max_j': meta.get('max_j'),
                'sum_j': meta.get('sum_j'),
                'mean_j': meta.get('mean_j'),
                'source_dir': str(exp_dir),
                'relative_dir': str(rel_under_root(exp_dir)),
            })
    return items


def write_readme(path: Path):
    text = (
        'This mirror package preserves original project/data path structure under one transfer root.\n'
        'Sa2VA/data/video_datas contains sampled raw dataset content only, not full datasets.\n'
        'The exact sampled train/eval expressions are listed in sample_inventory.json under raw_dataset_samples.\n'
        'The exact sampled offline cache expression directories are listed in sample_inventory.json under offline_cache_samples.\n'
    )
    path.write_text(text, encoding='utf-8')


def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for src, exclude in FILTERED_DIRS:
        copy_dir_filtered(src, exclude)
    for src in SINGLE_FILES:
        if src.exists():
            copy_file(src)

    raw_inventory = {}
    for dataset in ['mevis', 'davis17', 'refytros']:
        eval_manifest_path = SUBSET_ROOT / dataset / 'eval' / 'subset_manifest.json'
        train_manifest_path = SUBSET_ROOT / dataset / 'train' / 'subset_manifest.json'
        eval_meta = load_json(eval_manifest_path)
        train_meta = load_json(train_manifest_path)

        for common_name in eval_meta.get('copied_output_common_files', []):
            safe_copy_if_exists(Path(eval_meta['source_output_root']) / common_name)

        for entry in eval_meta['selected_expressions']:
            copy_eval_sample(entry, eval_meta)
        for entry in train_meta['selected_expressions']:
            copy_train_sample(entry, train_meta)

        raw_inventory[dataset] = {
            'eval': {
                'count': len(eval_meta['selected_expressions']),
                'unique_video_count': len({x['video_id'] for x in eval_meta['selected_expressions']}),
                'selected_expressions': eval_meta['selected_expressions'],
                'source_meta': eval_meta['source_meta'],
            },
            'train': {
                'count': len(train_meta['selected_expressions']),
                'unique_video_count': len({x['video_id'] for x in train_meta['selected_expressions']}),
                'selected_expressions': train_meta['selected_expressions'],
                'source_meta': train_meta['source_meta'],
            }
        }

    offline_cache_roots = {
        'mevis': PROJECT_ROOT / 'crf_sam3_attn' / 'offline_cache_mevis' / 'train',
        'davis17': PROJECT_ROOT / 'crf_sam3_attn' / 'offline_cache_davis17' / 'train',
        'refytros': PROJECT_ROOT / 'crf_sam3_attn' / 'offline_cache_refytros' / 'train',
    }
    offline_inventory = {}
    for dataset, root in offline_cache_roots.items():
        items = collect_offline_cache_items(root, dataset)
        selected = spread_select(items, OFFLINE_CACHE_COUNTS[dataset])
        for item in selected:
            copy_dir_exact(Path(item['source_dir']))
        offline_inventory[dataset] = {
            'count': len(selected),
            'unique_video_count': len({x['video_id'] for x in selected}),
            'source_root': str(root),
            'selected_expressions': selected,
        }

    inventory = {
        'package_root': str(OUT_ROOT),
        'raw_dataset_samples': raw_inventory,
        'offline_cache_samples_total': OFFLINE_CACHE_TARGET_TOTAL,
        'offline_cache_samples': offline_inventory,
    }
    with (OUT_ROOT / 'sample_inventory.json').open('w', encoding='utf-8') as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)

    summary = {
        'output_root': str(OUT_ROOT),
        'raw_dataset_sample_counts': {k: {'eval': v['eval']['count'], 'train': v['train']['count']} for k, v in raw_inventory.items()},
        'offline_cache_sample_counts': {k: v['count'] for k, v in offline_inventory.items()},
        'notes_file': 'README_transfer_mirror.txt',
        'inventory_file': 'sample_inventory.json',
    }
    with (OUT_ROOT / 'transfer_mirror_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_readme(OUT_ROOT / 'README_transfer_mirror.txt')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
