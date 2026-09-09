#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import pickle
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as cocomask
except Exception:  # pragma: no cover
    cocomask = None

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLIT_NAMES = ['train', 'val', 'valid', 'valid_u', 'validation', 'test']
DEFAULT_DATA_ROOT = Path('/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas')
DEFAULT_REPO_ROOT = Path('/mnt/csip-500/share/kongjiawei/Sa2VA-main')


class FrameResolver:
    def __init__(self, data_root: Path, repo_root: Path):
        self.data_root = data_root.resolve()
        self.repo_root = repo_root.resolve()
        self.allowed_roots = [self.data_root, self.repo_root]
        self.dataset_roots = self._discover_dataset_roots()
        self._meta_cache: Dict[str, Dict] = {}
        self._mask_dict_cache: Dict[str, Any] = {}
        self._mask_dict_mode_cache: Dict[str, str] = {}
        self._expr_index_cache: Dict[str, Dict[Tuple[str, int], int]] = {}
        self._expr_mask_cache: Dict[Tuple[str, str, str, int, str], Dict[int, np.ndarray]] = {}
        self.offline_roots = self._discover_offline_roots()

    def _discover_dataset_roots(self) -> Dict[str, List[Path]]:
        groups = {
            'davis17': ['davis17', 'DAVIS17'],
            'mevis': ['mevis', 'MeViS', 'MeViSv2'],
            'refytvos': ['Ref-Youtube-VOS', 'ref-youtube-vos', 'ref_youtube_vos'],
        }
        out: Dict[str, List[Path]] = {}
        for key, names in groups.items():
            roots = []
            for name in names:
                p = self.data_root / name
                if p.is_dir():
                    roots.append(p.resolve())
            out[key] = roots
        return out

    def _discover_offline_roots(self) -> Dict[str, Path]:
        candidates = {
            'offline_cache_davis17': self.repo_root / 'crf_sam3_attn' / 'offline_cache_davis17',
            'offline_cache_mevis': self.repo_root / 'crf_sam3_attn' / 'offline_cache_mevis',
            'offline_cache_refytros': self.repo_root / 'crf_sam3_attn' / 'offline_cache_refytros',
            'offline_cache_refytros_teacher_prompt_failures': self.repo_root / 'crf_sam3_attn' / 'offline_cache_refytros_teacher_prompt_failures',
            'offline_cache_refytros_gemma4_teacher_prompt_failures': self.repo_root / 'crf_sam3_attn' / 'offline_cache_refytros_gemma4_teacher_prompt_failures',
        }
        return {k: v.resolve() for k, v in candidates.items() if v.exists()}

    def _image_files(self, folder: Path) -> List[Path]:
        return [
            p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

    def _safe_relative(self, path: Optional[Path], base: Optional[Path] = None) -> Optional[str]:
        if path is None:
            return None
        path = path.resolve()
        bases = [base.resolve()] if base is not None else self.allowed_roots
        for root in bases:
            try:
                return str(path.relative_to(root))
            except ValueError:
                continue
        return str(path)

    def _iter_candidate_dirs(self, dataset_key: str, root: Path, video_id: str) -> Iterable[tuple[str, Path]]:
        yield ('root', root / video_id)
        yield ('root/JPEGImages', root / 'JPEGImages' / video_id)
        for split in SPLIT_NAMES:
            yield (split, root / split / video_id)
            yield (f'{split}/JPEGImages', root / split / 'JPEGImages' / video_id)
            yield (f'meta_expressions/{split}/JPEGImages', root / 'meta_expressions' / split / 'JPEGImages' / video_id)
        if dataset_key == 'refytvos':
            yield ('nested21/root', root / '21' / video_id)
            yield ('nested21/train/JPEGImages', root / '21' / 'train' / 'JPEGImages' / video_id)

    def _find_recursive(self, roots: List[Path], video_id: str, limit: int = 10) -> List[Path]:
        matches: List[Path] = []
        for root in roots:
            try:
                for cur, _, _ in os.walk(root):
                    if os.path.basename(cur) != video_id:
                        continue
                    frame_dir = Path(cur)
                    if self._image_files(frame_dir):
                        matches.append(frame_dir.resolve())
                        if len(matches) >= limit:
                            return matches
            except Exception:
                continue
        return matches

    def _validate_path_under_allowed_roots(self, path_str: str, must_be_file: bool = False, must_be_dir: bool = False) -> Optional[Path]:
        try:
            path = Path(path_str).resolve()
        except Exception:
            return None
        if must_be_file and not path.is_file():
            return None
        if must_be_dir and not path.is_dir():
            return None
        for root in self.allowed_roots:
            try:
                path.relative_to(root)
                return path
            except ValueError:
                continue
        return None

    def validate_file(self, path_str: str) -> Optional[Path]:
        path = self._validate_path_under_allowed_roots(path_str, must_be_file=True)
        if path is None or path.suffix.lower() not in IMAGE_EXTS:
            return None
        return path

    def _split_hint_from_frame_dir(self, root: Path, frame_dir: Path) -> Optional[str]:
        try:
            rel = frame_dir.resolve().relative_to(root.resolve())
        except ValueError:
            return None
        parts = rel.parts
        if parts and parts[0] in SPLIT_NAMES:
            return parts[0]
        return None

    def _candidate_meta_paths(self, dataset_key: str, root: Path, split_hint: Optional[str]) -> List[Path]:
        candidates: List[Path] = []
        if split_hint:
            candidates.extend([
                root / split_hint / 'meta_expressions.json',
                root / 'meta_expressions' / split_hint / 'meta_expressions.json',
            ])
        for split in SPLIT_NAMES:
            candidates.extend([
                root / split / 'meta_expressions.json',
                root / 'meta_expressions' / split / 'meta_expressions.json',
            ])
        candidates.append(root / 'meta_expressions.json')
        seen: set[str] = set()
        unique = []
        for p in candidates:
            real = str(p)
            if real in seen:
                continue
            seen.add(real)
            unique.append(p)
        return unique

    def _load_meta(self, dataset_key: str, root: Path, video_id: str, split_hint: Optional[str]) -> tuple[Optional[Dict], Optional[Path]]:
        for meta_path in self._candidate_meta_paths(dataset_key, root, split_hint):
            if not meta_path.exists():
                continue
            key = str(meta_path.resolve())
            if key not in self._meta_cache:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self._meta_cache[key] = json.load(f)
            meta = self._meta_cache[key]
            if video_id in meta.get('videos', {}):
                return meta, meta_path.resolve()
        return None, None

    @staticmethod
    def _expr_text(exp: Any) -> str:
        if isinstance(exp, str):
            return exp
        if isinstance(exp, dict):
            for key in ('exp', 'expression', 'text', 'sentence', 'sent'):
                value = exp.get(key)
                if isinstance(value, str):
                    return value
        return ''

    @staticmethod
    def _ensure_int_list(value: Any) -> List[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            out = []
            for x in value:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out
        try:
            return [int(value)]
        except Exception:
            return []

    def _load_expression_meta(
        self,
        dataset_key: str,
        root: Path,
        video_id: str,
        exp_id: Optional[int],
        split_hint: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if exp_id is None:
            return None
        meta, meta_path = self._load_meta(dataset_key, root, video_id, split_hint)
        if meta is None:
            return None
        exp = meta.get('videos', {}).get(video_id, {}).get('expressions', {}).get(str(exp_id))
        if exp is None:
            return None
        obj_ids = self._ensure_int_list(exp.get('obj_id') if isinstance(exp, dict) else None)
        anno_ids = self._ensure_int_list(exp.get('anno_id') if isinstance(exp, dict) else None)
        return {
            'exp_id': exp_id,
            'expression': self._expr_text(exp),
            'obj_ids': obj_ids,
            'anno_ids': anno_ids,
            'meta_path': str(meta_path) if meta_path else None,
            'meta_path_rel': self._safe_relative(meta_path, self.data_root) if meta_path else None,
            'raw': exp if isinstance(exp, dict) else {'exp': exp},
        }

    def _resolve_mask_dir(self, dataset_key: str, root: Path, frame_dir: Path, video_id: str) -> Optional[Path]:
        frame_dir = frame_dir.resolve()
        candidates: List[Path] = []
        try:
            rel = frame_dir.relative_to(root.resolve())
        except ValueError:
            rel = None

        if rel is not None:
            parts = list(rel.parts)
            if 'JPEGImages' in parts:
                idx = parts.index('JPEGImages')
                replaced = parts[:]
                replaced[idx] = 'Annotations'
                candidates.append(root / Path(*replaced))
            if len(parts) >= 2 and parts[0] in SPLIT_NAMES:
                split = parts[0]
                candidates.append(root / split / 'Annotations' / video_id)

        candidates.extend([
            frame_dir.parent.with_name('Annotations') / video_id,
            frame_dir.parent.parent / 'Annotations' / video_id if frame_dir.parent.parent != frame_dir.parent else frame_dir.parent / 'Annotations' / video_id,
            root / 'Annotations' / video_id,
            root / 'train' / 'Annotations' / video_id,
            root / 'valid' / 'Annotations' / video_id,
            root / 'val' / 'Annotations' / video_id,
        ])

        if dataset_key == 'refytvos':
            candidates.extend([
                root / '21' / 'train' / 'Annotations' / video_id,
                root / '21' / 'Annotations' / video_id,
            ])

        seen: set[str] = set()
        for candidate in candidates:
            try:
                real = str(candidate.resolve())
            except Exception:
                continue
            if real in seen:
                continue
            seen.add(real)
            if candidate.is_dir() and self._image_files(candidate):
                return candidate.resolve()
        return None

    def _candidate_mask_dict_paths(self, root: Path, split_hint: Optional[str]) -> List[Path]:
        candidates: List[Path] = []
        if split_hint:
            candidates.extend([
                root / split_hint / 'mask_dict.pkl',
                root / split_hint / 'mask_dict.json',
            ])
        for split in SPLIT_NAMES:
            candidates.extend([
                root / split / 'mask_dict.pkl',
                root / split / 'mask_dict.json',
            ])
        candidates.extend([
            root / 'mask_dict.pkl',
            root / 'mask_dict.json',
        ])
        seen = set()
        out = []
        for p in candidates:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _load_mask_dict(self, root: Path, split_hint: Optional[str]) -> tuple[Optional[Any], Optional[str], Optional[Path]]:
        for path in self._candidate_mask_dict_paths(root, split_hint):
            if not path.exists():
                continue
            key = str(path.resolve())
            if key not in self._mask_dict_cache:
                if path.suffix == '.pkl':
                    with open(path, 'rb') as f:
                        self._mask_dict_cache[key] = pickle.load(f)
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._mask_dict_cache[key] = json.load(f)
            if key not in self._mask_dict_mode_cache:
                meta, _ = self._load_meta('unknown', root, next(iter(self._meta_cache.values()))['videos'] if False else '', split_hint)
                # mode inference is independent of dataset_key string.
                mode = self._infer_mask_dict_mode(root, split_hint, self._mask_dict_cache[key])
                self._mask_dict_mode_cache[key] = mode
            return self._mask_dict_cache[key], self._mask_dict_mode_cache[key], path.resolve()
        return None, None, None

    def _infer_mask_dict_mode(self, root: Path, split_hint: Optional[str], mask_dict: Dict[Any, Any]) -> str:
        if not isinstance(mask_dict, dict) or not mask_dict:
            return 'obj_or_anno'
        int_keys = []
        for key in mask_dict.keys():
            try:
                int_keys.append(int(key))
            except Exception:
                return 'obj_or_anno'
        total_expr = self._count_total_expressions(root, split_hint)
        if total_expr > 0 and len(int_keys) == total_expr and min(int_keys) == 0 and max(int_keys) == total_expr - 1:
            return 'expression_index'
        return 'obj_or_anno'

    def _count_total_expressions(self, root: Path, split_hint: Optional[str]) -> int:
        count = 0
        seen_meta = set()
        for meta_path in self._candidate_meta_paths('unknown', root, split_hint):
            if not meta_path.exists():
                continue
            key = str(meta_path.resolve())
            if key in seen_meta:
                continue
            seen_meta.add(key)
            if key not in self._meta_cache:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self._meta_cache[key] = json.load(f)
            meta = self._meta_cache[key]
            count = sum(len(v.get('expressions', {})) for v in meta.get('videos', {}).values())
            if count > 0:
                return count
        return 0

    def _build_expression_global_index(self, root: Path, split_hint: Optional[str]) -> Dict[Tuple[str, int], int]:
        cache_key = f'{root.resolve()}::{split_hint or ""}'
        if cache_key in self._expr_index_cache:
            return self._expr_index_cache[cache_key]
        index_map: Dict[Tuple[str, int], int] = {}
        idx = 0
        for meta_path in self._candidate_meta_paths('unknown', root, split_hint):
            if not meta_path.exists():
                continue
            key = str(meta_path.resolve())
            if key not in self._meta_cache:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self._meta_cache[key] = json.load(f)
            meta = self._meta_cache[key]
            for vid, video_meta in meta.get('videos', {}).items():
                for exp_id in video_meta.get('expressions', {}).keys():
                    index_map[(str(vid), int(exp_id))] = idx
                    idx += 1
            if index_map:
                break
        self._expr_index_cache[cache_key] = index_map
        return index_map

    @staticmethod
    def _decode_mask_rle(mask_rle: Any) -> Optional[np.ndarray]:
        if cocomask is None or not mask_rle:
            return None
        if isinstance(mask_rle, (list, tuple)):
            merged = None
            for item in mask_rle:
                decoded = FrameResolver._decode_mask_rle(item)
                if decoded is None:
                    continue
                merged = decoded.astype(np.uint8) if merged is None else np.logical_or(merged > 0, decoded > 0).astype(np.uint8)
            return merged
        if isinstance(mask_rle, dict):
            counts = mask_rle.get('counts')
            if isinstance(counts, str):
                mask_rle = dict(mask_rle)
                mask_rle['counts'] = counts.encode('utf-8')
        decoded = cocomask.decode(mask_rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2).astype(np.uint8)
        return decoded.astype(np.uint8)

    def _load_expression_mask_sequence(
        self,
        dataset_key: str,
        root: Path,
        video_id: str,
        exp_id: int,
        split_hint: Optional[str],
        exp_meta: Dict[str, Any],
    ) -> tuple[Dict[int, np.ndarray], str, Optional[Path]]:
        cache_key = (dataset_key, str(root.resolve()), video_id, int(exp_id), split_hint or '')
        if cache_key in self._expr_mask_cache:
            return self._expr_mask_cache[cache_key], 'cache', None

        mask_dict, mode, mask_dict_path = self._load_mask_dict(root, split_hint)
        if not isinstance(mask_dict, dict) or not mask_dict:
            self._expr_mask_cache[cache_key] = {}
            return {}, 'none', None

        gt_masks: Dict[int, np.ndarray] = {}
        if mode == 'expression_index':
            expr_index = self._build_expression_global_index(root, split_hint).get((video_id, int(exp_id)))
            if expr_index is None:
                self._expr_mask_cache[cache_key] = {}
                return {}, mode, mask_dict_path
            seq = mask_dict.get(str(expr_index))
            if seq is None:
                seq = mask_dict.get(expr_index)
            if not isinstance(seq, list):
                self._expr_mask_cache[cache_key] = {}
                return {}, mode, mask_dict_path
            for frame_idx, item in enumerate(seq):
                decoded = self._decode_mask_rle(item)
                if decoded is not None:
                    gt_masks[frame_idx] = decoded
        else:
            target_ids = exp_meta.get('anno_ids') or exp_meta.get('obj_ids') or []
            if not target_ids:
                self._expr_mask_cache[cache_key] = {}
                return {}, mode, mask_dict_path
            max_len = 0
            seqs = []
            for target_id in target_ids:
                seq = mask_dict.get(str(target_id))
                if seq is None:
                    seq = mask_dict.get(target_id)
                if isinstance(seq, list):
                    seqs.append(seq)
                    max_len = max(max_len, len(seq))
            for frame_idx in range(max_len):
                merged = None
                for seq in seqs:
                    if frame_idx >= len(seq):
                        continue
                    decoded = self._decode_mask_rle(seq[frame_idx])
                    if decoded is None:
                        continue
                    merged = decoded.astype(np.uint8) if merged is None else np.logical_or(merged > 0, decoded > 0).astype(np.uint8)
                if merged is not None:
                    gt_masks[frame_idx] = merged

        self._expr_mask_cache[cache_key] = gt_masks
        return gt_masks, mode or 'obj_or_anno', mask_dict_path

    def _make_offline_meta_summaries(self, dataset_key: str, video_id: str, exp_id: Optional[int]) -> List[Dict[str, Any]]:
        if exp_id is None:
            return []
        rows = []
        for name, root in self.offline_roots.items():
            found = None
            for split in SPLIT_NAMES:
                meta_path = root / split / video_id / str(exp_id) / 'meta.json'
                if meta_path.exists():
                    found = meta_path
                    break
            if found is None:
                continue
            with open(found, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            iou_map = meta.get('iou_map', {})
            max_iou = 0.0
            try:
                vals = [float(v) for v in iou_map.values()]
                max_iou = max(vals) if vals else 0.0
            except Exception:
                pass
            rows.append({
                'cache_name': name,
                'meta_path': str(found),
                'meta_path_rel': self._safe_relative(found, self.repo_root),
                'category': meta.get('category'),
                'sam3_prompt_source': meta.get('sam3_prompt_source'),
                'sam3_prompt_text': meta.get('sam3_prompt_text'),
                'max_iou': max_iou,
                'obj_ids': meta.get('obj_ids', []),
            })
        return rows

    def _build_record(self, dataset_key: str, location: str, root: Path, frame_dir: Path, exp_id: Optional[int]) -> dict:
        video_id = frame_dir.name
        split_hint = self._split_hint_from_frame_dir(root, frame_dir)
        frames = self._image_files(frame_dir)
        mask_dir = self._resolve_mask_dir(dataset_key, root, frame_dir, video_id)
        raw_mask_files = self._image_files(mask_dir) if mask_dir else []
        raw_mask_by_stem = {p.stem: p for p in raw_mask_files}

        exp_meta = self._load_expression_meta(dataset_key, root, video_id, exp_id, split_hint)
        expr_mask_sequence: Dict[int, np.ndarray] = {}
        expr_mask_source = 'none'
        expr_mask_path_rel = None
        expr_mask_stems: set[str] = set()
        if exp_meta is not None:
            expr_mask_sequence, expr_mask_source, expr_mask_path = self._load_expression_mask_sequence(
                dataset_key, root, video_id, exp_id, split_hint, exp_meta
            )
            expr_mask_path_rel = self._safe_relative(expr_mask_path) if expr_mask_path else None
            for idx in expr_mask_sequence.keys():
                if 0 <= idx < len(frames):
                    expr_mask_stems.add(frames[idx].stem)
                else:
                    expr_mask_stems.add(f'{idx:05d}')

        pairs = []
        paired_count = 0
        expr_paired_count = 0
        for frame in frames:
            raw_mask = raw_mask_by_stem.get(frame.stem)
            if raw_mask is not None:
                paired_count += 1
            expr_mask_url = None
            if exp_meta is not None and frame.stem in expr_mask_stems:
                expr_paired_count += 1
                expr_mask_url = (
                    f"/expr-mask?dataset={quote(dataset_key)}&video_id={quote(video_id)}&exp_id={exp_meta['exp_id']}"
                    f"&frame_path={quote(str(frame.resolve()))}"
                )
            pairs.append({
                'frame_name': frame.name,
                'frame_url': f"/file?path={quote(str(frame.resolve()))}",
                'mask_name': raw_mask.name if raw_mask else None,
                'mask_url': f"/file?path={quote(str(raw_mask.resolve()))}" if raw_mask else None,
                'expr_mask_url': expr_mask_url,
            })

        return {
            'dataset': dataset_key,
            'location': location,
            'video_id': video_id,
            'frame_dir': str(frame_dir.resolve()),
            'frame_dir_rel': self._safe_relative(frame_dir),
            'frame_count': len(frames),
            'mask_dir': str(mask_dir) if mask_dir else None,
            'mask_dir_rel': self._safe_relative(mask_dir) if mask_dir else None,
            'mask_count': len(raw_mask_files),
            'paired_count': paired_count,
            'exp_meta': exp_meta,
            'expr_mask_source': expr_mask_source,
            'expr_mask_path_rel': expr_mask_path_rel,
            'expr_mask_count': len(expr_mask_stems) if expr_mask_stems else len(expr_mask_sequence),
            'expr_paired_count': expr_paired_count,
            'offline_meta_hits': self._make_offline_meta_summaries(dataset_key, video_id, exp_id),
            'pairs': pairs,
        }

    def resolve(self, video_id: str, dataset: str = 'auto', exp_id: Optional[int] = None) -> List[dict]:
        dataset = dataset.lower().strip()
        if not video_id:
            return []

        selected_keys = list(self.dataset_roots.keys()) if dataset in {'', 'auto', 'all'} else [dataset]
        results: List[dict] = []
        seen_dirs: set[str] = set()

        for dataset_key in selected_keys:
            roots = self.dataset_roots.get(dataset_key, [])
            for root in roots:
                for location, candidate in self._iter_candidate_dirs(dataset_key, root, video_id):
                    if not candidate.is_dir():
                        continue
                    frames = self._image_files(candidate)
                    if not frames:
                        continue
                    real = str(candidate.resolve())
                    if real in seen_dirs:
                        continue
                    seen_dirs.add(real)
                    results.append(self._build_record(dataset_key, location, root, candidate.resolve(), exp_id))

        if results:
            return results

        fallback_keys = selected_keys if dataset not in {'', 'auto', 'all'} else list(self.dataset_roots.keys())
        for dataset_key in fallback_keys:
            roots = self.dataset_roots.get(dataset_key, [])
            for root in roots:
                for frame_dir in self._find_recursive([root], video_id):
                    real = str(frame_dir.resolve())
                    if real in seen_dirs:
                        continue
                    seen_dirs.add(real)
                    results.append(self._build_record(dataset_key, 'recursive', root, frame_dir.resolve(), exp_id))
        return results

    def build_expression_mask_png(self, dataset_key: str, video_id: str, exp_id: int, frame_path: Path) -> Optional[bytes]:
        frame_path = frame_path.resolve()
        frame_dir = frame_path.parent
        roots = self.dataset_roots.get(dataset_key, [])
        root = None
        for candidate in roots:
            try:
                frame_dir.relative_to(candidate.resolve())
                root = candidate.resolve()
                break
            except ValueError:
                continue
        if root is None:
            return None
        split_hint = self._split_hint_from_frame_dir(root, frame_dir)
        exp_meta = self._load_expression_meta(dataset_key, root, video_id, exp_id, split_hint)
        if exp_meta is None:
            return None

        expr_seq, _, _ = self._load_expression_mask_sequence(dataset_key, root, video_id, exp_id, split_hint, exp_meta)
        frames = self._image_files(frame_dir)
        frame_idx = None
        for idx, candidate in enumerate(frames):
            if candidate.name == frame_path.name:
                frame_idx = idx
                break
        if frame_idx is None:
            try:
                frame_idx = int(frame_path.stem)
            except Exception:
                return None

        mask = expr_seq.get(frame_idx)
        if mask is not None:
            binary = (mask > 0).astype(np.uint8) * 255
            return self._mask_png_bytes(binary)

        mask_dir = self._resolve_mask_dir(dataset_key, root, frame_dir, video_id)
        target_ids = exp_meta.get('anno_ids') or exp_meta.get('obj_ids') or []
        if expr_seq:
            return None
        if mask_dir and target_ids:
            raw_mask_path = mask_dir / f'{frame_path.stem}.png'
            if raw_mask_path.exists():
                raw = np.array(Image.open(raw_mask_path))
                binary = np.isin(raw, target_ids).astype(np.uint8) * 255
                return self._mask_png_bytes(binary)
        return None

    @staticmethod
    def _mask_png_bytes(mask: np.ndarray) -> bytes:
        image = Image.fromarray(mask.astype(np.uint8), mode='L')
        bio = io.BytesIO()
        image.save(bio, format='PNG')
        return bio.getvalue()


HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Dataset Frame + Mask Viewer</title>
  <style>
    :root {
      --bg: #f3f1ea;
      --panel: #fffdf7;
      --ink: #1e1a17;
      --muted: #6f655d;
      --line: #d8cfc4;
      --accent: #0f6d5f;
      --warn: #d95f3f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, \"Times New Roman\", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217,95,63,.08), transparent 28%),
        radial-gradient(circle at top right, rgba(15,109,95,.08), transparent 30%),
        linear-gradient(180deg, #f8f5ee, var(--bg));
    }
    .wrap { max-width: 1680px; margin: 0 auto; padding: 28px 24px 48px; }
    .hero { display: grid; gap: 18px; margin-bottom: 26px; }
    h1 { margin: 0; font-size: clamp(28px, 4vw, 48px); line-height: 1; letter-spacing: -.03em; }
    .sub { color: var(--muted); font-size: 16px; max-width: 88ch; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,.04);
    }
    form { display: grid; grid-template-columns: minmax(220px, 1.2fr) 140px 180px 120px; gap: 12px; }
    input, select, button {
      width: 100%; border-radius: 14px; border: 1px solid var(--line);
      padding: 12px 14px; font-size: 16px; background: #fff;
    }
    button { border: none; background: var(--accent); color: white; font-weight: 600; cursor: pointer; }
    button:hover { background: #0c5c50; }
    .status { margin: 14px 0 0; color: var(--muted); min-height: 24px; }
    .results { display: grid; gap: 20px; margin-top: 28px; }
    .match-head { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: baseline; margin-bottom: 14px; }
    .tag {
      display: inline-flex; align-items: center; gap: 6px;
      border: 1px solid var(--line); border-radius: 999px; padding: 5px 10px;
      background: #fff; color: var(--muted); font-size: 13px;
    }
    .meta { color: var(--muted); font-size: 14px; margin-bottom: 10px; word-break: break-word; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-bottom: 16px; }
    .meta-card { border: 1px solid var(--line); border-radius: 16px; background: #fff; padding: 12px 14px; }
    .meta-card h3 { margin: 0 0 10px; font-size: 16px; }
    .kv { display: grid; gap: 6px; font-size: 14px; color: var(--muted); }
    .kv strong { color: var(--ink); }
    .frame-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; }
    .pair-card { border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: #fff; }
    .pair-images { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); }
    .slot { background: #f7f1e8; min-height: 180px; display: flex; flex-direction: column; }
    .slot-label { padding: 6px 10px; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--line); background: rgba(255,255,255,.8); }
    .slot a { display: block; }
    img { width: 100%; height: 220px; object-fit: contain; display: block; background: #ede5d8; }
    .missing-mask {
      flex: 1; display: grid; place-items: center; color: var(--warn); font-size: 13px; padding: 12px;
      background: repeating-linear-gradient(135deg, #f6efe5, #f6efe5 12px, #efe5d8 12px, #efe5d8 24px);
      text-align: center;
    }
    .caption { padding: 10px 12px; font-size: 13px; color: var(--muted); word-break: break-all; }
    .empty { padding: 24px; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 16px; }
    code { background: rgba(0,0,0,.04); padding: 2px 6px; border-radius: 8px; }
    @media (max-width: 760px) {
      form { grid-template-columns: 1fr; }
      .pair-images { grid-template-columns: 1fr; }
      img { height: 160px; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1>Dataset Frame + Mask Viewer</h1>
      <div class=\"sub\">Query a <code>video_id</code>. If you also provide <code>exp_id</code>, the page switches to an expression-level branch: it shows the matched referring expression, object metadata, offline-cache summaries, and an expression-specific mask when that dataset provides enough annotation information.</div>
    </section>

    <section class=\"panel\">
      <form id=\"search-form\">
        <input id=\"video-id\" name=\"video_id\" placeholder=\"video_id, e.g. 05d7715782\" autocomplete=\"off\" required />
        <input id=\"exp-id\" name=\"exp_id\" placeholder=\"exp_id (optional)\" autocomplete=\"off\" />
        <select id=\"dataset\" name=\"dataset\">
          <option value=\"auto\">auto</option>
          <option value=\"refytvos\">refytvos</option>
          <option value=\"davis17\">davis17</option>
          <option value=\"mevis\">mevis</option>
        </select>
        <button type=\"submit\">Open</button>
      </form>
      <div class=\"status\" id=\"status\"></div>
    </section>

    <section class=\"results\" id=\"results\"></section>
  </div>

  <script>
    const statusEl = document.getElementById('status');
    const resultsEl = document.getElementById('results');
    const form = document.getElementById('search-form');
    const videoInput = document.getElementById('video-id');
    const expInput = document.getElementById('exp-id');
    const datasetSelect = document.getElementById('dataset');

    function setStatus(msg) { statusEl.textContent = msg; }

    function renderMeta(match) {
      const exp = match.exp_meta;
      const hits = match.offline_meta_hits || [];
      const blocks = [];
      if (exp) {
        blocks.push(`
          <article class=\"meta-card\">
            <h3>Expression Branch</h3>
            <div class=\"kv\">
              <div><strong>exp_id:</strong> ${exp.exp_id}</div>
              <div><strong>expression:</strong> ${exp.expression || ''}</div>
              <div><strong>obj_ids:</strong> ${JSON.stringify(exp.obj_ids || [])}</div>
              <div><strong>anno_ids:</strong> ${JSON.stringify(exp.anno_ids || [])}</div>
              <div><strong>meta:</strong> <code>${exp.meta_path_rel || ''}</code></div>
              <div><strong>expr mask source:</strong> ${match.expr_mask_source}</div>
              <div><strong>expr mask frames:</strong> ${match.expr_mask_count}</div>
              <div><strong>expr paired with RGB:</strong> ${match.expr_paired_count}</div>
              <div><strong>expr mask path:</strong> <code>${match.expr_mask_path_rel || 'n/a'}</code></div>
            </div>
          </article>
        `);
      }
      hits.forEach((hit) => {
        blocks.push(`
          <article class=\"meta-card\">
            <h3>${hit.cache_name}</h3>
            <div class=\"kv\">
              <div><strong>category:</strong> ${hit.category ?? ''}</div>
              <div><strong>sam3 prompt source:</strong> ${hit.sam3_prompt_source ?? ''}</div>
              <div><strong>sam3 prompt text:</strong> ${hit.sam3_prompt_text ?? ''}</div>
              <div><strong>max_iou:</strong> ${hit.max_iou}</div>
              <div><strong>obj_ids:</strong> ${JSON.stringify(hit.obj_ids || [])}</div>
              <div><strong>meta:</strong> <code>${hit.meta_path_rel || ''}</code></div>
            </div>
          </article>
        `);
      });
      return blocks.length ? `<div class=\"meta-grid\">${blocks.join('')}</div>` : '';
    }

    function render(matches, videoId, expId) {
      resultsEl.innerHTML = '';
      if (!matches.length) {
        resultsEl.innerHTML = `<div class=\"empty\">No frame folder found for <code>${videoId}</code>${expId ? ` / <code>${expId}</code>` : ''}.</div>`;
        return;
      }
      matches.forEach((match) => {
        const section = document.createElement('section');
        section.className = 'panel';
        const useExprMask = !!match.exp_meta;
        const pairsHtml = match.pairs.map((pair) => {
          const chosenMaskUrl = useExprMask ? pair.expr_mask_url : pair.mask_url;
          const missingText = useExprMask ? 'No expression-specific mask for this frame' : 'No matching raw mask file for this frame';
          const maskLabel = useExprMask ? 'Mask (exp)' : 'Mask (raw)';
          return `
            <article class=\"pair-card\">
              <div class=\"pair-images\">
                <div class=\"slot\">
                  <div class=\"slot-label\">RGB</div>
                  <a href="${pair.frame_url}" target="_blank" rel="noreferrer">
                    <img loading="lazy" src="${pair.frame_url}" alt="${pair.frame_name}" />
                  </a>
                </div>
                <div class=\"slot\">
                  <div class=\"slot-label\">${maskLabel}</div>
                  ${chosenMaskUrl ? `
                    <a href="${chosenMaskUrl}" target="_blank" rel="noreferrer">
                      <img loading="lazy" src="${chosenMaskUrl}" alt="mask for ${pair.frame_name}" />
                    </a>
                  ` : `<div class=\"missing-mask\">${missingText}</div>`}
                </div>
              </div>
              <div class=\"caption\">${pair.frame_name}</div>
            </article>
          `;
        }).join('');

        const maskMeta = match.mask_dir_rel
          ? `<div class=\"meta\">Raw masks: <code>${match.mask_dir_rel}</code></div>`
          : `<div class=\"meta\">Raw masks: not found for this video folder</div>`;

        section.innerHTML = `
          <div class=\"match-head\">
            <strong>${match.dataset}</strong>
            <span class=\"tag\">${match.location}</span>
            <span class=\"tag\">${match.frame_count} frames</span>
            <span class=\"tag\">${match.mask_count} raw masks</span>
            ${match.exp_meta ? `<span class=\"tag\">exp_id ${match.exp_meta.exp_id}</span>` : ''}
          </div>
          <div class=\"meta\">Frames: <code>${match.frame_dir_rel}</code></div>
          ${maskMeta}
          ${renderMeta(match)}
          <div class=\"frame-grid\">${pairsHtml}</div>
        `;
        resultsEl.appendChild(section);
      });
    }

    async function search(videoId, dataset, expId) {
      setStatus('Resolving frame, expression branch, and mask folders...');
      resultsEl.innerHTML = '';
      const params = new URLSearchParams({ video_id: videoId, dataset });
      if (expId !== '') params.set('exp_id', expId);
      const url = `/api/resolve?${params.toString()}`;
      const res = await fetch(url);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setStatus(`Found ${data.matches.length} matching folder(s).`);
      render(data.matches, videoId, expId);
      history.replaceState(null, '', `/?${params.toString()}`);
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const videoId = videoInput.value.trim();
      const expId = expInput.value.trim();
      const dataset = datasetSelect.value;
      if (!videoId) return;
      try {
        await search(videoId, dataset, expId);
      } catch (err) {
        setStatus(`Request failed: ${err.message}`);
      }
    });

    const params = new URLSearchParams(window.location.search);
    const videoId = params.get('video_id');
    const expId = params.get('exp_id') || '';
    const dataset = params.get('dataset') || 'auto';
    if (dataset) datasetSelect.value = dataset;
    if (videoId) {
      videoInput.value = videoId;
      expInput.value = expId;
      search(videoId, dataset, expId).catch((err) => setStatus(`Request failed: ${err.message}`));
    }
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    resolver: FrameResolver = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            return self._send_html(HTML_PAGE)
        if parsed.path == '/api/resolve':
            return self._handle_resolve(parsed)
        if parsed.path == '/file':
            return self._handle_file(parsed)
        if parsed.path == '/expr-mask':
            return self._handle_expr_mask(parsed)
        self.send_error(HTTPStatus.NOT_FOUND, 'Not Found')

    def log_message(self, fmt: str, *args):
        return

    def _send_html(self, content: str):
        data = content.encode('utf-8')
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_resolve(self, parsed):
        qs = parse_qs(parsed.query)
        video_id = qs.get('video_id', [''])[0].strip()
        dataset = qs.get('dataset', ['auto'])[0].strip() or 'auto'
        exp_id_raw = qs.get('exp_id', [''])[0].strip()
        exp_id = int(exp_id_raw) if exp_id_raw else None
        if not video_id:
            return self._send_json({'error': 'video_id is required'}, status=400)
        matches = self.resolver.resolve(video_id, dataset=dataset, exp_id=exp_id)
        return self._send_json({'video_id': video_id, 'dataset': dataset, 'exp_id': exp_id, 'matches': matches})

    def _handle_file(self, parsed):
        qs = parse_qs(parsed.query)
        path_str = unquote(qs.get('path', [''])[0])
        file_path = self.resolver.validate_file(path_str)
        if file_path is None:
            return self._send_json({'error': 'invalid file path'}, status=404)
        ctype, _ = mimetypes.guess_type(str(file_path))
        ctype = ctype or 'application/octet-stream'
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_expr_mask(self, parsed):
        qs = parse_qs(parsed.query)
        dataset = qs.get('dataset', [''])[0].strip().lower()
        video_id = qs.get('video_id', [''])[0].strip()
        exp_id_raw = qs.get('exp_id', [''])[0].strip()
        frame_path_str = unquote(qs.get('frame_path', [''])[0])
        if not dataset or not video_id or not exp_id_raw or not frame_path_str:
            return self._send_json({'error': 'dataset, video_id, exp_id, and frame_path are required'}, status=400)
        try:
            exp_id = int(exp_id_raw)
        except ValueError:
            return self._send_json({'error': 'invalid exp_id'}, status=400)
        frame_path = self.resolver.validate_file(frame_path_str)
        if frame_path is None:
            return self._send_json({'error': 'invalid frame path'}, status=404)
        data = self.resolver.build_expression_mask_png(dataset, video_id, exp_id, frame_path)
        if data is None:
            return self._send_json({'error': 'expression-specific mask not found'}, status=404)
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description='View dataset frame folders and expression-specific masks by video_id in a local browser.')
    parser.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT))
    parser.add_argument('--repo-root', default=str(DEFAULT_REPO_ROOT))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    repo_root = Path(args.repo_root)
    if not data_root.is_dir():
        raise SystemExit(f'data root not found: {data_root}')
    if not repo_root.is_dir():
        raise SystemExit(f'repo root not found: {repo_root}')

    resolver = FrameResolver(data_root, repo_root)
    ViewerHandler.resolver = resolver
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f'Dataset Frame Viewer running at http://{args.host}:{args.port}')
    print(f'Data root: {resolver.data_root}')
    print(f'Repo root: {resolver.repo_root}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
