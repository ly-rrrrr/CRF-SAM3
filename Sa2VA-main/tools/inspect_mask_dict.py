#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def load_payload(path: Path) -> Dict[Any, Any]:
    if path.suffix == '.pkl':
        with path.open('rb') as f:
            payload = pickle.load(f)
    elif path.suffix == '.json':
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
    else:
        raise ValueError(f'Unsupported file type: {path}')
    if not isinstance(payload, dict):
        raise TypeError(f'Expected dict payload, got {type(payload).__name__}')
    return payload


def load_meta(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except Exception:
            return obj.decode('latin1')
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


def key_sort_value(key: Any):
    try:
        return (0, int(key))
    except Exception:
        return (1, str(key))


def canonicalize_payload(payload: Dict[Any, Any]) -> Dict[Any, Any]:
    return {key: payload[key] for key in sorted(payload.keys(), key=key_sort_value)}


def summarize(payload: Dict[Any, Any]) -> Dict[str, Any]:
    keys = list(payload.keys())
    first_key = keys[0] if keys else None
    first_value = payload[first_key] if first_key is not None else None
    summary = {
        'entries': len(payload),
        'first_10_keys': [str(k) for k in keys[:10]],
        'first_key_type': type(first_key).__name__ if first_key is not None else None,
        'first_value_type': type(first_value).__name__ if first_value is not None else None,
    }
    if isinstance(first_value, list):
        summary['first_value_len'] = len(first_value)
        summary['first_item_type'] = type(first_value[0]).__name__ if first_value else None
        summary['first_item_preview'] = to_jsonable(first_value[0]) if first_value else None
    return summary


def iter_expression_index(meta: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    idx = 0
    for video_id, video_meta in meta.get('videos', {}).items():
        frames = video_meta.get('frames', [])
        for exp_id, exp in video_meta.get('expressions', {}).items():
            yield {
                'global_index': idx,
                'video_id': str(video_id),
                'exp_id': int(exp_id),
                'expression': exp.get('exp', ''),
                'obj_id': exp.get('obj_id', []),
                'anno_id': exp.get('anno_id', []),
                'num_video_frames': len(frames),
            }
            idx += 1


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def write_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_jsonl_entries(path: Path, payload: Dict[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for key, value in payload.items():
            row = {
                'key': str(key),
                'value': to_jsonable(value),
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write('\n')


def write_index_jsonl(path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in iter_expression_index(meta):
            f.write(json.dumps(row, ensure_ascii=False))
            f.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect or export mask_dict.pkl/json so you can examine it yourself.')
    parser.add_argument('--mask-dict-path', required=True, help='Path to mask_dict.pkl or mask_dict.json')
    parser.add_argument('--meta-path', default=None, help='Optional meta_expressions.json path for exporting global-index mapping')
    parser.add_argument('--sorted', action='store_true', help='Use canonical numeric-key ordering for summary and dump outputs')
    parser.add_argument('--print-summary', action='store_true', help='Print a compact summary to stdout')
    parser.add_argument('--print-key', default=None, help='Print one raw entry by key to stdout')
    parser.add_argument('--dump-json', default=None, help='Dump the entire mask_dict as one JSON file')
    parser.add_argument('--dump-jsonl', default=None, help='Dump one mask_dict entry per JSONL line')
    parser.add_argument('--dump-index-jsonl', default=None, help='Dump global expression index mapping as JSONL; requires --meta-path')
    parser.add_argument('--rewrite-sorted-pkl', default=None, help='Write a new .pkl whose top-level dict insertion order is canonically sorted by key')
    args = parser.parse_args()

    mask_dict_path = Path(args.mask_dict_path).resolve()
    payload = load_payload(mask_dict_path)
    meta = load_meta(Path(args.meta_path).resolve() if args.meta_path else None)
    payload_view = canonicalize_payload(payload) if args.sorted or args.rewrite_sorted_pkl else payload

    if args.print_summary or not any([args.print_key, args.dump_json, args.dump_jsonl, args.dump_index_jsonl, args.rewrite_sorted_pkl]):
        print(json.dumps({
            'mask_dict_path': str(mask_dict_path),
            'sorted_view': bool(args.sorted),
            **summarize(payload_view),
        }, ensure_ascii=False, indent=2))

    if args.print_key is not None:
        value = payload.get(args.print_key)
        if value is None:
            try:
                value = payload.get(int(args.print_key))
            except Exception:
                value = None
        if value is None:
            raise KeyError(f'Key not found: {args.print_key}')
        print(json.dumps({
            'key': str(args.print_key),
            'value': to_jsonable(value),
        }, ensure_ascii=False, indent=2))

    if args.dump_json:
        write_json(Path(args.dump_json).resolve(), to_jsonable(payload_view))

    if args.dump_jsonl:
        write_jsonl_entries(Path(args.dump_jsonl).resolve(), payload_view)

    if args.dump_index_jsonl:
        if meta is None:
            raise ValueError('--dump-index-jsonl requires --meta-path')
        write_index_jsonl(Path(args.dump_index_jsonl).resolve(), meta)

    if args.rewrite_sorted_pkl:
        write_pickle(Path(args.rewrite_sorted_pkl).resolve(), canonicalize_payload(payload))


if __name__ == '__main__':
    main()
