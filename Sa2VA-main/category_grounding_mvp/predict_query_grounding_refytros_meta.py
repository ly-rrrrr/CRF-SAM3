#!/usr/bin/env python3
"""Predict RefYTROS meta categories with the trained query-grounding modules.

The output is a category cache consumed by CRFSAM3Pipeline/ConceptExtractor.
This script does not use GT masks; it samples RGB frames from each video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_refytvos_prompt_vs_query_grounding import (  # noqa: E402
    build_query_context,
    build_query_messages,
    load_query_config,
    pad_visual_mask,
    parse_category,
    resolve_device,
)
from train_query_grounding_qwen3vl import Qwen3VLQueryGroundingModel  # noqa: E402


def sample_frames(frame_names: List[str], num_frames: int) -> List[str]:
    if not frame_names:
        return []
    if len(frame_names) <= num_frames:
        return list(frame_names)
    idxs = []
    seen = set()
    for i in range(num_frames):
        idx = round(i * (len(frame_names) - 1) / max(num_frames - 1, 1))
        if idx not in seen:
            idxs.append(idx)
            seen.add(idx)
    for idx in range(len(frame_names)):
        if len(idxs) >= num_frames:
            break
        if idx not in seen:
            idxs.append(idx)
            seen.add(idx)
    return [frame_names[i] for i in sorted(idxs[:num_frames])]


def load_meta_records(data_root: str, split: str, meta_file: str, frames_per_sample: int) -> List[Dict[str, Any]]:
    meta_path = Path(data_root) / split / meta_file
    payload = json.loads(meta_path.read_text(encoding='utf-8'))
    records: List[Dict[str, Any]] = []
    for video_id in sorted(payload['videos'].keys()):
        video = payload['videos'][video_id]
        frames = [str(x) for x in video.get('frames', [])]
        picked = sample_frames(frames, frames_per_sample)
        frame_paths = [str(Path(data_root) / split / 'JPEGImages' / video_id / f'{fr}.jpg') for fr in picked]
        exprs = video.get('expressions', {})
        exp_ids = sorted(exprs.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        for exp_id in exp_ids:
            exp = exprs[exp_id].get('exp', '')
            records.append({
                'sample_id': f'refytros:{split}:{video_id}:{exp_id}',
                'video_id': video_id,
                'exp_id': str(exp_id),
                'expression': exp,
                'frames': picked,
                'frame_paths': frame_paths,
            })
    return records


def load_query_model(args: argparse.Namespace) -> Tuple[Any, Qwen3VLQueryGroundingModel]:
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else 'auto',
        device_map=args.device_map or None,
        local_files_only=True,
    ).eval()
    if not args.device_map:
        target_device = torch.device(getattr(args, 'device', 'cuda') if torch.cuda.is_available() else 'cpu')
        qwen.to(target_device)
    cfg = load_query_config(args.query_module_dir)
    qg = Qwen3VLQueryGroundingModel(
        qwen=qwen,
        num_queries=int(cfg.get('num_queries', 8)),
        query_dim=int(cfg.get('query_dim', 1024)),
        conditioner_layers=int(cfg.get('conditioner_layers', 2)),
        decoder_layers=int(cfg.get('decoder_layers', 2)),
        num_heads=int(cfg.get('num_heads', 8)),
        mask_dim=int(cfg.get('mask_dim', 256)),
        mask_logit_scale=cfg.get('mask_logit_scale'),
        modulator_bottleneck=int(cfg.get('modulator_bottleneck', 512)),
        lambda_mask=0.0,
        lambda_cat=0.0,
    )
    state = torch.load(Path(args.query_module_dir) / 'query_grounding_modules.pt', map_location='cpu')
    qg.expression_conditioner.load_state_dict(state['expression_conditioner'])
    qg.query_decoder.load_state_dict(state['query_decoder'])
    qg.visual_to_query.load_state_dict(state['visual_to_query'])
    qg.mask_head.load_state_dict(state['mask_head'])
    qg.modulator.load_state_dict(state['modulator'])
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    first_device = next((p.device for p in qwen.parameters() if p.device.type == 'cuda'), resolve_device(qwen))
    for module in (qg.expression_conditioner, qg.query_decoder, qg.visual_to_query, qg.mask_head, qg.modulator):
        module.to(device=first_device, dtype=dtype)
        module.eval()
    qg.eval()
    return processor, qg


@torch.no_grad()
def predict_one(
    processor: Any,
    qg: Qwen3VLQueryGroundingModel,
    record: Dict[str, Any],
    max_new_tokens: int,
) -> Tuple[str, str, Dict[str, float]]:
    prompt = processor.apply_chat_template(
        build_query_messages(record),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
        processor_kwargs={'do_sample_frames': False},
    ).to(resolve_device(qg.qwen))
    expr = processor.tokenizer(record['expression'], return_tensors='pt', add_special_tokens=True)
    batch: Dict[str, torch.Tensor] = {
        'input_ids': prompt['input_ids'],
        'attention_mask': prompt['attention_mask'],
        'expression_input_ids': expr['input_ids'].to(prompt['input_ids'].device),
        'expression_attention_mask': expr['attention_mask'].to(prompt['input_ids'].device),
    }
    for key in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'second_per_grid_ts'):
        if key in prompt and torch.is_tensor(prompt[key]):
            batch[key] = prompt[key]
    visual_pos_masks, modulated_visual, deepstack_visual_embeds, stats = build_query_context(qg, batch)

    qwen_model = qg.qwen.model
    input_ids = prompt['input_ids'].to(resolve_device(qg.qwen))
    attention = prompt['attention_mask'].to(input_ids.device)
    image_grid_thw = batch.get('image_grid_thw')
    video_grid_thw = batch.get('video_grid_thw')
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(input_ids.device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(input_ids.device)
    eos_ids = set()
    if isinstance(processor.tokenizer.eos_token_id, int):
        eos_ids.add(int(processor.tokenizer.eos_token_id))
    generated: List[int] = []
    for _ in range(max_new_tokens):
        if generated:
            gen = torch.tensor([generated], dtype=input_ids.dtype, device=input_ids.device)
            cur_ids = torch.cat([input_ids, gen], dim=1)
            cur_attention = torch.cat([attention, torch.ones_like(gen)], dim=1)
        else:
            cur_ids = input_ids
            cur_attention = attention
        embeds = qwen_model.get_input_embeddings()(cur_ids)
        cur_visual_mask = pad_visual_mask(visual_pos_masks.to(cur_ids.device), cur_ids.shape[1])
        embeds = qg._scatter_visual(embeds, cur_visual_mask, modulated_visual)
        position_ids, rope_deltas = qwen_model.get_rope_index(
            cur_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=cur_attention,
        )
        qwen_model.rope_deltas = rope_deltas
        outputs = qwen_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=cur_attention,
            inputs_embeds=embeds,
            visual_pos_masks=cur_visual_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = qg.qwen.lm_head(outputs.last_hidden_state[:, -1, :])
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        if next_id in eos_ids:
            break
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        if '}' in text:
            break
    raw = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return parse_category(raw), raw, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--query-module-dir', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', default='valid')
    p.add_argument('--meta-file', default='meta_expressions_challenge.json')
    p.add_argument('--output-json', required=True)
    p.add_argument('--frames-per-sample', type=int, default=3)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--device-map', default='auto', choices=['', 'auto'])
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    records = load_meta_records(args.data_root, args.split, args.meta_file, args.frames_per_sample)
    if args.max_samples and args.max_samples > 0:
        records = records[:args.max_samples]
    processor, qg = load_query_model(args)
    out_records = []
    for idx, record in enumerate(records, 1):
        pred, raw, stats = predict_one(processor, qg, record, args.max_new_tokens)
        out = dict(record)
        out.update({
            'pred_category': pred,
            'raw_response': raw,
            'query_stats': stats,
            'status': 'ok' if pred else 'error',
        })
        out_records.append(out)
        print(f"[{idx}/{len(records)}] {record['video_id']}/{record['exp_id']} pred={pred or '<empty>'}", flush=True)
    payload = {
        'summary': {
            'total': len(out_records),
            'valid': sum(1 for r in out_records if r.get('pred_category')),
            'data_root': args.data_root,
            'split': args.split,
            'meta_file': args.meta_file,
            'query_module_dir': args.query_module_dir,
        },
        'records': out_records,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload['summary'], ensure_ascii=False, indent=2), flush=True)
    print(f'[output] {out_path}', flush=True)


if __name__ == '__main__':
    main()
