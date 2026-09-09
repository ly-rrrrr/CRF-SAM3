import argparse
import gc
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.visualize_lalign_heatmap import (
    build_matcher,
    draw_label,
    ensure_dir,
    load_candidate_sample,
    load_yaml,
    make_grid,
    make_row,
    mask_contour,
    normalize_map,
    overlay_heatmap,
    resolve_path,
    save_rgb,
    to_uint8_heatmap,
    upsample_grid,
)
from crf_sam3_attn.train_data import load_base_config


def release_model(model):
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_layers(text: str) -> List[int]:
    vals = []
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError('No layers parsed from --layers')
    return vals


def _token_positions(matcher, inputs, capture):
    prompt_len = int(inputs['input_ids'].shape[1])
    prompt_ids = inputs['input_ids'].to(capture.device)[0, :prompt_len]
    text_positions = (prompt_ids != matcher.image_pad_id).nonzero(as_tuple=False).squeeze(1)
    image_positions = (prompt_ids == matcher.image_pad_id).nonzero(as_tuple=False).squeeze(1)
    return prompt_len, text_positions, image_positions


def extract_feature_probe(matcher, sample) -> Dict:
    inputs, prompt_len = matcher._prepare_inputs(sample.annotated_frames, sample.expression, answer=None)
    matcher._loss_capture = None
    with torch.no_grad():
        outputs = matcher.model(
            **inputs,
            output_hidden_states=False,
            output_attentions=False,
            logits_to_keep=1,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = matcher._loss_capture
    image_grid_thw = inputs.get('image_grid_thw')
    if hidden_states is None or image_grid_thw is None:
        raise RuntimeError('Failed to capture hidden states or image_grid_thw in feature probe.')

    prompt_ids = inputs['input_ids'].to(hidden_states.device)[0, :prompt_len]
    text_positions = (prompt_ids != matcher.image_pad_id).nonzero(as_tuple=False).squeeze(1)
    image_positions = (prompt_ids == matcher.image_pad_id).nonzero(as_tuple=False).squeeze(1)
    if text_positions.numel() == 0 or image_positions.numel() == 0:
        raise RuntimeError('Could not locate text/image token positions in feature probe.')

    hidden = hidden_states[0]
    text_vec = hidden[text_positions].mean(dim=0, keepdim=True)
    image_feats = hidden[image_positions]

    if matcher.align_similarity == 'cosine':
        text_vec = F.normalize(text_vec, dim=-1)
        image_feats = F.normalize(image_feats, dim=-1)
        sims = (image_feats @ text_vec.t()).squeeze(1)
    else:
        sims = (image_feats * text_vec).sum(dim=-1)

    tau = max(float(matcher.align_tau), 1e-6)
    merge_length = matcher.merge_size ** 2
    image_grid = image_grid_thw.detach().cpu().tolist()

    frame_results = []
    offset = 0
    for idx, grid in enumerate(image_grid):
        if idx >= len(sample.masks):
            break
        t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        num_tokens = (t * h * w) // merge_length
        frame_sims = sims[offset: offset + num_tokens]
        offset += num_tokens
        grid_h = h // matcher.merge_size
        grid_w = w // matcher.merge_size
        if frame_sims.numel() == 0 or grid_h <= 0 or grid_w <= 0:
            continue

        mask = sample.masks[idx]
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, size=(grid_h, grid_w), mode='nearest')
        mask_grid = (mask_t.squeeze(0).squeeze(0) > 0).float().to(frame_sims.device)

        prob_grid = F.softmax(frame_sims / tau, dim=0).view(grid_h, grid_w)
        masked_prob_grid = prob_grid * mask_grid
        total_mass = float(prob_grid.sum().detach().cpu())
        inside_mass = float(masked_prob_grid.sum().detach().cpu())
        frame_results.append({
            'grid': prob_grid.float().detach().cpu().numpy(),
            'inside_mass': inside_mass,
            'inside_ratio': inside_mass / max(total_mass, 1e-8),
        })

    prob_yes = prob_no = None
    pred = None
    if hasattr(outputs, 'logits') and outputs.logits is not None:
        logits = outputs.logits[0, -1]
        yes_no_logits = torch.stack([logits[matcher.yes_token_id], logits[matcher.no_token_id]], dim=0)
        yes_no_probs = F.softmax(yes_no_logits.float(), dim=0)
        prob_yes = float(yes_no_probs[0].detach().cpu())
        prob_no = float(yes_no_probs[1].detach().cpu())
        pred = 'Yes' if prob_yes >= prob_no else 'No'

    return {
        'prob_yes': prob_yes,
        'prob_no': prob_no,
        'pred': pred,
        'frame_results': frame_results,
    }


def extract_decision_probe(matcher, sample, local_frame_idx: Optional[int] = None) -> Dict:
    matcher.model.zero_grad(set_to_none=True)
    if local_frame_idx is None:
        frames = sample.annotated_frames
        masks = sample.masks
        frame_indices = sample.frame_indices
    else:
        frames = [sample.annotated_frames[local_frame_idx]]
        masks = [sample.masks[local_frame_idx]]
        frame_indices = [sample.frame_indices[local_frame_idx]]
    inputs, prompt_len = matcher._prepare_inputs(frames, sample.expression, answer=None)
    matcher._loss_capture = None
    outputs = matcher.model(
        **inputs,
        output_hidden_states=False,
        output_attentions=False,
        logits_to_keep=1,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = matcher._loss_capture
    image_grid_thw = inputs.get('image_grid_thw')
    if hidden_states is None or image_grid_thw is None:
        raise RuntimeError('Failed to capture hidden states or image_grid_thw in decision probe.')
    hidden_states.retain_grad()

    logits = outputs.logits[0, -1]
    score = logits[matcher.yes_token_id] - logits[matcher.no_token_id]
    score.backward()

    grad = hidden_states.grad
    if grad is None:
        raise RuntimeError('No gradient captured for decision probe.')

    prompt_ids = inputs['input_ids'].to(hidden_states.device)[0, :prompt_len]
    image_positions = (prompt_ids == matcher.image_pad_id).nonzero(as_tuple=False).squeeze(1)
    hidden = hidden_states[0]
    grad = grad[0]
    image_feats = hidden[image_positions]
    image_grads = grad[image_positions]
    attr = F.relu((image_feats.float() * image_grads.float()).sum(dim=-1))

    merge_length = matcher.merge_size ** 2
    image_grid = image_grid_thw.detach().cpu().tolist()
    frame_results = []
    offset = 0
    for idx, grid in enumerate(image_grid):
        if idx >= len(masks):
            break
        t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        num_tokens = (t * h * w) // merge_length
        frame_attr = attr[offset: offset + num_tokens]
        offset += num_tokens
        grid_h = h // matcher.merge_size
        grid_w = w // matcher.merge_size
        if frame_attr.numel() == 0 or grid_h <= 0 or grid_w <= 0:
            continue

        mask = masks[idx]
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, size=(grid_h, grid_w), mode='nearest')
        mask_grid = (mask_t.squeeze(0).squeeze(0) > 0).float().to(frame_attr.device)

        total = frame_attr.sum().clamp_min(1e-8)
        frame_attr_grid = frame_attr.view(grid_h, grid_w)
        inside_mass = float((frame_attr_grid * mask_grid).sum().detach().cpu())
        total_mass = float(total.detach().cpu())
        frame_results.append({
            'grid': frame_attr_grid.float().detach().cpu().numpy(),
            'inside_mass': inside_mass,
            'inside_ratio': inside_mass / max(total_mass, 1e-8),
        })

    matcher.model.zero_grad(set_to_none=True)
    return {
        'score_margin': float(score.detach().cpu()),
        'frame_results': frame_results,
        'frame_indices': frame_indices,
    }


def render_probe_grid(raw_frame, mask, grid, colormap='turbo', alpha=0.55, dim_outside=1.0):
    h, w = raw_frame.shape[:2]
    contour = mask_contour(mask)
    up = upsample_grid(grid, (h, w))
    norm = normalize_map(up)
    heat = to_uint8_heatmap(norm, colormap)
    if dim_outside < 1.0:
        overlay = overlay_heatmap(raw_frame, heat, alpha=alpha, mask=mask, mask_contour=contour, dim_outside=dim_outside)
    else:
        overlay = overlay_heatmap(raw_frame, heat, alpha=alpha, mask_contour=contour)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='Layer-wise Phase-2 analysis with feature response and decision attribution.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--exp-id', required=True)
    parser.add_argument('--obj-id', required=True, type=int)
    parser.add_argument('--finetuned-model-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--layers', default='-1,-2,-4,-8')
    parser.add_argument('--num-frames', type=int, default=None)
    parser.add_argument('--frame-mode', choices=['sampled', 'all'], default='sampled')
    parser.add_argument('--mask-style', choices=['mask', 'bbox'], default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dtype', default=None)
    parser.add_argument('--colormap', default='turbo', choices=['turbo', 'magma', 'inferno', 'viridis', 'jet'])
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    layers = parse_layers(args.layers)
    config_path = os.path.abspath(args.config)
    train_cfg = load_yaml(config_path)
    base_cfg, _ = load_base_config(train_cfg, config_path, _REPO_ROOT)
    model_cfg = dict(train_cfg.get('model', {}))
    data_cfg = train_cfg.get('data', {})
    pipeline_cfg = base_cfg.get('pipeline', {})

    base_model_path = base_cfg['model']['mllm_path']
    num_frames = int(args.num_frames or data_cfg.get('num_frames', 8))
    mask_style = args.mask_style or data_cfg.get('mask_style', 'mask')
    cache_root = resolve_path(args.cache_root, _REPO_ROOT)
    sample = load_candidate_sample(
        cache_root=cache_root,
        split=args.split,
        video_id=args.video_id,
        exp_id=args.exp_id,
        obj_id=args.obj_id,
        num_frames=num_frames,
        frame_mode=args.frame_mode,
        mask_style=mask_style,
    )

    summary = {
        'video_id': args.video_id,
        'exp_id': args.exp_id,
        'obj_id': args.obj_id,
        'expression': sample.expression,
        'frame_indices': sample.frame_indices,
        'layers': layers,
        'feature_probe': {},
        'decision_probe': {},
    }

    probes = [('before', base_model_path), ('after', os.path.abspath(args.finetuned_model_path))]
    layer_panels = []

    for layer in layers:
        model_cfg_layer = dict(model_cfg)
        model_cfg_layer['attn_loss_layer'] = layer
        per_layer = {'feature': {}, 'decision': {}}
        rendered_layers = {'feature': {}, 'decision': {}}
        for tag, model_path in probes:
            matcher = None
            try:
                matcher = build_matcher(
                    model_path=model_path,
                    model_cfg=model_cfg_layer,
                    pipeline_cfg=pipeline_cfg,
                    device_override=args.device,
                    dtype_override=args.dtype,
                    adapter_path=None,
                )
                feat = extract_feature_probe(matcher, sample)
                per_layer['feature'][tag] = {
                    'prob_yes': feat['prob_yes'],
                    'pred': feat['pred'],
                    'mean_inside_ratio': float(np.mean([x['inside_ratio'] for x in feat['frame_results']])) if feat['frame_results'] else None,
                    'frame_metrics': [
                        {
                            'frame_index': sample.frame_indices[idx],
                            'inside_ratio': item['inside_ratio'],
                            'inside_mass': item['inside_mass'],
                        }
                        for idx, item in enumerate(feat['frame_results'])
                    ],
                }
                rendered_layers['feature'][tag] = feat['frame_results']
            finally:
                release_model(matcher)

        feat_delta = per_layer['feature']['after']['mean_inside_ratio'] - per_layer['feature']['before']['mean_inside_ratio']

        best_idx = int(np.argmax([
            rendered_layers['feature']['after'][i]['inside_ratio'] - rendered_layers['feature']['before'][i]['inside_ratio']
            for i in range(len(sample.frame_indices))
        ]))
        frame_idx = sample.frame_indices[best_idx]

        for tag, model_path in probes:
            matcher = None
            try:
                matcher = build_matcher(
                    model_path=model_path,
                    model_cfg=model_cfg_layer,
                    pipeline_cfg=pipeline_cfg,
                    device_override=args.device,
                    dtype_override=args.dtype,
                    adapter_path=None,
                )
                dec = extract_decision_probe(matcher, sample, local_frame_idx=best_idx)
                per_layer['decision'][tag] = {
                    'score_margin': dec['score_margin'],
                    'mean_inside_ratio': float(np.mean([x['inside_ratio'] for x in dec['frame_results']])) if dec['frame_results'] else None,
                    'frame_metrics': [
                        {
                            'frame_index': dec['frame_indices'][idx],
                            'inside_ratio': item['inside_ratio'],
                            'inside_mass': item['inside_mass'],
                        }
                        for idx, item in enumerate(dec['frame_results'])
                    ],
                }
                rendered_layers['decision'][tag] = dec['frame_results']
            finally:
                release_model(matcher)

        dec_delta = per_layer['decision']['after']['mean_inside_ratio'] - per_layer['decision']['before']['mean_inside_ratio']
        summary['feature_probe'][str(layer)] = per_layer['feature']
        summary['decision_probe'][str(layer)] = per_layer['decision']
        summary['feature_probe'][str(layer)]['delta_mean_inside_ratio'] = feat_delta
        summary['decision_probe'][str(layer)]['delta_mean_inside_ratio'] = dec_delta

        raw = sample.raw_frames[best_idx]
        mask = sample.masks[best_idx]
        feat_before = render_probe_grid(raw, mask, rendered_layers['feature']['before'][best_idx]['grid'], args.colormap, 0.55, 1.0)
        feat_after = render_probe_grid(raw, mask, rendered_layers['feature']['after'][best_idx]['grid'], args.colormap, 0.55, 1.0)
        dec_before = render_probe_grid(raw, mask, rendered_layers['decision']['before'][0]['grid'], args.colormap, 0.55, 1.0)
        dec_after = render_probe_grid(raw, mask, rendered_layers['decision']['after'][0]['grid'], args.colormap, 0.55, 1.0)
        raw_mask = raw.copy()
        contour = mask_contour(mask)
        raw_mask[contour > 0] = np.array([255, 255, 255], dtype=np.uint8)
        panel = make_row([
            draw_label(raw_mask, f'Layer {layer} | Frame {frame_idx} | Candidate'),
            draw_label(feat_before, f'Layer {layer} | Before Feature'),
            draw_label(feat_after, f'Layer {layer} | After Feature'),
            draw_label(dec_before, f'Layer {layer} | Before Decision'),
            draw_label(dec_after, f'Layer {layer} | After Decision'),
        ], pad=12, bg=255)
        save_rgb(os.path.join(args.output_dir, f'layer_{layer}_bestframe.png'), panel)
        layer_panels.append(panel)

    feat_rank = sorted(
        [(int(k), v['delta_mean_inside_ratio']) for k, v in summary['feature_probe'].items()],
        key=lambda x: x[1], reverse=True,
    )
    dec_rank = sorted(
        [(int(k), v['delta_mean_inside_ratio']) for k, v in summary['decision_probe'].items()],
        key=lambda x: x[1], reverse=True,
    )
    summary['feature_probe_ranking'] = feat_rank
    summary['decision_probe_ranking'] = dec_rank
    summary['recommended_layer'] = {
        'feature': feat_rank[0][0],
        'decision': dec_rank[0][0],
    }

    if layer_panels:
        save_rgb(os.path.join(args.output_dir, 'layerwise_overview.png'), make_grid(layer_panels, pad=14, bg=255))

    concise = {
        'main_file': 'layerwise_overview.png',
        'recommended_layer': summary['recommended_layer'],
        'how_to_read': {
            'feature_probe': '对应 L_align 当前实现的 hidden-state similarity 响应。看 before/after inside_ratio 的提升。',
            'decision_probe': '对应最终 yes-no 判别 margin 的梯度归因。更接近最终判别真正依赖的空间证据。',
        },
    }
    with open(os.path.join(args.output_dir, 'README_main.json'), 'w', encoding='utf-8') as f:
        json.dump(concise, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
