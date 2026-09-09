import copy
import gc
import json
import os
import random
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from crf_sam3_attn.dataset import _draw_bbox, _get_frame_pad, _list_frame_files, _overlay_mask, _sample_indices
from crf_sam3_attn.train_data import load_base_config, load_yaml, resolve_dataset_specs
from crf_sam3_attn.model import Qwen3VLCandidateMatcher
from crf_sam3_attn.model_resume import Qwen3VLCandidateMatcher as Qwen3VLCandidateMatcherResume


@dataclass
class CandidateData:
    obj_id: int
    frames: List[Image.Image]
    masks: List[np.ndarray]
    obj_iou: float
    frame_indices: List[int]


@dataclass
class ExpressionData:
    dataset_name: str
    video_id: str
    exp_id: str
    expression: str
    candidates: List[CandidateData]
    pos_obj_ids: List[int]
    neg_obj_ids: List[int]
    iou_map: Dict[int, float]
    meta_path: str


@dataclass
class AnalysisContext:
    repo_root: str
    config_path: str
    train_cfg: Dict
    base_cfg: Dict
    model_cfg: Dict
    pipeline_cfg: Dict
    dataset_name: str
    cache_root: str
    split: str
    num_frames: int
    mask_style: str
    pos_iou: float
    neg_iou: float
    listwise_max_candidates: Optional[int]
    base_model_path: str


def release_model(model) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def _resolve_dataset_spec(train_cfg: Dict, base_cfg: Dict, config_path: str, repo_root: str, dataset_name: Optional[str]) -> Tuple[Dict, Dict]:
    specs = resolve_dataset_specs(
        global_base_cfg=base_cfg,
        data_cfg=train_cfg.get("data", {}),
        train_cfg_dir=os.path.dirname(os.path.abspath(config_path)),
        repo_root=repo_root,
    )
    if dataset_name is None:
        return specs[0]
    for dataset_base_cfg, dataset_train_cfg in specs:
        if str(dataset_train_cfg.get("dataset_name")) == str(dataset_name):
            return dataset_base_cfg, dataset_train_cfg
    available = [str(cfg.get("dataset_name")) for _, cfg in specs]
    raise KeyError(f"dataset_name={dataset_name!r} not found; available={available}")


def load_analysis_context(
    config_path: str,
    repo_root: str,
    dataset_name: Optional[str] = None,
    cache_root: Optional[str] = None,
    split: Optional[str] = None,
    num_frames: Optional[int] = None,
    mask_style: Optional[str] = None,
) -> AnalysisContext:
    config_path = os.path.abspath(config_path)
    train_cfg = load_yaml(config_path)
    base_cfg, _ = load_base_config(train_cfg, config_path, repo_root)
    dataset_base_cfg, dataset_train_cfg = _resolve_dataset_spec(train_cfg, base_cfg, config_path, repo_root, dataset_name)

    model_cfg = copy.deepcopy(train_cfg.get("model", {}))
    pipeline_cfg = copy.deepcopy(base_cfg.get("pipeline", {}))
    dataset_name = str(dataset_train_cfg.get("dataset_name") or dataset_name or "dataset")
    default_cache_root = dataset_train_cfg.get("offline_cache_root", "crf_sam3_attn/offline_cache")
    cache_root = cache_root or default_cache_root
    if not os.path.isabs(cache_root):
        cache_root = os.path.abspath(os.path.join(repo_root, cache_root))
    split = str(split or dataset_train_cfg.get("split") or train_cfg.get("data", {}).get("split", "train"))
    num_frames = int(num_frames or dataset_train_cfg.get("num_frames") or train_cfg.get("data", {}).get("num_frames", 4))
    mask_style = str(mask_style or dataset_train_cfg.get("mask_style") or train_cfg.get("data", {}).get("mask_style", "mask"))
    base_model_path = dataset_base_cfg["model"]["mllm_path"]
    listwise_max_candidates = dataset_train_cfg.get("listwise_max_candidates")
    if listwise_max_candidates is not None:
        try:
            listwise_max_candidates = int(listwise_max_candidates)
        except (TypeError, ValueError):
            listwise_max_candidates = None

    return AnalysisContext(
        repo_root=repo_root,
        config_path=config_path,
        train_cfg=train_cfg,
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        pipeline_cfg=pipeline_cfg,
        dataset_name=dataset_name,
        cache_root=cache_root,
        split=split,
        num_frames=num_frames,
        mask_style=mask_style,
        pos_iou=float(dataset_train_cfg.get("pos_iou", train_cfg.get("data", {}).get("pos_iou", 0.5))),
        neg_iou=float(dataset_train_cfg.get("neg_iou", train_cfg.get("data", {}).get("neg_iou", 0.1))),
        listwise_max_candidates=listwise_max_candidates,
        base_model_path=base_model_path,
    )


def discover_expression_dirs(cache_root: str, split: str) -> List[str]:
    pattern = os.path.join(cache_root, str(split), '*', '*')
    exp_dirs = [p for p in glob(pattern) if os.path.isfile(os.path.join(p, 'meta.json'))]
    exp_dirs.sort()
    return exp_dirs


def sample_expression_dirs(exp_dirs: Sequence, max_expressions: Optional[int], seed: int):
    items = list(exp_dirs)
    if max_expressions is None or max_expressions <= 0 or len(items) <= max_expressions:
        return items
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:max_expressions]


def _trim_candidate_ids(obj_ids: List[int], iou_map: Dict[int, float], max_candidates: Optional[int]) -> List[int]:
    if not max_candidates:
        return obj_ids
    if max_candidates <= 0 or len(obj_ids) <= max_candidates:
        return obj_ids
    return sorted(obj_ids, key=lambda oid: iou_map.get(int(oid), 0.0), reverse=True)[:max_candidates]


def load_expression_data(exp_dir: str, context: AnalysisContext, frame_mode: str = 'sampled') -> Optional[ExpressionData]:
    meta_path = os.path.join(exp_dir, 'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    video_id = str(meta['video_id']) if 'video_id' in meta else os.path.basename(os.path.dirname(exp_dir))
    exp_id = str(meta['exp_id']) if 'exp_id' in meta else os.path.basename(exp_dir)
    expression = meta.get('expression', '')
    iou_map_raw = meta.get('iou_map', {})
    iou_map = {int(k): float(v) for k, v in iou_map_raw.items()}
    if not iou_map:
        return None

    obj_ids = meta.get('obj_ids') or list(iou_map.keys())
    obj_ids = [int(obj_id) for obj_id in obj_ids]
    obj_ids = _trim_candidate_ids(obj_ids, iou_map, context.listwise_max_candidates)
    if not obj_ids:
        return None

    video_path = meta.get('video_path')
    if not video_path:
        return None
    frame_files = _list_frame_files(video_path)
    if not frame_files:
        return None

    frame_indices_map = meta.get('frame_indices', {})
    frame_pad = int(meta.get('frame_pad', _get_frame_pad(video_path)))
    frame_cache: Dict[int, np.ndarray] = {}
    candidates: List[CandidateData] = []

    for obj_id in obj_ids:
        frames_with_obj = frame_indices_map.get(str(obj_id)) or frame_indices_map.get(obj_id)
        if not frames_with_obj:
            continue
        frame_indices = [int(x) for x in frames_with_obj]
        if frame_mode == 'sampled':
            frame_indices = _sample_indices(frame_indices, context.num_frames)
        mask_dir = os.path.join(exp_dir, 'masks', f'obj_{obj_id}')
        if not os.path.isdir(mask_dir):
            continue

        frames: List[Image.Image] = []
        masks: List[np.ndarray] = []
        valid_indices: List[int] = []
        for frame_idx in frame_indices:
            if frame_idx >= len(frame_files):
                continue
            mask_name = f"{frame_idx:0{frame_pad}d}.png" if frame_pad > 0 else f"{frame_idx}.png"
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                continue
            frame = frame_cache.get(frame_idx)
            if frame is None:
                frame = np.array(Image.open(frame_files[frame_idx]).convert('RGB'))
                frame_cache[frame_idx] = frame
            mask = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)
            if context.mask_style == 'bbox':
                annotated = _draw_bbox(frame, mask)
            else:
                annotated = _overlay_mask(frame, mask, alpha=0.5)
            frames.append(Image.fromarray(annotated))
            masks.append(mask)
            valid_indices.append(frame_idx)

        if not frames:
            continue

        candidates.append(CandidateData(
            obj_id=obj_id,
            frames=frames,
            masks=masks,
            obj_iou=float(iou_map.get(obj_id, 0.0)),
            frame_indices=valid_indices,
        ))

    if not candidates:
        return None

    pos_obj_ids = [c.obj_id for c in candidates if c.obj_iou >= context.pos_iou]
    neg_obj_ids = [c.obj_id for c in candidates if c.obj_iou <= context.neg_iou]
    return ExpressionData(
        dataset_name=context.dataset_name,
        video_id=video_id,
        exp_id=exp_id,
        expression=expression,
        candidates=candidates,
        pos_obj_ids=pos_obj_ids,
        neg_obj_ids=neg_obj_ids,
        iou_map=iou_map,
        meta_path=meta_path,
    )


def resolve_model_runs(base_model_path: str, finetuned_model_path: Optional[str]) -> List[Tuple[str, str, Optional[str]]]:
    runs = [('before', base_model_path, None)]
    if finetuned_model_path:
        finetuned_abs = os.path.abspath(finetuned_model_path)
        if os.path.exists(os.path.join(finetuned_abs, 'adapter_config.json')):
            runs.append(('after', base_model_path, finetuned_abs))
        else:
            runs.append(('after', finetuned_abs, None))
    return runs


def build_runtime_matcher(context: AnalysisContext, model_path: str, adapter_path: Optional[str], device: Optional[str], dtype: Optional[str], layer_override: Optional[int] = None):
    model_cfg = copy.deepcopy(context.model_cfg)
    if layer_override is not None:
        model_cfg['attn_loss_layer'] = int(layer_override)
    return build_matcher(model_path, model_cfg, context.pipeline_cfg, device, dtype, adapter_path)


def build_matcher(model_path: str, model_cfg: Dict, pipeline_cfg: Dict, device: Optional[str], dtype: Optional[str], adapter_path: Optional[str] = None):
    dtype_map = {
        None: torch.bfloat16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'fp32': torch.float32,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map.get(str(dtype).lower() if dtype is not None else None, torch.bfloat16)
    kwargs = {
        'model_path': model_path,
        'attn_loss_weight': model_cfg.get('attn_loss_weight', 0.0),
        'attn_loss_layer': model_cfg.get('attn_loss_layer', -1),
        'attn_loss_eps': model_cfg.get('attn_loss_eps', 1e-6),
        'freeze_visual_encoder': model_cfg.get('freeze_visual_encoder', True),
        'lora_cfg': model_cfg.get('lora_cfg'),
        'attn_implementation': model_cfg.get('attn_implementation', 'sdpa'),
        'loss_type': model_cfg.get('loss_type', 'feature'),
        'align_tau': model_cfg.get('align_tau', 1.0),
        'align_similarity': model_cfg.get('align_similarity', 'cosine'),
        'score_type': model_cfg.get('score_type', 'mask_align'),
        'mask_context_alpha': model_cfg.get('mask_context_alpha', 0.8),
        'context_mode': model_cfg.get('context_mode', 'full'),
        'listwise_tau': model_cfg.get('listwise_tau', 1.0),
        'listwise_neg_policy': model_cfg.get('listwise_neg_policy', 'all'),
        'listwise_force_pos': model_cfg.get('listwise_force_pos', True),
        'hard_neg_weight': model_cfg.get('hard_neg_weight', 0.0),
        'hard_neg_margin': model_cfg.get('hard_neg_margin', 0.2),
        'hard_neg_pos_strategy': model_cfg.get('hard_neg_pos_strategy', 'max'),
        'debug_logits': model_cfg.get('debug_logits', False),
        'debug_logits_topk': model_cfg.get('debug_logits_topk', 5),
        'device_map': model_cfg.get('device_map'),
        'max_memory': model_cfg.get('max_memory'),
        'offload_folder': model_cfg.get('offload_folder'),
        'offload_state_dict': model_cfg.get('offload_state_dict', True),
        'low_cpu_mem_usage': model_cfg.get('low_cpu_mem_usage', True),
        'device': device or model_cfg.get('device', 'cuda'),
        'dtype': torch_dtype,
    }
    if adapter_path:
        kwargs.pop('lora_cfg', None)
        kwargs['lora_resume_path'] = adapter_path
        return Qwen3VLCandidateMatcherResume(**kwargs).eval()
    return Qwen3VLCandidateMatcher(**kwargs).eval()


def run_candidate_forward(matcher, expression: str, frames: List[Image.Image], masks: List[np.ndarray]) -> Dict:
    inputs, prompt_len = matcher._prepare_inputs(frames, expression, answer=None)
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
    prob_score = matcher._compute_yesno_score(outputs, prompt_len, score_type='prob')
    logit_score = matcher._compute_yesno_score(outputs, prompt_len, score_type='logit')
    return {
        'inputs': inputs,
        'prompt_len': prompt_len,
        'outputs': outputs,
        'capture': matcher._loss_capture,
        'prob_score': None if prob_score is None else float(prob_score.detach().cpu()),
        'logit_score': None if logit_score is None else float(logit_score.detach().cpu()),
    }


def compute_align_margin_metrics(matcher, input_ids: torch.Tensor, prompt_len: int, hidden_states: torch.Tensor, image_grid_thw: torch.Tensor, masks: List[np.ndarray]) -> List[Dict[str, float]]:
    if hidden_states is None or image_grid_thw is None or hidden_states.dim() != 3:
        return []
    prompt_ids = input_ids.to(hidden_states.device)[0, :prompt_len]
    text_positions = (prompt_ids != matcher.image_pad_id).nonzero(as_tuple=False).squeeze(1)
    image_positions = (prompt_ids == matcher.image_pad_id).nonzero(as_tuple=False).squeeze(1)
    if text_positions.numel() == 0 or image_positions.numel() == 0:
        return []

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
    frame_metrics = []
    offset = 0
    for idx, grid in enumerate(image_grid):
        if idx >= len(masks):
            break
        t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
        num_tokens = (t * h * w) // merge_length
        frame_sims = sims[offset: offset + num_tokens]
        offset += num_tokens
        grid_h = h // matcher.merge_size
        grid_w = w // matcher.merge_size
        if frame_sims.numel() == 0 or grid_h <= 0 or grid_w <= 0:
            continue
        mask = masks[idx]
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_t = F.interpolate(mask_t, size=(grid_h, grid_w), mode='nearest')
        mask_bool = (mask_t.squeeze(0).squeeze(0) > 0).view(-1).to(frame_sims.device)
        if mask_bool.numel() != frame_sims.numel() or not mask_bool.any():
            continue
        probs = F.softmax(frame_sims / tau, dim=0)
        outside = ~mask_bool
        mean_in_sim = float(frame_sims[mask_bool].mean().detach().cpu())
        mean_out_sim = float(frame_sims[outside].mean().detach().cpu()) if outside.any() else 0.0
        mean_in_prob = float(probs[mask_bool].mean().detach().cpu())
        mean_out_prob = float(probs[outside].mean().detach().cpu()) if outside.any() else 0.0
        inside_mass = float((probs * mask_bool.float()).sum().detach().cpu())
        total_mass = float(probs.sum().detach().cpu())
        in_vals = frame_sims[mask_bool]
        out_vals = frame_sims[outside]
        if out_vals.numel() == 0:
            auc_sim = 0.5
        else:
            comp = in_vals[:, None] - out_vals[None, :]
            auc_sim = float(((comp > 0).float().mean() + 0.5 * (comp == 0).float().mean()).detach().cpu())
        frame_metrics.append({
            'mean_in_sim': mean_in_sim,
            'mean_out_sim': mean_out_sim,
            'margin_sim': mean_in_sim - mean_out_sim,
            'mean_in_prob': mean_in_prob,
            'mean_out_prob': mean_out_prob,
            'margin_prob': mean_in_prob - mean_out_prob,
            'auc_sim': auc_sim,
            'inside_mass': inside_mass,
            'inside_ratio': inside_mass / max(total_mass, 1e-8),
        })
    return frame_metrics
