import argparse
import json
import os
import sys
import random
import time
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

# Ensure repo root is on PYTHONPATH when running as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
    
from crf_sam3_attn.dataset import CRFSAM3AttnDataset
from crf_sam3_attn.model import Qwen3VLCandidateMatcher
from crf_sam3_attn.train_data import (
    build_training_dataset,
    load_base_config,
    prepare_training_offline_cache,
)




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def passthrough_collate(batch):
    if len(batch) != 1:
        raise ValueError("Expected batch_size=1 for passthrough_collate.")
    return batch[0]

def _to_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach())
    return float(value)


def _grad_norm(params):
    norms = []
    for p in params:
        if p.grad is None:
            continue
        norms.append(p.grad.detach().norm(2).float().cpu())
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2))


def _is_oom_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _short_text(text, max_chars: int = 160) -> str:
    if text is None:
        return ""
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_candidate_supervision(batch, out, max_items: int = 12) -> str:
    candidates = batch.get("candidates", []) or []
    scores = out.get("scores", {}) or {}
    pos_ids = set(int(x) for x in (batch.get("pos_obj_ids") or out.get("pos_obj_ids") or []))
    neg_ids = set(int(x) for x in (batch.get("neg_obj_ids") or out.get("neg_obj_ids") or []))

    rows = []
    for cand in candidates:
        obj_id = int(cand.get("obj_id"))
        obj_iou = float(cand.get("obj_iou", 0.0))
        if obj_id in pos_ids:
            tag = "P"
        elif obj_id in neg_ids:
            tag = "N"
        else:
            tag = "M"
        score = scores.get(obj_id)
        if score is None:
            score = scores.get(str(obj_id))
        score = None if score is None else float(score)
        rows.append((obj_id, obj_iou, tag, score))

    rows.sort(key=lambda x: (x[3] if x[3] is not None else -1e9), reverse=True)
    rows = rows[:max_items]
    if not rows:
        return "no-candidates"

    parts = []
    for obj_id, obj_iou, tag, score in rows:
        score_text = "NA" if score is None else f"{score:.3f}"
        parts.append(f"{obj_id}:{tag}:iou={obj_iou:.3f}:s={score_text}")
    return " | ".join(parts)


def _save_debug(step, batch, out, debug_dir, prompt):
    step_dir = os.path.join(debug_dir, f"step_{step:06d}")
    os.makedirs(step_dir, exist_ok=True)

    frames = batch.get("frames", [])
    masks = batch.get("masks", [])
    if (not frames) and "candidates" in batch:
        candidates = batch.get("candidates", [])
        scores = out.get("scores", {})
        best_id = None
        if scores:
            best_id = max(scores.items(), key=lambda item: item[1])[0]
        if best_id is not None:
            for cand in candidates:
                if int(cand.get("obj_id")) == int(best_id):
                    frames = cand.get("frames", [])
                    masks = cand.get("masks", [])
                    break
    for i, frame in enumerate(frames):
        frame.save(os.path.join(step_dir, f"frame_{i:02d}.png"))
    for i, mask in enumerate(masks):
        if torch.is_tensor(mask):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask
        mask_np = (mask_np > 0).astype(np.uint8) * 255
        Image.fromarray(mask_np).save(os.path.join(step_dir, f"mask_{i:02d}.png"))

    meta = {
        "dataset_name": batch.get("dataset_name"),
        "video_id": batch.get("video_id"),
        "exp_id": batch.get("exp_id"),
        "obj_id": batch.get("obj_id"),
        "obj_iou": batch.get("obj_iou"),
        "label": batch.get("label"),
        "expression": batch.get("expression"),
        "prompt": prompt,
        "frame_indices": batch.get("frame_indices"),
        "prob_yes": _to_float(out.get("prob_yes")),
        "prob_no": _to_float(out.get("prob_no")),
        "pred": out.get("pred"),
        "scores": out.get("scores"),
        "pos_obj_ids": out.get("pos_obj_ids"),
        "neg_obj_ids": out.get("neg_obj_ids"),
    }
    with open(os.path.join(step_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="CRF-SAM3 attention training")
    parser.add_argument("--config", type=str, default="crf_sam3_attn/config.yaml")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    train_cfg = load_yaml(config_path)
    base_cfg, _ = load_base_config(train_cfg, config_path, _REPO_ROOT)
    data_cfg = train_cfg.get("data", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    if data_cfg.get("offline_prepare", False):
        prepare_results = prepare_training_offline_cache(
            global_base_cfg=base_cfg,
            data_cfg=data_cfg,
            train_cfg_dir=config_dir,
            repo_root=_REPO_ROOT,
        )
        total_prepared = sum(item["prepared"] for item in prepare_results)
        for item in prepare_results:
            print(
                f"[OfflinePrepare] dataset={item['name']} | split={item['split']} | "
                f"prepared={item['prepared']} | cache_root={item['offline_cache_root']}"
            )
        print(f"[OfflinePrepare] total_prepared={total_prepared}")
        return

    dataset, dataset_infos = build_training_dataset(
        global_base_cfg=base_cfg,
        data_cfg=data_cfg,
        train_cfg_dir=config_dir,
        repo_root=_REPO_ROOT,
    )
    for item in dataset_infos:
        print(
            f"[dataset] name={item['name']} | split={item['split']} | "
            f"raw_samples={item.get('raw_samples', item['samples'])} | "
            f"repeats={item.get('repeats', 1)} | "
            f"effective_samples={item.get('effective_samples', item['samples'])} | "
            f"cache_root={item['offline_cache_root']}"
        )
    if len(dataset_infos) > 1:
        print(f"[dataset] total_samples={len(dataset)} | datasets={len(dataset_infos)} | mix=concat")
    batch_size = int(train_cfg.get("train", {}).get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("This training script currently supports batch_size=1 only.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("train", {}).get("num_workers", 2)),
        pin_memory=True,
        collate_fn=passthrough_collate,
    )

    model_cfg = train_cfg.get("model", {})
    model = Qwen3VLCandidateMatcher(
        model_path=base_cfg["model"]["mllm_path"],
        attn_loss_weight=float(model_cfg.get("attn_loss_weight", 0.3)),
        attn_loss_layer=int(model_cfg.get("attn_loss_layer", -1)),
        attn_loss_eps=float(model_cfg.get("attn_loss_eps", 1e-6)),
        freeze_visual_encoder=bool(model_cfg.get("freeze_visual_encoder", True)),
        lora_cfg=model_cfg.get("lora"),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        loss_type=model_cfg.get("loss_type", "attn"),
        align_tau=float(model_cfg.get("align_tau", 1.0)),
        align_similarity=model_cfg.get("align_similarity", "cosine"),
        score_type=model_cfg.get("score_type", "mask_align"),
        mask_context_alpha=float(model_cfg.get("mask_context_alpha", 0.8)),
        context_mode=model_cfg.get("context_mode", "full"),
        listwise_tau=float(model_cfg.get("listwise_tau", 1.0)),
        listwise_neg_policy=model_cfg.get("listwise_neg_policy", "all"),
        listwise_force_pos=bool(model_cfg.get("listwise_force_pos", True)),
        hard_neg_weight=float(model_cfg.get("hard_neg_weight", 0.0)),
        hard_neg_margin=float(model_cfg.get("hard_neg_margin", 0.2)),
        hard_neg_pos_strategy=model_cfg.get("hard_neg_pos_strategy", "max"),
        debug_logits=bool(model_cfg.get("debug_logits", False)),
        debug_logits_topk=int(model_cfg.get("debug_logits_topk", 5)),
        device_map=model_cfg.get("device_map"),
        max_memory=model_cfg.get("max_memory"),
        offload_folder=model_cfg.get("offload_folder"),
        offload_state_dict=bool(model_cfg.get("offload_state_dict", True)),
        low_cpu_mem_usage=bool(model_cfg.get("low_cpu_mem_usage", True)),
        device=base_cfg["pipeline"].get("device", "cuda"),
        dtype=getattr(torch, base_cfg["pipeline"].get("dtype", "bfloat16")),
    )

    lr = float(train_cfg.get("train", {}).get("lr", 1e-5))
    weight_decay = float(train_cfg.get("train", {}).get("weight_decay", 0.0))
    grad_accum = int(train_cfg.get("train", {}).get("grad_accum", 1))
    epochs = int(train_cfg.get("train", {}).get("epochs", 1))
    max_steps = int(train_cfg.get("train", {}).get("max_steps", -1))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    output_dir = train_cfg.get("train", {}).get("output_dir", "crf_sam3_attn_outputs")
    os.makedirs(output_dir, exist_ok=True)

    log_every = int(train_cfg.get("train", {}).get("log_every", 10))
    save_every = int(train_cfg.get("train", {}).get("save_every", 200))
    save_merged = bool(train_cfg.get("train", {}).get("save_merged", False))
    merged_dir = train_cfg.get("train", {}).get("merged_output_dir", "crf_sam3_attn_outputs_merged")

    debug_cfg = train_cfg.get("debug", {})
    debug_enable = bool(debug_cfg.get("enable", False))
    debug_log_every = int(debug_cfg.get("log_every", log_every))
    debug_save_every = int(debug_cfg.get("save_every", 0))
    debug_max_save = int(debug_cfg.get("max_save", 50))
    debug_dir = debug_cfg.get("save_dir", "crf_sam3_attn_debug")
    debug_candidate_topk = int(debug_cfg.get("candidate_topk", 12))
    debug_expr_chars = int(debug_cfg.get("expr_max_chars", 160))
    if debug_enable and debug_save_every > 0:
        os.makedirs(debug_dir, exist_ok=True)

    step = 0
    saved_debug = 0
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            start = time.time()
            try:
                out = model(batch)
                loss = out["loss"] / grad_accum
                loss.backward()
            except RuntimeError as err:
                if not _is_oom_error(err):
                    raise
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cand_cnt = len(batch.get("candidates", [])) if isinstance(batch, dict) else -1
                print(
                    f"[oom-skip] step {step} | dataset={batch.get('dataset_name')} "
                    f"video={batch.get('video_id')} exp={batch.get('exp_id')} "
                    f"candidates={cand_cnt} | cleared cache and skipped batch"
                )
                step += 1
                if max_steps > 0 and step >= max_steps:
                    break
                continue

            grad_norm = None
            if debug_enable and (step % debug_log_every == 0):
                grad_norm = _grad_norm(trainable_params)

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                loss_attn = out["loss_attn"]
                loss_rank = out.get("loss_rank")
                loss_attn_val = _to_float(loss_attn) if loss_attn is not None else 0.0
                loss_rank_val = _to_float(loss_rank) if loss_rank is not None else 0.0
                print(
                    f"step {step} | dataset {batch.get('dataset_name')} | "
                    f"loss {float(out['loss'].detach()):.4f} | "
                    f"cls {float(out['loss_cls'].detach()):.4f} | "
                    f"rank {loss_rank_val:.4f} | "
                    f"attn {loss_attn_val:.4f} | time {time.time() - start:.2f}s"
                )

            if debug_enable and step % debug_log_every == 0:
                prompt = model._build_prompt(batch.get("expression", ""))
                if "candidates" in batch:
                    scores = out.get("scores", {}) or {}
                    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                    top_scores = ", ".join(
                        [f"{obj_id}:{score:.3f}" for obj_id, score in sorted_scores[:5]]
                    )
                    print(
                        f"[debug] dataset={batch.get('dataset_name')} "
                        f"video={batch.get('video_id')} exp={batch.get('exp_id')} "
                        f"candidates={len(scores)} pos={out.get('pos_obj_ids')} neg={out.get('neg_obj_ids')} "
                        f"grad_norm={grad_norm:.4f}"
                    )
                    print(
                        f"[debug] expr: {_short_text(batch.get('expression'), max_chars=debug_expr_chars)}"
                    )
                    print(
                        f"[debug] supervision: "
                        f"{_format_candidate_supervision(batch, out, max_items=debug_candidate_topk)}"
                    )
                    if top_scores:
                        print(f"[debug] scores_top: {top_scores}")
                else:
                    label = batch.get("label")
                    label_text = "Yes" if int(label) == 1 else "No"
                    obj_iou = batch.get("obj_iou")
                    print(
                        f"[debug] dataset={batch.get('dataset_name')} "
                        f"video={batch.get('video_id')} exp={batch.get('exp_id')} obj={batch.get('obj_id')} "
                        f"iou={obj_iou} label={label_text} pred={out.get('pred')} "
                        f"p_yes={_to_float(out.get('prob_yes'))} p_no={_to_float(out.get('prob_no'))} "
                        f"grad_norm={grad_norm:.4f}"
                    )
                if debug_cfg.get("log_prompt", True):
                    print(f"[debug] expression: {batch.get('expression')}")
                    print(f"[debug] prompt: {prompt}")

                if debug_save_every > 0 and step % debug_save_every == 0 and saved_debug < debug_max_save:
                    _save_debug(step, batch, out, debug_dir, prompt)
                    saved_debug += 1

            if save_every > 0 and step > 0 and step % save_every == 0:
                save_path = os.path.join(output_dir, f"checkpoint_{step}")
                os.makedirs(save_path, exist_ok=True)
                model.save(save_path)

            step += 1
            if max_steps > 0 and step >= max_steps:
                break
        if max_steps > 0 and step >= max_steps:
            break

    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save(final_path)

    if save_merged:
        base_model = model.model
        if hasattr(base_model, "merge_and_unload"):
            merged = base_model.merge_and_unload()
        elif hasattr(base_model, "base_model") and hasattr(base_model.base_model, "merge_and_unload"):
            merged = base_model.base_model.merge_and_unload()
        else:
            raise RuntimeError("merge_and_unload not available; check PEFT model wrapper.")
        merged.save_pretrained(merged_dir)
        model.tokenizer.save_pretrained(merged_dir)
        if hasattr(model, "processor") and model.processor is not None:
            model.processor.save_pretrained(merged_dir)
        else:
            print("[warn] processor not found; merged dir may be missing preprocessor_config.json")


if __name__ == "__main__":
    main()
