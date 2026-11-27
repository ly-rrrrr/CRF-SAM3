#!/usr/bin/env python3
"""Stage 5 (2 GPUs): Verify joint training, a data-parallel shell around the repo's trainer.

The repo's crf_sam3_attn/train_resume.py is single-process. This file runs the SAME model
(crf_sam3_attn.model_resume.Qwen3VLCandidateMatcher), the SAME dataset builder
(crf_sam3_attn.train_data.build_training_dataset: one mixed dataset, per-dataset repeats, like Sa2VA) and
the SAME checkpoint format (adapter + training_state.pt, so train_resume.py can also resume it), with:
  * one process per GPU (torchrun), each holding a full model copy (no device_map sharding);
  * a DistributedSampler over the mixed dataset (each rank sees a different 1/world of it per epoch);
  * gradients averaged across ranks before every optimizer step (all_reduce of the LoRA grads only),
    so effective batch = world x grad_accum;
  * LoRA weights broadcast from rank 0 at start (identical init), rank 0 logs and saves;
  * true resume: --auto-resume picks the latest checkpoint_* with training_state.pt; the sampler skips
    the batches already done without loading them.
Nothing in the repo is modified.

  torchrun --nproc_per_node=2 -m v2.train_verify_ddp --repo $REPO --config runs/verify/train.yaml --auto-resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


class SkipSampler(Sampler):
    """DistributedSampler for one epoch, minus the first `skip` indices of this rank."""

    def __init__(self, base: DistributedSampler, skip: int = 0):
        self.base, self.skip = base, int(skip)

    def __iter__(self):
        return iter(list(self.base)[self.skip:])

    def __len__(self):
        return max(0, len(self.base) - self.skip)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--auto-resume", action="store_true")
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--max-oom-skips", type=int, default=50,
                    help="CUDA OOM on any rank -> BOTH ranks drop that step (logged to oom_skips_rank*.jsonl); "
                         "stop when more than this many steps were dropped in the whole run")
    a = ap.parse_args(argv)
    config_path = os.path.abspath(a.config)
    sys.path.insert(0, os.path.abspath(a.repo))
    os.chdir(a.repo)          # the repo's configs use repo-relative paths

    from crf_sam3_attn import train_resume as tr
    from crf_sam3_attn.model_resume import Qwen3VLCandidateMatcher
    from crf_sam3_attn.train_data import build_training_dataset, load_base_config

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    main_proc = rank == 0

    def log(*x):
        if main_proc:
            print(*x, flush=True)

    train_cfg = tr.load_yaml(config_path)
    base_cfg, _ = load_base_config(train_cfg, config_path, os.path.abspath(a.repo))
    data_cfg, ts, model_cfg = train_cfg.get("data", {}), train_cfg.get("train", {}), train_cfg.get("model", {})
    seed = int(train_cfg.get("seed", 42))
    tr.set_seed(seed)

    from v2.server_dataset import install
    install()
    dataset, infos = build_training_dataset(global_base_cfg=base_cfg, data_cfg=data_cfg,
                                            train_cfg_dir=os.path.dirname(config_path), repo_root=os.path.abspath(a.repo))
    for it in infos:
        log(f"[dataset] name={it['name']} raw={it.get('raw_samples')} repeats={it.get('repeats')} "
            f"effective={it.get('effective_samples')} cache={it['offline_cache_root']}")
    log(f"[dataset] total={len(dataset)} world={world} per_rank={len(dataset) // world}")

    grad_accum = int(ts.get("grad_accum", 1))
    if grad_accum != 1:
        raise SystemExit("This validated server recipe requires grad_accum=1 per GPU")
    epochs, max_steps = int(ts.get("epochs", 1)), int(ts.get("max_steps", -1))
    save_every, log_every = int(ts.get("save_every", 200)), int(ts.get("log_every", 10))
    output_dir = os.path.abspath(ts["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    class _A:  # what train_resume's resolvers read
        resume_from = a.resume_from or None
        auto_resume = a.auto_resume
        init_adapter_from = None
    resume_dir = tr._resolve_resume_dir(_A, train_cfg, output_dir)
    init_dir = tr._resolve_init_adapter_dir(_A, model_cfg, os.path.dirname(config_path))
    load_dir = resume_dir or init_dir
    log(f"[train] resume={resume_dir} init_adapter={init_dir}")

    model = Qwen3VLCandidateMatcher(
        model_path=base_cfg["model"]["mllm_path"],
        attn_loss_weight=float(model_cfg.get("attn_loss_weight", 0.3)),
        attn_loss_layer=int(model_cfg.get("attn_loss_layer", -1)),
        attn_loss_eps=float(model_cfg.get("attn_loss_eps", 1e-6)),
        freeze_visual_encoder=bool(model_cfg.get("freeze_visual_encoder", True)),
        lora_cfg=model_cfg.get("lora"), lora_resume_path=load_dir,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        loss_type=model_cfg.get("loss_type", "attn"), align_tau=float(model_cfg.get("align_tau", 1.0)),
        align_similarity=model_cfg.get("align_similarity", "cosine"),
        score_type=model_cfg.get("score_type", "mask_align"),
        mask_context_alpha=float(model_cfg.get("mask_context_alpha", 0.8)),
        context_mode=model_cfg.get("context_mode", "full"), listwise_tau=float(model_cfg.get("listwise_tau", 1.0)),
        listwise_neg_policy=model_cfg.get("listwise_neg_policy", "all"),
        listwise_force_pos=bool(model_cfg.get("listwise_force_pos", False)),
        hard_neg_weight=float(model_cfg.get("hard_neg_weight", 0.0)),
        hard_neg_margin=float(model_cfg.get("hard_neg_margin", 0.2)),
        hard_neg_pos_strategy=model_cfg.get("hard_neg_pos_strategy", "max"),
        debug_logits=False, debug_logits_topk=5,
        device_map=None, device=f"cuda:{local}",
        dtype=getattr(torch, base_cfg["pipeline"].get("dtype", "bfloat16")),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    if ddp:
        for p in params:
            dist.broadcast(p.data, src=0)
    optimizer = torch.optim.AdamW(params, lr=float(ts.get("lr", 1e-5)), weight_decay=float(ts.get("weight_decay", 0.0)))

    tr.set_seed(seed + rank)  # independent dropout streams after identical parameter broadcast
    start_epoch = start_batch = global_step = opt_step = oom_skips = 0
    if resume_dir is not None:
        st = torch.load(os.path.join(resume_dir, "training_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        tr._move_optimizer_state_to_param_device(optimizer)
        states = st.get("rank_rng_states")
        tr._restore_rng_state(states[rank] if states else st.get("rng_state"))
        start_epoch, start_batch = int(st.get("epoch", 0)), int(st.get("batch_in_epoch", 0))
        global_step, opt_step = int(st.get("global_step", 0)), int(st.get("optimizer_step", 0))
        oom_skips = int(st.get("oom_skips", 0))
        if int(st.get("grad_accum", grad_accum)) != grad_accum or int(st.get("world_size", world)) != world:
            raise SystemExit("resume needs the same grad_accum and number of GPUs as the checkpoint")
        log(f"[resume] epoch={start_epoch} batch_in_epoch={start_batch} global_step={global_step} opt_step={opt_step}")

    def sync_and_step():
        if ddp:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError("Non-finite gradient; stopped without saving")
        torch.nn.utils.clip_grad_norm_(params, float(ts.get("max_grad_norm", 1.0)), error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def save(path, epoch, batch_in_epoch, completed_steps):
        rng = tr._capture_rng_state()
        states = [None] * world
        if ddp:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        if main_proc:
            tmp = path + ".incomplete"
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            os.makedirs(tmp)
            model.save(tmp)
            torch.save({"epoch": epoch, "batch_in_epoch": batch_in_epoch,
                        "global_step": completed_steps, "optimizer_step": opt_step,
                        "grad_accum": grad_accum, "world_size": world, "oom_skips": oom_skips,
                        "optimizer": optimizer.state_dict(), "rank_rng_states": states,
                        "rng_state": states[0]}, os.path.join(tmp, "training_state.pt"))
            if os.path.exists(path):
                raise RuntimeError(f"Refusing to overwrite existing checkpoint {path}")
            os.rename(tmp, path)
            print(f"[checkpoint] {path}", flush=True)
        if ddp:
            dist.barrier()

    if len(dataset) < world:
        raise RuntimeError("Training dataset smaller than world size")
    model.train()
    t_run = time.time()
    done = False
    last_epoch, last_batch = start_epoch, start_batch
    for epoch in range(start_epoch, epochs):
        base = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
        base.set_epoch(epoch)
        skip = start_batch if epoch == start_epoch else 0
        loader = DataLoader(dataset, batch_size=1, sampler=SkipSampler(base, skip),
                            num_workers=int(ts.get("num_workers", 2)), pin_memory=True, collate_fn=tr.passthrough_collate)
        n_epoch = len(base)
        for bi, batch in enumerate(loader, start=skip):
            t0 = time.time()
            out = None
            # flags[0] = CUDA OOM on this rank, flags[1] = non-finite loss on this rank. One collective per
            # step AFTER forward/backward, so a rank that failed never skips a collective the other one makes.
            flags = torch.zeros(2, device=f"cuda:{local}")
            try:
                out = model(batch)
                loss = out["loss"] / grad_accum
                if not torch.isfinite(loss.detach()).all():
                    flags[1] = 1
                elif loss.requires_grad:
                    loss.backward()
            except RuntimeError as err:
                if not tr._is_oom_error(err):
                    raise
                out = None
                flags[0] = 1
            if ddp:
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            if flags[1].item():
                raise RuntimeError(f"Non-finite loss at step {global_step} ({batch.get('dataset_name')} "
                                   f"{batch.get('video_id')}/{batch.get('exp_id')}); stopped without saving")
            if flags[0].item():
                # same decision on every rank: drop this step's gradients everywhere, record which sample did it
                optimizer.zero_grad(set_to_none=True)
                out = loss = None
                torch.cuda.empty_cache()
                oom_skips += 1
                with open(os.path.join(output_dir, f"oom_skips_rank{rank}.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"global_step": global_step, "epoch": epoch, "batch_in_epoch": bi,
                                        "dataset": batch.get("dataset_name"), "video_id": batch.get("video_id"),
                                        "exp_id": batch.get("exp_id"), "cands": len(batch.get("candidates", []))},
                                       ensure_ascii=False) + "\n")
                if main_proc:
                    print(f"[oom-skip] step {global_step} dropped on all ranks ({oom_skips} so far)", flush=True)
                if oom_skips > a.max_oom_skips:
                    raise RuntimeError(f"{oom_skips} OOM-dropped steps > --max-oom-skips {a.max_oom_skips}; "
                                       f"see {output_dir}/oom_skips_rank*.jsonl")
            elif (global_step + 1) % grad_accum == 0:
                sync_and_step()
                opt_step += 1
                if save_every > 0 and opt_step % save_every == 0:
                    save(os.path.join(output_dir, f"checkpoint_{global_step + 1}"), epoch, bi + 1, global_step + 1)
            if main_proc and out is not None and global_step % log_every == 0:
                el = time.time() - t_run
                left = (n_epoch * (epochs - epoch) - bi - 1) * el / max(1, global_step + 1 - start_batch)
                print(f"step {global_step} | {bi + 1}/{n_epoch} | dataset {batch.get('dataset_name')} | "
                      f"loss {float(out['loss'].detach()):.4f} | cls {float(out['loss_cls'].detach()):.4f} | "
                      f"cands {len(batch.get('candidates', []))} | {time.time() - t0:.2f}s | eta {left / 3600:.1f} h",
                      flush=True)
            global_step += 1
            last_epoch, last_batch = epoch, bi + 1
            if max_steps > 0 and global_step >= max_steps:
                done = True
                break
        if done:
            break
    if global_step % grad_accum:
        raise RuntimeError("Unflushed accumulation: use grad_accum=1 for this release")
    save(os.path.join(output_dir, "final"), last_epoch, last_batch, global_step)
    if main_proc:
        with open(os.path.join(output_dir, "final", "run_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"global_step": global_step, "optimizer_step": opt_step, "oom_skips": oom_skips,
                       "world_size": world}, f, indent=2)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
