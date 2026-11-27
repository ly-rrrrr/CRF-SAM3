#!/usr/bin/env python3
"""Stage 5a (CPU): the Verify joint-training yaml, from the caches stage 4 actually wrote.

* One mixed dataset over the four train caches (repo's build_training_dataset), Sa2VA-style repeats
  computed by v2/mix.py from what is in the caches (videos V, expressions E with a meta.json).
* Same labels for every dataset: iou_map in the cache is J_best, pos_iou 0.5 / neg_iou 0.1;
  listwise_force_pos off (no cached expression lacks a positive); one hard-negative setting for all.
* listwise_max_candidates = the largest pool in the caches, so the trainer never trims by GT IoU.
* Model / optimiser recipe = the paper's joint config (config_joint_train_train_dms_recall24_keep7_filter_failures_20260623),
  F = 4 frames per candidate, fresh LoRA on Qwen3-VL-4B-Instruct.

  python tools/make_verify_config.py --repo $REPO --cache runs/verify/cache --out runs/verify/train.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from v2.mix import repeats  # noqa: E402

BASE = {"mevis": "configs/crf_sam3_mevis_train_base.yaml", "davis17": "configs/crf_sam3_davis_train_base.yaml",
        "refytvos": "configs/crf_sam3_refytros_train_base.yaml", "revos": "configs/crf_sam3_revos_train_base.yaml"}
RECIPE = "crf_sam3_attn/config_joint_train_train_dms_recall24_keep7_filter_failures_20260623.yaml"


def scan(root: Path):
    vids, n, pool = set(), 0, 0
    for m in root.rglob("meta.json"):
        meta = json.loads(m.read_text(encoding="utf-8"))
        vids.add(meta["video_id"])
        n += 1
        pool = max(pool, len(meta.get("obj_ids") or []))
    return len(vids), n, pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cache", required=True, help="dir with <dataset>/train_root/train/...")
    ap.add_argument("--datasets", default="mevis,davis17,refytvos,revos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--output-dir", default="", help="checkpoint dir (default: next to --out)")
    ap.add_argument("--grad-accum", type=int, default=1, help="per GPU; effective batch = GPUs x this")
    ap.add_argument("--lr", type=float, default=0.0, help="0 = the paper recipe's")
    ap.add_argument("--hard-neg-weight", type=float, default=0.3)
    ap.add_argument("--save-every", type=int, default=1000, help="optimizer steps")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N samples per GPU")
    a = ap.parse_args(argv)

    repo, cache, out = Path(a.repo).resolve(), Path(a.cache).resolve(), Path(a.out).resolve()
    recipe = yaml.safe_load((repo / RECIPE).read_text(encoding="utf-8"))
    counts, pools, roots = {}, {}, {}
    for d in a.datasets.split(","):
        r = cache / d / "train_root"
        v, e, p = scan(r / "train")
        if not e:
            raise SystemExit(f"no cached samples under {r}/train; run stage 4 for {d}")
        counts[d], pools[d], roots[d] = (v, e), p, str(r)
    rep = repeats(counts)
    model = dict(recipe["model"])
    model.update(listwise_force_pos=False, listwise_neg_policy="threshold", hard_neg_weight=a.hard_neg_weight,
                 hard_neg_margin=0.1, hard_neg_pos_strategy="max", device_map=None)
    train = dict(recipe["train"])
    train.update(output_dir=a.output_dir or str(out.parent / "ckpt"), grad_accum=a.grad_accum,
                 save_every=a.save_every, save_merged=False, num_workers=0, epochs=1, max_steps=a.max_steps)
    if a.lr:
        train["lr"] = a.lr
    per_ds = []
    for d in counts:
        per_ds.append({"name": "refytros" if d == "refytvos" else d, "base_config": BASE[d],
                       "offline_cache_root": roots[d], "repeats": rep[d]["r"],
                       "pos_iou": 0.5, "neg_iou": 0.1, "listwise_neg_policy": "threshold",
                       "hard_neg_weight": a.hard_neg_weight, "hard_neg_margin": 0.1, "hard_neg_pos_strategy": "max"})
    data = {"split": "train", "num_frames": 4, "mask_style": "mask", "pos_iou": 0.5, "neg_iou": 0.1,
            "sample_neg_prob": 0.5, "listwise": True, "listwise_max_candidates": max(64, max(pools.values())),
            "max_videos": None, "max_video_frames": None, "max_expressions": None, "cache_sam3": False,
            "category_source": "concept", "offline_mode": True, "offline_prepare": False,
            "offline_overwrite": False, "offline_strict": True, "datasets": per_ds}
    cfg = {"base_config": BASE["mevis"], "seed": 42, "model": model, "train": train, "data": data,
           "debug": {"enable": False}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# written by crf_sam3_v2/tools/make_verify_config.py\n"
                   + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    total = sum(r["samples"] for r in rep.values())
    (out.with_suffix(".summary.json")).write_text(json.dumps({"repeats": rep, "max_pool": pools,
                                                               "samples_per_epoch": total}, indent=2), encoding="utf-8")
    for d, r in rep.items():
        print(f"[verify-cfg] {d}: videos {r['videos']} expressions {r['expressions']} -> repeats {r['r']} "
              f"({r['samples']} samples), max pool {pools[d]}")
    print(f"[verify-cfg] {total} samples per epoch -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
