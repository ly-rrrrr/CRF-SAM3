#!/usr/bin/env python3
"""One mainline yaml for all four datasets (and for the Verify training caches).

Starts from the ReVOS 0806 config (the only paper config whose Verify prompt matches training) and fixes
the settings that must be identical at train and test time:
  * discrimination.prompt_type = yesno_images -> configs/discrimination_prompt_yesno_train_aligned_20260806.txt,
    frames sent as separate images, exactly crf_sam3_attn/model.py::_build_prompt used in training;
  * pipeline.num_tube_frames = 4 (training num_frames = 4);
  * SAM3-DMS recall 24 / keep 7 by SAM3 score sum (checked, not changed);
  * no backup words / semantic expansion (mainline turns backups off; expansion must be absent);
  * Ground words are passed at run time (category_override_cache_path is cleared here).
--adapter sets the Verify LoRA (leave empty for the cache stage: Verify does not score there).

  python tools/make_mainline_yaml.py --repo $REPO --adapter runs/verify/ckpt/final --out runs/verify/mainline.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SRC = "configs/crf_sam3_config_revos_valid_revos_query_ranker_20260806.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve()
    cfg = yaml.safe_load((repo / a.src).read_text(encoding="utf-8"))
    cfg.setdefault("pipeline", {})["num_tube_frames"] = a.frames
    cfg.setdefault("discrimination", {})["prompt_type"] = "yesno_images"
    cfg["discrimination"]["backend"] = "mllm"
    if a.adapter:
        cfg.setdefault("model", {})["mllm_adapter_path"] = str(Path(a.adapter).resolve())
    ce = cfg.setdefault("concept_extraction", {})
    ce.pop("category_override_cache_path", None)
    rec = cfg.get("sam3_recall") or {}
    problems = []
    if cfg.get("semantic_query_expander") or (cfg.get("concept_extraction", {}).get("semantic_query_expander")):
        problems.append("semantic_query_expander is set")
    vt = cfg.get("visual_tube", {})
    if vt.get("sampling_strategy", "uniform") != "uniform":
        problems.append(f"visual_tube.sampling_strategy={vt.get('sampling_strategy')}")
    if rec.get("backend") != "dms":
        problems.append("sam3_recall.backend must be dms")
    if rec.get("candidate_recall_top_k") != 24 or rec.get("candidate_keep_top_k") != 7:
        problems.append("SAM3 recall/keep must be 24/7")
    if problems:
        raise SystemExit("source yaml is not the paper setting: " + "; ".join(problems))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# written by crf_sam3_v2/tools/make_mainline_yaml.py from {a.src}\n"
                   + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    keys = {k: rec.get(k) for k in rec if "top_k" in k or k in ("backend",)}
    print(f"[yaml] {out}: frames {a.frames}, prompt yesno_images, adapter {cfg.get('model', {}).get('mllm_adapter_path')}, "
          f"sam3_recall {keys}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
