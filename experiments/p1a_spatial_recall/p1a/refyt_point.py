#!/usr/bin/env python3
"""Ref-YouTube-VOS: paper mainline + heatmap point seeds, in one pass, with masks written by frame position.

Per expression (everything except step 3 is the repo's own CRFSAM3Pipeline, built from the mainline yaml):
  1. pipeline.predict(): Ground word (category cache of the mainline) -> SAM3 text recall (recall24/keep7)
     -> visual tubes -> Verify scores -> top num_targets.                               => row "base"
     Backup words are switched off unless --keep-backup (paper rule: no backup words).
  2. Ground v2 heatmap (gv2.predict npz) on its 3 frames: blobs not covered by the recalled candidates
     (coverage < 0.5), top --budget blobs by mass, their peaks.
  3. Each peak -> SAM3 tracker positive click on that frame (same SAM3 predictor as step 1, no word)
     -> full-video tube; skipped if the point is already on a candidate, dropped if tube IoU > --dup-iou
     with an existing one; at most --max-new tubes.
  4. The new tubes are scored by the SAME Verify (same style rule: colour words -> bbox); the top
     num_targets of (candidates + new tubes) is the row "point".
Both rows are written as <out>/<row>/Annotations/<video>/<exp>/<frame>.png keyed by frame POSITION
(the repo writer looks up int(frame_name) first, which misplaces 16% of Ref-YTVOS frames).
With --gt, per-expression recall diagnostics (best candidate J before / after the point tubes) are logged.

  cd p1a_spatial_recall && python -m p1a.refyt_point --repo $REPO --config <mainline yaml> \
      --data-root .../Ref-Youtube-VOS --heat-dir <gv2 heat npz dir> --out runs/refyt_v2 --gt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .common import tube_iou, video_j
from .seeds import select_seeds


def _bool(m) -> np.ndarray:
    a = np.asarray(m)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return a > 0


def candidates_of(all_masks: Dict[int, Dict[int, np.ndarray]]) -> Dict[int, Dict[int, np.ndarray]]:
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for t, fm in all_masks.items():
        for o, m in fm.items():
            out.setdefault(int(o), {})[int(t)] = _bool(m)
    return out


def select(scores: Dict[int, float], k: int) -> List[int]:
    """SemanticDiscriminator.score_and_filter's rule: ids ascending, stable sort by score desc, top k."""
    valid = [i for i in sorted(scores) if scores[i] is not None]
    return sorted(valid, key=lambda i: scores[i], reverse=True)[: max(int(k), 0)]


def on_any(pt, masks) -> bool:
    for m in masks:
        h, w = m.shape
        x, y = min(int(pt[0] * w), w - 1), min(int(pt[1] * h), h - 1)
        if m[y, x]:
            return True
    return False


def load_heat(path: Path, frames: List[str]):
    """npz from gv2.predict; frame indices re-mapped to positions in the SORTED frame list via frame_stems."""
    z = np.load(path)
    R = z["responses"].astype(np.float32)
    grid, merge = z["image_grid_thw"], int(z["merge"])
    if "frame_stems" in z:
        pos = {s: i for i, s in enumerate(frames)}
        fidx = [pos[str(s)] for s in z["frame_stems"]]
    else:
        fidx = [int(x) for x in z["frame_indices"]]
    return R, grid, merge, fidx


def write_pngs(out_root: Path, video_id: str, exp_id: str, frames: List[str], masks: Dict[int, Dict[int, np.ndarray]],
               h: int, w: int, make_instance_mask) -> None:
    from PIL import Image

    d = out_root / "Annotations" / video_id / str(exp_id)
    d.mkdir(parents=True, exist_ok=True)
    for pos, name in enumerate(frames):
        Image.fromarray(make_instance_mask(masks.get(pos, {}), h, w)).save(d / f"{name}.png")


def build_pipeline(a):
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    import crf_sam3_demo as demo
    from run_crf_sam3_dataset import load_config
    from run_crf_sam3_refytros_dataset import _resolve_prompt_template

    cfg = load_config(a.config)
    m, vt, dc = cfg["model"], cfg.get("visual_tube", {}), cfg.get("discrimination", {})
    if a.adapter:
        m["mllm_adapter_path"] = a.adapter
    if vt.get("sampling_strategy", "uniform") == "qframe" or dc.get("backend", "mllm") != "mllm":
        raise SystemExit("refyt_point supports the mainline setting only (uniform tubes, mllm scorer)")
    pipe = demo.CRFSAM3Pipeline(
        mllm_path=m.get("mllm_path"), base_mllm_path=a.base_mllm or m.get("base_mllm_path"),
        mllm_adapter_path=m.get("mllm_adapter_path"), sam3_checkpoint=a.sam3_checkpoint or m["sam3_checkpoint"],
        num_tube_frames=int(a.num_tube_frames or cfg.get("pipeline", {}).get("num_tube_frames", 8)),
        model_type=m.get("model_type", "qwen3"), prompt_type=dc.get("prompt_type", "yesno"),
        prompt_template_path=_resolve_prompt_template(dc.get("prompt_type", "yesno")),
        visualization_style=vt.get("visualization_style", "mask"), sampling_strategy="uniform",
        qframe_config=None, concept_use_qframe=False, concept_config=cfg.get("concept_extraction", {}),
        discriminator_backend="mllm", reranker_cfg=dc.get("reranker", {}),
        sam3_recall_cfg=cfg.get("sam3_recall", {}) or {},
    )
    if not a.keep_backup:
        orig = pipe.concept_extractor.extract_category

        def no_backup(expr, video_path=None, *args, **kw):
            c, n, r = orig(expr, video_path, *args, **kw)
            r = dict(r or {})
            r["backup_categories"] = []
            return c, n, r

        pipe.concept_extractor.extract_category = no_backup
    return pipe, demo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True, help="mainline yaml, e.g. configs/crf_sam3_config_query_grounding_refytros_valid_20260701.yaml")
    ap.add_argument("--adapter", default="", help="Verify LoRA (the mainline script passes checkpoint_39000)")
    ap.add_argument("--base-mllm", default="")
    ap.add_argument("--sam3-checkpoint", default="")
    ap.add_argument("--num-tube-frames", type=int, default=0)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--meta-file", default="meta_expressions_challenge.json")
    ap.add_argument("--heat-dir", required=True)
    ap.add_argument("--keep-backup", action="store_true", help="allow the repo's backup words when recall is empty")
    ap.add_argument("--agg", default="max")
    ap.add_argument("--abs-thr", type=float, default=0.5)
    ap.add_argument("--rel-thr", type=float, default=0.6)
    ap.add_argument("--coverage-thr", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=3)
    ap.add_argument("--dup-iou", type=float, default=0.7)
    ap.add_argument("--gt", action="store_true", help="log recall diagnostics from <data-root>/<split>/Annotations")
    ap.add_argument("--limit", type=int, default=0, help="first N videos")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for k in ("data_root", "heat_dir", "out", "config", "repo"):
        setattr(a, k, str(Path(getattr(a, k)).resolve()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.data_root) / a.split / a.meta_file).read_text(encoding="utf-8"))["videos"]
    vids = sorted(meta)[: a.limit] if a.limit else sorted(meta)
    res_path = out / "scores.jsonl"
    done = set()
    if res_path.is_file():
        done = {json.loads(l)["key"] for l in res_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [(v, e) for v in vids for e in meta[v]["expressions"] if f"{v}/{e}" not in done]
    print(f"[refyt] {sum(len(meta[v]['expressions']) for v in vids)} expressions, {len(todo)} to run", flush=True)
    if not todo:
        return

    pipe, demo = build_pipeline(a)
    from run_crf_sam3_dataset import _infer_video_hw, _make_instance_mask
    from .common import PngDirGT
    from .sam3_ops import SAM3Session
    import torch

    gt_loader = PngDirGT(str(Path(a.data_root) / a.split / "Annotations" / "{video_id}" / "{exp_id}" / "{frame}.png")) if a.gt else None
    t_start, n_err = time.time(), 0
    for n, (vid, eid) in enumerate(todo, 1):
        key = f"{vid}/{eid}"
        t0 = time.time()
        try:
            video_path = os.path.join(a.data_root, a.split, "JPEGImages", vid)
            frames = sorted(str(x) for x in meta[vid]["frames"])
            h, w = (int(x) for x in _infer_video_hw(video_path, frames))
            expr = meta[vid]["expressions"][eid].get("exp", "")

            final_masks, _, det = pipe.predict(video_path, expr, return_details=True)
            all_masks = det.get("all_masks") or {}
            k = max(int(det.get("num_targets") or 1), 1)
            category = det.get("category") or det.get("primary_category") or ""
            scores = {int(o): (None if d.get("score") is None else float(d["score"])) for o, d in (det.get("scores") or {}).items()}
            base_sel = [int(x) for x in det.get("selected_candidates", [])]
            cands = candidates_of(all_masks)

            heat = Path(a.heat_dir) / vid / f"{eid}.npz"
            rec = {"key": key, "video_id": vid, "exp_id": str(eid), "expression": expr, "category": category,
                   "num_targets": k, "base_ids": sorted(cands), "base_selected": base_sel}
            new: List[Dict[int, np.ndarray]] = []
            if heat.is_file():
                R, grid, merge, fidx = load_heat(heat, frames)
                sel = select_seeds(R, grid, fidx, cands, merge=merge, agg=a.agg, abs_thr=a.abs_thr,
                                   rel_thr=a.rel_thr, coverage_thr=a.coverage_thr, budget=a.budget)
                blobs = sorted(sel["uncovered"], key=lambda b: -b.mass)[: a.budget]
                rec.update(n_blobs=len(sel["blobs"]), n_uncovered=len(sel["uncovered"]), seeds=[b.to_json() for b in blobs])
                if blobs:
                    with SAM3Session(pipe.sam3_recall.predictor, video_path) as sess:
                        sess.warm_text = category or None
                        base = list(cands.values())
                        info = []
                        for b in blobs:
                            t = b.frame_idx
                            for pt in b.peaks:
                                if len(new) >= a.max_new:
                                    break
                                if on_any(pt, [tb[t] for tb in base + new if t in tb]):
                                    continue
                                tube = sess.track_seed(t, points=[pt], labels=[1])
                                if tube and not any(tube_iou(tube, o) > a.dup_iou for o in base + new):
                                    new.append(tube)
                                    info.append({"frame": t, "point": pt})
                        rec["point_info"] = info
            else:
                rec["no_heatmap"] = True
            pids = [1000 + i for i in range(len(new))]
            if new:
                pm: Dict[int, Dict[int, np.ndarray]] = {}
                for oid, tube in zip(pids, new):
                    for t, m in tube.items():
                        pm.setdefault(int(t), {})[oid] = m
                style = "bbox" if demo.contains_color_attribute(expr) else pipe._original_visualization_style
                pipe.tube_builder.visualization_style = style
                try:
                    tubes = pipe.tube_builder.build_tubes(video_path, pm, expr)
                    _, d2 = pipe.discriminator.score_and_filter(tubes, expr, k, return_details=True, visualization_style=style)
                finally:
                    pipe.tube_builder.visualization_style = pipe._original_visualization_style
                for o, d in d2.items():
                    scores[int(o)] = None if d.get("score") is None else float(d["score"])
            point_sel = select(scores, k) if new else base_sel
            rec.update(point_ids=pids, scores={str(o): s for o, s in scores.items()}, point_selected=point_sel)

            write_pngs(out / "base", vid, eid, frames, final_masks, h, w, _make_instance_mask)
            tubes_all = dict(cands)
            tubes_all.update(zip(pids, new))
            pmasks: Dict[int, Dict[int, np.ndarray]] = {}
            for o in point_sel:
                for t, m in tubes_all.get(o, {}).items():
                    pmasks.setdefault(int(t), {})[o] = m
            write_pngs(out / "point", vid, eid, frames, pmasks, h, w, _make_instance_mask)

            if gt_loader is not None:
                gt = gt_loader(vid, eid, frames)
                jb = [video_j(t, gt) for t in cands.values()]
                jp = [video_j(t, gt) for t in new]
                rec.update(pool_j_base=max(jb, default=0.0), pool_j_point=max(jb + jp, default=0.0), j_point_tubes=jp)
            rec["sec"] = round(time.time() - t0, 1)
            with open(res_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            el = time.time() - t_start
            print(f"[{n}/{len(todo)}] {key} '{category}' k={k} cands={len(cands)} +pt={len(new)} "
                  f"sel {base_sel}->{point_sel} {rec['sec']}s (eta {el / n * (len(todo) - n) / 60:.0f} min)", flush=True)
        except torch.OutOfMemoryError:
            raise
        except Exception as e:  # not written to scores.jsonl, so a re-run retries it
            if "out of memory" in str(e).lower():
                raise
            n_err += 1
            import traceback

            print(f"[error] {key}: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if n_err:
        print(f"[refyt] {n_err} expressions failed; re-run the same command to retry them")


def summary(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", action="append", default=[], help="name=evaluation json of another run")
    a = ap.parse_args(argv)
    out = Path(a.out)
    L = ["# Ref-YouTube-VOS valid：主线 vs 主线 + 热力图点种子（都按帧位置写掩码）", "",
         "| 结果 | J | F | J&F（表达式级） | J&F（视频级） |", "|---|---|---|---|---|"]
    rows = [tuple(x.split("=", 1)) for x in a.ref] + [("主线（本脚本，无 backup 词）", str(out / "base" / "evaluation_results_refytros_png.json")),
                                                     ("主线 + 热力图点种子", str(out / "point" / "evaluation_results_refytros_png.json"))]
    for name, p in rows:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            e, v = d["overall_expression_level"], d["overall_video_level"]
            L.append(f"| {name} | {e['J']:.2f} | {e['F']:.2f} | **{e['J&F']:.2f}** | {v['J&F']:.2f} |")
        except (FileNotFoundError, KeyError):
            L.append(f"| {name} | 未找到 {p} | | | |")
    sp = out / "scores.jsonl"
    if sp.is_file():
        rs = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(rs)
        trig = sum(1 for r in rs if r.get("point_ids"))
        ch = sum(1 for r in rs if sorted(r["point_selected"]) != sorted(r["base_selected"]))
        L += ["", f"- 已跑 {n} 条；补了点种子轨迹的 {trig} 条（平均每条 +{sum(len(r.get('point_ids', [])) for r in rs) / max(n, 1):.2f} 个候选）；"
                  f"Verify 选择因此改变的 {ch} 条。",
              f"- 0 候选的表达式：{sum(1 for r in rs if not r['base_ids'])} 条。"]
        if rs and "pool_j_base" in rs[0]:
            b = np.array([r["pool_j_base"] for r in rs])
            p = np.array([r["pool_j_point"] for r in rs])
            L.append(f"- 候选池 Recall@0.5：{100 * (b >= 0.5).mean():.2f}% → {100 * (p >= 0.5).mean():.2f}%；"
                     f"池内最好候选的平均 J：{100 * b.mean():.2f} → {100 * p.mean():.2f}。")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(sys.argv[2:])
    else:
        main()
