#!/usr/bin/env python3
"""CRF-SAM3 v2 mainline: one pass per expression, same code for every dataset.

  1. Ground v2 (gv2.predict, run beforehand by run_v2.sh) gives the category word + a token heatmap
     on 3 uniform frames. The word is fed to the pipeline through the yaml's category_override_cache_path.
  2. CRFSAM3Pipeline.predict (the repo's own code, built from the dataset's mainline yaml):
     Qwen3-VL extractor (num_targets) -> SAM3 text recall (recall/keep from the yaml) -> visual tubes
     -> Verify scores -> top num_targets.                                         => row "words"
     Backup words are off (paper rule) unless --keep-backup.
  3. Heatmap point seeds: blobs of the heatmap not covered by the recalled candidates (coverage < 0.5),
     top --budget blobs by mass, their peaks -> SAM3 tracker positive click -> full-video tube
     (skipped if the point is already on a candidate, dropped if tube IoU > --dup-iou, <= --max-new tubes).
  4. The new tubes are scored by the SAME Verify; top num_targets of (candidates + new tubes) => row "final"
Masks are written keyed by SAM3 frame POSITION (sorted JPEG order), never by int(frame_name).

  python -m v2.mainline run --dataset refytvos --repo $REPO --config <yaml> --data-root <Ref-Youtube-VOS> \
      --categories <gv2 categories json> --heat-dir <gv2 npz dir> --out runs/refytvos
  python -m v2.mainline export  --dataset revos --data-root <revos> --out runs/revos   # ReVOS: merge -> results.json
  python -m v2.mainline summary --dataset refytvos --out runs/refytvos --ref "name=eval json"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROWS = ("words", "final")
POINT_BASE = 1000


# ----------------------------------------------------------------------------- datasets
class Dataset:
    name = ""
    eval_json = ""

    def __init__(self, data_root: str, split: str = "valid", meta_file: str = "") -> None:
        self.root = Path(data_root)
        self.split = split
        self.meta_file = meta_file or self.default_meta

    @property
    def meta_path(self) -> Path:
        raise NotImplementedError

    def video_dir(self, vid: str) -> str:
        raise NotImplementedError

    def videos(self) -> Dict:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))["videos"]

    def gt_loader(self):
        raise NotImplementedError

    def write(self, out: Path, row: str, vid: str, eid: str, meta_frames: List[str], pos_of: Dict[str, int],
              masks: Dict[int, Dict[int, np.ndarray]], h: int, w: int, expr: str) -> None:
        raise NotImplementedError

    def is_written(self, out: Path, row: str, vid: str, eid: str, meta_frames: List[str]) -> bool:
        raise NotImplementedError


class RefYTVOS(Dataset):
    """<root>/<split>/JPEGImages/<vid>/<frame>.jpg ; output <out>/<row>/Annotations/<vid>/<eid>/<frame>.png"""
    name = "refytvos"
    default_meta = "meta_expressions_challenge.json"
    eval_json = "evaluation_results_refytros_png.json"

    @property
    def meta_path(self) -> Path:
        return self.root / self.split / self.meta_file

    def video_dir(self, vid: str) -> str:
        return str(self.root / self.split / "JPEGImages" / vid)

    def gt_loader(self):
        from p1a.common import PngDirGT

        return PngDirGT(str(self.root / self.split / "Annotations" / "{video_id}" / "{exp_id}" / "{frame}.png"))

    def write(self, out, row, vid, eid, meta_frames, pos_of, masks, h, w, expr):
        from PIL import Image
        from run_crf_sam3_dataset import _make_instance_mask

        d = out / row / "Annotations" / vid / str(eid)
        d.mkdir(parents=True, exist_ok=True)
        for name in meta_frames:
            fm = masks.get(pos_of[name], {}) if name in pos_of else {}
            Image.fromarray(_make_instance_mask(fm, h, w)).save(d / f"{name}.png")

    def is_written(self, out, row, vid, eid, meta_frames):
        d = out / row / "Annotations" / vid / str(eid)
        return all((d / f"{n}.png").is_file() for n in meta_frames)


class ReVOS(Dataset):
    """<root>/<vid>/<frame>.jpg ; output <out>/<row>/pred/<vid>/<eid>.json, merged by `export` into results.json"""
    name = "revos"
    default_meta = "meta_expressions_valid_.json"
    eval_json = "revos_valid_official.json"

    @property
    def meta_path(self) -> Path:
        return self.root / self.meta_file

    def video_dir(self, vid: str) -> str:
        return str(self.root / vid)

    def gt_loader(self):
        from p1a.common import ReVOSGT

        return ReVOSGT(str(self.root / "mask_dict.json"), str(self.meta_path))

    @staticmethod
    def pred_path(out: Path, row: str, vid: str, eid: str) -> Path:
        return out / row / "pred" / vid / f"{eid}.json"

    def write(self, out, row, vid, eid, meta_frames, pos_of, masks, h, w, expr):
        import run_crf_sam3_revos_dataset as rv

        rles = [rv._encode_mask_to_rle(rv._compose_binary_mask(masks.get(pos_of[n], {}) if n in pos_of else {}, h, w))
                for n in meta_frames]
        p = self.pred_path(out, row, vid, eid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"video_id": vid, "exp_id": eid, "exp": expr, "frames": meta_frames,
                                 "prediction_masks": rles}), encoding="utf-8")

    def is_written(self, out, row, vid, eid, meta_frames):
        return self.pred_path(out, row, vid, eid).is_file()


DATASETS = {"refytvos": RefYTVOS, "revos": ReVOS}


# ----------------------------------------------------------------------------- helpers
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


def select(scores: Dict[int, Optional[float]], k: int) -> List[int]:
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


def load_heat(path: Path, pos_of: Dict[str, int]):
    """npz from gv2.predict; its frames are mapped to SAM3 positions through frame_stems."""
    z = np.load(path)
    R = z["responses"].astype(np.float32)
    fidx = [pos_of[str(s)] for s in z["frame_stems"]]
    return R, z["image_grid_thw"], int(z["merge"]), fidx


def masks_of(tubes: Dict[int, Dict[int, np.ndarray]], ids: List[int]) -> Dict[int, Dict[int, np.ndarray]]:
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for o in ids:
        for t, m in tubes.get(o, {}).items():
            out.setdefault(int(t), {})[o] = m
    return out


def word_coverage(categories: str, ds: Dataset, videos: Dict) -> tuple:
    """(#expressions with a Ground v2 word, total, first missing) under the pipeline's lookup key."""
    recs = json.loads(Path(categories).read_text(encoding="utf-8"))
    recs = recs.get("records", recs) if isinstance(recs, dict) else recs
    have = {(str(r.get("video_id", "")).strip(), str(r.get("expression", "")).strip())
            for r in recs if str(r.get("pred_category") or r.get("category") or "").strip()}
    n = ok = 0
    first = None
    for vid, v in videos.items():
        for eid, e in v["expressions"].items():
            n += 1
            if (vid, str(e.get("exp", "")).strip()) in have:
                ok += 1
            elif first is None:
                first = f"{vid}/{eid}"
    return ok, n, first


def install_recall_cache(recall, max_bytes: float) -> None:
    """Memoise SAM3 text recall per (video, word, budgets) for the current video only.

    Expressions of one video often share the word, and predict() re-runs the same deterministic SAM3
    propagation for each. Results above max_bytes are not kept (huge videos just run as before)."""
    orig = recall.recall_all_instances_stream
    state = {"video": None, "items": {}}

    def cached(video_path, category, **kw):
        key = (category, tuple(sorted((k, v) for k, v in kw.items() if k != "session_id")))
        if state["video"] != video_path:
            state.update(video=video_path, items={})
        if key in state["items"]:
            for f, m, s in state["items"][key]:
                yield f, dict(m), dict(s)
            return
        keep, size = [], 0
        for f, m, s in orig(video_path, category, **kw):
            if keep is not None:
                size += sum(int(np.asarray(x).nbytes) for x in m.values())
                if size <= max_bytes:
                    keep.append((f, dict(m), dict(s)))
                else:
                    keep = None
            yield f, m, s
        if keep is not None:
            state["items"][key] = keep

    recall.recall_all_instances_stream = cached


def build_pipeline(a):
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    import crf_sam3_demo as demo
    from run_crf_sam3_dataset import load_config
    from run_crf_sam3_refytros_dataset import _resolve_prompt_template

    cfg = load_config(a.config)
    m, vt, dc = cfg["model"], cfg.get("visual_tube", {}), cfg.get("discrimination", {})
    ce = dict(cfg.get("concept_extraction", {}))
    ce["category_override_cache_path"] = a.categories  # Ground v2 words
    if a.adapter:
        m["mllm_adapter_path"] = a.adapter
    if vt.get("sampling_strategy", "uniform") == "qframe" or dc.get("backend", "mllm") != "mllm":
        raise SystemExit("mainline v2 supports the paper setting only (uniform tubes, mllm scorer)")
    pipe = demo.CRFSAM3Pipeline(
        mllm_path=m.get("mllm_path"), base_mllm_path=a.base_mllm or m.get("base_mllm_path"),
        mllm_adapter_path=m.get("mllm_adapter_path"), visual_tube_temp_dir=str(Path(a.out) / "visual_tubes_temp"),
        sam3_checkpoint=a.sam3_checkpoint or m["sam3_checkpoint"],
        num_tube_frames=int(a.num_tube_frames or cfg.get("pipeline", {}).get("num_tube_frames", 8)),
        model_type=m.get("model_type", "qwen3"), prompt_type=dc.get("prompt_type", "yesno"),
        prompt_template_path=_resolve_prompt_template(dc.get("prompt_type", "yesno")),
        visualization_style=vt.get("visualization_style", "mask"), sampling_strategy="uniform",
        qframe_config=None, concept_use_qframe=False, concept_config=ce,
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
    if a.recall_cache_gb > 0:
        install_recall_cache(pipe.sam3_recall, a.recall_cache_gb * 1e9)
    offload = bool((cfg.get("sam3_recall") or {}).get("offload_video_to_cpu", False))
    return pipe, demo, offload


# ----------------------------------------------------------------------------- run (GPU)
def run(a) -> None:
    ds = DATASETS[a.dataset](a.data_root, a.split, a.meta_file)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = ds.videos()
    ok, n, first = word_coverage(a.categories, ds, videos)
    print(f"[v2] Ground v2 words: {ok}/{n} expressions", flush=True)
    if ok < n and not a.allow_missing_words:
        raise SystemExit(f"{n - ok} expressions have no Ground v2 word in {a.categories} (first {first}); "
                         "finish the heat step first, or pass --allow-missing-words (they fall back to the extractor's word)")
    vids = sorted(videos)[: a.limit] if a.limit else sorted(videos)
    res_path = out / "scores.jsonl"
    done = set()
    if res_path.is_file():
        done = {json.loads(l)["key"] for l in res_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [(v, e) for v in vids for e in videos[v]["expressions"] if f"{v}/{e}" not in done]
    print(f"[v2] {a.dataset}: {sum(len(videos[v]['expressions']) for v in vids)} expressions, {len(todo)} to run", flush=True)
    if not todo:
        return

    pipe, demo, offload = build_pipeline(a)
    from run_crf_sam3_dataset import _infer_video_hw
    from p1a.common import list_frames, tube_iou, video_j
    from p1a.sam3_ops import SAM3Session
    from p1a.seeds import select_seeds
    import torch

    gt_loader = ds.gt_loader() if a.gt else None
    t_start, n_err = time.time(), 0
    for n_done, (vid, eid) in enumerate(todo, 1):
        key = f"{vid}/{eid}"
        t0 = time.time()
        try:
            video_path = ds.video_dir(vid)
            meta_frames = [str(x) for x in videos[vid]["frames"]]
            sam3_frames = list_frames(video_path)          # SAM3's frame order = mask dict keys
            pos_of = {s: i for i, s in enumerate(sam3_frames)}
            h, w = (int(x) for x in _infer_video_hw(video_path, meta_frames))
            expr = videos[vid]["expressions"][eid].get("exp", "")

            final_masks, _, det = pipe.predict(video_path, expr, return_details=True)
            all_masks = det.get("all_masks") or {}
            k = max(int(det.get("num_targets") or 1), 1)
            category = det.get("category") or det.get("primary_category") or ""
            scores = {int(o): (None if d.get("score") is None else float(d["score"]))
                      for o, d in (det.get("scores") or {}).items()}
            words_sel = [int(x) for x in det.get("selected_candidates", [])]
            cands = candidates_of(all_masks)
            rec = {"key": key, "video_id": vid, "exp_id": str(eid), "expression": expr, "category": category,
                   "num_targets": k, "cand_ids": sorted(cands), "words_selected": words_sel}

            new: List[Dict[int, np.ndarray]] = []
            heat = Path(a.heat_dir) / vid / f"{eid}.npz"
            if heat.is_file():
                R, grid, merge, fidx = load_heat(heat, pos_of)
                sel = select_seeds(R, grid, fidx, cands, merge=merge, agg=a.agg, abs_thr=a.abs_thr,
                                   rel_thr=a.rel_thr, coverage_thr=a.coverage_thr, budget=a.budget)
                blobs = sorted(sel["uncovered"], key=lambda b: -b.mass)[: a.budget]
                rec.update(n_blobs=len(sel["blobs"]), n_uncovered=len(sel["uncovered"]), seeds=[b.to_json() for b in blobs])
                if blobs:
                    info = []
                    with SAM3Session(pipe.sam3_recall.predictor, video_path, offload) as sess:
                        sess.warm_text = category or None
                        base = list(cands.values())
                        for b in blobs:
                            t = b.frame_idx
                            for pt in b.peaks:
                                if len(new) >= a.max_new:
                                    break
                                if on_any(pt, [tb[t] for tb in base + new if t in tb]):
                                    continue
                                tube = sess.track_seed(t, points=[pt], labels=[1])
                                if tube and not any(tube_iou(tube, o) > a.dup_iou for o in base + new):
                                    new.append({int(f): _bool(m) for f, m in tube.items()})
                                    info.append({"frame": t, "point": [float(x) for x in pt]})
                    rec["point_info"] = info
            else:
                rec["no_heatmap"] = True

            pids = [POINT_BASE + i for i in range(len(new))]
            if new:
                pm = masks_of(dict(zip(pids, new)), pids)
                style = "bbox" if demo.contains_color_attribute(expr) else pipe._original_visualization_style
                pipe.tube_builder.visualization_style = style
                try:
                    tubes = pipe.tube_builder.build_tubes(video_path, pm, expr)
                    _, d2 = pipe.discriminator.score_and_filter(tubes, expr, k, return_details=True, visualization_style=style)
                finally:
                    pipe.tube_builder.visualization_style = pipe._original_visualization_style
                for o, d in d2.items():
                    scores[int(o)] = None if d.get("score") is None else float(d["score"])
            final_sel = select(scores, k) if new else words_sel
            rec.update(point_ids=pids, scores={str(o): s for o, s in scores.items()}, final_selected=final_sel)

            tubes_all = dict(cands)
            tubes_all.update(zip(pids, new))
            ds.write(out, "words", vid, eid, meta_frames, pos_of, final_masks, h, w, expr)
            ds.write(out, "final", vid, eid, meta_frames, pos_of, masks_of(tubes_all, final_sel), h, w, expr)

            if gt_loader is not None:
                gt = gt_loader(vid, eid, sam3_frames)
                jb = [video_j(t, gt) for t in cands.values()]
                jp = [video_j(t, gt) for t in new]
                rec.update(pool_j_words=max(jb, default=0.0), pool_j_final=max(jb + jp, default=0.0), j_point_tubes=jp)
            rec["sec"] = round(time.time() - t0, 1)
            with open(res_path, "a", encoding="utf-8") as f:  # written last: a sample counts as done only here
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            el = time.time() - t_start
            print(f"[{n_done}/{len(todo)}] {key} '{category}' k={k} cands={len(cands)} +pt={len(new)} "
                  f"sel {words_sel}->{final_sel} {rec['sec']}s (eta {el / n_done * (len(todo) - n_done) / 3600:.1f} h)", flush=True)
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
        print(f"[v2] {n_err} expressions failed; re-run the same command to retry them", flush=True)
        raise SystemExit(3)


# ----------------------------------------------------------------------------- export (ReVOS) / summary
def export(a) -> int:
    """ReVOS: merge per-expression predictions into <out>/<row>/results.json (missing -> empty mask)."""
    ds = DATASETS[a.dataset](a.data_root, a.split, a.meta_file)
    out = Path(a.out)
    videos = ds.videos()
    rc = 0
    if a.dataset != "revos":
        for row in ROWS:
            miss = [f"{v}/{e}" for v, d in videos.items() for e in d["expressions"]
                    if not ds.is_written(out, row, v, e, [str(x) for x in d["frames"]])]
            (out / row).mkdir(parents=True, exist_ok=True)
            (out / row / "missing.txt").write_text("\n".join(miss), encoding="utf-8")
            print(f"[export] {row}: missing {len(miss)}")
            rc = rc or (2 if miss else 0)
        return rc
    from p1a.common import encode_rle

    for row in ROWS:
        res, miss = {}, []
        for vid, v in videos.items():
            frames = [str(x) for x in v["frames"]]
            for eid, info in v["expressions"].items():
                p = ReVOS.pred_path(out, row, vid, eid)
                if p.is_file():
                    res.setdefault(vid, {})[eid] = json.loads(p.read_text(encoding="utf-8"))
                else:
                    miss.append(f"{vid}/{eid}")
                    empty = encode_rle(np.zeros((int(v["height"]), int(v["width"])), bool))
                    res.setdefault(vid, {})[eid] = {"video_id": vid, "exp_id": eid, "exp": info.get("exp", ""),
                                                    "frames": frames, "prediction_masks": [empty] * len(frames)}
        d = out / row
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
        (d / "missing.txt").write_text("\n".join(miss), encoding="utf-8")
        print(f"[export] {row}: {sum(len(x) for x in res.values())} expressions -> {d / 'results.json'}; missing {len(miss)}")
        rc = rc or (2 if miss else 0)
    return rc


NAMES = {"words": "Ground v2 词（不加点种子）", "final": "新主线：Ground v2 词 + 热力图点种子"}


def _row(dataset: str, name: str, d: Dict) -> str:
    if dataset == "revos":
        return (f"| {name} | {d['referring']['JF']:.2f} | {d['reason']['JF']:.2f} | {d['overall']['J']:.2f} | "
                f"{d['overall']['F']:.2f} | **{d['overall']['JF']:.2f}** |")
    e, v = d["overall_expression_level"], d["overall_video_level"]
    return f"| {name} | {e['J']:.2f} | {e['F']:.2f} | **{e['J&F']:.2f}** | {v['J&F']:.2f} |"


def summary(a) -> None:
    out = Path(a.out)
    ds_cls = DATASETS[a.dataset]
    if a.dataset == "revos":
        head = ["# ReVOS valid（官方 eval_revos.py）", "",
                "| 结果 | Referring J&F | Reasoning J&F | Overall J | Overall F | Overall J&F |", "|---|---|---|---|---|---|"]
    else:
        head = ["# Ref-YouTube-VOS valid（eval_refytros_png.py）", "",
                "| 结果 | J | F | J&F（表达式级） | J&F（视频级） |", "|---|---|---|---|---|"]
    L = list(head)
    rows = [tuple(x.split("=", 1)) for x in a.ref] + [(NAMES[r], str(out / r / ds_cls.eval_json)) for r in ROWS]
    for name, p in rows:
        try:
            L.append(_row(a.dataset, name, json.loads(Path(p).read_text(encoding="utf-8"))))
        except (FileNotFoundError, KeyError) as e:
            L.append(f"| {name} | 未找到或格式不对：{p} ({type(e).__name__}) |")
    sp = out / "scores.jsonl"
    if sp.is_file():
        rs = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(rs)
        trig = sum(1 for r in rs if r.get("point_ids"))
        ch = sum(1 for r in rs if sorted(r["final_selected"]) != sorted(r["words_selected"]))
        picked = sum(1 for r in rs if any(i >= POINT_BASE for i in r["final_selected"]))
        L += ["", f"- 已跑 {n} 条；补了点种子轨迹的 {trig} 条（平均每条 +{sum(len(r.get('point_ids', [])) for r in rs) / max(n, 1):.2f} 个候选）；"
                  f"Verify 选择因此改变的 {ch} 条，其中选中点种子轨迹的 {picked} 条。",
              f"- SAM3 文本召回 0 候选的表达式：{sum(1 for r in rs if not r['cand_ids'])} 条；没有热力图文件的：{sum(1 for r in rs if r.get('no_heatmap'))} 条（应为 0）。"]
        if rs and "pool_j_words" in rs[0]:
            b = np.array([r["pool_j_words"] for r in rs])
            p = np.array([r["pool_j_final"] for r in rs])
            L.append(f"- 候选池 Recall@0.5：{100 * (b >= 0.5).mean():.2f}% → {100 * (p >= 0.5).mean():.2f}%；"
                     f"池内最好候选的平均 J：{100 * b.mean():.2f} → {100 * p.mean():.2f}。")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", "export", "summary"])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--data-root", default="")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--meta-file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--categories", default="", help="Ground v2 category json (gv2.predict --output-json)")
    ap.add_argument("--heat-dir", default="", help="Ground v2 heatmap npz dir (gv2.predict --heat-dir)")
    ap.add_argument("--allow-missing-words", action="store_true")
    ap.add_argument("--adapter", default="", help="override the yaml's Verify LoRA")
    ap.add_argument("--base-mllm", default="")
    ap.add_argument("--sam3-checkpoint", default="")
    ap.add_argument("--num-tube-frames", type=int, default=0, help="0 = the yaml's value")
    ap.add_argument("--keep-backup", action="store_true", help="allow the repo's backup words when recall is empty")
    ap.add_argument("--recall-cache-gb", type=float, default=4.0, help="0 = off")
    ap.add_argument("--agg", default="max")
    ap.add_argument("--abs-thr", type=float, default=0.5)
    ap.add_argument("--rel-thr", type=float, default=0.6)
    ap.add_argument("--coverage-thr", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=3)
    ap.add_argument("--dup-iou", type=float, default=0.7)
    ap.add_argument("--gt", action="store_true", help="log candidate-pool recall against the GT")
    ap.add_argument("--limit", type=int, default=0, help="first N videos")
    ap.add_argument("--ref", action="append", default=[], help="summary: 'name=evaluation json' of another run")
    a = ap.parse_args(argv)
    for k in ("data_root", "heat_dir", "out", "config", "repo", "categories"):
        if getattr(a, k):
            setattr(a, k, str(Path(getattr(a, k)).resolve()))
    if a.step == "run":
        for k in ("repo", "config", "data_root", "categories", "heat_dir"):
            if not getattr(a, k):
                raise SystemExit(f"run needs --{k.replace('_', '-')}")
        run(a)
    elif a.step == "export":
        raise SystemExit(export(a))
    else:
        summary(a)


if __name__ == "__main__":
    main()
