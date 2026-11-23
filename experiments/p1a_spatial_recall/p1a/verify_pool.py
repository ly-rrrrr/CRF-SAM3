#!/usr/bin/env python3
"""Verify re-eval on fixed candidate pools: final ReVOS J&F for old-word pool / v2-word pool / v2 pool + point tubes.

Runs the paper's own Verify (CRFSAM3Pipeline's ConceptExtractor for num_targets, VisualTubeBuilder,
SemanticDiscriminator from the same yaml) but replaces SAM3 text recall with pools that were already computed:

  old       : text-recall pool of the old Ground word (ground_v2 step 4)   -> sanity check vs the paper (59.74)
  v2        : text-recall pool of the v2 Ground word (ground_v2 step 4)
  v2_point  : v2 pool + heatmap point tubes (p1a point mode, saved with --save-tubes)

Every candidate of the union is scored ONCE (the yes/no scorer scores each tube independently, so taking the
top num_targets inside each subset equals running the pipeline on that subset). SAM3 is not loaded.
No backup words, no semantic expansion: a 0-candidate pool gives an empty prediction.

  predict : GPU. python -m p1a.verify_pool predict --repo $REPO --config <yaml> ... --out runs/verify_v2
  merge   : CPU. writes <out>/<row>/results.json in the ReVOS format (missing expressions -> empty masks, reported)
  fidelity: CPU. 'old' row vs the paper run's scores.json on the same candidates (run after a short predict)
  compare : CPU. reads <out>/<row>/revos_valid_official.json (from tools/eval/eval_revos.py) -> <out>/compare.md
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .common import decode_rle, encode_rle, norm_word, pool_file, tube_from_rle  # before predict() chdirs to $REPO

ROWS = ("old", "v2", "v2_point")
OLD_BASE, POINT_BASE = 500, 1000


# ----------------------------------------------------------------------------- inputs
def load_words(per_expression: str) -> Dict[str, Dict]:
    out = {}
    for r in csv.DictReader(open(per_expression, encoding="utf-8")):
        out[f"{r['video_id']}/{r['exp_id']}"] = r
    return out


def load_pool_raw(pool_dir: str, video_id: str, word: str) -> Optional[Dict[int, Dict[int, np.ndarray]]]:
    """{cid: {t: mask}} keeping every frame SAM3 returned (also empty masks), as the pipeline's all_masks does."""
    if not norm_word(word):
        return {}
    p = pool_file(pool_dir, video_id, word)
    if not p.is_file():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k, seq in d.get("rles", {}).items():
        tube = {}
        for t, r in enumerate(seq):
            if r is None:
                continue
            m = decode_rle(r)
            if m is not None:
                tube[t] = m
        if tube:
            out[int(k)] = tube
    return out


def load_point_tubes(point_dir: str, video_id: str, exp_id: str) -> List[Dict[int, np.ndarray]]:
    p = Path(point_dir) / "tubes" / video_id / f"{exp_id}.json"
    if not p.is_file():
        return []
    return [tube_from_rle(t) for t in json.loads(p.read_text(encoding="utf-8")).get("point", [])]


def load_point_done(point_dir: str) -> set:
    p = Path(point_dir) / "results.jsonl"
    if not p.is_file():
        return set()
    return {json.loads(l)["key"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def _fit(m: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.asarray(m).astype(bool)
    if m.shape != (h, w):
        import cv2

        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def union_candidates(old: Dict, new: Dict, points: List, same_word: bool, h: int, w: int):
    """-> all_masks {t: {oid: mask}}, ids {row-part: [oid]}"""
    all_masks: Dict[int, Dict[int, np.ndarray]] = {}
    ids = {"old": [], "v2": [], "point": []}

    def put(oid, tube):
        for t, m in tube.items():
            all_masks.setdefault(int(t), {})[oid] = _fit(m, h, w)

    for cid, tube in sorted(new.items()):
        put(int(cid), tube)
        ids["v2"].append(int(cid))
    if same_word:
        ids["old"] = list(ids["v2"])
    else:
        for cid, tube in sorted(old.items()):
            put(OLD_BASE + int(cid), tube)
            ids["old"].append(OLD_BASE + int(cid))
    for i, tube in enumerate(points):
        put(POINT_BASE + i, tube)
        ids["point"].append(POINT_BASE + i)
    return all_masks, ids


def row_ids(ids: Dict[str, List[int]], row: str) -> List[int]:
    return {"old": ids["old"], "v2": ids["v2"], "v2_point": ids["v2"] + ids["point"]}[row]


def select(scores: Dict[int, float], cand: List[int], k: int) -> List[int]:
    """Same rule as SemanticDiscriminator.score_and_filter: sort scored ids by score (stable, ids ascending), top k."""
    valid = [i for i in sorted(cand) if scores.get(i) is not None]
    return sorted(valid, key=lambda i: scores[i], reverse=True)[: max(int(k), 0)]


def masks_of(all_masks: Dict, keep: List[int]) -> Dict[int, Dict[int, np.ndarray]]:
    keep = set(keep)
    return {t: {o: m for o, m in fm.items() if o in keep} for t, fm in all_masks.items()}


def pred_path(out: Path, row: str, video_id: str, exp_id: str) -> Path:
    return out / "pred" / row / video_id / f"{exp_id}.json"


# ----------------------------------------------------------------------------- predict (GPU)
def build_pipeline(repo: str, config: str, temp_dir: str):
    sys.path.insert(0, repo)
    os.chdir(repo)
    import crf_sam3_demo as demo
    from run_crf_sam3_dataset import load_config
    from run_crf_sam3_revos_dataset import _resolve_prompt_template

    demo.build_sam3_recall_backend = lambda *a, **k: None  # pools are fixed; SAM3 is never loaded
    cfg = load_config(config)
    m, vt, dc = cfg["model"], cfg.get("visual_tube", {}), cfg.get("discrimination", {})
    pipe = demo.CRFSAM3Pipeline(
        mllm_path=m.get("mllm_path"), base_mllm_path=m.get("base_mllm_path"),
        mllm_adapter_path=m.get("mllm_adapter_path"), visual_tube_temp_dir=temp_dir,
        sam3_checkpoint=m.get("sam3_checkpoint"),
        num_tube_frames=int(cfg.get("pipeline", {}).get("num_tube_frames", 8)),
        model_type=m.get("model_type", "qwen3"), prompt_type=dc.get("prompt_type", "yesno"),
        prompt_template_path=_resolve_prompt_template(dc.get("prompt_type", "yesno")),
        visualization_style=vt.get("visualization_style", "mask"),
        sampling_strategy=vt.get("sampling_strategy", "uniform"), qframe_config=None,
        concept_use_qframe=False, concept_config=cfg.get("concept_extraction", {}),
        discriminator_backend=dc.get("backend", "mllm"), reranker_cfg=dc.get("reranker", {}),
        sam3_recall_cfg=cfg.get("sam3_recall", {}) or {},
    )
    if dc.get("backend", "mllm") != "mllm" or vt.get("sampling_strategy", "uniform") == "qframe":
        raise SystemExit("verify_pool supports the paper setting only (mllm yes/no scorer, uniform tubes)")
    return pipe, demo


def predict(a) -> None:
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))["videos"]
    words = load_words(a.per_expression)
    point_done = load_point_done(a.point_dir) if a.point_dir else set()
    rows = [r for r in a.rows.split(",") if r]
    if "v2_point" in rows and not a.point_dir:
        raise SystemExit("row v2_point needs --point-dir")
    scores_path = out / "scores.jsonl"
    done = set()
    if scores_path.is_file():
        done = {json.loads(l)["key"] for l in scores_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    jobs = [(v, e) for v in meta for e in meta[v]["expressions"]]
    jobs = [j for i, j in enumerate(jobs) if i % a.num_shards == a.shard]
    if a.limit:
        jobs = jobs[: a.limit]
    todo = [j for j in jobs if f"{j[0]}/{j[1]}" not in done]
    if "v2_point" in rows:
        miss = [j for j in todo if f"{j[0]}/{j[1]}" not in point_done]
        if miss and not a.allow_missing_point:
            raise SystemExit(f"{len(miss)} expressions have no point-mode result in {a.point_dir}/results.jsonl "
                             f"(first {miss[0][0]}/{miss[0][1]}). Finish the point run first, or pass --allow-missing-point.")
    print(f"[verify] {len(jobs)} expressions, {len(jobs) - len(todo)} done, {len(todo)} to run; rows={rows}", flush=True)
    if not todo:
        return

    pipe, demo = build_pipeline(a.repo, a.config, str(out / "visual_tubes_temp"))
    from run_crf_sam3_dataset import _infer_video_hw
    from run_crf_sam3_revos_dataset import _build_revos_rles
    import torch

    hw_cache: Dict[str, tuple] = {}
    t_start, n_err = time.time(), 0
    for n, (vid, eid) in enumerate(todo, 1):
        key = f"{vid}/{eid}"
        info = meta[vid]["expressions"][eid]
        expr = info.get("exp", "")
        frames = [str(x) for x in meta[vid]["frames"]]
        video_path = os.path.join(a.revos_root, vid)
        t0 = time.time()
        try:
            if vid not in hw_cache:
                hw_cache = {vid: _infer_video_hw(video_path, frames)}
            h, w = (int(x) for x in hw_cache[vid])
            wr = words.get(key)
            if wr is None:
                raise KeyError(f"{key} not in {a.per_expression}")
            old_w, new_w = wr["old_word"], wr["new_word"]
            old = load_pool_raw(a.pool_dir, vid, old_w)
            new = load_pool_raw(a.pool_dir, vid, new_w)
            if old is None or new is None:
                raise FileNotFoundError(f"pool missing for {key}: old '{old_w}' {old is not None}, new '{new_w}' {new is not None}")
            points = load_point_tubes(a.point_dir, vid, eid) if "v2_point" in rows else []
            same = " ".join(old_w.lower().split()) == " ".join(new_w.lower().split())
            all_masks, ids = union_candidates(old, new, points, same, h, w)

            category, num_targets, _ = pipe.concept_extractor.extract_category(expr, video_path)
            num_targets = max(int(num_targets or 1), 1)
            has_color = demo.contains_color_attribute(expr)
            style = "bbox" if has_color else pipe._original_visualization_style
            scores: Dict[int, float] = {}
            if all_masks:
                pipe.tube_builder.visualization_style = style
                try:
                    tubes = pipe.tube_builder.build_tubes(video_path, all_masks, expr)
                    _, details = pipe.discriminator.score_and_filter(tubes, expr, num_targets, return_details=True,
                                                                     visualization_style=style)
                finally:
                    pipe.tube_builder.visualization_style = pipe._original_visualization_style
                scores = {int(o): (None if d.get("score") is None else float(d["score"])) for o, d in details.items()}
            sel = {}
            for row in rows:
                keep = select(scores, row_ids(ids, row), num_targets)
                sel[row] = keep
                p = pred_path(out, row, vid, eid)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"video_id": vid, "exp_id": eid, "exp": expr, "frames": frames,
                                         "prediction_masks": _build_revos_rles(masks_of(all_masks, keep), frames, h, w)}),
                             encoding="utf-8")
            rec = {"key": key, "video_id": vid, "exp_id": eid, "type_id": int(info.get("type_id", -1)),
                   "expression": expr, "old_word": old_w, "new_word": new_w, "extractor_category": category,
                   "num_targets": num_targets, "style": style, "ids": ids,
                   "scores": {str(k): v for k, v in scores.items()}, "selected": sel,
                   "sec": round(time.time() - t0, 1)}
            with open(scores_path, "a", encoding="utf-8") as f:  # written last: a sample counts as done only here
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            el = time.time() - t_start
            print(f"[{n}/{len(todo)}] {key} k={num_targets} old={len(ids['old'])} v2={len(ids['v2'])} "
                  f"pt={len(ids['point'])} sel={sel} {rec['sec']}s (eta {el / n * (len(todo) - n) / 3600:.1f} h)", flush=True)
        except torch.OutOfMemoryError:
            raise
        except Exception as e:  # not written to scores.jsonl, so a re-run retries it
            if "out of memory" in str(e).lower():
                raise
            n_err += 1
            print(f"[error] {key}: {type(e).__name__}: {e}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if n_err:
        print(f"[verify] {n_err} expressions failed; re-run the same command to retry them")


# ----------------------------------------------------------------------------- merge / compare (CPU)
def merge(a) -> int:
    out = Path(a.out)
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))["videos"]
    rc = 0
    for row in [r for r in a.rows.split(",") if r]:
        res, missing = {}, []
        for vid, v in meta.items():
            frames = [str(x) for x in v["frames"]]
            for eid, info in v["expressions"].items():
                p = pred_path(out, row, vid, eid)
                if p.is_file():
                    res.setdefault(vid, {})[eid] = json.loads(p.read_text(encoding="utf-8"))
                else:
                    missing.append(f"{vid}/{eid}")
                    empty = encode_rle(np.zeros((int(v["height"]), int(v["width"])), bool))
                    res.setdefault(vid, {})[eid] = {"video_id": vid, "exp_id": eid, "exp": info.get("exp", ""),
                                                    "frames": frames, "prediction_masks": [empty] * len(frames)}
        d = out / row
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
        (d / "missing.txt").write_text("\n".join(missing), encoding="utf-8")
        n = sum(len(v) for v in res.values())
        print(f"[merge] {row}: {n} expressions -> {d / 'results.json'}; missing (empty mask) {len(missing)}")
        if missing:
            rc = 2
    return rc


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def fidelity(a) -> int:
    """Compare the 'old' row with the paper run's scores.json on the same candidates.

    The yes/no score is sigmoid(logit_yes - logit_no) from bf16 logits, so scores move in steps of
    0.125-0.25 in logit space (0.4378 = sigmoid(-0.25), 0.5312 = sigmoid(0.125)). A different GPU / torch /
    kernel can shift a logit by one or two steps. We therefore compare in logit space: the pipeline counts as
    reproduced when logit differences stay within a couple of bf16 steps, there is no systematic shift, and
    every changed selection is a near-tie in the paper run itself.
    """
    hist = json.loads(Path(a.paper_scores).read_text(encoding="utf-8"))
    hist = {f"{r['video_id']}/{r['exp_id']}": r for r in hist if isinstance(r, dict)}
    recs = [json.loads(l) for l in (Path(a.out) / "scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = same_sel = skipped = 0
    dl, exact, flips, rows = [], 0, [], []
    for r in recs:
        h = hist.get(r["key"])
        if h is None or str(h.get("used_category_query", "")).strip().lower() != " ".join(r["old_word"].lower().split()):
            skipped += 1  # not in the paper run, or the paper run fell back to a backup word
            continue
        n += 1
        back = {i: (i - OLD_BASE if i >= OLD_BASE else i) for i in r["ids"]["old"]}
        hs = {int(k): (v.get("score") if isinstance(v, dict) else v) for k, v in (h.get("scores") or {}).items()}
        for i, oi in back.items():
            ours = r["scores"].get(str(i))
            if ours is not None and hs.get(oi) is not None:
                d = _logit(ours) - _logit(hs[oi])
                dl.append(d)
                exact += abs(float(ours) - float(hs[oi])) < 1e-3
                rows.append((abs(d), r["key"], oi, float(hs[oi]), float(ours)))
        ours_sel = sorted(back[i] for i in r["selected"]["old"])
        paper_sel = sorted(int(x) for x in h.get("selected_candidates", []))
        if ours_sel == paper_sel:
            same_sel += 1
        else:
            # gap, in the paper's own scores, between the last selected and the best non-selected candidate
            ps = sorted((_logit(v) for v in hs.values() if v is not None), reverse=True)
            k = len(paper_sel)
            gap = ps[k - 1] - ps[k] if 0 < k < len(ps) else float("inf")
            flips.append((r["key"], paper_sel, ours_sel, gap,
                          {oi: round(float(hs[oi]), 4) for oi in sorted(hs) if hs[oi] is not None},
                          {back[i]: round(float(r["scores"][str(i)]), 4) for i in sorted(back) if r["scores"].get(str(i)) is not None}))
    d = np.array(dl) if dl else np.zeros(1)
    ad = np.abs(d)
    print(f"[fidelity] compared {n} expressions with the paper run (skipped {skipped}: backup word or not found)")
    print(f"[fidelity] same selection: {same_sel}/{n}; candidates with identical score: {exact}/{len(dl)}")
    print(f"[fidelity] logit diff (ours - paper): mean {d.mean():+.3f}, |diff| median {np.median(ad):.3f}, "
          f"p90 {np.percentile(ad, 90):.3f}, max {ad.max():.3f}  (one bf16 step = 0.125-0.25)")
    for x in sorted(rows, reverse=True)[:5]:
        print(f"[fidelity]   largest: {x[1]} cand {x[2]}: paper {x[3]:.4f} ours {x[4]:.4f} (logit diff {x[0]:.3f})")
    for key, ps_, os_, gap, hp, ho in flips:
        print(f"[fidelity]   changed selection {key}: paper {ps_} ours {os_}; paper's own logit gap {gap:.3f}")
        print(f"[fidelity]     paper scores {hp}")
        print(f"[fidelity]     our scores   {ho}")
    ok = (n > 0 and len(flips) <= max(1, 0.1 * n) and abs(d.mean()) <= 0.1 and float(np.percentile(ad, 90)) <= 0.25
          and float(ad.max()) <= 0.75 and all(f[3] <= 0.5 for f in flips))
    print("[fidelity] OK: differences are bf16 rounding only; this Verify reproduces the paper pipeline" if ok else
          "[fidelity] MISMATCH: send me this output before running the full set")
    return 0 if ok else 1


def compare(a) -> None:
    out = Path(a.out)
    names = {"paper": "论文结果（原流程，含 backup 词）", "old": "旧词池（本脚本复现）",
             "v2": "v2 词池", "v2_point": "v2 词池 + 热力图点种子"}
    runs = []
    if a.paper_json and Path(a.paper_json).is_file():
        runs.append(("paper", json.loads(Path(a.paper_json).read_text(encoding="utf-8"))))
    for row in [r for r in a.rows.split(",") if r]:
        p = out / row / "revos_valid_official.json"
        if p.is_file():
            runs.append((row, json.loads(p.read_text(encoding="utf-8"))))
    L = ["# ReVOS valid 最终 J&F（官方 eval_revos.py）", "",
         "| 候选池 | Referring J&F | Reasoning J&F | Overall J | Overall F | Overall J&F | 缺失（按空掩码计） |",
         "|---|---|---|---|---|---|---|"]
    for row, r in runs:
        miss = ""
        mp = out / row / "missing.txt"
        if mp.is_file():
            miss = str(len([x for x in mp.read_text(encoding="utf-8").splitlines() if x.strip()]))
        L.append(f"| {names.get(row, row)} | {r['referring']['JF']:.2f} | {r['reason']['JF']:.2f} | "
                 f"{r['overall']['J']:.2f} | {r['overall']['F']:.2f} | **{r['overall']['JF']:.2f}** | {miss} |")
    sp = out / "scores.jsonl"
    if sp.is_file():
        recs = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len(recs)
        pt = sum(1 for r in recs if any(i >= POINT_BASE for i in r["selected"].get("v2_point", [])))
        ch = sum(1 for r in recs if r["selected"].get("v2_point") != r["selected"].get("v2"))
        mism = sum(1 for r in recs if " ".join(str(r["extractor_category"]).lower().split())
                   != " ".join(str(r["old_word"]).lower().split()))
        L += ["", f"- 已打分 {n} 条；v2_point 与 v2 选择不同 {ch} 条，其中选中了点种子轨迹的 {pt} 条。",
              f"- 自检：抽取器给出的类别词与旧词不一致 {mism} 条（应为 0 或极少；不一致说明配置里的类别缓存不是论文那份）。",
              "- 旧词池一行不含 backup 词和语义扩展，所以会比论文数字略低（差距应在约 1 个点以内）；若差得多，说明本脚本的 Verify 与原流程不一致，先别看另外两行。"]
    (out / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["predict", "merge", "compare", "fidelity"])
    ap.add_argument("--repo", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--revos-root", default="")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--per-expression", default="")
    ap.add_argument("--pool-dir", default="")
    ap.add_argument("--point-dir", default="", help="p1a point run dir holding results.jsonl and tubes/")
    ap.add_argument("--allow-missing-point", action="store_true")
    ap.add_argument("--rows", default=",".join(ROWS))
    ap.add_argument("--paper-scores", default="", help="historical scores.json (for fidelity)")
    ap.add_argument("--paper-json", default="", help="historical revos_valid_official.json for the compare table")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.step == "predict":
        for k in ("repo", "config", "revos_root", "per_expression", "pool_dir"):
            if not getattr(a, k):
                raise SystemExit(f"predict needs --{k.replace('_', '-')}")
        a.out = str(Path(a.out).resolve())
        for k in ("meta", "per_expression", "pool_dir", "point_dir", "revos_root", "config"):
            if getattr(a, k):
                setattr(a, k, str(Path(getattr(a, k)).resolve()))
        predict(a)
    elif a.step == "merge":
        sys.exit(merge(a))
    elif a.step == "fidelity":
        sys.exit(fidelity(a))
    else:
        compare(a)


if __name__ == "__main__":
    main()
