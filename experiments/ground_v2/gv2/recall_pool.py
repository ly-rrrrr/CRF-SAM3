#!/usr/bin/env python3
"""Step 4: does the v2 category word give SAM3 a better candidate pool on ReVOS valid?

Runs the paper's SAM3 recall protocol (DMSConceptRecall.recall_all_instances_stream, recall 24 -> keep 7 by
summed score, the same call as paper_assets/.../replay_recall.py) with the OLD Ground word and the NEW v2 word,
then scores both pools against the GT with the same code, so the comparison is apples to apples.

SAM3 output depends only on (video, word), so each unique pair is run once and shared by every expression
that uses it (about 1,650 runs for 5,822 expressions; 3-15 s each).

  # GPU, can be sharded over several cards; resumable (finished pools are skipped)
  python -m gv2.recall_pool recall --repo $REPO --sam3-ckpt <sam3.pt> --revos-root $REVOS_ROOT \
      --old-csv <all_cases_diagnostics.csv> --new-json runs/gv2_revos/revos_valid_categories.json \
      --out runs/pool_revos --shard 0 --num-shards 2
  # CPU, after all shards finished
  python -m gv2.recall_pool score --revos-root $REVOS_ROOT --old-csv ... --new-json ... --out runs/pool_revos

Outputs in --out: pools/<hash>.json.gz (top-7 candidate RLEs per pair; reusable for Verify),
per_expression.csv, report.md, report.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

TYPE_NAME = {0: "Referring", 1: "Reasoning", 2: "Other/type=2"}


# ----------------------------------------------------------------------------- inputs
def norm_word(w: Any) -> str:
    return " ".join(str(w or "").strip().lower().split())


def load_words(old_csv: str, new_json: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """(video_id, exp_id) -> {old, new, type_id, expression}."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(old_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["video_id"], str(r["exp_id"]))] = {
                "old": norm_word(r["category"]), "new": None, "type_id": int(r["type_id"]),
                "expression": r.get("expression", ""), "old_n_archived": r.get("n")}
    new = json.loads(Path(new_json).read_text(encoding="utf-8"))
    recs = new["records"] if isinstance(new, dict) else new
    for r in recs:
        k = (r["video_id"], str(r["exp_id"]))
        e = out.setdefault(k, {"old": None, "type_id": int(r.get("type_id", -1)), "expression": r.get("expression", "")})
        e["new"] = norm_word(r.get("pred_category") or r.get("category"))
    return out


def pool_key(video_id: str, word: str) -> str:
    return hashlib.sha1(f"{video_id}\t{word}".encode("utf-8")).hexdigest()[:20]


def jobs_of(words: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs = set()
    for (vid, _), e in words.items():
        for w in (e.get("old"), e.get("new")):
            if w:
                pairs.add((vid, w))
    return sorted(pairs)


# ----------------------------------------------------------------------------- RLE / J
def encode(m: np.ndarray) -> Optional[Dict[str, Any]]:
    from pycocotools import mask as cm

    if not m.any():
        return None
    r = cm.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def decode(r: Any) -> Optional[np.ndarray]:
    if not r:
        return None
    from pycocotools import mask as cm

    r = dict(r)
    if isinstance(r.get("counts"), str):
        r["counts"] = r["counts"].encode("ascii")
    m = cm.decode(r)
    return (m.any(axis=2) if m.ndim == 3 else m).astype(bool)


def region_j(pred: List[Optional[np.ndarray]], gt: List[Optional[np.ndarray]]) -> float:
    """Mean per-frame IoU over all annotated frames; both empty = 1 (ReVOS / DAVIS convention)."""
    vals = []
    for t, g in enumerate(gt):
        p = pred[t] if t < len(pred) else None
        g_any = g is not None and g.any()
        p_any = p is not None and p.any()
        if not g_any and not p_any:
            vals.append(1.0)
        elif not g_any or not p_any:
            vals.append(0.0)
        else:
            if p.shape != g.shape:
                import cv2

                p = cv2.resize(p.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            vals.append(float((p & g).sum() / (p | g).sum()))
    return float(np.mean(vals)) if vals else 0.0


# ----------------------------------------------------------------------------- recall (GPU)
class DMSRecaller:
    """Thin wrapper over the repo's own recall (identical call to replay_recall.py)."""

    def __init__(self, repo: str, sam3_ckpt: str, mem_fraction: float):
        import torch

        sys.path.insert(0, repo)
        from crf_sam3_demo import DMSConceptRecall  # noqa: E402

        if mem_fraction > 0:
            torch.cuda.set_per_process_memory_fraction(mem_fraction, 0)
        self.torch = torch
        self.model = DMSConceptRecall(str(Path(repo) / "SAM3-DMS"), sam3_ckpt, gpus_to_use=[0],
                                      async_loading_frames=True, offload_video_to_cpu=True,
                                      offload_output_to_cpu_for_eval=True)

    def __call__(self, video_dir: str, word: str, top_k: int, keep: int):
        torch = self.torch
        masks: Dict[int, Dict[int, np.ndarray]] = {}
        scores: Dict[int, float] = defaultdict(float)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t, ms, ss in self.model.recall_all_instances_stream(video_dir, word, candidate_top_k=top_k,
                                                                     candidate_keep_top_k=keep):
                masks[t] = ms
                for k in ms:
                    scores[k] += float(ss[k])
        return masks, dict(scores)


def run_recall(a, recaller: Optional[Callable] = None) -> None:
    root = Path(a.revos_root)
    meta = json.loads((root / a.meta).read_text(encoding="utf-8"))["videos"]
    words = load_words(a.old_csv, a.new_json)
    jobs = jobs_of(words)
    jobs = [j for i, j in enumerate(jobs) if i % a.num_shards == a.shard]
    if a.limit > 0:
        jobs = jobs[: a.limit]
    pool_dir = Path(a.out) / "pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (pool_dir / f"{pool_key(*j)}.json.gz").exists()]
    print(f"[pool] shard {a.shard}/{a.num_shards}: {len(jobs)} (video, word) pairs, {len(jobs) - len(todo)} done, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        return
    recaller = recaller or DMSRecaller(a.repo, a.sam3_ckpt, a.mem_fraction)
    t0 = time.time()
    for n, (vid, word) in enumerate(todo, 1):
        v = meta[vid]
        nf = len(v["frames"])
        h, w = int(v.get("height", 0)), int(v.get("width", 0))
        t1 = time.time()
        try:
            masks, scores = recaller(str(root / vid), word, a.recall_top_k, a.keep_top_k)
            err = None
        except Exception as e:  # keep going; the pair is recorded as failed (0 candidates) and can be re-run
            masks, scores, err = {}, {}, f"{type(e).__name__}: {e}"
        keep = sorted(scores, key=lambda k: (-scores[k], k))[: a.keep_top_k]
        rles = {}
        for k in keep:
            seq = []
            for t in range(nf):
                m = masks.get(t, {}).get(k)
                seq.append(encode(np.asarray(m).astype(bool)) if m is not None else None)
            rles[str(k)] = seq
        payload = {"video_id": vid, "word": word, "n_frames": nf, "height": h, "width": w, "keep": [str(k) for k in keep],
                   "score_sums": {str(k): scores[k] for k in keep}, "rles": rles, "error": err,
                   "config": {"recall_top_k": a.recall_top_k, "keep_top_k": a.keep_top_k}}
        if err is None:  # failed pairs are not written, so a re-run retries them
            tmp = pool_dir / f"{pool_key(vid, word)}.json.gz.tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(payload, f)
            tmp.replace(pool_dir / f"{pool_key(vid, word)}.json.gz")
        el = time.time() - t0
        print(f"[pool {n}/{len(todo)}] {vid} '{word}' -> {len(keep)} cands, {time.time() - t1:.1f}s"
              f"{'  ERROR ' + err if err else ''}  (eta {el / n * (len(todo) - n) / 60:.0f} min)", flush=True)


# ----------------------------------------------------------------------------- score (CPU)
def gt_tube(mask_dict: Dict[str, Any], vmeta: Dict[str, Any], exp_id: str) -> List[Optional[np.ndarray]]:
    info = vmeta["expressions"][str(exp_id)]
    aids = info.get("anno_id", [])
    aids = aids if isinstance(aids, (list, tuple)) else [aids]
    out = []
    for j in range(len(vmeta["frames"])):
        merged = None
        for aid in aids:
            seq = mask_dict.get(str(aid))
            m = decode(seq[j]) if seq is not None and j < len(seq) else None
            if m is not None:
                merged = m if merged is None else (merged | m)
        out.append(merged)
    return out


def load_pool(pool_dir: Path, vid: str, word: Optional[str], cache: Dict) -> Optional[Dict[str, Any]]:
    if not word:
        return {"keep": [], "rles": {}}
    key = pool_key(vid, word)
    if key not in cache:
        p = pool_dir / f"{key}.json.gz"
        if not p.exists():
            cache[key] = None
        else:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
            d["tubes"] = {k: [decode(r) for r in seq] for k, seq in d["rles"].items()}
            cache[key] = d
    return cache[key]


def pool_metrics(pool: Dict[str, Any], gt: List[Optional[np.ndarray]]) -> Tuple[int, float, Optional[str]]:
    js = {k: region_j(t, gt) for k, t in pool.get("tubes", {}).items()}
    if not js:
        return 0, 0.0, None
    best = max(js, key=js.get)
    return len(js), js[best], best


def summarise(rows: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    js = np.array([r[f"{side}_oracle_j"] for r in rows])
    nc = np.array([r[f"{side}_n"] for r in rows])
    return {"n": n, "recall@0.5": float((js >= 0.5).mean()), "unavailable": float((js < 0.5).mean()),
            "oracle_j": float(js.mean()), "zero_cand": float((nc == 0).mean()), "mean_cands": float(nc.mean())}


def run_score(a) -> None:
    root = Path(a.revos_root)
    meta = json.loads((root / a.meta).read_text(encoding="utf-8"))["videos"]
    print("[score] loading mask_dict ...", flush=True)
    mask_dict = json.loads((root / a.mask_dict).read_text(encoding="utf-8"))
    words = load_words(a.old_csv, a.new_json)
    out = Path(a.out)
    pool_dir = out / "pools"
    rows, missing = [], 0
    by_video: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for (vid, eid), e in words.items():
        by_video[vid].append((eid, e))
    for vi, vid in enumerate(sorted(by_video), 1):
        cache: Dict[str, Any] = {}  # pools of one video only, to bound memory
        for eid, e in by_video[vid]:
            gt = gt_tube(mask_dict, meta[vid], eid)
            row = {"video_id": vid, "exp_id": eid, "type_id": e["type_id"], "source": vid.split("/")[0],
                   "old_word": e.get("old") or "", "new_word": e.get("new") or "",
                   "changed": (e.get("old") or "") != (e.get("new") or ""), "expression": e.get("expression", "")}
            ok = True
            for side in ("old", "new"):
                pool = load_pool(pool_dir, vid, e.get(side), cache)
                if pool is None:
                    ok = False
                    break
                row[f"{side}_n"], row[f"{side}_oracle_j"], row[f"{side}_best"] = pool_metrics(pool, gt)
            if not ok:
                missing += 1
                continue
            row["old_n_archived"] = e.get("old_n_archived")
            rows.append(row)
        if vi % 50 == 0:
            print(f"[score] {vi}/{len(by_video)} videos", flush=True)
    if missing:
        print(f"[score] WARNING: {missing} expressions skipped because a pool is missing (recall not finished?)")

    with open(out / "per_expression.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["video_id", "exp_id", "type_id", "source", "old_word", "new_word", "changed", "old_n", "new_n",
                "old_oracle_j", "new_oracle_j", "old_best", "new_best", "old_n_archived", "expression"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    groups = {"Referring": lambda r: r["type_id"] == 0, "Reasoning": lambda r: r["type_id"] == 1,
              "Official (Ref+Rea)": lambda r: r["type_id"] in (0, 1), "Other/type=2": lambda r: r["type_id"] == 2,
              "All": lambda r: True}
    rep: Dict[str, Any] = {"n_rows": len(rows), "missing": missing, "groups": {}, "changed_only": {}, "by_source": {}}
    for g, f in groups.items():
        sub = [r for r in rows if f(r)]
        rep["groups"][g] = {"old": summarise(sub, "old"), "new": summarise(sub, "new")}
        ch = [r for r in sub if r["changed"]]
        rep["changed_only"][g] = {"old": summarise(ch, "old"), "new": summarise(ch, "new")}
    for s in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == s and r["type_id"] in (0, 1)]
        rep["by_source"][s] = {"old": summarise(sub, "old"), "new": summarise(sub, "new")}
    off = [r for r in rows if r["type_id"] in (0, 1)]
    tr = defaultdict(int)
    for r in off:
        tr[(r["old_oracle_j"] >= 0.5, r["new_oracle_j"] >= 0.5)] += 1
    rep["transition_official"] = {"old_ok_new_ok": tr[(True, True)], "old_ok_new_fail": tr[(True, False)],
                                  "old_fail_new_ok": tr[(False, True)], "old_fail_new_fail": tr[(False, False)]}
    arch = [r for r in rows if str(r.get("old_n_archived") or "").strip() not in ("", "None")]
    if arch:
        rep["sanity_old_n_matches_archived"] = float(np.mean([int(float(r["old_n_archived"])) == r["old_n"] for r in arch]))
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    pct = lambda x: f"{100 * x:.2f}%"
    L = ["# ReVOS valid 候选池：旧 Ground 词 vs v2 词（同一套 SAM3 召回与打分代码）", "",
         f"表达式数 {len(rows)}（缺池子跳过 {missing}）。Recall@0.5 = 池中至少一个候选的区域 J ≥ 0.5；"
         "不可用 = 1 − Recall@0.5；Oracle J = 池中最好候选的区域 J 的平均。", ""]
    if "sanity_old_n_matches_archived" in rep:
        L += [f"自检：旧词重跑的候选数与历史记录一致的比例 {pct(rep['sanity_old_n_matches_archived'])}"
              "（应接近 100%，否则说明召回设置和论文结果不一致）", ""]

    def table(title: str, d: Dict[str, Any]) -> None:
        L.extend([f"## {title}", "", "| 分组 | n | Recall@0.5 旧 → 新 | 不可用 旧 → 新 | Oracle J 旧 → 新 | 0 候选 旧 → 新 | 平均候选数 旧 → 新 |",
                  "|---|---:|---|---|---|---|---|"])
        for g, v in d.items():
            o, nw = v["old"], v["new"]
            if not o.get("n"):
                continue
            L.append(f"| {g} | {o['n']} | {pct(o['recall@0.5'])} → **{pct(nw['recall@0.5'])}** | "
                     f"{pct(o['unavailable'])} → {pct(nw['unavailable'])} | {100 * o['oracle_j']:.2f} → {100 * nw['oracle_j']:.2f} | "
                     f"{pct(o['zero_cand'])} → {pct(nw['zero_cand'])} | {o['mean_cands']:.2f} → {nw['mean_cands']:.2f} |")
        L.append("")

    table("全部表达式", rep["groups"])
    table("只看换了词的表达式", rep["changed_only"])
    table("按来源（仅 Referring + Reasoning）", rep["by_source"])
    t = rep["transition_official"]
    L += ["## 逐条变化（Referring + Reasoning）", "",
          f"- 旧可用 → 新可用：{t['old_ok_new_ok']}", f"- 旧可用 → 新不可用（变差）：{t['old_ok_new_fail']}",
          f"- 旧不可用 → 新可用（救回）：{t['old_fail_new_ok']}", f"- 都不可用：{t['old_fail_new_fail']}", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["recall", "score"])
    ap.add_argument("--revos-root", required=True)
    ap.add_argument("--meta", default="meta_expressions_valid_.json")
    ap.add_argument("--mask-dict", default="mask_dict.json")
    ap.add_argument("--old-csv", required=True, help="old Ground words: all_cases_diagnostics.csv (video_id, exp_id, category, type_id, n)")
    ap.add_argument("--new-json", required=True, help="gv2.predict output (records with pred_category)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="")
    ap.add_argument("--sam3-ckpt", default="")
    ap.add_argument("--recall-top-k", type=int, default=24)
    ap.add_argument("--keep-top-k", type=int, default=7)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs of this shard (smoke test)")
    ap.add_argument("--mem-fraction", type=float, default=0.0, help="torch per-process GPU memory cap (0 = none)")
    a = ap.parse_args(argv)
    if a.stage == "recall":
        if not a.repo or not a.sam3_ckpt:
            ap.error("recall needs --repo and --sam3-ckpt")
        run_recall(a)
    else:
        run_score(a)


if __name__ == "__main__":
    main()
