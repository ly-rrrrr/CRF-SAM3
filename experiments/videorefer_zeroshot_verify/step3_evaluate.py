#!/usr/bin/env python
"""Step 3 (CPU): compare VideoRefer zero-shot Verify with the current Qwen3-VL Verify.

Everything is evaluated on the SAME replayed candidate sets and with the SAME
number of selected targets k (= len(historical selected_candidates), i.e. the
Ground stage's num_targets), so only the Verify decision differs.

Methods
  qwen          : historical Qwen3-VL+LoRA selection (scores.json), masks from replay
  vr:<label>    : top-k by VideoRefer score (one per --vr argument)
  fuse:<label>  : top-k by 0.5*P_qwen + 0.5*P_videorefer (late fusion, no training)
  sam3_native   : top-k by SAM3 native score sum (no Verify at all)
  random        : expected J&F of a uniformly random k-subset
  oracle        : top-k by per-candidate J&F against GT (upper bound of Verify)

Reported
  * ReVOS official-style J / F / J&F (referring, reasoning, overall = mean of the
    two), on (a) the evaluated contested subset and (b) the full valid set, where
    every non-evaluated expression keeps its official historical per-expression
    J/F from revos_valid_official.csv (identical for all methods, choice forced).
  * Candidate discrimination on k=1 expressions with an answerable oracle
    (best candidate J >= --answerable_j): top-1 accuracy, MRR, pairwise AUC.
  * Agreement / win-loss between Qwen and VideoRefer and their per-expression
    "pick the better of the two" upper bound (complementarity).
  * Breakdowns by expression type, #candidates and source dataset.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.io_utils import decode_rle, read_json_gz, read_jsonl  # noqa: E402

TYPE_NAME = {0: "referring", 1: "reasoning"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--cand_dir", default=None)
    p.add_argument("--revos_root", required=True)
    p.add_argument("--expression_file", default="meta_expressions_valid_.json")
    p.add_argument("--mask_dict", default=None, help="default: <revos_root>/mask_dict.json")
    p.add_argument("--official_csv", required=True, help="historical revos_valid_official.csv")
    p.add_argument("--vr", action="append", default=[], help="label=path/to/step2_output.jsonl (repeatable)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--answerable_j", type=float, default=0.5)
    p.add_argument("--include_id_mismatch", action="store_true")
    p.add_argument("--require_pixel_exact", action="store_true",
                   help="only evaluate expressions whose replayed selection is pixel-identical to the archive")
    p.add_argument("--random_samples", type=int, default=16)
    p.add_argument("--fusion_weight", type=float, default=0.5, help="weight of VideoRefer in fuse:*")
    p.add_argument("--export_results", action="store_true",
                   help="also write ReVOS results.json per method (needs --historical_results)")
    p.add_argument("--historical_results", default=None)
    return p.parse_args()


# ----------------------------------------------------------------------------
# metrics (official ReVOS code, copied from the CRF-SAM3 repo)
# ----------------------------------------------------------------------------
from common.revos_metrics import db_eval_boundary, db_eval_iou  # noqa: E402

G = {}


def _init_worker(mask_dict_path, cand_dir):
    import json as _json
    G["mask_dict"] = _json.load(open(mask_dict_path)) if mask_dict_path else None
    G["cand_dir"] = cand_dir
    G["cand_cache"] = {}


def _gt_video(anno_ids, n, h, w):
    md = G["mask_dict"]
    gt = np.zeros((n, h, w), dtype=np.uint8)
    for t in range(n):
        for a in anno_ids:
            rle = md[str(a)][t]
            if rle:
                gt[t] += decode_rle(rle)
    return gt


def _cand(cf):
    cache = G["cand_cache"]
    if cf not in cache:
        if len(cache) > 3:
            cache.clear()
        cache[cf] = read_json_gz(os.path.join(G["cand_dir"], cf))
    return cache[cf]


def _cand_video(cand, oid, n, h, w):
    out = np.zeros((n, h, w), dtype=np.uint8)
    for t, rle in cand["masks"].get(str(oid), {}).items():
        out[int(t)] = decode_rle(rle)
    return out


def _jf(gt, pred):
    j = float(db_eval_iou(gt, pred).mean())
    f = float(db_eval_boundary(gt, pred).mean())
    # official per-expression rounding (2 decimals, in %)
    return round(100 * j, 2), round(100 * f, 2)


def eval_expression(job):
    """job = (key, cand_file, anno_ids, subsets) -> metrics per candidate and per subset."""
    key, cf, anno_ids, subsets = job
    try:
        cand = _cand(cf)
        n, h, w = len(cand["frames"]), cand["height"], cand["width"]
        gt = (_gt_video(anno_ids, n, h, w) > 0).astype(np.uint8)
        per = {}
        vids = {}
        for oid in cand["candidate_ids"]:
            pv = _cand_video(cand, oid, n, h, w)
            vids[oid] = pv
            per[str(oid)] = _jf(gt, pv)
        sub = {}
        for s in subsets:
            s = tuple(sorted(int(x) for x in s))
            if len(s) == 0:
                sub[",".join(map(str, s))] = _jf(gt, np.zeros_like(gt))
            elif len(s) == 1:
                sub[str(s[0])] = per[str(s[0])]
            else:
                u = np.zeros_like(gt)
                for oid in s:
                    u |= vids[oid]
                sub[",".join(map(str, s))] = _jf(gt, u)
        return key, {"cand": per, "sub": sub}, None
    except Exception as e:  # pragma: no cover
        return key, None, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------
# selection helpers
# ----------------------------------------------------------------------------

def topk(scores: dict, k: int, tie: dict = None):
    tie = tie or {}
    ids = sorted(scores, key=lambda x: (-scores[x], -tie.get(x, 0.0), int(x)))
    return tuple(sorted(int(x) for x in ids[:k]))


def subset_key(s):
    return ",".join(str(x) for x in sorted(s))


def agg_official(rows, field_j, field_f):
    """ReVOS official aggregation: per-type mean, overall = mean of referring & reasoning."""
    out = {}
    for t, name in TYPE_NAME.items():
        rs = [r for r in rows if r["type_id"] == t]
        if rs:
            j = float(np.mean([r[field_j] for r in rs]))
            f = float(np.mean([r[field_f] for r in rs]))
            out[name] = {"J": j, "F": f, "JF": (j + f) / 2, "n": len(rs)}
    if "referring" in out and "reasoning" in out:
        o = {m: (out["referring"][m] + out["reasoning"][m]) / 2 for m in ("J", "F", "JF")}
    elif out:
        o = dict(next(iter(out.values())))
    else:
        o = {"J": float("nan"), "F": float("nan"), "JF": float("nan")}
    o["n"] = len(rows)
    out["overall"] = o
    return out


def ranking_stats(rows, method_scores_field, answerable_j):
    """k=1, answerable expressions: top-1 acc, MRR, pairwise AUC."""
    acc, mrr, auc_num, auc_den, n = 0.0, 0.0, 0.0, 0, 0
    for r in rows:
        if r["k"] != 1:
            continue
        cj = {oid: v[0] for oid, v in r["cand_metrics"].items()}
        cjf = {oid: (v[0] + v[1]) / 2 for oid, v in r["cand_metrics"].items()}
        if not cj or max(cj.values()) < 100 * answerable_j:
            continue
        sc = r[method_scores_field]
        if not sc:
            continue
        best = max(cjf.values())
        good = {o for o, v in cjf.items() if v >= best - 1e-6}
        order = sorted(sc, key=lambda x: -sc[x])
        n += 1
        acc += float(order[0] in good)
        rank = min(order.index(o) for o in good if o in order) + 1 if any(o in order for o in good) else len(order)
        mrr += 1.0 / rank
        pos = [o for o in sc if cj.get(o, 0) >= 100 * answerable_j]
        neg = [o for o in sc if cj.get(o, 0) < 100 * answerable_j]
        for a in pos:
            for b in neg:
                auc_den += 1
                auc_num += 1.0 if sc[a] > sc[b] else (0.5 if sc[a] == sc[b] else 0.0)
    return {"n": n, "top1_acc": acc / n if n else None, "mrr": mrr / n if n else None,
            "pair_auc": auc_num / auc_den if auc_den else None, "pairs": auc_den}


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cand_dir = args.cand_dir or os.path.join(os.path.dirname(os.path.abspath(args.manifest)), "candidates")
    mask_dict = args.mask_dict or os.path.join(args.revos_root, "mask_dict.json")
    meta = json.load(open(os.path.join(args.revos_root, args.expression_file)))["videos"]

    official = {}
    with open(args.official_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            official[f"{row['video_name']}||{row['exp_id']}"] = (float(row["J"]), float(row["F"]))

    manifest = {r["key"]: r for r in read_jsonl(args.manifest)}

    vr_runs = {}
    for spec in args.vr:
        label, path = spec.split("=", 1)
        vr_runs[label] = {r["key"]: r["scores"] for r in read_jsonl(path) if r.get("status") == "ok"}
        print(f"[step3] {label}: {len(vr_runs[label])} scored expressions")

    # ---------------- evaluated subset
    eval_rows = []
    for key, r in manifest.items():
        if not r["contested"] or not r.get("cand_file"):
            continue
        if not args.include_id_mismatch and not r.get("ids_match"):
            continue
        if args.require_pixel_exact and r.get("selected_exact_frames") != r.get("num_frames"):
            continue
        if any(key not in s for s in vr_runs.values()):
            continue  # compare only where every VideoRefer run has scores
        eval_rows.append(r)
    print(f"[step3] evaluated contested expressions: {len(eval_rows)}")

    # ---------------- selections per expression
    rng = random.Random(0)
    native_cache = {}
    jobs = []
    for r in eval_rows:
        cf = r["cand_file"]
        if cf not in native_cache:
            c = read_json_gz(os.path.join(cand_dir, cf))
            native_cache.clear()
            native_cache[cf] = {k: float(v) for k, v in c["native_score_sums"].items()}
        native = native_cache[cf]
        ids = [str(x) for x in r["all_candidates"]]
        k = r["k"]
        sels = {"qwen": tuple(sorted(r["selected_candidates"])),
                "sam3_native": topk({i: native.get(i, 0.0) for i in ids}, k)}
        q = {i: r["qwen_scores"].get(i, 0.0) for i in ids}
        for label, runs in vr_runs.items():
            v = {i: runs[r["key"]].get(i, 0.0) for i in ids}
            sels[f"vr:{label}"] = topk(v, k, tie=native)
            w = args.fusion_weight
            sels[f"fuse:{label}"] = topk({i: (1 - w) * q[i] + w * v[i] for i in ids}, k, tie=native)
        rand = []
        if k > 1:
            for _ in range(args.random_samples):
                rand.append(tuple(sorted(int(x) for x in rng.sample(ids, k))))
        r["_sels"], r["_rand"] = sels, rand
        subsets = set(s for s in sels.values() if len(s) > 1) | set(rand)
        jobs.append((r["key"], cf, r["anno_id"], sorted(subsets)))

    # ---------------- metrics
    results, errors = {}, {}
    print(f"[step3] computing J/F for {len(jobs)} expressions with {args.workers} workers ...")
    jobs.sort(key=lambda j: j[1])  # group by candidate file for the per-worker cache
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker, initargs=(mask_dict, cand_dir)) as pool:
            for key, res, err in pool.imap_unordered(eval_expression, jobs, chunksize=4):
                (results.__setitem__(key, res) if res is not None else errors.__setitem__(key, err))
    else:
        _init_worker(mask_dict, cand_dir)
        for job in jobs:
            key, res, err = eval_expression(job)
            (results.__setitem__(key, res) if res is not None else errors.__setitem__(key, err))
    if errors:
        print(f"[step3] {len(errors)} expressions failed, e.g. {next(iter(errors.items()))}")

    methods = ["qwen"] + [f"vr:{l}" for l in vr_runs] + [f"fuse:{l}" for l in vr_runs] + \
              ["sam3_native", "random", "oracle"]
    per_rows = []
    for r in eval_rows:
        res = results.get(r["key"])
        if res is None:
            continue
        cm = res["cand"]
        row = {"key": r["key"], "video_id": r["video_id"], "exp_id": r["exp_id"], "type_id": r["type_id"],
               "source": r["video_id"].split("/")[0], "n_cand": len(r["all_candidates"]), "k": r["k"],
               "expression": r["expression"], "cand_metrics": cm, "cand_file": r["cand_file"],
               "qwen_scores": {i: r["qwen_scores"].get(i, 0.0) for i in map(str, r["all_candidates"])}}
        for label in vr_runs:
            row[f"vr:{label}_scores"] = {i: vr_runs[label][r["key"]].get(i, 0.0) for i in map(str, r["all_candidates"])}
        for m, s in r["_sels"].items():
            jfv = cm[str(s[0])] if len(s) == 1 else res["sub"][subset_key(s)]
            row[f"{m}_sel"], row[f"{m}_J"], row[f"{m}_F"] = subset_key(s), jfv[0], jfv[1]
        if r["k"] == 1:
            row["random_J"] = float(np.mean([v[0] for v in cm.values()]))
            row["random_F"] = float(np.mean([v[1] for v in cm.values()]))
            best = max(cm, key=lambda o: (cm[o][0] + cm[o][1]))
            row["oracle_sel"], row["oracle_J"], row["oracle_F"] = best, cm[best][0], cm[best][1]
        else:
            vals = [res["sub"][subset_key(s)] for s in r["_rand"]]
            row["random_J"] = float(np.mean([v[0] for v in vals]))
            row["random_F"] = float(np.mean([v[1] for v in vals]))
            # greedy oracle for k>1: top-k candidates by individual J&F (approximate upper bound)
            top = sorted(cm, key=lambda o: -(cm[o][0] + cm[o][1]))[: r["k"]]
            s = tuple(sorted(int(x) for x in top))
            sub = res["sub"].get(subset_key(s))
            if sub is None:  # not precomputed; fall back to the best single-candidate value
                sub = cm[top[0]]
            row["oracle_sel"], row["oracle_J"], row["oracle_F"] = subset_key(s), sub[0], sub[1]
        off = official.get(r["key"])
        row["qwen_official_J"], row["qwen_official_F"] = (off if off else (None, None))
        per_rows.append(row)

    summary = {"n_evaluated": len(per_rows), "n_failed": len(errors), "methods": {}}

    # (a) contested subset
    for m in methods:
        summary["methods"][m] = {"subset": agg_official(per_rows, f"{m}_J", f"{m}_F")}

    # (b) full ReVOS valid: non-evaluated expressions keep the official per-expression values
    full_base = []
    evaluated = {r["key"] for r in per_rows}
    for vid, v in meta.items():
        for eid, e in v["expressions"].items():
            key = f"{vid}||{eid}"
            if key in evaluated:
                continue
            off = official.get(key)
            if off is None:
                continue
            full_base.append({"type_id": int(e["type_id"]), "J": off[0], "F": off[1]})
    for m in methods + ["qwen_official"]:
        rows = list(full_base) + [{"type_id": r["type_id"], "J": r[f"{m}_J"], "F": r[f"{m}_F"]}
                                  for r in per_rows if r.get(f"{m}_J") is not None]
        summary["methods"].setdefault(m, {})["full_valid"] = agg_official(rows, "J", "F")
    summary["full_valid_note"] = ("non-evaluated expressions (forced choice, no replay or id mismatch) use the "
                                  "historical official per-expression J/F, identical for all methods")

    # replay fidelity: qwen on replay vs official CSV on the evaluated subset
    paired = [r for r in per_rows if r["qwen_official_J"] is not None]
    if paired:
        d = [((r["qwen_J"] + r["qwen_F"]) - (r["qwen_official_J"] + r["qwen_official_F"])) / 2 for r in paired]
        summary["replay_fidelity"] = {
            "n": len(paired), "mean_abs_JF_diff": float(np.mean(np.abs(d))),
            "max_abs_JF_diff": float(np.max(np.abs(d))),
            "frac_within_0.01": float(np.mean(np.abs(d) <= 0.01 + 1e-9))}

    # ranking quality
    rank = {"qwen": ranking_stats(per_rows, "qwen_scores", args.answerable_j)}
    for label in vr_runs:
        rank[f"vr:{label}"] = ranking_stats(per_rows, f"vr:{label}_scores", args.answerable_j)
    summary["ranking_k1_answerable"] = rank

    # agreement / complementarity
    comp = {}
    for label in vr_runs:
        m = f"vr:{label}"
        jfq = np.array([(r["qwen_J"] + r["qwen_F"]) / 2 for r in per_rows])
        jfv = np.array([(r[f"{m}_J"] + r[f"{m}_F"]) / 2 for r in per_rows])
        same = np.mean([r["qwen_sel"] == r[f"{m}_sel"] for r in per_rows]) if per_rows else float("nan")
        best_of_two = [{"type_id": r["type_id"],
                        "J": r["qwen_J"] if (r["qwen_J"] + r["qwen_F"]) >= (r[f"{m}_J"] + r[f"{m}_F"]) else r[f"{m}_J"],
                        "F": r["qwen_F"] if (r["qwen_J"] + r["qwen_F"]) >= (r[f"{m}_J"] + r[f"{m}_F"]) else r[f"{m}_F"]}
                       for r in per_rows]
        comp[m] = {"same_selection_rate": float(same),
                   "videorefer_better(>1JF)": int(np.sum(jfv - jfq > 1)),
                   "qwen_better(>1JF)": int(np.sum(jfq - jfv > 1)),
                   "tie(<=1JF)": int(np.sum(np.abs(jfv - jfq) <= 1)),
                   "best_of_two_subset": agg_official(best_of_two, "J", "F")}
    summary["qwen_vs_videorefer"] = comp

    # breakdowns (subset J&F overall-style mean of per-expression JF, no type averaging)
    def bucket_n(n):
        return str(n) if n < 5 else "5+"
    groups = {"by_type": lambda r: TYPE_NAME.get(r["type_id"], str(r["type_id"])),
              "by_num_candidates": lambda r: bucket_n(r["n_cand"]),
              "by_source": lambda r: r["source"],
              "by_k": lambda r: "k=1" if r["k"] == 1 else "k>1"}
    br = {}
    for gname, fn in groups.items():
        g = defaultdict(list)
        for r in per_rows:
            g[fn(r)].append(r)
        br[gname] = {b: {"n": len(rs), **{m: float(np.mean([(x[f"{m}_J"] + x[f"{m}_F"]) / 2 for x in rs]))
                                          for m in methods}}
                     for b, rs in sorted(g.items())}
    summary["breakdown_mean_JF"] = br

    # ---------------- write
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2, ensure_ascii=False)
    cols = ["key", "type_id", "source", "n_cand", "k", "expression"] + \
           [f"{m}_{x}" for m in methods for x in ("sel", "J", "F")] + ["qwen_official_J", "qwen_official_F",
                                                                     "cand_metrics", "qwen_scores"] + \
           [f"vr:{l}_scores" for l in vr_runs]
    with open(os.path.join(args.out_dir, "per_expression.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in per_rows:
            rr = dict(r)
            for c in ("cand_metrics", "qwen_scores") + tuple(f"vr:{l}_scores" for l in vr_runs):
                rr[c] = json.dumps(rr.get(c), ensure_ascii=False)
            wr.writerow(rr)
    write_report(args, summary, methods, vr_runs)

    if args.export_results:
        export_results(args, per_rows, methods, cand_dir, meta)
    print(f"[step3] done -> {args.out_dir}/report.md")


def write_report(args, s, methods, vr_runs):
    L = []
    L.append("# VideoRefer-7B 零样本 Verify 对照结果\n")
    L.append(f"评测表达式（有竞争、重放 id 一致）：{s['n_evaluated']}，失败：{s['n_failed']}\n")
    if "replay_fidelity" in s:
        rf = s["replay_fidelity"]
        L.append(f"重放保真度：Qwen 历史选择在重放候选上的 J&F 与官方 CSV 的平均绝对差 "
                 f"{rf['mean_abs_JF_diff']:.4f}，最大 {rf['max_abs_JF_diff']:.2f}，"
                 f"差值 ≤0.01 的比例 {rf['frac_within_0.01']:.3f}（n={rf['n']}）\n")
    L.append("\n## 1. 有竞争子集（ReVOS 官方聚合：overall = referring 与 reasoning 的平均）\n")
    L.append("| 方法 | referring J&F | reasoning J&F | overall J | overall F | overall J&F |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for m in methods:
        a = s["methods"][m]["subset"]
        L.append(f"| {m} | {fmt(a.get('referring', {}).get('JF'))} | {fmt(a.get('reasoning', {}).get('JF'))} | "
                 f"{fmt(a['overall']['J'])} | {fmt(a['overall']['F'])} | {fmt(a['overall']['JF'])} |")
    L.append("\n## 2. 全量 ReVOS valid（非评测表达式沿用历史官方逐表达式指标）\n")
    L.append("| 方法 | referring J&F | reasoning J&F | overall J&F |")
    L.append("|---|---:|---:|---:|")
    for m in ["qwen_official"] + methods:
        a = s["methods"][m]["full_valid"]
        L.append(f"| {m} | {fmt(a.get('referring', {}).get('JF'))} | {fmt(a.get('reasoning', {}).get('JF'))} | "
                 f"{fmt(a['overall']['JF'])} |")
    L.append("\n## 3. 候选判别（k=1 且 oracle 候选 J ≥ %.2f）\n" % args.answerable_j)
    L.append("| 方法 | n | Top-1 准确率 | MRR | 成对 AUC |")
    L.append("|---|---:|---:|---:|---:|")
    for m, r in s["ranking_k1_answerable"].items():
        L.append(f"| {m} | {r['n']} | {fmt(r['top1_acc'], 4)} | {fmt(r['mrr'], 4)} | {fmt(r['pair_auc'], 4)} |")
    if s.get("qwen_vs_videorefer"):
        L.append("\n## 4. Qwen 与 VideoRefer 的一致性与互补性\n")
        L.append("| 运行 | 选择相同比例 | VideoRefer 更好(>1) | Qwen 更好(>1) | 持平 | 二者择优 overall J&F |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for m, c in s["qwen_vs_videorefer"].items():
            L.append(f"| {m} | {c['same_selection_rate']:.3f} | {c['videorefer_better(>1JF)']} | "
                     f"{c['qwen_better(>1JF)']} | {c['tie(<=1JF)']} | {fmt(c['best_of_two_subset']['overall']['JF'])} |")
    L.append("\n## 5. 分组（每条表达式 J&F 的简单平均）\n")
    for g, d in s["breakdown_mean_JF"].items():
        L.append(f"\n### {g}\n")
        L.append("| 组 | n | " + " | ".join(methods) + " |")
        L.append("|---|---:|" + "---:|" * len(methods))
        for b, v in d.items():
            L.append(f"| {b} | {v['n']} | " + " | ".join(fmt(v[m]) for m in methods) + " |")
    L.append("\n说明：fuse:* 是 Qwen 与 VideoRefer 概率的固定权重平均，仅作互补性参考，未调参也未训练；"
             "oracle 在 k>1 时为按单候选 J&F 取前 k 的近似上界。\n")
    open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8").write("\n".join(L))


def export_results(args, per_rows, methods, cand_dir, meta):
    """ReVOS results.json per method for tools/eval/eval_revos.py (official evaluator)."""
    if not args.historical_results:
        print("[step3] --export_results needs --historical_results; skipped")
        return
    from common.io_utils import encode_rle
    hist = json.load(open(args.historical_results))
    by_key = {r["key"]: r for r in per_rows}
    for m in [x for x in methods if x not in ("random",)]:
        out = json.loads(json.dumps(hist))  # deep copy; unchanged entries stay historical
        cache = {}
        for key, r in by_key.items():
            vid, eid = r["video_id"], str(r["exp_id"])
            cf = r["cand_file"]
            if cf not in cache:
                if len(cache) > 8:
                    cache.clear()
                cache[cf] = read_json_gz(os.path.join(cand_dir, cf))
            cand = cache[cf]
            sel = [int(x) for x in str(r[f"{m}_sel"]).split(",") if x != ""]
            n, h, w = len(cand["frames"]), cand["height"], cand["width"]
            rles = []
            for t in range(n):
                u = np.zeros((h, w), dtype=np.uint8)
                for oid in sel:
                    rr = cand["masks"].get(str(oid), {}).get(str(t))
                    if rr is not None:
                        u |= decode_rle(rr)
                rles.append(encode_rle(u))
            out.setdefault(vid, {}).setdefault(eid, {"video_id": vid, "exp_id": eid,
                                                     "frames": [str(x) for x in meta[vid]["frames"]]})
            out[vid][eid]["prediction_masks"] = rles
        name = m.replace(":", "_")
        d = os.path.join(args.out_dir, "results_json", name)
        os.makedirs(d, exist_ok=True)
        json.dump(out, open(os.path.join(d, "results.json"), "w"))
        print(f"[step3] wrote {d}/results.json  (run tools/eval/eval_revos.py on it)")


if __name__ == "__main__":
    main()
