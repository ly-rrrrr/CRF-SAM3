#!/usr/bin/env python3
"""Merge run_p1a results on the ReVOS subsets with ground_v2's per_expression.csv into one table.

Samples that were not run keep their step-4 pool J (adding tubes can only raise the oracle J), so the full-set
numbers are exact for zero/fail once those subsets are finished; the 'ok' subset is a random sample used only to
estimate how many tubes are added to pools that were already fine.

  python -m p1a.revos_summary --per-expression .../per_expression.csv --runs runs/revos_v2 --out runs/revos_v2/summary.md
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .revos_subsets import subset_of

TYPES = {"Referring": {0}, "Reasoning": {1}, "Official (Ref+Rea)": {0, 1}, "All": {0, 1, 2}}


def load_results(run_dirs):
    res = {}
    for d in run_dirs:
        p = Path(d) / "results.jsonl"
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    res[r["key"]] = r
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-expression", required=True)
    ap.add_argument("--runs", required=True, help="dir holding run_zero/ run_fail/ run_ok/ (each with results.jsonl)")
    ap.add_argument("--modes", default="rescue,point,oracle_gt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    modes = [m for m in a.modes.split(",") if m]
    rows = list(csv.DictReader(open(a.per_expression, encoding="utf-8")))
    res = load_results(sorted(Path(a.runs).glob("run_*")))
    by_sub = defaultdict(list)
    for r in rows:
        r["subset"] = subset_of(r)
        r["res"] = res.get(f"{r['video_id']}/{r['exp_id']}")
        by_sub[r["subset"]].append(r)

    L = ["# 热力图辅助扩展候选池：ReVOS valid（v2 词的候选池 + v2 热力图）", ""]
    L += ["## 各子集", "", "| 子集 | 总数 | 已跑 | 触发率 | 方式 | 救回（J*≥0.5） | 子集 Recall 前 → 后 | 平均新增候选 |",
          "|---|---:|---:|---:|---|---:|---|---:|"]
    added_per = {}
    for sub in ("zero", "fail", "ok"):
        rs = [r for r in by_sub[sub] if r["res"]]
        if not rs:
            L.append(f"| {sub} | {len(by_sub[sub])} | 0 | - | - | - | - | - |")
            continue
        trig = np.mean([bool(r["res"].get("triggered")) for r in rs])
        for m in modes:
            got = [r for r in rs if m in r["res"]]
            if not got:
                continue
            before = np.mean([r["res"]["j_old"] >= 0.5 for r in got])
            after = np.mean([r["res"][m]["j_star"] >= 0.5 for r in got])
            saved = sum(r["res"]["j_old"] < 0.5 <= r["res"][m]["j_star"] for r in got)
            add = float(np.mean([r["res"][m]["added"] for r in got]))
            added_per[(sub, m)] = add
            L.append(f"| {sub} | {len(by_sub[sub])} | {len(got)} | {trig:.1%} | {m} | {saved} | {before:.1%} → {after:.1%} | {add:.2f} |")
    L.append("")
    L += ["## 全量 Recall@0.5（未跑的样本保持第 4 步的值）", "",
          "| 分组 | n | 第 4 步（v2 词） | + " + " | + ".join(modes) + " |", "|---|---:|---|" + "---|" * len(modes)]
    for g, ts in TYPES.items():
        sub_rows = [r for r in rows if int(r["type_id"]) in ts]
        if not sub_rows:
            continue
        base = np.mean([float(r["new_oracle_j"]) >= 0.5 for r in sub_rows])
        cells = []
        for m in modes:
            v = [((r["res"][m]["j_star"] if (r["res"] and m in r["res"]) else float(r["new_oracle_j"])) >= 0.5) for r in sub_rows]
            cells.append(f"{np.mean(v):.2%}")
        L.append(f"| {g} | {len(sub_rows)} | {base:.2%} | " + " | ".join(cells) + " |")
    L.append("")
    done = {s: sum(1 for r in by_sub[s] if r["res"]) for s in by_sub}
    L.append(f"已跑：zero {done.get('zero', 0)}/{len(by_sub['zero'])}，fail {done.get('fail', 0)}/{len(by_sub['fail'])}，"
             f"ok {done.get('ok', 0)}（抽样）。zero 和 fail 没跑完时，全量数字是下限。")
    for m in modes:
        if m == "oracle_gt":
            continue
        tot = sum(added_per.get((s, m), 0.0) * len(by_sub[s]) for s in ("zero", "fail", "ok"))
        if tot:
            L.append(f"- {m}：估计每条表达式平均多出 {tot / len(rows):.2f} 个候选（ok 子集按抽样外推），这是 Verify 要多承受的干扰。")
    mism = [r for r in rows if r["res"] and abs(r["res"]["j_old"] - float(r["new_oracle_j"])) > 0.02]
    L += ["", f"自检：已跑样本里，本脚本算的原池 J* 与第 4 步差异 > 0.02 的条数：{len(mism)}（应接近 0）",
          "", "oracle_gt 是用 GT 框做种子的上限，只用来判断「SAM3 被告知是哪个之后能做多好」，不是方法。"]
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
