#!/usr/bin/env python3
"""Draw training curves from the ``[gv2] {json}`` lines that gv2.train prints (and 03_train.sh tees
into ``<run>.log``), and print a short health check. Works while training is still running.

  python -m gv2.plot_log runs/gv2_revos.log                      # -> runs/gv2_revos.png
  python -m gv2.plot_log runs/gv2_revos.log runs/gv2_rowB.log --labels E B --out runs/compare.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

PANELS: List[Tuple[str, List[str]]] = [
    ("total loss", ["loss", "eval_loss"]),
    ("heatmap loss (weighted BCE + Dice)", ["loss_mask", "eval_loss_mask"]),
    ("BCE / Dice", ["loss_bce", "loss_dice"]),
    ("category-word loss (CE)", ["loss_cat", "eval_loss_cat"]),
    ("grad norm (before clipping)", ["grad_norm"]),
    ("learning rate", ["learning_rate"]),
]


def parse(path: str) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]], List[str]]:
    """Returns train rows and eval rows keyed by step (a later line for the same step wins, so a
    re-run appended to the same log replaces the old values), plus warning lines."""
    train, evals, warns = {}, {}, []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        # tqdm may prepend its progress bar to the JSON record on the same line.
        start = line.find("[gv2] ")
        if start < 0:
            continue
        line = line[start:]
        if line.startswith("[gv2] WARNING") or line.startswith("[gv2] STOPPED"):
            warns.append(line)
        if not line.startswith("[gv2] {"):
            continue
        try:
            row, _ = json.JSONDecoder().raw_decode(line[len("[gv2] "):])
        except json.JSONDecodeError:
            continue
        step = int(row.get("step", -1))
        if "loss" in row:
            train[step] = row
        elif "eval_loss" in row:
            evals[step] = row
    return train, evals, warns


def ema(vals: List[float], alpha: float) -> List[float]:
    out, cur = [], None
    for v in vals:
        if not math.isfinite(v):
            out.append(float("nan"))
            continue
        cur = v if cur is None else alpha * v + (1 - alpha) * cur
        out.append(cur)
    return out


def health(name: str, train: Dict[int, Dict[str, float]], evals: Dict[int, Dict[str, float]]) -> List[str]:
    lines = [f"== {name}: {len(train)} train logs, {len(evals)} eval logs, last step {max(train) if train else '-'}"]
    steps = sorted(train)
    for key in ("loss", "loss_mask", "loss_cat"):
        vals = [train[s][key] for s in steps if key in train[s]]
        if len(vals) < 4:
            continue
        bad = sum(not math.isfinite(v) for v in vals)
        q = max(1, len(vals) // 5)
        head = sum(vals[:q]) / q
        tail = [v for v in vals[-q:] if math.isfinite(v)]
        tail_m = sum(tail) / len(tail) if tail else float("nan")
        trend = "down" if tail_m < head * 0.95 else ("UP" if tail_m > head * 1.05 else "flat")
        lines.append(f"  {key:10s} first 20% avg {head:.4f} -> last 20% avg {tail_m:.4f}  [{trend}]"
                     + (f"  NON-FINITE x{bad}" if bad else ""))
    g = [train[s]["grad_norm"] for s in steps if "grad_norm" in train[s]]
    if g:
        fin = sorted(v for v in g if math.isfinite(v))
        med = fin[len(fin) // 2] if fin else float("nan")
        spikes = sum(v > 10 * med for v in fin) if fin else 0
        lines.append(f"  grad_norm  median {med:.3f}, max {max(fin) if fin else float('nan'):.3f}, "
                     f">10x median: {spikes}" + (f", NON-FINITE x{len(g) - len(fin)}" if len(fin) < len(g) else ""))
    if evals:
        es = sorted(evals)
        ev = [evals[s]["eval_loss"] for s in es]
        best = min(range(len(ev)), key=lambda i: ev[i] if math.isfinite(ev[i]) else float("inf"))
        lines.append(f"  eval_loss  " + ", ".join(f"{s}:{v:.4f}" for s, v in zip(es, ev))
                     + f"  (best at step {es[best]})")
        if best < len(ev) - 2:
            lines.append("  note: eval_loss has not improved for the last 2 evals (possible overfitting)")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--smooth", type=float, default=0.2, help="EMA weight of the newest point (1 = no smoothing)")
    args = ap.parse_args(argv)
    labels = args.labels or [Path(p).stem for p in args.logs]
    runs = [(lab, *parse(p)) for lab, p in zip(labels, args.logs)]

    for lab, train, evals, warns in runs:
        print("\n".join(health(lab, train, evals)))
        for w in warns:
            print("  " + w)

    if not any(train or evals for _, train, evals, _ in runs):
        raise SystemExit("No training/evaluation records found; no empty plot was written.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed: health check above only (pip install matplotlib for the PNG)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    for ax, (title, keys) in zip(axes.flat, PANELS):
        for ci, (lab, train, evals, _) in enumerate(runs):
            col = colors[ci % len(colors)]
            for ki, key in enumerate(keys):
                src = evals if key.startswith("eval_") else train
                steps = [s for s in sorted(src) if key in src[s]]
                vals = [float(src[s][key]) for s in steps]
                if not steps:
                    continue
                name = key if len(runs) == 1 else f"{lab} {key}"
                if key.startswith("eval_"):
                    ax.plot(steps, vals, "o--", color=col, ms=4, lw=1.2, label=name)
                elif key == "learning_rate":
                    ax.plot(steps, vals, color=col, lw=1.5, label=name)
                else:
                    ls = "-" if ki == 0 else ":"
                    ax.plot(steps, vals, ls, color=col, alpha=0.25, lw=0.8)
                    ax.plot(steps, ema(vals, args.smooth), ls, color=col, lw=1.8, label=name)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8)
        if title.startswith("grad norm") and ax.lines:
            ax.set_yscale("log")
    fig.tight_layout()
    out = args.out or str(Path(args.logs[0]).with_suffix(".png"))
    fig.savefig(out, dpi=120)
    print(f"\ncurves -> {out}")


if __name__ == "__main__":
    main()
