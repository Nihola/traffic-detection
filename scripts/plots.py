#!/usr/bin/env python
"""Report figures/tables from results/*_metrics.json and results/speed_*.json.
Produces: iou_curve.png, per_class_ap50.png, conditions.csv, summary.csv, speed_vs_accuracy.png"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--exclude", nargs="*", default=["sanity"], help="run names to skip")
    a = ap.parse_args()

    M = {}
    for p in sorted(a.results.glob(f"*_{a.split}_metrics.json")):
        m = json.loads(p.read_text())
        if m["name"] not in a.exclude:
            M[m["name"]] = m
    if not M:
        raise SystemExit("no metrics files found")

    # IoU-threshold analysis
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n, m in M.items():
        d = m["overall"].get("ap_per_iou")
        if d:
            ax.plot([float(k) for k in d], [v or 0 for v in d.values()], marker="o", label=n)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_title("Accuracy vs localisation strictness")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.results / "iou_curve.png", dpi=150)

    # per-class AP50
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(M)
    for i, (n, m) in enumerate(M.items()):
        pc = m["overall"].get("per_class", {})
        ax.bar(np.arange(len(CLASSES)) + i * w, [(pc.get(c) or {}).get("ap50") or 0 for c in CLASSES], w, label=n)
    ax.set_xticks(np.arange(len(CLASSES)) + 0.4 - w / 2)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-class AP50")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.results / "per_class_ap50.png", dpi=150)

    # tables
    f = lambda v: "" if v is None else f"{v:.4f}"
    with open(a.results / "summary.csv", "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["model", "AP50", "AP75", "AP[.5:.95]", "AP_small", "AP_medium", "AP_large"])
        for n, m in M.items():
            o = m["overall"]
            w_.writerow([n] + [f(o[k]) for k in ["ap50", "ap75", "ap", "ap_small", "ap_medium", "ap_large"]])
    conds = sorted({c for m in M.values() for c in m["conditions"]})
    with open(a.results / "conditions.csv", "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["condition", "n_images"] + [f"{n}_AP50" for n in M] + [f"{n}_AP" for n in M])
        for c in conds:
            n_img = next(m["conditions"][c]["n_images"] for m in M.values() if c in m["conditions"])
            w_.writerow([c, n_img] + [f((m["conditions"].get(c) or {}).get("ap50")) for m in M.values()]
                        + [f((m["conditions"].get(c) or {}).get("ap")) for m in M.values()])

    # speed vs accuracy (needs speed_<name>_<prec>.json with the same name as the metrics file)
    S = [json.loads(p.read_text()) for p in sorted(a.results.glob("speed_*.json"))]
    S = [s for s in S if s["name"] in M]
    if S:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, key, lab in [(axes[0], "ap50", "AP@0.5"), (axes[1], "ap", "AP@[.5:.95]")]:
            for s in S:
                y = M[s["name"]]["overall"][key]
                x = s["end_to_end"]["fps"]
                ax.scatter(x, y, s=70)
                ax.annotate(f"{s['name']} ({s['precision']})", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
            ax.set_xlabel("FPS (end-to-end, batch 1)")
            ax.set_ylabel(lab)
            ax.grid(alpha=.3)
        axes[0].set_title(f"Speed vs accuracy ({S[0]['gpu']})")
        fig.tight_layout()
        fig.savefig(a.results / "speed_vs_accuracy.png", dpi=150)
    else:
        print("no speed_*.json matching a metrics name: skipped speed_vs_accuracy.png")
    print("wrote figures/tables to", a.results)


if __name__ == "__main__":
    main()
