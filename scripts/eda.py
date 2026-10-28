#!/usr/bin/env python
"""EDA figures for the report: class counts, box-size distribution, objects per image, conditions."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/eda"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    data = {s: json.loads((a.data_root / "annotations" / f"{s}.json").read_text()) for s in splits}

    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.27
    for i, s in enumerate(splits):
        c = Counter(x["category_id"] for x in data[s]["annotations"] if not x["iscrowd"])
        ax.bar(np.arange(len(CLASSES)) + i * w, [c[k + 1] for k in range(len(CLASSES))], w, label=s)
    ax.set_xticks(np.arange(len(CLASSES)) + w)
    ax.set_xticklabels(CLASSES)
    ax.set_yscale("log")
    ax.set_ylabel("boxes (log scale)")
    ax.set_title("Class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "class_counts.png", dpi=150)

    areas = np.array([x["area"] for x in data["train"]["annotations"] if not x["iscrowd"]])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.sqrt(areas), bins=60)
    for t, lab in [(32, "small<32px"), (96, "large>96px")]:
        ax.axvline(t, color="r", ls="--")
    ax.set_xlabel("sqrt(box area) in px (original 1280x720)")
    ax.set_title("Box size distribution (train)")
    fig.tight_layout()
    fig.savefig(a.out / "box_sizes.png", dpi=150)
    print(f"train boxes: small(<32px) {np.mean(np.sqrt(areas) < 32):.1%}, medium {np.mean((np.sqrt(areas) >= 32) & (np.sqrt(areas) < 96)):.1%}, "
          f"large {np.mean(np.sqrt(areas) >= 96):.1%}")

    per_img = Counter(x["image_id"] for x in data["train"]["annotations"] if not x["iscrowd"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(list(per_img.values()), bins=40)
    ax.set_xlabel("objects per image")
    ax.set_title("Objects per image (train)")
    fig.tight_layout()
    fig.savefig(a.out / "objects_per_image.png", dpi=150)

    rows = []
    for s in splits:
        for key in ["timeofday", "weather"]:
            for k, v in Counter(i[key] for i in data[s]["images"]).items():
                rows.append(f"{s},{key},{k},{v}")
    (a.out / "conditions.csv").write_text("split,attribute,value,n_images\n" + "\n".join(rows) + "\n")
    print("EDA written to", a.out)


if __name__ == "__main__":
    main()
