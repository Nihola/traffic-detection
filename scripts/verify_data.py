#!/usr/bin/env python
"""Quality gate: run this BEFORE anyone trains. Exits non-zero if any check fails.

Checks: no image shared between splits, COCO vs YOLO annotation counts identical, every image has a
label file, boxes inside the image, class ids in range. Also saves side-by-side visualisations
(left = COCO json boxes, right = YOLO txt boxes) so you can eyeball that both formats agree.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2

from common import CLASSES, COLORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--n_vis", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/data_check"))
    a = ap.parse_args()
    root, errors = a.data_root, []
    a.out.mkdir(parents=True, exist_ok=True)

    coco, names = {}, {}
    for s in ["train", "val", "test"]:
        coco[s] = json.loads((root / "annotations" / f"{s}.json").read_text())
        names[s] = {i["file_name"] for i in coco[s]["images"]}
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        if names[s1] & names[s2]:
            errors.append(f"LEAKAGE: {len(names[s1] & names[s2])} images shared by {s1} and {s2}")

    for s in coco:
        imgs = {i["id"]: i for i in coco[s]["images"]}
        n_coco = sum(1 for x in coco[s]["annotations"] if not x["iscrowd"])
        n_yolo = 0
        for im in imgs.values():
            lab = root / "labels" / s / (Path(im["file_name"]).stem + ".txt")
            if not (root / "images" / s / im["file_name"]).exists():
                errors.append(f"{s}: missing image {im['file_name']}")
            if not lab.exists():
                errors.append(f"{s}: missing label file for {im['file_name']}")
                continue
            for line in lab.read_text().split("\n"):
                if not line.strip():
                    continue
                c, cx, cy, w, h = line.split()
                c, vals = int(c), [float(v) for v in (cx, cy, w, h)]
                n_yolo += 1
                if not 0 <= c < len(CLASSES):
                    errors.append(f"{s}: bad class id {c} in {lab.name}")
                if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    errors.append(f"{s}: out-of-range box in {lab.name}: {line}")
        for x in coco[s]["annotations"]:
            im = imgs[x["image_id"]]
            bx, by, bw, bh = x["bbox"]
            if bw <= 0 or bh <= 0 or bx < 0 or by < 0 or bx + bw > im["width"] + 0.05 or by + bh > im["height"] + 0.05:
                errors.append(f"{s}: bad COCO box {x['id']}: {x['bbox']}")
                break
            if not 1 <= x["category_id"] <= len(CLASSES):
                errors.append(f"{s}: bad COCO category {x['category_id']}")
                break
        if n_coco != n_yolo:
            errors.append(f"{s}: non-crowd boxes differ (COCO {n_coco} vs YOLO {n_yolo})")
        print(f"{s:5s}: {len(imgs):6d} images | {n_coco:7d} boxes (COCO == YOLO: {n_coco == n_yolo})")

    rng = random.Random(0)
    for k, im in enumerate(rng.sample(coco["train"]["images"], min(a.n_vis, len(coco["train"]["images"])))):
        img = cv2.imread(str(root / "images" / "train" / im["file_name"]))
        left, right = img.copy(), img.copy()
        for x in coco["train"]["annotations"]:
            if x["image_id"] == im["id"]:
                bx, by, bw, bh = map(int, x["bbox"])
                cv2.rectangle(left, (bx, by), (bx + bw, by + bh), COLORS[x["category_id"] - 1], 3)
        lab = root / "labels" / "train" / (Path(im["file_name"]).stem + ".txt")
        for line in lab.read_text().split("\n"):
            if line.strip():
                c, cx, cy, w, h = line.split()
                c, cx, cy, w, h = int(c), float(cx) * im["width"], float(cy) * im["height"], \
                    float(w) * im["width"], float(h) * im["height"]
                cv2.rectangle(right, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                              COLORS[c], 3)
                cv2.putText(right, CLASSES[c], (int(cx - w / 2), int(cy - h / 2) - 5), 0, 0.9, COLORS[c], 2)
        cv2.imwrite(str(a.out / f"vis_{k:02d}.jpg"), cv2.resize(cv2.hconcat([left, right]), None, fx=0.5, fy=0.5))
    print(f"saved visual checks to {a.out}/ (open them: the two halves must show identical boxes)")

    if errors:
        print(f"\nFAILED with {len(errors)} problems:")
        print("\n".join(errors[:30]))
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
