#!/usr/bin/env python
"""Generate a tiny synthetic dataset in BDD100K label format, to test the whole pipeline
(prepare -> verify -> train -> evaluate) in minutes without downloading anything."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

COLS = {"pedestrian": (60, 76, 231), "car": (113, 204, 46), "bus": (219, 152, 52),
        "truck": (15, 196, 241), "motorcycle": (182, 89, 155)}


def make(root: Path, split: str, n: int, rng):
    img_dir = root / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n):
        night = rng.random() < 0.3
        weather = rng.choice(["clear", "clear", "overcast", "rainy"])
        img = np.full((720, 1280, 3), 40 if night else 150, np.uint8)
        img += np.random.RandomState(i).randint(0, 20, img.shape, dtype=np.uint8)
        labels = [{"category": "traffic light", "box2d": {"x1": 5, "y1": 5, "x2": 30, "y2": 40}},
                  {"category": "lane/road curb", "poly2d": []}]          # must be ignored
        for _ in range(rng.randint(1, 6)):
            cat = rng.choice(list(COLS))
            w, h = rng.randint(40, 300), rng.randint(40, 200)
            x1, y1 = rng.randint(0, 1280 - w), rng.randint(0, 720 - h)
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), COLS[cat], -1)
            labels.append({"category": cat, "attributes": {"crowd": False},
                           "box2d": {"x1": float(x1), "y1": float(y1), "x2": float(x1 + w), "y2": float(y1 + h)}})
        name = f"{split}_{i:05d}.jpg"
        cv2.imwrite(str(img_dir / name), img)
        entries.append({"name": name, "attributes": {"weather": weather, "scene": "city street",
                        "timeofday": "night" if night else "daytime"}, "labels": labels})
    (root / f"labels_{split}.json").write_text(json.dumps(entries))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--n_val", type=int, default=60)
    a = ap.parse_args()
    rng = random.Random(0)
    make(a.out, "train", a.n_train, rng)
    make(a.out, "val", a.n_val, rng)
    print("fake BDD written to", a.out)
