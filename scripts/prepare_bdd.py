#!/usr/bin/env python
"""Build the project dataset (bdd5) from BDD100K detection labels.

Splits (BDD100K *test* labels are not public, so we never use the official test set):
  train / val : disjoint, stratified samples of the OFFICIAL TRAIN split
  test        : sample of the OFFICIAL VAL split, with guaranteed minimum rainy / night images

Output layout (one source of truth -> two formats):
  OUT/images/{split}/*.jpg
  OUT/labels/{split}/*.txt        YOLO format (class 0-4, normalised cx cy w h, crowd boxes skipped)
  OUT/annotations/{split}.json    COCO format (category ids 1-5, crowd boxes kept with iscrowd=1)
  OUT/splits/{split}.txt          file lists (commit these to Git)
  OUT/dataset_stats.json
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from common import BDD_ALIAS, CLASSES, IMG_H, IMG_W


def convert_entry(entry):
    """BDD entry -> list of COCO-style annotation dicts (xywh, absolute pixels)."""
    anns = []
    for lab in entry.get("labels") or []:
        cid = BDD_ALIAS.get(lab.get("category"))
        box = lab.get("box2d")
        if cid is None or box is None:      # other classes, lanes, drivable areas
            continue
        x1, y1 = max(0.0, box["x1"]), max(0.0, box["y1"])
        x2, y2 = min(float(IMG_W), box["x2"]), min(float(IMG_H), box["y2"])
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2:                  # degenerate boxes break training / give NaN losses
            continue
        crowd = int(bool((lab.get("attributes") or {}).get("crowd", False)))
        anns.append(dict(category_id=cid, bbox=[round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                         area=round(w * h, 2), iscrowd=crowd))
    return anns


def collect(json_path, img_dir):
    with open(json_path) as f:
        data = json.load(f)
    items, missing = [], 0
    for e in data:
        anns = convert_entry(e)
        if not anns:                        # keep only images with >= 1 target object
            continue
        if not (img_dir / e["name"]).exists():
            missing += 1
            continue
        attr = e.get("attributes") or {}
        items.append(dict(name=e["name"], weather=attr.get("weather", "undefined"),
                          timeofday=attr.get("timeofday", "undefined"), anns=anns))
    print(f"[{json_path.name}] {len(data)} entries -> {len(items)} usable ({missing} listed but image file missing)")
    return items


def stratified_sample(items, n, rng):
    """Proportional allocation over (timeofday, weather) groups (largest-remainder rounding)."""
    n = min(n, len(items))
    if n <= 0:
        return []
    groups = defaultdict(list)
    for it in items:
        groups[(it["timeofday"], it["weather"])].append(it)
    total = len(items)
    quota = {k: len(v) * n / total for k, v in groups.items()}
    base = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - base[k], reverse=True)[: n - sum(base.values())]:
        base[k] += 1
    out = []
    for k, v in groups.items():
        rng.shuffle(v)
        out += v[: base[k]]
    return out


def sample_test(items, n, min_rainy, min_night, rng):
    """Guarantee minimum rainy / night counts (otherwise condition analysis is statistically meaningless)."""
    chosen = {}

    def top_up(pred, k):
        have = sum(1 for c in chosen.values() if pred(c))
        pool = [it for it in items if pred(it) and it["name"] not in chosen]
        rng.shuffle(pool)
        for it in pool[: max(0, k - have)]:
            chosen[it["name"]] = it

    top_up(lambda it: it["weather"] == "rainy", min_rainy)
    top_up(lambda it: it["timeofday"] == "night", min_night)
    rest = [it for it in items if it["name"] not in chosen]
    return list(chosen.values()) + stratified_sample(rest, n - len(chosen), rng)


def write_split(split, items, src_dir, out, id_offset):
    img_out, lab_out = out / "images" / split, out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)

    images, annotations, names = [], [], []
    ann_id, img_id = 1, id_offset
    counts = Counter()
    for it in tqdm(items, desc=f"writing {split}"):
        src = src_dir / it["name"]
        with Image.open(src) as im:
            w, h = im.size
        if (w, h) != (IMG_W, IMG_H):
            print(f"  skip {it['name']}: size {w}x{h} != {IMG_W}x{IMG_H}")
            continue
        shutil.copy2(src, img_out / it["name"])
        img_id += 1
        images.append(dict(id=img_id, file_name=it["name"], width=w, height=h,
                           weather=it["weather"], timeofday=it["timeofday"]))
        lines = []
        for a in it["anns"]:
            annotations.append(dict(id=ann_id, image_id=img_id, **a))
            ann_id += 1
            if a["iscrowd"]:
                continue
            counts[CLASSES[a["category_id"] - 1]] += 1
            x, y, bw, bh = a["bbox"]
            lines.append(f"{a['category_id'] - 1} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
        (lab_out / (Path(it["name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        names.append(it["name"])

    coco = dict(images=images, annotations=annotations,
                categories=[dict(id=i + 1, name=c) for i, c in enumerate(CLASSES)])
    (out / "annotations" / f"{split}.json").write_text(json.dumps(coco))
    (out / "splits" / f"{split}.txt").write_text("\n".join(sorted(names)) + "\n")
    stats = dict(n_images=len(images), n_boxes=dict(counts),
                 weather=dict(Counter(i["weather"] for i in images)),
                 timeofday=dict(Counter(i["timeofday"] for i in images)))
    return stats, img_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_json", required=True, type=Path, help="BDD100K detection labels, train")
    ap.add_argument("--val_json", required=True, type=Path, help="BDD100K detection labels, val")
    ap.add_argument("--train_img_dir", required=True, type=Path)
    ap.add_argument("--val_img_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--min_rainy_test", type=int, default=300)
    ap.add_argument("--min_night_test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zip", action="store_true", help="also create OUT.zip (upload this to Drive)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    train_pool = collect(a.train_json, a.train_img_dir)
    val_pool = collect(a.val_json, a.val_img_dir)

    sel = stratified_sample(train_pool, a.n_train + a.n_val, rng)
    val_items = stratified_sample(sel, a.n_val, rng)
    val_names = {i["name"] for i in val_items}
    train_items = [i for i in sel if i["name"] not in val_names]
    test_items = sample_test(val_pool, a.n_test, a.min_rainy_test, a.min_night_test, rng)

    all_stats, offset = {}, 0
    for split, items, src in [("train", train_items, a.train_img_dir),
                              ("val", val_items, a.train_img_dir),
                              ("test", test_items, a.val_img_dir)]:
        all_stats[split], offset = write_split(split, items, src, a.out, offset)   # ids unique across splits

    (a.out / "dataset_stats.json").write_text(json.dumps(all_stats, indent=2))
    print(json.dumps(all_stats, indent=2))
    if all_stats["test"]["weather"].get("rainy", 0) < 100:
        print("WARNING: very few rainy test images; rain analysis will be unreliable.")
    if a.zip:
        shutil.make_archive(str(a.out), "zip", root_dir=a.out.parent, base_dir=a.out.name)
        print(f"created {a.out}.zip")


if __name__ == "__main__":
    main()
