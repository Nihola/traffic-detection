#!/usr/bin/env python
"""Unified evaluation: ONE script scores EVERY model (never use framework-internal mAP for the comparison).

Examples
  python evaluate.py --data_root /content/data/bdd5 --split test --name frcnn --model frcnn --ckpt runs/frcnn/best.pt
  python evaluate.py ... --name yolov8s --model yolo --ckpt runs/yolov8s_640/weights/best.pt
  python evaluate.py ... --name frcnn_zeroshot --model frcnn --zero_shot      # COCO-pretrained, no fine-tuning
  python evaluate.py ... --name sanity --gt_as_pred                           # must print AP = 1.000

Shared inference settings for all models: conf 0.001, NMS IoU 0.5, max 100 detections / image.
Outputs: results/{name}_{split}_dets.json (COCO results) and results/{name}_{split}_metrics.json
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import CLASSES

ZERO = dict(ap=0.0, ap50=0.0, ap75=0.0, ap_small=0.0, ap_medium=0.0, ap_large=0.0, ar100=0.0)


def _mean(x):
    x = np.asarray(x)
    x = x[x > -1]
    return float(x.mean()) if x.size else None


def load_gt(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def run_eval(gt, dets, img_ids=None):
    """Returns a COCOeval object, or None if there are no detections."""
    if not dets:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox")
        if img_ids is not None:
            E.params.imgIds = sorted(img_ids)
        E.evaluate()
        E.accumulate()
        E.summarize()
    return E


def summarize(E, with_details=True):
    """COCO stats -> dict. `precision` array layout: [IoU(10), recall(101), class, area(4), maxDets(3)]."""
    if E is None:
        return dict(ZERO)
    s = [None if v < 0 else float(v) for v in E.stats]
    out = dict(ap=s[0], ap50=s[1], ap75=s[2], ap_small=s[3], ap_medium=s[4], ap_large=s[5], ar100=s[8])
    if with_details:
        P = E.eval["precision"]
        out["ap_per_iou"] = {f"{0.5 + 0.05 * t:.2f}": _mean(P[t, :, :, 0, 2]) for t in range(P.shape[0])}
        out["per_class"] = {}
        for k, cid in enumerate(E.params.catIds):
            out["per_class"][CLASSES[cid - 1]] = dict(ap=_mean(P[:, :, k, 0, 2]), ap50=_mean(P[0, :, k, 0, 2]))
    return out


def condition_subsets(gt, min_images=20):
    """image-id subsets per BDD attribute value (e.g. weather=rainy, timeofday=night)."""
    subsets = {}
    for key in ["timeofday", "weather"]:
        for v in sorted({im.get(key) for im in gt.dataset["images"] if im.get(key)}):
            ids = [im["id"] for im in gt.dataset["images"] if im.get(key) == v]
            if len(ids) >= min_images:
                subsets[f"{key}={v}"] = ids
    return subsets


def score_all(gt, dets):
    res = dict(overall=summarize(run_eval(gt, dets)), conditions={})
    for name, ids in condition_subsets(gt).items():
        m = summarize(run_eval(gt, dets, ids), with_details=False)
        m["n_images"] = len(ids)
        res["conditions"][name] = m
    return res


def to_coco_dets(image_id, boxes, scores, classes):
    """boxes xyxy (pixels), classes 0..4 -> COCO result dicts with xywh and category 1..5."""
    out = []
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        out.append(dict(image_id=int(image_id), category_id=int(c) + 1,
                        bbox=[round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                        score=round(float(s), 5)))
    return out


def build_detector(a):
    from detectors import FRCNNDetector, YOLODetector       # lazy: torch is not needed for --gt_as_pred
    common = dict(conf=a.conf, nms_iou=a.nms_iou, max_det=a.max_det, half=a.half, zero_shot=a.zero_shot)
    if a.model == "frcnn":
        return FRCNNDetector(a.ckpt, proposals=a.proposals, **common)
    return YOLODetector(a.ckpt, imgsz=a.imgsz, **common)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", choices=["frcnn", "yolo"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--zero_shot", action="store_true", help="COCO-pretrained weights, COCO->our class mapping")
    ap.add_argument("--gt_as_pred", action="store_true", help="sanity test: ground truth as detections")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--max_det", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--proposals", type=int, default=1000, help="Faster R-CNN RPN post-NMS proposals at test time")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (debug)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    gt = load_gt(a.data_root / "annotations" / f"{a.split}.json")
    images = sorted(gt.dataset["images"], key=lambda x: x["id"])[: a.limit]

    if a.gt_as_pred:
        dets = [dict(image_id=x["image_id"], category_id=x["category_id"], bbox=x["bbox"], score=1.0)
                for x in gt.dataset["annotations"] if not x["iscrowd"]]
    else:
        import cv2
        from tqdm import tqdm
        assert a.model, "--model is required unless --gt_as_pred"
        assert a.ckpt or a.zero_shot, "--ckpt is required unless --zero_shot"
        det = build_detector(a)
        det.detect(cv2.imread(str(a.data_root / "images" / a.split / images[0]["file_name"])))   # warm-up
        dets = []
        for im in tqdm(images, desc=f"inference [{a.name}]"):
            frame = cv2.imread(str(a.data_root / "images" / a.split / im["file_name"]))   # BGR
            assert frame is not None, im["file_name"]
            b, s, c = det.detect(frame)
            dets += to_coco_dets(im["id"], b, s, c)

    (a.out / f"{a.name}_{a.split}_dets.json").write_text(json.dumps(dets))
    ids = [im["id"] for im in images]
    overall_E = run_eval(gt, dets, ids)
    res = score_all(gt, dets) if a.limit is None else dict(overall=summarize(overall_E), conditions={})
    res.update(name=a.name, split=a.split, n_images=len(images), n_detections=len(dets),
               settings=dict(conf=a.conf, nms_iou=a.nms_iou, max_det=a.max_det, imgsz=a.imgsz,
                             proposals=a.proposals, zero_shot=a.zero_shot, ckpt=a.ckpt))
    (a.out / f"{a.name}_{a.split}_metrics.json").write_text(json.dumps(res, indent=2))

    o = res["overall"]
    f = lambda v: "  n/a" if v is None else f"{v:.3f}"
    print(f"\n[{a.name}/{a.split}] AP@[.5:.95]={f(o['ap'])}  AP50={f(o['ap50'])}  AP75={f(o['ap75'])}  "
          f"small={f(o['ap_small'])}  medium={f(o['ap_medium'])}  large={f(o['ap_large'])}")
    for k, v in o.get("per_class", {}).items():
        print(f"   {k:11s} AP={f(v['ap'])}  AP50={f(v['ap50'])}")
    for k, v in res["conditions"].items():
        print(f"   {k:22s} n={v['n_images']:4d}  AP50={f(v['ap50'])}  AP={f(v['ap'])}")


if __name__ == "__main__":
    main()
