#!/usr/bin/env python
"""Latency / FPS benchmark. Run ALL models in ONE session on ONE GPU and record the GPU name.

  python benchmark_speed.py --name frcnn   --model frcnn --ckpt runs/frcnn/best.pt --image some_test.jpg
  python benchmark_speed.py --name yolov8s --model yolo  --ckpt runs/yolov8s_640/weights/best.pt --image some_test.jpg --half

Protocol: batch size 1, real image (NMS cost depends on content), 50 warm-up + 300 timed runs, deployment
confidence (0.25, NOT the 0.001 used for mAP: that floods NMS and inflates post-processing time).
Two levels are reported (label them in the report):
  model_postprocess : input tensor already on GPU -> detections (forward + built-in NMS/box decoding)
                      NOTE: Faster R-CNN gets the full 1280x720 tensor and resizes internally (its resize is part of
                      the model); YOLO gets a pre-resized 640x384 tensor, so its letterbox is excluded.
  end_to_end        : BGR numpy frame -> numpy boxes (colour conversion, resize, transfers, NMS, GPU->CPU copy)
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from detectors import FRCNNDetector, YOLODetector


def stats(ts):
    ts = np.asarray(ts)
    return dict(mean_ms=float(ts.mean()), median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)),
                std_ms=float(ts.std()), fps=float(1000.0 / ts.mean()))


@torch.inference_mode()
def bench(fn, warmup, iters):
    cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if cuda:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()          # without this you time only the kernel *launch*, not the work
            ts.append(s.elapsed_time(e))
        else:
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
    return stats(ts)


def bench_e2e(fn, warmup, iters):
    """perf_counter timing; detect() copies results to CPU, which forces GPU synchronisation."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return stats(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="must equal the --name used in evaluate.py (links speed to mAP)")
    ap.add_argument("--model", choices=["frcnn", "yolo"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--zero_shot", action="store_true")
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--proposals", type=int, default=1000)
    ap.add_argument("--half", action="store_true", help="FP16")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    if not cuda:
        print("WARNING: no GPU found. CPU numbers are NOT comparable to GPU numbers.")
    kw = dict(conf=a.conf, nms_iou=a.nms_iou, half=a.half, zero_shot=a.zero_shot)
    det = FRCNNDetector(a.ckpt, proposals=a.proposals, **kw) if a.model == "frcnn" else \
        YOLODetector(a.ckpt, imgsz=a.imgsz, **kw)
    frame = cv2.imread(a.image)
    assert frame is not None, a.image
    dev = det.dev if a.model == "frcnn" else det.device

    if a.model == "frcnn":
        x = det.to_tensor(frame)
        model_fn = lambda: det.forward_tensor(x)
    else:
        h = int(math.ceil(a.imgsz * frame.shape[0] / frame.shape[1] / 32) * 32)
        rgb = cv2.cvtColor(cv2.resize(frame, (a.imgsz, h)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div_(255).to(dev)
        model_fn = lambda: det.predict_raw(x)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    res = dict(name=a.name, model=a.model, precision="fp16" if det.half else "fp32",
               gpu=torch.cuda.get_device_name(0) if cuda else "cpu", torch=torch.__version__,
               python=platform.python_version(), conf=a.conf, proposals=a.proposals if a.model == "frcnn" else None,
               imgsz=a.imgsz, warmup=a.warmup, iters=a.iters, params_million=det.param_count() / 1e6)
    res["model_postprocess"] = bench(model_fn, a.warmup, a.iters)
    res["end_to_end"] = bench_e2e(lambda: det.detect(frame), a.warmup, a.iters)
    res["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20 if cuda else None

    res["gflops"] = None                                   # approximate; state the tool in the report
    try:
        if a.model == "yolo":
            from ultralytics.utils.torch_utils import get_flops
            res["gflops"], res["gflops_tool"] = float(get_flops(det.m.model, a.imgsz)), f"ultralytics get_flops @{a.imgsz}"
        else:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                det.forward_tensor(x)
            res["gflops"], res["gflops_tool"] = fc.get_total_flops() / 1e9, "torch FlopCounterMode (conv/matmul only)"
    except Exception as e:                                 # FLOP counters are fragile for detection models
        res["gflops_error"] = repr(e)

    path = a.out / f"speed_{a.name}_{res['precision']}.json"
    path.write_text(json.dumps(res, indent=2))
    m, e = res["model_postprocess"], res["end_to_end"]
    print(f"[{a.name} {res['precision']} on {res['gpu']}]")
    print(f"  model+postprocess: {m['mean_ms']:.1f} ms (p95 {m['p95_ms']:.1f})  -> {m['fps']:.1f} FPS")
    print(f"  end-to-end       : {e['mean_ms']:.1f} ms (p95 {e['p95_ms']:.1f})  -> {e['fps']:.1f} FPS")
    print(f"  params {res['params_million']:.1f} M | GFLOPs {res['gflops']} | peak mem {res['peak_gpu_mem_mb']} MB -> {path}")


if __name__ == "__main__":
    main()
