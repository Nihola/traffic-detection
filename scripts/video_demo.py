#!/usr/bin/env python
"""Real-time detection demo: video file / webcam / stream, boxes + confidence + FPS overlay.

  python video_demo.py --model yolo  --ckpt best.pt --source dashcam.mp4 --out results/demo_yolo.mp4
  python video_demo.py --model frcnn --ckpt frcnn_best.pt --source 0 --show        # local webcam (not on Colab)

Colab has no display/webcam: use --out only, then re-encode so browsers can play it:
  ffmpeg -y -i results/demo_yolo.mp4 -vcodec libx264 -pix_fmt yuv420p results/demo_yolo_h264.mp4
"""
import argparse
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from common import CLASSES, COLORS
from detectors import load_detector


class LatestFrameGrabber:
    """Reads in a background thread and keeps only the newest frame, so slow inference never builds up lag
    on live sources (webcam / RTSP). Not used for files: there every frame is processed."""

    def __init__(self, src):
        self.cap, self.q, self.stop = cv2.VideoCapture(src), queue.Queue(maxsize=1), False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                self.q.put(None)
                return
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(f)

    def read(self):
        f = self.q.get()
        return (f is not None), f

    def release(self):
        self.stop = True
        self.cap.release()


def draw(frame, boxes, scores, cls, header):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, cls):
        col = COLORS[int(c) % len(COLORS)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, col, 2)
        label = f"{CLASSES[int(c)]} {s:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - bl - 4), (p1[0] + tw + 4, p1[1]), col, -1)
        cv2.putText(frame, label, (p1[0] + 2, p1[1] - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["frcnn", "yolo"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="video path, stream URL, or webcam index (0)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--show", action="store_true", help="cv2.imshow window (local machine only)")
    ap.add_argument("--max_frames", type=int, default=None)
    a = ap.parse_args()

    kw = dict(conf=a.conf, half=a.half)
    if a.model == "yolo":
        kw["imgsz"] = a.imgsz
    det = load_detector(a.model, a.ckpt, **kw)

    src = int(a.source) if a.source.isdigit() else a.source
    live = isinstance(src, int) or str(src).startswith(("rtsp://", "http://", "https://"))
    cap = LatestFrameGrabber(src) if live else cv2.VideoCapture(src)
    src_fps = (cap.cap if live else cap).get(cv2.CAP_PROP_FPS) or 30.0

    writer, times, n = None, deque(maxlen=30), 0
    ok, frame = cap.read()
    if ok:
        det.detect(frame)                                   # warm-up (cuDNN autotune, lazy init) before timing
    while ok:
        t0 = time.perf_counter()
        boxes, scores, cls = det.detect(frame)
        t_inf = time.perf_counter() - t0
        times.append(time.perf_counter() - t0)
        fps = len(times) / max(sum(times), 1e-9)            # smoothed over last 30 frames
        out = draw(frame, boxes, scores, cls, f"{det.name} | {fps:.1f} FPS | {t_inf * 1000:.0f} ms | {len(boxes)} objects")
        if a.out is not None:
            if writer is None:
                a.out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                         (out.shape[1], out.shape[0]))
            writer.write(out)
        if a.show:
            cv2.imshow("detections (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
        ok, frame = cap.read()
    cap.release()
    if writer:
        writer.release()
        print("saved", a.out)
    if times:
        print(f"processed {n} frames, mean smoothed FPS (last window) = {len(times) / sum(times):.1f}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
