"""Common Detector interface used by evaluation, benchmarking and the video demo.

    detect(frame_bgr: HxWx3 uint8 BGR numpy) -> (boxes[N,4] xyxy in frame pixels, scores[N], class_ids[N] in 0..4)

Colour order: OpenCV frames are BGR. torchvision needs RGB (converted here); Ultralytics expects BGR numpy (not converted).
"""
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from common import CLASSES, NUM_FRCNN_CLASSES

# COCO category -> our class index (0..4), for zero-shot baselines of COCO-pretrained models
TV_COCO_TO_OURS = {1: 0, 3: 1, 6: 2, 8: 3, 4: 4}      # torchvision 91-id scheme: person, car, bus, truck, motorcycle
ULTRA_COCO_TO_OURS = {0: 0, 2: 1, 5: 2, 7: 3, 3: 4}   # ultralytics 80-id scheme


def build_frcnn(pretrained=True, score_thresh=0.001, nms_iou=0.5, max_det=100, proposals=1000,
                min_size=360, max_size=640, fine_tune_head=True):
    """Faster R-CNN ResNet50-FPN v2. 1280x720 input -> resized to 640x360 inside the model (transform is built in).

    pretrained=True,  fine_tune_head=True : COCO weights, new 6-way head (training start point)
    pretrained=True,  fine_tune_head=False: original 91-class COCO model (zero-shot baseline)
    pretrained=False                       : random init with 6-way head (for loading our checkpoint)
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    kw = dict(min_size=min_size, max_size=max_size, box_score_thresh=score_thresh, box_nms_thresh=nms_iou,
              box_detections_per_img=max_det, rpn_post_nms_top_n_test=proposals)
    if pretrained:
        m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **kw)
        if fine_tune_head:
            m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,
                                                          NUM_FRCNN_CLASSES)
    else:
        m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=NUM_FRCNN_CLASSES, **kw)
    return m


class Detector(ABC):
    name: str

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray):
        ...

    def param_count(self) -> int:
        return 0


def _remap(boxes, scores, cls, mapping):
    keep = np.array([c in mapping for c in cls], dtype=bool)
    return boxes[keep], scores[keep], np.array([mapping[c] for c in cls[keep]], dtype=int)


class FRCNNDetector(Detector):
    def __init__(self, ckpt=None, conf=0.4, nms_iou=0.5, max_det=100, proposals=1000, half=False,
                 zero_shot=False, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and self.dev == "cuda"
        self.conf, self.zero_shot = conf, zero_shot
        self.name = "FasterRCNN-R50-FPNv2" + ("-zeroshot" if zero_shot else "")
        if zero_shot:
            self.m = build_frcnn(True, conf, nms_iou, max_det, proposals, fine_tune_head=False)
        else:
            self.m = build_frcnn(False, conf, nms_iou, max_det, proposals)
            sd = torch.load(ckpt, map_location="cpu")
            self.m.load_state_dict(sd["model"] if "model" in sd else sd)
        self.m.to(self.dev).eval()

    def to_tensor(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).float().div_(255)

    @torch.inference_mode()
    def forward_tensor(self, x):
        with torch.autocast(self.dev, dtype=torch.float16, enabled=self.half):
            return self.m([x])[0]

    def detect(self, frame_bgr):
        o = self.forward_tensor(self.to_tensor(frame_bgr))
        boxes, scores = o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy()
        cls = o["labels"].cpu().numpy()
        if self.zero_shot:
            return _remap(boxes, scores, cls, TV_COCO_TO_OURS)
        return boxes, scores, cls - 1                       # torchvision 1..5 -> 0..4

    def param_count(self):
        return sum(p.numel() for p in self.m.parameters())


class YOLODetector(Detector):
    def __init__(self, ckpt=None, conf=0.4, nms_iou=0.5, max_det=100, imgsz=640, half=False,
                 zero_shot=False, device=None):
        from ultralytics import YOLO
        cuda = torch.cuda.is_available()
        self.device = device or ("cuda:0" if cuda else "cpu")
        self.half = half and cuda
        self.conf, self.nms_iou, self.max_det, self.imgsz, self.zero_shot = conf, nms_iou, max_det, imgsz, zero_shot
        self.m = YOLO("yolov8s.pt" if zero_shot else ckpt)
        self.name = "YOLOv8s" + ("-zeroshot" if zero_shot else "")

    def predict_raw(self, src):
        """src: BGR numpy frame OR a BCHW float tensor in [0,1] (RGB, sides multiple of 32)."""
        return self.m.predict(src, imgsz=self.imgsz, conf=self.conf, iou=self.nms_iou, max_det=self.max_det,
                              half=self.half, device=self.device, verbose=False)[0]

    def detect(self, frame_bgr):
        b = self.predict_raw(frame_bgr).boxes
        boxes, scores = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()   # xyxy, NOT xywh (which is centre-based)
        cls = b.cls.cpu().numpy().astype(int)                        # already 0..4 for our fine-tuned model
        if self.zero_shot:
            return _remap(boxes, scores, cls, ULTRA_COCO_TO_OURS)
        return boxes, scores, cls

    def param_count(self):
        return sum(p.numel() for p in self.m.model.parameters())


def load_detector(kind, ckpt, **kw):
    return (FRCNNDetector if kind == "frcnn" else YOLODetector)(ckpt, **kw)
