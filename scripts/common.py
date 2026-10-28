"""Shared constants and helpers: the single source of truth for the project contract.

Class-id convention (see CONTRACT.md):
  * COCO JSON / torchvision : 1..5  (0 = background)
  * YOLO txt / Ultralytics  : 0..4  -> yolo_id = coco_id - 1
  * Detector.detect() output: 0..4  (index into CLASSES)
"""
import random

import numpy as np

CLASSES = ["pedestrian", "car", "bus", "truck", "motorcycle"]
NUM_FG = len(CLASSES)               # 5 foreground classes
NUM_FRCNN_CLASSES = NUM_FG + 1      # + background for torchvision

# BDD100K category name -> COCO-style id (1-based). Older releases call pedestrians "person".
BDD_ALIAS = {"pedestrian": 1, "person": 1, "car": 2, "bus": 3, "truck": 4, "motorcycle": 5}

IMG_W, IMG_H = 1280, 720            # all BDD100K detection images
SEED = 0

# BGR colours for drawing (one per class)
COLORS = [(60, 76, 231), (113, 204, 46), (219, 152, 52), (15, 196, 241), (182, 89, 155)]


def set_seed(seed: int = SEED, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
