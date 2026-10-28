"""COCO-json detection dataset for torchvision models (used by train_frcnn.py)."""
import json
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def build_train_aug():
    """Flip + brightness/contrast/gamma jitter (dark values simulate night, bright ones glare)."""
    return A.Compose(
        [A.HorizontalFlip(p=0.5),
         A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.3), contrast_limit=0.2, p=0.6),
         A.RandomGamma(gamma_limit=(70, 140), p=0.3)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3))


class CocoDetDataset(Dataset):
    """Returns (image float32 CHW RGB in [0,1], target{boxes xyxy float32 [N,4], labels int64 1..5, image_id}).
    Do NOT normalise or resize here: the torchvision model's built-in transform does both and maps
    predictions back to original pixel coordinates."""

    def __init__(self, root, split, train=False, limit=None):
        self.root, self.split = Path(root), split
        d = json.loads((self.root / "annotations" / f"{split}.json").read_text())
        self.images = sorted(d["images"], key=lambda x: x["id"])[:limit]
        self.anns = defaultdict(list)
        for a in d["annotations"]:
            if not a.get("iscrowd", 0):
                self.anns[a["image_id"]].append(a)
        self.tf = build_train_aug() if train else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = np.array(Image.open(self.root / "images" / self.split / info["file_name"]).convert("RGB"))
        H, W = img.shape[:2]
        boxes, labels = [], []
        for a in self.anns[info["id"]]:
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(a["category_id"])
        if self.tf is not None:
            out = self.tf(image=img, bboxes=boxes, labels=labels)
            img, boxes, labels = out["image"], [list(b) for b in out["bboxes"]], list(out["labels"])
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255)
        target = dict(boxes=torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),   # [0,4] if no objects
                      labels=torch.as_tensor(labels, dtype=torch.int64),
                      image_id=torch.tensor(info["id"]))
        return t, target


def collate(batch):
    return tuple(zip(*batch))
