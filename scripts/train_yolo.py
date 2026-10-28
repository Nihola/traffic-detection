#!/usr/bin/env python
"""Fine-tune YOLOv8 on bdd5 with Ultralytics.

  python train_yolo.py --data_root /content/data/bdd5 --project /content/drive/MyDrive/traffic/runs --name yolov8s_640
  python train_yolo.py ... --mosaic 0.0 --name yolov8s_640_nomosaic     # closer to the torchvision pipeline (ablation)
  python train_yolo.py ... --resume                                      # after a Colab disconnect
Best weights: <project>/<name>/weights/best.pt (chosen on VAL by Ultralytics' fitness).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from common import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--name", default="yolov8s_640")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mosaic", type=float, default=1.0, help="Ultralytics default 1.0; 0.0 disables mosaic")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of train images (smoke tests)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    last = a.project / a.name / "weights" / "last.pt"
    if a.resume:
        YOLO(str(last)).train(resume=True)
        return

    # write the dataset yaml with the CURRENT location of the data (it differs between Drive and /content)
    root = a.data_root.resolve()
    yaml = root / "bdd5.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")

    YOLO(a.weights).train(data=str(yaml), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, seed=a.seed,
                          project=str(a.project), name=a.name, exist_ok=True, patience=a.patience,
                          workers=a.workers, mosaic=a.mosaic, fraction=a.fraction, plots=True)
    print("best weights:", a.project / a.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
