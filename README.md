# Single-Stage vs Two-Stage Object Detection for Traffic Monitoring

Faster R-CNN (two-stage) vs YOLOv8 (single-stage), fine-tuned on a BDD100K subset (5 classes), compared on
accuracy (AP50, AP75, AP@[.5:.95], per class, per condition), speed (FPS) and real-time behaviour.
Read `CONTRACT.md` first.

## Order of work

| # | Step | Command | Gate (must pass) |
|---|------|---------|------|
| 0 | Smoke-test pipeline, no download | `python scripts/make_fake_bdd.py --out /tmp/fakebdd` then `prepare_bdd.py` on it | runs end to end |
| 1 | Build dataset | `python scripts/prepare_bdd.py --train_json ... --val_json ... --train_img_dir ... --val_img_dir ... --out data/bdd5 --zip` | prints stats |
| 2 | Verify data | `python scripts/verify_data.py --data_root data/bdd5` | `ALL CHECKS PASSED`, open `results/data_check/*.jpg` |
| 3 | EDA | `python scripts/eda.py --data_root data/bdd5` | figures in `results/eda` |
| 4 | Evaluator sanity | `python scripts/evaluate.py --data_root data/bdd5 --name sanity --gt_as_pred` | **AP = 1.000** |
| 5 | Overfit tests | `train_frcnn.py --limit 16 --val_limit 16 --epochs 40` and `train_yolo.py --fraction 0.002 --epochs 30` | loss near 0, val AP50 climbs high |
| 6 | Zero-shot baselines | `evaluate.py --split val --name frcnn_zeroshot --model frcnn --zero_shot` (and yolo) | numbers recorded |
| 7 | Full training | `train_frcnn.py`, `train_yolo.py` (and `--mosaic 0.0` variant) | both beat zero-shot on val |
| 8 | Final evaluation | `evaluate.py --split test ...` for every model | metrics JSON per model |
| 9 | Speed benchmark | `benchmark_speed.py ...` all models, one session, fp32 and `--half` | JSON per model |
| 10 | Figures | `python scripts/plots.py` | `results/*.png, *.csv` |
| 11 | Demo | `python scripts/video_demo.py --model yolo --ckpt ... --source clip.mp4 --out results/demo.mp4` | video plays |

Getting BDD100K: register at the official BDD100K site (bdd-data.berkeley.edu) and download the "100K Images" and the
detection labels ("Detection 2020 Labels", e.g. `det_train.json` / `det_val.json`). Mirrors exist on Kaggle, but folder names
differ, so always pass the four paths explicitly. Older label files (`bdd100k_labels_images_*.json`) also work.

## Layout
`scripts/common.py` constants | `prepare_bdd.py`, `verify_data.py`, `eda.py` data | `det_dataset.py`, `train_frcnn.py`, `train_yolo.py` training |
`detectors.py` common `Detector` interface | `evaluate.py` unified metrics | `benchmark_speed.py` FPS | `video_demo.py` | `plots.py`.
`notebooks/00_colab_bootstrap.ipynb` is the per-session Colab setup.

## Rules that prevent silent bugs
- Never normalise/resize images yourself for torchvision models; never feed RGB to Ultralytics via numpy conversion.
- Always select checkpoints on val; never look at test until final evaluation.
- Never quote framework-internal mAP; use `evaluate.py`.
- Record GPU name (the benchmark JSON does) and compare speeds only from one session.
