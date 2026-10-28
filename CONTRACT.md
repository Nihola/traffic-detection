# Project Contract (change only by team agreement + a commit to this file)

| Item | Decision |
|---|---|
| Classes (fixed order) | `pedestrian, car, bus, truck, motorcycle` |
| Class ids | COCO json / torchvision: **1-5** (0 = background). YOLO txt: **0-4**. `yolo_id = coco_id - 1`. `Detector.detect()` returns **0-4** |
| Box formats | Internally **xyxy, original pixels**. COCO json = `[x, y, w, h]` top-left. YOLO txt = normalised `cx cy w h`. Convert only at file/model boundaries |
| Colour order | OpenCV frames = **BGR**. torchvision input = RGB (converted inside `FRCNNDetector`). Ultralytics = BGR numpy |
| Splits | train/val = disjoint stratified samples of official BDD train; **test = sample of official BDD val** (BDD test labels are private). Created once by `prepare_bdd.py` (seed 0); `splits/*.txt` committed |
| Model selection | on **val** only. **Test** is used only for the final report numbers |
| Resolution | 1280x720 source. Primary comparison: long side **640** (FRCNN `min_size=360,max_size=640`; YOLO `imgsz=640`) |
| Evaluation | `evaluate.py` for ALL models: conf 0.001, NMS IoU 0.5, max 100 dets/img, pycocotools |
| Speed | `benchmark_speed.py`: same GPU + session for all models, batch 1, 50 warm-up + 300 runs, conf 0.25 |
| Seed | 0 |
| Models | Faster R-CNN R50-FPN **v2** (COCO-pretrained), YOLOv8s (COCO-pretrained); optional: YOLOv8m, SSD |
| Known confounder | Ultralytics uses mosaic augmentation by default; torchvision pipeline does not. Report both `mosaic=1.0` and `mosaic=0.0` |
