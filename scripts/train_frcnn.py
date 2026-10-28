#!/usr/bin/env python
"""Fine-tune Faster R-CNN ResNet50-FPN v2 (COCO-pretrained) on bdd5. Resumable (Colab disconnects!).

  python train_frcnn.py --data_root /content/data/bdd5 --out /content/drive/MyDrive/traffic/runs/frcnn
  python train_frcnn.py ... --resume                       # continue from out/last.pt
  python train_frcnn.py ... --limit 16 --epochs 40 --val_limit 16   # OVERFIT TEST: loss must go near zero

Checkpoints: last.pt (model+optimizer+scheduler+scaler+epoch, every epoch), best.pt (model state_dict,
best val AP50). Model selection uses VAL only; the test split is never touched here.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import set_seed
from det_dataset import CocoDetDataset, collate
from detectors import build_frcnn
from evaluate import load_gt, run_eval, summarize, to_coco_dets


@torch.inference_mode()
def validate(model, loader, gt, amp, dev):
    model.eval()
    dets, ids = [], []
    for imgs, tgts in loader:
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            outs = model([i.to(dev) for i in imgs])
        for o, t in zip(outs, tgts):
            ids.append(int(t["image_id"]))
            dets += to_coco_dets(int(t["image_id"]), o["boxes"].float().cpu().numpy(),
                                 o["scores"].float().cpu().numpy(), (o["labels"] - 1).cpu().numpy())
    return summarize(run_eval(gt, dets, ids), with_details=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01, help="for batch 8; scaled linearly with batch size")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup_iters", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="train on first N images (overfit test)")
    ap.add_argument("--val_limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (dev == "cuda") and not a.no_amp
    set_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))

    train_ds = CocoDetDataset(a.data_root, "train", train=True, limit=a.limit)
    val_ds = CocoDetDataset(a.data_root, "val", train=False, limit=a.val_limit)
    g = torch.Generator().manual_seed(a.seed)
    train_dl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate,
                          drop_last=len(train_ds) >= a.batch, pin_memory=True, generator=g,
                          persistent_workers=a.workers > 0)
    val_dl = DataLoader(val_ds, 4, shuffle=False, num_workers=a.workers, collate_fn=collate)
    gt = load_gt(a.data_root / "annotations" / "val.json")

    model = build_frcnn(pretrained=True).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = a.lr * a.batch / 8
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=a.wd)
    total = a.epochs * len(train_dl)
    warm = min(a.warmup_iters, max(1, total // 5))

    def lr_lambda(it):                                  # linear warm-up then cosine decay (per iteration)
        if it < warm:
            return 0.001 + (1 - 0.001) * it / warm
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start, best = 0, -1.0
    if a.resume and (a.out / "last.pt").exists():
        ck = torch.load(a.out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start} (best val AP50 so far {best:.4f})")

    log_path = a.out / "log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,val_ap50,val_ap,val_ap75,lr,epoch_sec\n")
    print(f"device={dev} amp={amp} train={len(train_ds)} val={len(val_ds)} iters/epoch={len(train_dl)} lr={lr}")

    for ep in range(start, a.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for it, (imgs, tgts) in enumerate(train_dl):
            imgs = [i.to(dev, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items() if k != "image_id"} for t in tgts]
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                losses = model(imgs, tgts)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                print(f"  non-finite loss at epoch {ep} iter {it}: {({k: float(v) for k, v in losses.items()})}; skipping")
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run, n = run + float(loss), n + 1
            if it % 50 == 0:
                print(f"  ep {ep} it {it}/{len(train_dl)} loss {float(loss):.3f} lr {opt.param_groups[0]['lr']:.5f}")
        m = validate(model, val_dl, gt, amp, dev)
        dt = time.time() - t0
        row = [ep, run / max(1, n), m["ap50"], m["ap"], m["ap75"], opt.param_groups[0]["lr"], dt]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"epoch {ep}: loss {row[1]:.4f} | val AP50 {m['ap50']:.4f} AP {m['ap']:.4f} | {dt / 60:.1f} min")
        if m["ap50"] > best:
            best = m["ap50"]
            torch.save(model.state_dict(), a.out / "best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep, best=best), a.out / "last.pt")
    print(f"done. best val AP50 = {best:.4f}  ->  {a.out / 'best.pt'}")


if __name__ == "__main__":
    main()
