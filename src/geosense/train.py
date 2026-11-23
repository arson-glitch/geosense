"""Shared training loop for our own models + export of the logits for evaluation.

Examples
  # quick check that everything runs (CPU is fine, ~1,000 training images, no export)
  python -m geosense.train --root ~/Downloads/compressed_dataset --model cnn_scratch --subset 1000 --epochs 2

  # CNN from scratch, full data (GPU)
  python -m geosense.train --root /path/compressed_dataset --model cnn_scratch

  # ResNet-50: 3 epochs head only, then fine-tune layer4
  python -m geosense.train --root /path/compressed_dataset --model resnet50 --stage1-epochs 3 --unfreeze-blocks 1

  # Colab session cut off? Same command + --resume continues from the last epoch.

  # At the very END of the project only: export test logits from the best checkpoint
  python -m geosense.train --root /path/compressed_dataset --model resnet50 --export-only --export val,test

What it does
  1. Data      : GeoSenseDataset (manifest.csv split, seed 42 - never re-split), transforms.py.
  2. Imbalance : class-weighted cross-entropy (default: square-root weights) or a weighted sampler.
  3. Training  : Adam, LR halved when val macro-F1 stops improving, early stopping on val macro-F1.
                 Transfer models: stage 1 head only, stage 2 last block(s) unfrozen with a 10x smaller LR.
  4. Saving    : checkpoints/<model>/last.pt every epoch (resume), best.pt = best val macro-F1,
                 results/<model>_history.csv (curves for the report).
  5. Export    : predictions/<model>_<split>_logits.npy, float32, shape (N, C),
                 rows in manifest order for that split, columns in classes.json order, raw logits.
                 Only val by default. Test only when asked explicitly (--export test).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .data import PROJECT_ROOT, GeoSenseDataset, load_classes
from .models import MODEL_NAMES, build_model, freeze_backbone, unfreeze_last_blocks
from .predictions import MODEL_NAMES as SHARED_NAMES, save_logits
from .transforms import DEFAULT_RESIZE, RESIZE_MODES, build_transforms, open_image

PRED_DIR = PROJECT_ROOT / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results"
CKPT_DIR = PROJECT_ROOT / "checkpoints"          # *.pt is git-ignored: never commit weights


class FastGeoSenseDataset(GeoSenseDataset):
    """Seif's GeoSenseDataset (same manifest, same order) with faster half-resolution decoding."""

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = open_image(self.root / row["path"])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


# ------------------------------------------------------------------ setup helpers
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():        # Apple silicon laptops
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(mode: str) -> torch.Tensor | None:
    """Weights for the loss. classes.json holds plain inverse frequency (Ghana ~110x USA),
    which can make training unstable; the square root (~10x) is a gentler default."""
    if mode in ("none", "sampler"):
        return None
    _, w = load_classes()
    w = torch.tensor(w, dtype=torch.float32)
    if mode == "sqrt_weights":
        w = w.sqrt()
    return w / w.mean()


def make_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """Each image drawn with probability 1/sqrt(count of its class): small countries seen more often."""
    counts = np.bincount(labels, minlength=num_classes)
    w = 1.0 / np.sqrt(counts[labels])
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)


def small_subset(ds, n: int, seed: int) -> Subset:
    """n images, stratified by country, for quick debugging runs."""
    df = ds.df
    frac = min(1.0, n / len(df))
    idx = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed)).index
    return Subset(ds, sorted(idx.tolist()))


def subset_labels(ds) -> np.ndarray:
    if isinstance(ds, Subset):
        return ds.dataset.df["label"].to_numpy()[ds.indices]
    return ds.df["label"].to_numpy()


# ------------------------------------------------------------------ one epoch / evaluation
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(x), y)
        if scaler is not None:                       # mixed precision on GPU: ~2x faster
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Raw logits (N, C) and true labels, in loader order (loader must NOT shuffle)."""
    model.eval()
    out, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        out.append(logits.float().cpu())
        ys.append(y)
    return torch.cat(out).numpy().astype(np.float32), torch.cat(ys).numpy()


def metrics(logits: np.ndarray, y: np.ndarray, criterion=None) -> dict:
    """Quick metrics to follow training. The official ones come from Maya's evaluation module."""
    num_classes = logits.shape[1]
    top5 = np.argsort(-logits, axis=1)[:, :5]
    m = {
        "acc": float((top5[:, 0] == y).mean()),
        "top5": float((top5 == y[:, None]).any(axis=1).mean()),
        "macro_f1": float(f1_score(y, top5[:, 0], average="macro", labels=range(num_classes), zero_division=0)),
    }
    if criterion is not None:
        m["loss"] = float(criterion(torch.from_numpy(logits), torch.from_numpy(y)).item())
    return m


# ------------------------------------------------------------------ export (the shared contract)
def export_logits(model, root, split, args, device, classes) -> Path:
    ds = FastGeoSenseDataset(root, split, transform=build_transforms(False, args.size, args.resize))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    logits, y = predict_logits(model, loader, device)
    assert logits.shape == (len(ds.df), len(classes)), logits.shape
    assert (y == ds.df["label"].to_numpy()).all(), "row order differs from manifest"
    if args.model in SHARED_NAMES:                     # Thea's checker: shape, dtype, no NaN
        path = save_logits(args.model, split, logits, PRED_DIR, manifest=ds.df, n_classes=len(classes))
    else:                                               # efficientnet_b0 is not in the shared list
        PRED_DIR.mkdir(exist_ok=True)
        path = PRED_DIR / f"{args.model}_{split}_logits.npy"
        np.save(path, logits)
    print(f"  exported {path}  shape={logits.shape}  {metrics(logits, y)}")
    return path


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Kaggle compressed_dataset folder")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--resize", default=DEFAULT_RESIZE, choices=RESIZE_MODES)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30, help="maximum; early stopping usually ends sooner")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="CNN / stage-1 learning rate")
    ap.add_argument("--ft-lr", type=float, default=1e-4, help="stage-2 (fine-tuning) learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4, help="epochs without val macro-F1 gain before stopping")
    ap.add_argument("--imbalance", default="sqrt_weights", choices=["sqrt_weights", "weights", "sampler", "none"])
    ap.add_argument("--stage1-epochs", type=int, default=3, help="transfer models: epochs with frozen backbone")
    ap.add_argument("--unfreeze-blocks", type=int, default=1, help="transfer models: blocks unfrozen in stage 2")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="use only N train images (debug); disables export")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<model>/last.pt")
    ap.add_argument("--export", default="val", help="splits to export after training, e.g. 'val' or 'val,test'")
    ap.add_argument("--export-only", action="store_true", help="skip training, export from best.pt")
    return ap.parse_args(argv)


def build_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    return opt, sched


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    classes, _ = load_classes()
    C = len(classes)
    transfer = args.model != "cnn_scratch"
    ckpt_dir = CKPT_DIR / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model}  device={device}  classes={C}  resize={args.resize}  size={args.size}")

    model = build_model(args.model, C, pretrained=not args.no_pretrained).to(device)

    if not args.export_only:
        # ---- data
        train_ds = FastGeoSenseDataset(args.root, "train", transform=build_transforms(True, args.size, args.resize))
        val_ds = FastGeoSenseDataset(args.root, "val", transform=build_transforms(False, args.size, args.resize))
        if args.subset:
            train_ds = small_subset(train_ds, args.subset, args.seed)
            val_ds = small_subset(val_ds, max(200, args.subset // 4), args.seed)
        labels = subset_labels(train_ds)
        sampler = make_sampler(labels, C) if args.imbalance == "sampler" else None
        kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.workers > 0)
        train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
        val_loader = DataLoader(val_ds, shuffle=False, **kw)
        print(f"train={len(train_ds)}  val={len(val_ds)}  imbalance={args.imbalance}")

        w = class_weights(args.imbalance)
        criterion = nn.CrossEntropyLoss(weight=None if w is None else w.to(device))
        val_criterion = nn.CrossEntropyLoss()        # unweighted, comparable across settings
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        # ---- stage 1 (transfer) or the only stage (CNN)
        if transfer:
            freeze_backbone(model)
        optimizer, scheduler = build_optimizer(model, args.lr, args.weight_decay)
        start, best_f1, bad, history = 0, -1.0, 0, []

        if args.resume and (ckpt_dir / "last.pt").exists():
            ck = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if transfer and ck["epoch"] > args.stage1_epochs:   # already inside stage 2
                unfreeze_last_blocks(model, args.unfreeze_blocks)
                optimizer, scheduler = build_optimizer(model, args.ft_lr, args.weight_decay)
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start, best_f1, bad, history = ck["epoch"], ck["best_f1"], ck["bad"], ck["history"]
            print(f"resumed after epoch {start} (best val macro-F1 {best_f1:.3f})")

        for epoch in range(start, args.epochs):
            if transfer and epoch == args.stage1_epochs:        # ---- stage 2: fine-tune
                unfreeze_last_blocks(model, args.unfreeze_blocks)
                optimizer, scheduler = build_optimizer(model, args.ft_lr, args.weight_decay)
                bad = 0
                print(f"stage 2: unfroze last {args.unfreeze_blocks} block(s), lr={args.ft_lr}")

            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            logits, y = predict_logits(model, val_loader, device)
            m = metrics(logits, y, val_criterion)
            scheduler.step(m["macro_f1"])
            row = {"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in m.items()},
                   "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0)}
            history.append(row)
            print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)

            if m["macro_f1"] > best_f1:
                best_f1, bad = m["macro_f1"], 0
                torch.save({"model": model.state_dict(), "config": vars(args), "classes": classes,
                            "epoch": epoch + 1, "val": m}, ckpt_dir / "best.pt")
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_f1": best_f1,
                        "bad": bad, "history": history, "config": vars(args)}, ckpt_dir / "last.pt")
            RESULTS_DIR.mkdir(exist_ok=True)
            suffix = f"_subset{args.subset}" if args.subset else ""
            pd.DataFrame(history).to_csv(RESULTS_DIR / f"{args.model}{suffix}_history.csv", index=False)

            in_stage1 = transfer and epoch + 1 <= args.stage1_epochs
            if bad >= args.patience and not in_stage1:
                print(f"early stopping: no val macro-F1 gain for {args.patience} epochs")
                break
        print(f"best val macro-F1 = {best_f1:.4f}")

    # ---- export from the best checkpoint
    if args.subset:
        print("subset run: export skipped (files must cover the full split)")
        return
    best = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    if best["classes"] != classes:
        raise RuntimeError("checkpoint classes differ from data/classes.json - retrain")
    for k in ("resize", "size"):                     # always predict with the training pre-processing
        setattr(args, k, best["config"][k])
    model.load_state_dict(best["model"])
    for split in [s.strip() for s in args.export.split(",") if s.strip()]:
        if split == "test":
            print("  WARNING: exporting TEST logits - only do this once, at the end of the project")
        export_logits(model, args.root, split, args, device, classes)


if __name__ == "__main__":
    main()
