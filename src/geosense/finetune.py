"""StreetCLIP fine-tuning (Faris + Louis): model 7 of the results table, ``streetclip_finetuned``.

Examples
  # smoke run on a laptop (~200 training images, CPU/MPS, no export)
  python -m geosense.finetune --root ~/Downloads/compressed_dataset --subset 200 --epochs 1 --workers 0

  # full run on the cluster GPU (default method: last 2 vision blocks)
  python -m geosense.finetune --root /path/compressed_dataset --ckpt-dir /scratch/me/checkpoints
  python -m geosense.finetune --root /path/compressed_dataset --method lora --lr 1e-4 --run-name ft_lora

  # job killed by the time limit? Same command + --resume continues after the last finished epoch.

  # At the very END of the project only: write the team files from the best checkpoint
  python -m geosense.finetune --root /path/compressed_dataset --run-name ft_lora --export-only \\
      --export val,test --final

What fine-tuning does here
  Thea's zero-shot model compares an image embedding with the text embedding of "A Street View photo in
  {country}." and her probe trains one layer on frozen embeddings. Here we also let part of the image
  encoder change, so the embeddings themselves adapt to our 56 countries and our UI-cropped screenshots.
  Model = StreetCLIP vision tower + projection + cosine head: logits = scale * cos(image_emb, W_c).
  --method head        only W is trained (an end-to-end linear probe, a sanity run)
  --method last_blocks W + the last N transformer blocks + final layer norm + projection
  --method lora        W + small low-rank adapters on the attention q/v projections (peft); the
                       430M original weights stay frozen and only ~1M new numbers are learned

Why each safeguard exists
  - Start from zero-shot: W = text embeddings of the country prompts, LoRA adapters start at zero, so
    the model at step 0 IS the zero-shot model. Epoch 0 of the history measures it with our pipeline.
  - Small learning rate (1e-5 backbone, 1e-4 head), warm-up then cosine decay, gradient clipping: big
    updates would destroy what StreetCLIP learned from 1.1M street images ("catastrophic forgetting").
  - Frozen parts stay in eval mode and build no gradient graph (cheaper, nothing drifts).
  - Checkpoints every epoch (only the trainable weights, so they are small): a cluster job can be cut
    off at any time and --resume continues; best.pt = best val macro-F1, early stopping after --patience.
  - Compare with zero-shot and the probe: best.pt starts as the epoch-0 (zero-shot) model, so if
    fine-tuning never beats it on val, the "fine-tuned" export is just zero-shot - and then it is not
    our final model. The test set is only exported once, at the end (--export test --final).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from . import streetclip
from .data import DATA_DIR, PROJECT_ROOT, GeoSenseDataset, load_classes
from .predictions import save_logits
from .preprocess import CLIP_MEAN, CLIP_STD, to_square

MODEL_NAME = "streetclip_finetuned"          # name in the team contract (predictions.MODEL_NAMES)
METHODS = ("head", "last_blocks", "lora")
RESULTS_DIR = PROJECT_ROOT / "results"
PRED_DIR = PROJECT_ROOT / "predictions"      # experiments go to PRED_DIR / "runs"
CKPT_DIR = PROJECT_ROOT / "checkpoints"      # never committed (git-ignored)
HISTORY_COLS = ["epoch", "train_loss", "val_loss", "val_acc", "val_top5", "val_macro_f1", "lr", "seconds"]


# ------------------------------------------------------------------ model
class StreetCLIPClassifier(nn.Module):
    """Vision tower + visual projection + cosine head. The text tower is not kept."""

    def __init__(self, clip, W, scale: float):
        super().__init__()
        self.vision_model, self.visual_projection = clip.vision_model, clip.visual_projection
        self.image_size = clip.config.vision_config.image_size
        W = torch.as_tensor(np.asarray(W), dtype=torch.float32)
        self.head = nn.Linear(W.shape[1], W.shape[0], bias=False)
        self.head.weight.data.copy_(W)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.backbone_frozen = False                 # set by set_trainable

    def embed(self, x):
        # towers by hand (as in streetclip.py): works in transformers 4.x and 5.x
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.backbone_frozen):
            return self.visual_projection(self.vision_model(pixel_values=x).pooler_output)

    def forward(self, x):
        z = F.normalize(self.embed(x).float(), dim=-1)
        return self.scale * z @ F.normalize(self.head.weight, dim=-1).T


def build_model(model_id=streetclip.STREETCLIP_ID, classes=None, templates="street_view", W=None,
                device="cpu", **_) -> StreetCLIPClassifier:
    """StreetCLIP as a classifier whose head starts at the zero-shot text weights.

    W: optional (C, D) init (e.g. Thea's probe weights); then no tokenizer / text tower is needed.
    Extra keyword arguments are ignored, so ``build_model(**vars(args))`` works.
    """
    from transformers import CLIPModel

    if W is None:
        clip, tok = streetclip.load_streetclip(model_id)
        classes = load_classes(Path(DATA_DIR) / "classes.json")[0] if classes is None else classes
        W = streetclip.text_weights(clip, tok, classes, streetclip.TEMPLATE_SETS[templates],
                                    str(get_device(device)))
    else:
        clip = CLIPModel.from_pretrained(model_id)
    model = StreetCLIPClassifier(clip.cpu(), W, scale=clip.logit_scale.exp().item())
    del clip                                         # drops the text tower
    return model


def set_trainable(model, method="last_blocks", unfreeze_blocks=2, lora_r=8, lora_alpha=16, lora_dropout=0.05):
    """Freeze everything, then unfreeze what ``method`` trains. Returns the model."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    if method == "last_blocks":
        for m in (*model.vision_model.encoder.layers[-unfreeze_blocks:], model.vision_model.post_layernorm,
                  model.visual_projection):
            m.requires_grad_(True)
    elif method == "lora":
        from peft import LoraConfig, inject_adapter_in_model   # lazy: the module imports without peft

        if not any("lora_" in n for n, _ in model.named_parameters()):
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules=["q_proj", "v_proj"])
            inject_adapter_in_model(cfg, model.vision_model)    # vision tower only
        for n, p in model.named_parameters():
            p.requires_grad_("lora_" in n or n.startswith("head."))
    model.backbone_frozen = method == "head"
    return model


def train_mode(model) -> None:
    """model.train(), but modules with only frozen weights stay in eval mode (no dropout drift)."""
    model.train()
    for m in model.modules():
        params = list(m.parameters())
        if params and not any(p.requires_grad for p in params):
            m.eval()


def trainable_names(model) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def trainable_state(model) -> dict:
    """Only the trained tensors: the frozen backbone is re-loaded from --model-id."""
    keep = set(trainable_names(model))
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def load_for_inference(ckpt_path, device="cpu") -> StreetCLIPClassifier:
    """Rebuild the model from a checkpoint (its stored config) and return it ready, in eval mode."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    W0 = np.zeros_like(ck["model"]["head.weight"].numpy())   # placeholder: the real head is in ck
    model = build_model(cfg["model_id"], ck["classes"], W=W0)
    set_trainable(model, cfg["method"], cfg["unfreeze_blocks"], cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"])
    expected = set(trainable_names(model))
    if set(ck["model"]) != expected:
        raise RuntimeError(f"checkpoint tensors differ from the rebuilt model: "
                           f"missing {sorted(expected - set(ck['model']))[:5]}, "
                           f"unexpected {sorted(set(ck['model']) - expected)[:5]}")
    model.load_state_dict(ck["model"], strict=False)
    model.config = cfg                               # training arguments (view_mode, run_name, ...)
    return model.to(device).eval()


# ------------------------------------------------------------------ data
def build_transforms(train: bool, size: int = 336, view_mode: str = "centre"):
    """Train: resize short side x1.1, random crop, colour jitter (no hue). NO horizontal flip.
    Eval: exactly Thea's view (to_square, bicubic), so the untrained model reproduces her zero-shot."""
    if train:
        first = [T.Resize(round(size * 1.1), interpolation=T.InterpolationMode.BICUBIC), T.RandomCrop(size),
                 T.ColorJitter(0.2, 0.2, 0.2)]
    else:
        first = [partial(to_square, size=size, mode=view_mode)]
    return T.Compose([*first, T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])


def small_subset(ds, n: int, seed: int) -> Subset:
    """n images, stratified by country, for quick debugging runs (same rule as Mohamed's train.py)."""
    df = ds.df.assign(label=ds.labels)
    idx = df.groupby("label").sample(frac=min(1.0, n / len(df)), random_state=seed).index
    return Subset(ds, sorted(idx.tolist()))


def class_weights(mode: str) -> torch.Tensor | None:
    """sqrt of inverse frequency by default (~10x between countries instead of ~110x)."""
    if mode == "none":
        return None
    w = torch.tensor(load_classes(Path(DATA_DIR) / "classes.json")[1], dtype=torch.float32)
    w = w.sqrt() if mode == "sqrt_weights" else w
    return w / w.mean()


# ------------------------------------------------------------------ helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    """Linear warm-up, then cosine decay to 0, stepped once per optimizer step."""
    warmup = max(1, round(warmup_frac * total_steps))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def atomic_save(obj, path: Path) -> None:
    """Write to a tmp file then rename: a job killed mid-write never leaves a broken checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Raw logits (N, C) and labels, in loader order (the loader must NOT shuffle)."""
    device = torch.device(device)
    model.eval()
    out, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out).numpy().astype(np.float32), torch.cat(ys).numpy()


def metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    """Quick metrics to follow training (the official ones come from Maya's evaluation module)."""
    top5 = np.argsort(-logits, axis=1)[:, :5]
    return {"loss": float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y).long())),
            "acc": float((top5[:, 0] == y).mean()), "top5": float((top5 == y[:, None]).any(axis=1).mean()),
            "macro_f1": float(f1_score(y, top5[:, 0], average="macro", labels=range(logits.shape[1]),
                                       zero_division=0))}


def export_logits(model, root, split, run_name, final, device, batch_size=64, workers=0,
                  view_mode="centre") -> Path:
    """Logits for a whole split, rows in splits.csv order. final=True -> the team file via save_logits."""
    ds = GeoSenseDataset(root, split, transform=build_transforms(False, model.image_size, view_mode),
                         data_dir=DATA_DIR)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=torch.device(device).type == "cuda")
    logits, y = predict_logits(model, loader, device)
    assert (y == ds.labels).all(), "row order differs from splits.csv"
    if final:
        path = save_logits(MODEL_NAME, split, logits, PRED_DIR, manifest=ds.df, n_classes=model.head.out_features)
    else:                                            # experiments never overwrite the team file
        path = Path(PRED_DIR) / "runs" / f"{run_name}_{split}_logits.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, logits)
    print(f"  exported {path}  shape={logits.shape}  {metrics(logits, y)}")
    return path


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.add_argument
    a("--root", required=True, help="Kaggle compressed_dataset folder")
    a("--run-name", default="streetclip_ft", help="names the checkpoint folder and history files")
    a("--method", default="last_blocks", choices=METHODS)
    a("--unfreeze-blocks", type=int, default=2, help="N for last_blocks")
    a("--lora-r", type=int, default=8)
    a("--lora-alpha", type=int, default=16)
    a("--lora-dropout", type=float, default=0.05)
    a("--lr", type=float, default=1e-5, help="backbone learning rate (LoRA runs: 1e-4)")
    a("--head-lr", type=float, default=1e-4)
    a("--weight-decay", type=float, default=0.05)
    a("--epochs", type=int, default=8, help="maximum; early stopping usually ends sooner")
    a("--patience", type=int, default=2, help="epochs without val macro-F1 gain before stopping")
    a("--warmup-frac", type=float, default=0.05)
    a("--batch-size", type=int, default=32)
    a("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch x accum)")
    a("--imbalance", default="sqrt_weights", choices=["sqrt_weights", "weights", "none"])
    a("--label-smoothing", type=float, default=0.0)
    a("--templates", default="street_view", choices=list(streetclip.TEMPLATE_SETS))
    a("--head-init", default="", help="optional .npy (C, D) head init instead of the text prompts")
    a("--view-mode", default="centre", choices=["centre", "squash"])
    a("--model-id", default=streetclip.STREETCLIP_ID, help="HF id or local folder")
    a("--subset", type=int, default=0, help="N train images, stratified (debug); disables export")
    a("--workers", type=int, default=4)
    a("--device", default="auto")
    a("--seed", type=int, default=42)
    a("--ckpt-dir", default=str(CKPT_DIR), help="parent of <run-name>/last.pt and best.pt")
    a("--resume", action="store_true", help="continue from <ckpt-dir>/<run-name>/last.pt")
    a("--export", default="val", help="splits to export from best.pt, e.g. 'val' or 'val,test'")
    a("--export-only", action="store_true", help="skip training, export from best.pt")
    a("--final", action="store_true", help="write predictions/streetclip_finetuned_<split>_logits.npy")
    a("--no-amp", action="store_true", help="disable mixed precision (CUDA only anyway)")
    return ap.parse_args(argv)


def train(args, device, ckpt_dir: Path) -> None:
    classes = load_classes(Path(DATA_DIR) / "classes.json")[0]
    W = np.load(args.head_init) if args.head_init else None
    model = build_model(args.model_id, classes, args.templates, W=W, device=str(device))
    set_trainable(model, args.method, args.unfreeze_blocks, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.to(device)
    size = model.image_size

    train_ds = GeoSenseDataset(args.root, "train", build_transforms(True, size), data_dir=DATA_DIR)
    val_ds = GeoSenseDataset(args.root, "val", build_transforms(False, size, args.view_mode), data_dir=DATA_DIR)
    if args.subset:
        train_ds = small_subset(train_ds, args.subset, args.seed)
        val_ds = small_subset(val_ds, max(200, args.subset // 4), args.seed)
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    head = [model.head.weight]
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [{"params": head, "lr": args.head_lr, "weight_decay": 0.0}]
    if backbone:
        groups.append({"params": backbone, "lr": args.lr, "weight_decay": args.weight_decay})
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    scheduler = make_scheduler(optimizer, args.epochs * steps_per_epoch, args.warmup_frac)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and not args.no_amp else None
    w = class_weights(args.imbalance)
    criterion = nn.CrossEntropyLoss(weight=None if w is None else w.to(device), label_smoothing=args.label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]

    suffix = f"_subset{args.subset}" if args.subset else ""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_train, n_total = sum(p.numel() for p in trainable), sum(p.numel() for p in model.parameters())
    with open(RESULTS_DIR / f"{args.run_name}{suffix}_config.json", "w") as fh:
        json.dump({**vars(args), "trainable_params": n_train, "total_params": n_total, "image_size": size,
                   "git_commit": git_commit()}, fh, indent=2)
    print(f"method={args.method}  device={device}  train={len(train_ds)}  val={len(val_ds)}  "
          f"trainable={n_train:,}/{n_total:,} ({100 * n_train / n_total:.2f}%)")

    start, best_f1, bad, history = 0, -1.0, 0, []
    if args.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start, best_f1, bad, history = ck["epoch"], ck["best_f1"], ck["bad"], ck["history"]
        print(f"resumed after epoch {start} (best val macro-F1 {best_f1:.4f})")

    def end_of_epoch(epoch, train_loss, t0):
        nonlocal best_f1, bad
        m = metrics(*predict_logits(model, val_loader, device))
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": m[k] for k in ("loss", "acc", "top5", "macro_f1")},
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        state = {"model": trainable_state(model), "config": vars(args), "classes": classes, "epoch": epoch}
        if m["macro_f1"] > best_f1:
            best_f1, bad = m["macro_f1"], 0
            atomic_save({**state, "val": m}, ckpt_dir / "best.pt")
        else:
            bad += 1
        atomic_save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict() if scaler else None, "best_f1": best_f1, "bad": bad,
                     "history": history}, ckpt_dir / "last.pt")
        pd.DataFrame(history, columns=HISTORY_COLS).to_csv(RESULTS_DIR / f"{args.run_name}{suffix}_history.csv",
                                                           index=False)

    if not history:                                  # epoch 0 = zero-shot reference, before any update
        end_of_epoch(0, float("nan"), time.time())
    for epoch in range(start + 1, args.epochs + 1):
        if bad >= args.patience:
            print(f"early stopping: no val macro-F1 gain for {args.patience} epochs")
            break
        torch.manual_seed(args.seed + epoch)         # same shuffling with or without --resume
        t0, total, n = time.time(), 0.0, 0
        train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
                loss = criterion(model(x), y)
            (scaler.scale(loss / args.accum) if scaler else loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or i + 1 == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total, n = total + loss.item() * len(y), n + len(y)
        end_of_epoch(epoch, total / max(n, 1), t0)
    print(f"best val macro-F1 = {best_f1:.4f} (epoch 0 = zero-shot: {history[0]['val_macro_f1']:.4f})")


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    # subset (debug) runs get their own folder: a full run must never --resume from a debug checkpoint
    ckpt_dir = Path(args.ckpt_dir).expanduser() / (args.run_name + (f"_subset{args.subset}" if args.subset else ""))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not args.export_only:
        train(args, device, ckpt_dir)
    if args.subset:
        print("subset run: export skipped (prediction files must cover the full split)")
        return
    model = load_for_inference(ckpt_dir / "best.pt", device)
    for split in [s.strip() for s in args.export.split(",") if s.strip()]:
        if split == "test":
            print("  WARNING: exporting TEST logits - only do this once, at the end of the project")
        export_logits(model, args.root, split, args.run_name, args.final, device, args.batch_size, args.workers,
                      model.config["view_mode"])         # always the view the model was selected with


if __name__ == "__main__":
    main()
