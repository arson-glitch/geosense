"""StreetCLIP: zero-shot and linear probe (Thea's part).

StreetCLIP (geolocal/StreetCLIP, CC-BY-NC-4.0) = CLIP ViT-L/14 @ 336 px trained on ~1.1M street-level
images. Image and text live in the same embedding space (Lecture 16):
  zero-shot : logits = scale * cos(image embedding, text embedding of "A Street View photo in {country}.")
  probe     : freeze the model, embed every image ONCE, train one small layer on the embeddings.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

STREETCLIP_ID = "geolocal/StreetCLIP"
TEMPLATE_SETS = {
    "street_view": ("A Street View photo in {}.",),
    "photo": ("A photo taken in {}.",),
    "plain": ("{}",),
    "ensemble": ("A Street View photo in {}.", "A photo taken in {}.", "A Google Street View image of a road in {}."),
}


def load_streetclip(model_id: str = STREETCLIP_ID):
    """Returns (CLIPModel, tokenizer). ~1.7 GB download the first time."""
    from transformers import AutoTokenizer, CLIPModel

    return CLIPModel.from_pretrained(model_id).eval(), AutoTokenizer.from_pretrained(model_id)


@torch.no_grad()
def text_weights(model, tokenizer, names, templates=TEMPLATE_SETS["street_view"], device="cpu") -> np.ndarray:
    """(C, D) unit-norm text embedding per country (averaged over prompt templates)."""
    model = model.to(device).eval()
    rows = []
    for name in names:
        tok = tokenizer([t.format(name) for t in templates], padding=True, truncation=True, return_tensors="pt").to(device)
        # tower + projection by hand: works in transformers 4.x and 5.x
        emb = model.text_projection(model.text_model(**tok).pooler_output)
        rows.append(F.normalize(F.normalize(emb, dim=-1).mean(0), dim=-1))
    return torch.stack(rows).cpu().numpy().astype("float32")


@torch.no_grad()
def image_embeddings(model, dataset, device="cpu", batch_size=64, num_workers=4, desc="embeddings") -> np.ndarray:
    """(N, D) unit-norm image embeddings, in dataset order. This is the slow step: do it once, then cache."""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device == "cuda")
    out, amp = [], device == "cuda"
    for x, _ in tqdm(loader, desc=desc):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            z = model.visual_projection(model.vision_model(pixel_values=x.to(device)).pooler_output)
        out.append(F.normalize(z.float(), dim=-1).cpu())
    return torch.cat(out).numpy().astype("float32")


def zero_shot_logits(Z: np.ndarray, W: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """logits_per_image: scale * cosine similarity (scale = CLIP's learned temperature, 100)."""
    return (scale * F.normalize(torch.from_numpy(Z), dim=-1) @ F.normalize(torch.from_numpy(W), dim=-1).T).numpy()


def class_weight_options(w_inverse: np.ndarray) -> dict[str, np.ndarray | None]:
    """Candidates for the weighted loss; the best one is chosen on validation."""
    sqrt = np.sqrt(w_inverse)
    return {"none": None, "sqrt": (sqrt / sqrt.mean()).astype("float32"), "inverse": np.asarray(w_inverse, "float32")}


def train_probe(Ztr, ytr, Zva, yva, init=None, class_weights=None, epochs=40, lr=1e-3, weight_decay=1e-4,
                batch_size=512, scale=100.0, seed=0, device="cpu", n_classes=None):
    """Linear softmax layer on frozen embeddings (cosine form: scale * cos(z, w_c)).

    init: text embeddings (start from the zero-shot classifier) or None (random start).
    Best epoch = highest validation macro-F1. Returns (W, history, best_val_macro_f1).
    """
    torch.manual_seed(seed)
    Ztr = F.normalize(torch.from_numpy(Ztr), dim=-1).to(device)
    Zva = F.normalize(torch.from_numpy(Zva), dim=-1).to(device)
    ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.long, device=device)
    n_classes = n_classes or int(max(np.max(ytr), np.max(yva)) + 1)
    W = torch.nn.Parameter(torch.from_numpy(init).clone().to(device) if init is not None
                           else torch.randn(n_classes, Ztr.shape[1], device=device) * 0.02)
    cw = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=weight_decay)
    best, best_W, hist = -1.0, W.detach().clone(), []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(Ztr), device=device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(scale * Ztr[idx] @ F.normalize(W, dim=-1).T, ytr_t[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (Zva @ F.normalize(W, dim=-1).T).argmax(1).cpu().numpy()
        f1 = float(f1_score(yva, pred, average="macro"))
        hist.append({"epoch": ep, "val_macro_f1": f1, "val_acc": float((pred == np.asarray(yva)).mean()),
                     "train_loss": float(loss.detach())})
        if f1 > best:
            best, best_W = f1, W.detach().clone()
    return best_W.cpu().numpy(), hist, best
