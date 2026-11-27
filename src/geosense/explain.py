"""Explanation maps (Maya): Grad-CAM for the CNNs, attention rollout for StreetCLIP.

    # CNN / ResNet  (layer from Mohamed: model.gradcam_layer() or geosense.models.gradcam_layer(model))
    cam = grad_cam(model, layer, x, class_idx=None)             # (H, W) in [0, 1], x = (1, 3, H, W)

    # StreetCLIP (HF CLIPModel, vision tower with eager attention)
    rollout = clip_attention_rollout(clip_model, pixel_values)   # (grid, grid) in [0, 1]

    overlay(image_rgb_uint8, heat) -> uint8 image for the app / report

Both maps explain the model's top class unless class_idx is given. They show which image
regions the model used; they are NOT proof of causality. Say so in the report.
"""
from __future__ import annotations

import numpy as np


def _norm01(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    return a / (a.max() + 1e-12)


def grad_cam(model, layer, x, class_idx: int | None = None) -> np.ndarray:
    """Grad-CAM: weight the layer's feature maps by the mean gradient of the class score."""
    import torch
    import torch.nn.functional as F

    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            model.zero_grad()
            logits[0, idx].backward()
    finally:
        h1.remove()
        h2.remove()
        model.train(was_training)
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return _norm01(cam[0, 0].detach().cpu().numpy())


def rollout_from_attentions(attentions, discard: float = 0.0) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema 2020) from a list of (1, heads, T, T) attention tensors.

    Average heads, add the identity (residual connection), re-normalise, multiply layer by layer.
    Returns the CLS-token row over the patch tokens as a (grid, grid) map in [0, 1].
    """
    result = None
    for a in attentions:
        a = np.asarray(a.detach().cpu() if hasattr(a, "detach") else a)[0].mean(0)
        if discard > 0:
            a = np.where(a < np.quantile(a, discard), 0, a)
        a = a + np.eye(a.shape[0])
        a = a / a.sum(-1, keepdims=True)
        result = a if result is None else a @ result
    cls_to_patches = result[0, 1:]
    grid = int(round(np.sqrt(len(cls_to_patches))))
    return _norm01(cls_to_patches.reshape(grid, grid))


def clip_attention_rollout(clip_model, pixel_values, discard: float = 0.0) -> np.ndarray:
    """Rollout for StreetCLIP's vision tower. Needs attn_implementation='eager' to return attentions:
    CLIPModel.from_pretrained('geolocal/StreetCLIP', attn_implementation='eager').
    """
    import torch

    with torch.no_grad():
        out = clip_model.vision_model(pixel_values=pixel_values, output_attentions=True)
    return rollout_from_attentions(out.attentions, discard)


def overlay(image_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heat map (any size) over an RGB uint8 image with the 'jet' colour map."""
    import matplotlib
    from PIL import Image

    h, w = image_rgb.shape[:2]
    heat_img = Image.fromarray((heat * 255).astype("uint8")).resize((w, h), Image.BILINEAR)
    rgb = (matplotlib.colormaps["jet"](np.asarray(heat_img) / 255.0)[..., :3] * 255).astype("uint8")
    return ((1 - alpha) * image_rgb + alpha * rgb).astype("uint8")
