"""Tests for the baselines + StreetCLIP part (run offline, no GPU, no real images)."""
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel

from geosense import load_classes, load_manifest
from geosense.baselines import colour_features, majority_logits
from geosense.predictions import MODEL_NAMES, full_class_columns, load_logits, save_logits, softmax
from geosense.preprocess import ViewDataset, crop_overlay, to_square
from geosense.streetclip import (class_weight_options, image_embeddings, text_weights, train_probe,
                                 zero_shot_logits)

CLASSES, WEIGHTS = load_classes()
C = len(CLASSES)


def test_crop_keeps_5_to_82_percent_of_width():
    out = crop_overlay(Image.new("RGB", (1536, 662)))
    assert out.size == (round(1536 * 0.82) - round(1536 * 0.05), 662)


def test_to_square_modes():
    img = Image.new("RGB", (1183, 662), (10, 20, 30))
    assert to_square(img, 336).size == (336, 336)
    assert to_square(img, 336, "squash").size == (336, 336)
    with pytest.raises(ValueError):
        to_square(img, 336, "bad")


def test_save_logits_checks_shape_order_and_values(tmp_path):
    m = load_manifest()
    n_val = int((m.split == "val").sum())
    p = save_logits("majority", "val", np.zeros((n_val, C)), tmp_path)
    assert p.name == "majority_val_logits.npy" and load_logits("majority", "val", tmp_path).dtype == np.float32
    with pytest.raises(ValueError):
        save_logits("majority", "val", np.zeros((n_val - 1, C)), tmp_path)       # wrong N
    with pytest.raises(ValueError):
        save_logits("majority", "val", np.zeros((n_val, C - 1)), tmp_path)       # wrong C
    with pytest.raises(ValueError):
        save_logits("majority", "val", np.full((n_val, C), np.nan), tmp_path)    # NaN
    with pytest.raises(ValueError):
        save_logits("my_model", "val", np.zeros((n_val, C)), tmp_path)           # name not in contract


def test_majority_logits_give_training_prior():
    m = load_manifest()
    y = m.loc[m.split == "train", "label"].to_numpy()
    p = softmax(majority_logits(y, C, 5))
    assert p.shape == (5, C) and p[0].argmax() == CLASSES.index("United States")
    assert abs(p[0].sum() - 1) < 1e-5


def test_full_class_columns_restores_missing_classes():
    p = full_class_columns(np.array([[0.7, 0.3]]), [1, 3], 5)
    assert p.tolist() == [[0, 0.7, 0, 0.3, 0]]


def test_colour_features_shape(mini):
    root, sub = mini
    X = colour_features(root, sub.path.iloc[:6], workers=1)
    assert X.shape[0] == 6 and X.shape[1] > 50 and np.isfinite(X).all()


def tiny_clip():
    cfg = CLIPConfig(projection_dim=16,
                     text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                      vocab_size=100, max_position_embeddings=16),
                     vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                        image_size=28, patch_size=14))
    return CLIPModel(cfg).eval()


class FakeTokenizer:
    def __call__(self, prompts, **kw):
        ids = torch.tensor([[1 + (sum(map(ord, p)) % 90), 2, 3, 4] for p in prompts])

        class Out(dict):
            def to(self, device):
                return self
        return Out(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_streetclip_pipeline_on_real_manifest_subset(mini):
    root, sub = mini
    model = tiny_clip()
    W = text_weights(model, FakeTokenizer(), CLASSES)
    assert W.shape == (C, 16) and np.allclose(np.linalg.norm(W, axis=1), 1, atol=1e-4)

    Z, y = {}, {}
    for s in ("train", "val"):
        d = sub[sub.split == s].reset_index(drop=True)
        Z[s] = image_embeddings(model, ViewDataset(root, d.path, size=28), "cpu", batch_size=16, num_workers=0)
        y[s] = d.label.to_numpy()
        assert Z[s].shape == (len(d), 16) and np.allclose(np.linalg.norm(Z[s], axis=1), 1, atol=1e-4)

    zs = zero_shot_logits(Z["val"], W)
    assert zs.shape == (len(y["val"]), C)

    opts = class_weight_options(np.array(WEIGHTS))
    assert set(opts) == {"none", "sqrt", "inverse"} and abs(opts["sqrt"].mean() - 1) < 1e-4
    for init in (None, W):
        Wp, hist, best = train_probe(Z["train"], y["train"], Z["val"], y["val"], init=init,
                                     class_weights=opts["sqrt"], epochs=3, device="cpu")
        assert Wp.shape == W.shape and 0 <= best <= 1 and len(hist) == 3
    out = zero_shot_logits(Z["val"], Wp)
    assert out.shape == (len(y["val"]), C)


def test_model_names_cover_my_five_models():
    assert {"majority", "histogram_lr", "histogram_rf", "streetclip_zeroshot", "streetclip_probe"} <= set(MODEL_NAMES)
