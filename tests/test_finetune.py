"""Tests for the StreetCLIP fine-tuning module (offline, CPU, tiny random CLIP, synthetic images)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import CLIPConfig, CLIPModel

from geosense import crop_ui, load_classes
from geosense import finetune as ft
from geosense.data import DATA_DIR
from geosense.preprocess import ViewDataset
from geosense.streetclip import text_weights, zero_shot_logits

CLASSES, _ = load_classes()
C, SIZE = len(CLASSES), 32


class FakeTokenizer:                                  # same idea as in test_thea_part.py
    def __call__(self, prompts, **kw):
        ids = torch.tensor([[1 + (sum(map(ord, p)) % 90), 2, 3, 4] for p in prompts])

        class Out(dict):
            def to(self, device):
                return self
        return Out(input_ids=ids, attention_mask=torch.ones_like(ids))


def tiny_clip():
    torch.manual_seed(0)
    cfg = CLIPConfig(projection_dim=16,
                     text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                      vocab_size=100, max_position_embeddings=16, bos_token_id=0, eos_token_id=2),
                     vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                        image_size=SIZE, patch_size=16))
    return CLIPModel(cfg).eval()


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """(model folder, W.npy, W): a saved tiny CLIP and its text weights (no tokenizer needed)."""
    d = tmp_path_factory.mktemp("tiny_clip")
    clip = tiny_clip()
    clip.save_pretrained(d)
    W = text_weights(clip, FakeTokenizer(), CLASSES)
    np.save(d / "W.npy", W)
    return d, d / "W.npy", W


@pytest.fixture
def env(mini, tmp_path, monkeypatch):
    """Point the module at the mini manifest and at temporary output folders."""
    root, sub = mini
    data = tmp_path / "data"
    data.mkdir()
    sub.drop(columns="label").to_csv(data / "splits.csv", index=False)
    for f in ("classes.json", "class_counts.csv"):
        shutil.copy(DATA_DIR / f, data / f)
    monkeypatch.setattr(ft, "DATA_DIR", data)
    monkeypatch.setattr(ft, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ft, "PRED_DIR", tmp_path / "predictions")
    return root, sub, tmp_path


def batch(mini, n=6):
    root, sub = mini
    ds = ViewDataset(root, sub[sub.split == "val"].path.iloc[:n], size=SIZE)
    return torch.stack([ds[i][0] for i in range(len(ds))])


def test_model_at_init_is_the_zero_shot_model(tiny, mini, monkeypatch):
    d, _, W = tiny
    x = batch(mini)
    model = ft.build_model(str(d), CLASSES, W=W).eval()
    with torch.no_grad():
        Z = model.embed(x).numpy()
        logits = model(x).numpy()
    assert logits.shape == (len(x), C)
    assert np.allclose(logits, zero_shot_logits(Z, W, model.scale.item()), atol=1e-4)
    # default path: head initialised from the prompts with the text tower, which is then dropped
    monkeypatch.setattr(ft.streetclip, "load_streetclip", lambda mid: (CLIPModel.from_pretrained(mid), FakeTokenizer()))
    model2 = ft.build_model(str(d), CLASSES)
    assert np.allclose(model2.head.weight.detach().numpy(), W, atol=1e-5)
    assert not any(n.startswith("text") for n, _ in model2.named_parameters())


def test_set_trainable_per_method(tiny, mini):
    d, _, W = tiny
    model = ft.set_trainable(ft.build_model(str(d), CLASSES, W=W), "head")
    assert ft.trainable_names(model) == ["head.weight"]

    model = ft.set_trainable(ft.build_model(str(d), CLASSES, W=W), "last_blocks", unfreeze_blocks=1)
    names = ft.trainable_names(model)
    assert "head.weight" in names and "visual_projection.weight" in names
    assert any(n.startswith("vision_model.post_layernorm") for n in names)
    assert any(n.startswith("vision_model.encoder.layers.1.") for n in names)
    assert not any(n.startswith(("vision_model.encoder.layers.0.", "vision_model.embeddings")) for n in names)
    ft.train_mode(model)
    assert not model.vision_model.encoder.layers[0].training and model.vision_model.encoder.layers[1].training

    x = batch(mini)
    model = ft.build_model(str(d), CLASSES, W=W).eval()
    with torch.no_grad():
        before = model(x)
    ft.set_trainable(model, "lora", lora_r=2, lora_alpha=4)
    names = ft.trainable_names(model)
    assert "head.weight" in names and len(names) > 1
    assert all(n == "head.weight" or ("lora_" in n and n.startswith("vision_model.")) for n in names)
    assert any("q_proj" in n for n in names) and any("v_proj" in n for n in names)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), before, atol=1e-5)   # LoRA starts at zero: still zero-shot


def test_transforms(mini):
    root, sub = mini
    path = sub.path.iloc[0]
    img = crop_ui(Image.open(root / path).convert("RGB"))
    thea = ViewDataset(root, [path], size=SIZE)[0][0]
    assert torch.allclose(ft.build_transforms(False, SIZE)(img), thea, atol=1e-5)
    tr = ft.build_transforms(True, SIZE)
    assert not any(isinstance(t, T.RandomHorizontalFlip) for t in tr.transforms)
    assert any(isinstance(t, T.ColorJitter) and t.hue is None for t in tr.transforms)
    assert tr(img).shape == (3, SIZE, SIZE)


def run(env, tiny, *extra):
    root, _, tmp = env
    ft.main(["--root", str(root), "--model-id", str(tiny[0]), "--head-init", str(tiny[1]), "--device", "cpu",
             "--workers", "0", "--batch-size", "32", "--ckpt-dir", str(tmp / "ckpt"), "--run-name", "t",
             "--method", "last_blocks", "--unfreeze-blocks", "1", "--lr", "1e-3", "--head-lr", "1e-2", *extra])
    return tmp


def test_train_resume_export_and_reload(env, tiny, monkeypatch):
    root, sub, tmp = env
    calls = []
    real = ft.train_mode
    monkeypatch.setattr(ft, "train_mode", lambda m: (calls.append(1), real(m)))
    run(env, tiny, "--epochs", "1", "--patience", "5")
    hist = pd.read_csv(tmp / "results" / "t_history.csv")
    assert list(hist.columns) == ft.HISTORY_COLS and hist.epoch.tolist() == [0, 1] and len(calls) == 1
    assert json.load(open(tmp / "results" / "t_config.json"))["trainable_params"] > 0
    ck = torch.load(tmp / "ckpt" / "t" / "last.pt", weights_only=False)
    model = ft.set_trainable(ft.build_model(str(tiny[0]), CLASSES, W=tiny[2]), "last_blocks", 1)
    assert set(ck["model"]) == set(ft.trainable_names(model))      # only trainable tensors are saved
    assert (tmp / "ckpt" / "t" / "best.pt").exists() and (tmp / "predictions" / "runs" / "t_val_logits.npy").exists()

    calls.clear()
    run(env, tiny, "--epochs", "2", "--patience", "5", "--resume")  # continues, does not redo epoch 1
    hist2 = pd.read_csv(tmp / "results" / "t_history.csv")
    assert hist2.epoch.tolist() == [0, 1, 2] and len(calls) == 1
    pd.testing.assert_frame_equal(hist2.iloc[:2], hist)
    ck = torch.load(tmp / "ckpt" / "t" / "last.pt", weights_only=False)
    assert ck["epoch"] == 2 and ck["scheduler"]["last_epoch"] == 2 * (len(sub[sub.split == "train"]) // 32)
    assert not np.allclose(ck["model"]["head.weight"].numpy(), tiny[2])     # training did update the head
    calls.clear()
    run(env, tiny, "--epochs", "2", "--patience", "5", "--resume")  # nothing left to do
    assert len(calls) == 0 and len(pd.read_csv(tmp / "results" / "t_history.csv")) == 3

    # exports: rows in splits.csv order; --final writes the team file through save_logits
    run(env, tiny, "--export-only", "--final")
    team = np.load(tmp / "predictions" / "streetclip_finetuned_val_logits.npy")
    runs = np.load(tmp / "predictions" / "runs" / "t_val_logits.npy")
    val = sub[sub.split == "val"].reset_index(drop=True)
    assert team.shape == (len(val), C) and team.dtype == np.float32 and np.allclose(team, runs)
    model = ft.load_for_inference(tmp / "ckpt" / "t" / "best.pt", "cpu")
    assert not model.training
    loader = DataLoader(ViewDataset(root, val.path, size=SIZE), batch_size=64)
    again = torch.cat([model(x).detach() for x, _ in loader]).numpy()
    assert np.allclose(again, team, atol=1e-4)


def test_lora_and_subset_run(env, tiny):
    root, _, tmp = env
    run(env, tiny, "--method", "lora", "--lora-r", "2", "--epochs", "1", "--subset", "60")
    assert pd.read_csv(tmp / "results" / "t_subset60_history.csv").epoch.tolist() == [0, 1]
    assert not (tmp / "predictions").exists()                       # subset runs never export
    model = ft.load_for_inference(tmp / "ckpt" / "t_subset60" / "last.pt", "cpu")
    assert any("lora_" in n for n in ft.trainable_names(model))
