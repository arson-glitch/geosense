"""Checks for transforms, models and the export contract (no images needed). Run with: pytest -q"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from geosense.models import MODEL_NAMES, build_model, freeze_backbone, unfreeze_last_blocks  # noqa: E402
from geosense.train import metrics  # noqa: E402
from geosense.transforms import RESIZE_MODES, CropUI, build_transforms, output_size  # noqa: E402

FAKE = Image.new("RGB", (1536, 662), (120, 160, 90))


def test_crop_ui_matches_shared_preprocess():
    from geosense.preprocess import crop_overlay
    assert CropUI()(FAKE).size == crop_overlay(FAKE).size == (round(1536 * 0.82) - round(1536 * 0.05), 662)


def test_default_resize_is_team_rule():
    from geosense.transforms import DEFAULT_RESIZE
    assert DEFAULT_RESIZE == "crop"


@pytest.mark.parametrize("resize", RESIZE_MODES)
@pytest.mark.parametrize("train", [True, False])
def test_output_shape(resize, train):
    x = build_transforms(train, 224, resize)(FAKE)
    assert tuple(x.shape) == (3, *output_size(224, resize))


def test_no_horizontal_flip_anywhere():
    for train in (True, False):
        for resize in RESIZE_MODES:
            steps = build_transforms(train, 224, resize).transforms
            assert not any(isinstance(s, T.RandomHorizontalFlip) for s in steps)


def test_eval_transform_is_deterministic():
    tf = build_transforms(False)
    assert torch.equal(tf(FAKE), tf(FAKE))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_output_one_logit_per_class(name):
    m = build_model(name, num_classes=56, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 224, 224)).shape == (2, 56)
        assert m(torch.randn(1, 3, 224, 448)).shape == (1, 56)       # "wide" works too


def test_freeze_then_unfreeze_resnet():
    m = build_model("resnet50", 56, pretrained=False)
    freeze_backbone(m)
    trainable = {n.split(".")[0] for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"fc"}
    unfreeze_last_blocks(m, 1)
    trainable = {n.split(".")[0] for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"fc", "layer4"}


def test_metrics():
    logits = np.eye(4, dtype=np.float32)
    m = metrics(logits, np.arange(4))
    assert m["acc"] == m["top5"] == m["macro_f1"] == 1.0


def test_training_and_export_on_tiny_fake_data(mini, tmp_path, monkeypatch):
    """End to end on Thea's synthetic mini dataset: train 1 epoch, export val logits in the shared format."""
    import geosense.train as tr
    root, sub = mini
    man = tmp_path / "manifest.csv"
    sub.to_csv(man, index=False)
    import shutil
    from geosense.data import DATA_DIR
    shutil.copy(DATA_DIR / "classes.json", tmp_path / "classes.json")
    orig = tr.FastGeoSenseDataset.__init__
    monkeypatch.setattr(tr.FastGeoSenseDataset, "__init__",
                        lambda self, r, split, transform=None, manifest=None: orig(self, r, split, transform, man))
    for name in ("PRED_DIR", "RESULTS_DIR", "CKPT_DIR"):
        monkeypatch.setattr(tr, name, tmp_path / name)
    tr.main(["--root", str(root), "--model", "cnn_scratch", "--epochs", "1", "--size", "64",
             "--batch-size", "16", "--workers", "0", "--device", "cpu"])
    out = np.load(tmp_path / "PRED_DIR" / "cnn_scratch_val_logits.npy")
    assert out.dtype == np.float32 and out.shape == ((sub.split == "val").sum(), 56)


def test_cache_gives_same_geometry_as_full_pipeline(tmp_path):
    """Cached small image (UI removed, height 258) -> same tensor shape and nearly the same pixels."""
    from geosense.cache import _one, cache_height
    from geosense.transforms import open_image
    rng = np.random.default_rng(0)
    arr = (np.linspace(0, 255, 1536)[None, :, None] * np.ones((662, 1, 3)) + rng.normal(0, 5, (662, 1536, 3)))
    (tmp_path / "src" / "X").mkdir(parents=True)
    Image.fromarray(arr.clip(0, 255).astype("uint8")).save(tmp_path / "src" / "X" / "a.jpg", quality=95)
    _one((tmp_path / "src", tmp_path / "cache", "X/a.jpg", cache_height()))
    small = open_image(tmp_path / "cache" / "X" / "a.jpg")
    assert small.height == cache_height()
    full = build_transforms(False)(open_image(tmp_path / "src" / "X" / "a.jpg"))
    cached = build_transforms(False, crop_ui=False)(small)
    assert full.shape == cached.shape
    assert (full - cached).abs().mean() < 0.05
