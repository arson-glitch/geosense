"""Checks for transforms, models and the export contract (no images needed). Run with: pytest -q"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from geosense.models import MODEL_NAMES, build_model, freeze_backbone, unfreeze_last_blocks  # noqa: E402
from geosense.train import metrics  # noqa: E402
from geosense.transforms import RESIZE_MODES, build_transforms, open_image, output_size  # noqa: E402

FAKE = Image.new("RGB", (1536, 662), (120, 160, 90))


def _real_size_jpeg(path):
    """A 1536x662 JPEG with a horizontal gradient, so a wrong crop shows up as a wrong colour."""
    arr = np.linspace(0, 255, 1536)[None, :, None] * np.ones((662, 1, 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype("uint8")).save(path, quality=95)
    return path


def test_open_image_crops_same_region_as_team_crop_ui(tmp_path):
    """Half-resolution decoding must not change WHICH part of the image is kept."""
    from geosense.data import crop_ui
    p = _real_size_jpeg(tmp_path / "a.jpg")
    full = crop_ui(Image.open(p).convert("RGB"))                 # team crop, full resolution
    half = open_image(p)                                          # our fast loader
    assert abs(half.width - full.width / 2) <= 1 and abs(half.height - full.height / 2) <= 1
    ref = np.asarray(full.resize(half.size), dtype=float)
    assert np.abs(np.asarray(half, dtype=float) - ref).mean() < 3


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
    import shutil
    from geosense.data import DATA_DIR
    sub.drop(columns="label").to_csv(tmp_path / "splits.csv", index=False)
    for f in ("classes.json", "class_counts.csv"):
        shutil.copy(DATA_DIR / f, tmp_path / f)
    orig = tr.FastGeoSenseDataset.__init__
    monkeypatch.setattr(tr.FastGeoSenseDataset, "__init__",
                        lambda self, r, split, transform=None, cached=False:
                        orig(self, r, split, transform, cached, data_dir=tmp_path))
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
    _real_size_jpeg(tmp_path / "src" / "X" / "a.jpg")
    _one((tmp_path / "src", tmp_path / "cache", "X/a.jpg", cache_height()))
    small = open_image(tmp_path / "cache" / "X" / "a.jpg", crop_ui=False)
    assert small.height == cache_height()
    full = build_transforms(False)(open_image(tmp_path / "src" / "X" / "a.jpg"))
    cached = build_transforms(False)(small)
    assert full.shape == cached.shape
    assert (full - cached).abs().mean() < 0.05
