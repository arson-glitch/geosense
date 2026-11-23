"""Checks that the shared data files follow the team contract. Run with: pytest -q"""
from PIL import Image

from geosense import crop_ui, load_classes, load_continents, load_splits

df = load_splits()
classes = load_classes()
continents = load_continents()


def test_columns_exactly_as_agreed():
    assert list(df.columns) == ["path", "country", "continent", "split"]


def test_splits_present_and_sized():
    share = df.split.value_counts(normalize=True)
    assert set(share.index) == {"train", "val", "test"}
    assert abs(share["train"] - 0.70) < 0.01
    assert abs(share["val"] - 0.15) < 0.01
    assert abs(share["test"] - 0.15) < 0.01


def test_every_image_listed_once():
    assert df.path.is_unique


def test_classes_sorted_and_match_splits():
    assert classes == sorted(classes)
    assert set(classes) == set(df.country)


def test_every_country_in_every_split():
    assert (df.groupby("split").country.nunique() == len(classes)).all()


def test_continents_cover_all_classes_and_match_splits():
    assert set(continents) == set(classes)
    assert (df.country.map(continents) == df.continent).all()


def test_min_images_per_class():
    assert df.country.value_counts().min() >= 100


def test_crop_ui_standard_image_is_5_to_82_percent():
    out = crop_ui(Image.new("RGB", (1536, 662)))
    assert out.size == (round(0.82 * 1536) - round(0.05 * 1536), 662)
