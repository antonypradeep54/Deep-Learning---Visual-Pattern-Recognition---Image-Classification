from pathlib import Path

import pytest
from PIL import Image
from src.data_pipeline import inventory, stratified_split


def _make_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def test_inventory_and_reproducible_splits(tmp_path):
    for i in range(20):
        _make_image(tmp_path / "Cat" / f"cat_{i}.png", (i, 1, 2))
        _make_image(tmp_path / "Dog" / f"dog_{i}.png", (1, i, 3))
    valid, rejected = inventory(tmp_path)
    assert len(valid) == 40
    assert rejected == []
    first = stratified_split(valid)
    assert first == stratified_split(list(reversed(valid)))
    for label in ("Cat", "Dog"):
        assert sum(x["label"] == label and x["split"] == "train" for x in first) == 14
        assert sum(x["label"] == label and x["split"] == "val" for x in first) == 3
        assert sum(x["label"] == label and x["split"] == "test" for x in first) == 3


def test_invalid_and_duplicate_images(tmp_path):
    _make_image(tmp_path / "Cat" / "cat_1.png", (20, 30, 40))
    (tmp_path / "Cat" / "cat_2.png").write_bytes((tmp_path / "Cat" / "cat_1.png").read_bytes())
    (tmp_path / "Dog").mkdir()
    (tmp_path / "Dog" / "dog_1.jpg").write_text("not an image")
    valid, rejected = inventory(tmp_path)
    assert len(valid) == 1
    assert {x["reason"].split(":")[0] for x in rejected} == {"duplicate_exact", "invalid_image"}


def test_conflicting_labels_are_rejected(tmp_path):
    _make_image(tmp_path / "Cat" / "cat_1.png", (20, 30, 40))
    (tmp_path / "Dog").mkdir()
    (tmp_path / "Dog" / "dog_1.png").write_bytes((tmp_path / "Cat" / "cat_1.png").read_bytes())
    valid, rejected = inventory(tmp_path)
    assert valid == []
    assert len(rejected) == 2
    assert all(x["reason"] == "duplicate_conflicting_label" for x in rejected)


def test_small_class_rejected():
    with pytest.raises(ValueError, match="At least 7"):
        stratified_split([{"path": f"cat_{i}", "label": "Cat"} for i in range(3)] +
                         [{"path": f"dog_{i}", "label": "Dog"} for i in range(7)])
