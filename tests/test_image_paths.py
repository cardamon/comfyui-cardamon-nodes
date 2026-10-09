import json
import os

import numpy as np
import pytest
from PIL import Image

from cardamon_nodes.images import (
    MAX_IMAGE_PATHS,
    CardamonNodesLoadImagePaths,
    clean_path,
)


def write_png(path, color, size=(32, 16), mode="RGB"):
    Image.new(mode, size, color).save(path)
    return str(path)


def load(*paths):
    return CardamonNodesLoadImagePaths.execute(json.dumps(list(paths))).args


def test_loads_one_image_per_path_and_pads_the_other_outputs(tmp_path):
    red = write_png(tmp_path / "red.png", (255, 0, 0))
    blue = write_png(tmp_path / "sub dir.png", (0, 0, 255), size=(8, 8))
    outputs = load(red, blue)
    assert len(outputs) == MAX_IMAGE_PATHS
    assert outputs[0].shape == (1, 16, 32, 3) and outputs[0][0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert outputs[1].shape == (1, 8, 8, 3) and outputs[1][0, 0, 0].tolist() == [0.0, 0.0, 1.0]
    assert all(o is None for o in outputs[2:])


def test_rgba_images_load_as_rgb(tmp_path):
    path = write_png(tmp_path / "a.png", (0, 255, 0, 128), mode="RGBA")
    (image,) = load(path)[:1]
    assert image.shape[-1] == 3


def test_animated_images_load_as_a_batch(tmp_path):
    path = str(tmp_path / "anim.gif")
    frames = [Image.new("RGB", (16, 16), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100)
    (image,) = load(path)[:1]
    assert image.shape[0] == 3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("  /tmp/a.png  ", "/tmp/a.png"),
        ('"/tmp/my image.png"', "/tmp/my image.png"),
        ("'/tmp/a.png'", "/tmp/a.png"),
        ("file:///tmp/my%20image.png", "/tmp/my image.png"),
    ],
)
def test_clean_path(text, expected):
    assert clean_path(text) == expected


def test_home_directory_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_png(tmp_path / "home.png", (1, 2, 3))
    assert clean_path("~/home.png") == str(tmp_path / "home.png")
    assert load("~/home.png")[0].shape == (1, 16, 32, 3)


def test_problems_are_reported_before_running(tmp_path):
    good = write_png(tmp_path / "good.png", (0, 0, 0))
    paths = json.dumps([good, "  ", str(tmp_path / "missing.png")])
    message = CardamonNodesLoadImagePaths.validate_inputs(paths)
    assert "path 2 is empty" in message and "image 3 not found" in message
    assert CardamonNodesLoadImagePaths.validate_inputs(json.dumps([good])) is True
    assert CardamonNodesLoadImagePaths.validate_inputs("[]") is True
    assert "not valid JSON" in CardamonNodesLoadImagePaths.validate_inputs("[")
    with pytest.raises(ValueError, match="not found"):
        CardamonNodesLoadImagePaths.execute(paths)


def test_fingerprint_changes_when_a_file_changes(tmp_path):
    path = write_png(tmp_path / "a.png", (0, 0, 0))
    paths = json.dumps([path])
    before = CardamonNodesLoadImagePaths.fingerprint_inputs(paths)
    os.utime(path, (1, 1))
    assert CardamonNodesLoadImagePaths.fingerprint_inputs(paths) != before


def test_schema_declares_the_maximum_number_of_image_outputs():
    schema = CardamonNodesLoadImagePaths.define_schema()
    assert [o.id for o in schema.outputs] == [f"image_{i}" for i in range(MAX_IMAGE_PATHS)]
    assert all(o.get_io_type() == "IMAGE" for o in schema.outputs)
