import pytest
import torch

import comfy.nested_tensor
import folder_paths
from cardamon_nodes.latent_files import (
    CardamonLoadLatentList,
    is_empty_latent,
    numbered_latent_files,
    save_latent,
)
from cardamon_nodes.minimax_h3 import (
    CardamonMiniMaxH3AddLatentGuide,
    CardamonMiniMaxH3ExtractLatent,
)


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    return tmp_path


def save_numbered(directory, *numbers):
    directory.mkdir(parents=True, exist_ok=True)
    for n in numbers:
        save_latent({"samples": torch.full((1,), float(n))}, directory / f"shot_{n:05}_.latent")


def load_list(count, subdir="shots", start_with_empty=False):
    prompts = [f"prompt {i}" for i in range(count)]
    return CardamonLoadLatentList.execute(prompts, [subdir], [start_with_empty]).args[0]


def values(latents):
    return [None if is_empty_latent(l) else int(l["samples"][0]) for l in latents]


def test_files_are_ordered_by_number(tmp_path):
    save_numbered(tmp_path, 10, 2, 1)
    (tmp_path / "notes.latent").touch()
    assert [p.rsplit("/", 1)[-1] for p in numbered_latent_files(str(tmp_path))] == [
        "shot_00001_.latent",
        "shot_00002_.latent",
        "shot_00010_.latent",
    ]


def test_loads_the_latest_latent_per_prompt(output_dir):
    save_numbered(output_dir / "shots", 1, 2, 3, 4, 5)
    assert values(load_list(3)) == [3, 4, 5]


def test_start_with_empty_shifts_by_one(output_dir):
    save_numbered(output_dir / "shots", 1, 2, 3, 4, 5)
    assert values(load_list(3, start_with_empty=True)) == [None, 3, 4]


def test_missing_latents_are_padded_at_the_end(output_dir):
    save_numbered(output_dir / "shots", 1, 2)
    assert values(load_list(3)) == [1, 2, None]
    assert values(load_list(4, start_with_empty=True)) == [None, 1, 2, None]


def test_missing_directory_gives_all_empty(output_dir):
    assert values(load_list(2, subdir="not_there", start_with_empty=True)) == [None, None]


def test_fingerprint_changes_when_latents_are_saved(output_dir):
    save_numbered(output_dir / "shots", 1)
    before = CardamonLoadLatentList.fingerprint_inputs(["shots"])
    save_numbered(output_dir / "shots", 2)
    assert CardamonLoadLatentList.fingerprint_inputs(["shots"]) != before


def test_subdir_outside_output_is_rejected(output_dir):
    assert CardamonLoadLatentList.validate_inputs("../elsewhere") is not True
    with pytest.raises(ValueError):
        load_list(1, subdir="../elsewhere")


def test_empty_latent_passes_through_extract_and_adds_no_guide():
    empty = {"samples": torch.zeros(0)}
    section, start, length = CardamonMiniMaxH3ExtractLatent.execute(empty, -22, 0).args
    assert is_empty_latent(section) and (start, length) == (0, 0)
    positive = [[torch.zeros(1, 1, 8), {}]]
    target = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 37, 4, 6), torch.zeros(1, 32, 2, 207)))}
    out = CardamonMiniMaxH3AddLatentGuide.execute(positive, target, section, 0, True, True)
    assert out.args[0] is positive
