import pytest
import torch

import comfy.nested_tensor
import comfy.utils
import folder_paths
from cardamon_nodes.latent_files import CardamonLoadLatent, CardamonSaveLatent, load_latent, save_latent


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    return tmp_path


def test_nested_latent_round_trip_keeps_simple_extras(tmp_path):
    video, audio = torch.randn(1, 24, 7, 4, 6), torch.randn(1, 32, 2, 37)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio)), "offset": 0.4,
              "noise_mask": torch.ones(1)}
    save_latent(latent, tmp_path / "a.latent")
    loaded = load_latent(tmp_path / "a.latent")
    assert torch.equal(loaded["samples"].tensors[0], video)
    assert torch.equal(loaded["samples"].tensors[1], audio)
    assert loaded["offset"] == 0.4
    assert "noise_mask" not in loaded


def test_plain_latent_round_trip(tmp_path):
    samples = torch.randn(2, 4, 8, 8)
    save_latent({"samples": samples}, tmp_path / "a.latent")
    assert torch.equal(load_latent(tmp_path / "a.latent")["samples"], samples)


def test_loads_builtin_save_latent_format(tmp_path):
    samples = torch.randn(1, 4, 8, 8)
    comfy.utils.save_torch_file({"latent_tensor": samples, "latent_format_version_0": torch.tensor([])},
                                str(tmp_path / "a.latent"))
    assert torch.equal(load_latent(tmp_path / "a.latent")["samples"], samples)


def test_save_node_writes_to_output_and_load_node_lists_it(output_dir):
    latent = {"samples": comfy.nested_tensor.NestedTensor((torch.randn(1, 24, 2, 4, 6), torch.randn(1, 32, 2, 8)))}
    result = CardamonSaveLatent.execute(latent, "latents/test")
    saved = result.ui["latents"][0]
    assert (saved["subfolder"], saved["filename"]) == ("latents", "test_00001_.latent")

    options = CardamonLoadLatent.define_schema().inputs[0].options
    assert options == ["latents/test_00001_.latent"]
    assert CardamonLoadLatent.validate_inputs(options[0]) is True
    loaded = CardamonLoadLatent.execute(options[0]).args[0]
    assert torch.equal(loaded["samples"].tensors[0], latent["samples"].tensors[0])


def test_load_node_refuses_paths_outside_output(output_dir):
    assert CardamonLoadLatent.validate_inputs("../secret.latent") != True
    with pytest.raises(ValueError):
        CardamonLoadLatent.execute("../secret.latent")
