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


class RecordingVae:
    """Decodes to frames whose value is the latent frame index; records what it was given."""

    def __init__(self, frames_per_token=1):
        self.frames_per_token = frames_per_token
        self.decoded = []

    def decode(self, samples):
        self.decoded.append(samples)
        if samples.ndim == 4:  # image latents: [batch, c, h, w] -> [batch, H, W, 3]
            return samples[:, :1].movedim(1, -1).repeat(1, 8, 8, 3)
        frames = samples[0, 0, :, 0, 0].repeat_interleave(self.frames_per_token)
        return frames.view(1, -1, 1, 1, 1).expand(1, -1, 8, 8, 3).clone()


def h3_latent(tokens):
    video = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1, 1).expand(2, 24, tokens, 4, 6).clone()
    return {"samples": comfy.nested_tensor.NestedTensor((video, torch.zeros(2, 32, 2, 10)))}


def test_h3_preview_decodes_only_the_last_two_chunks_of_the_last_item():
    from cardamon_nodes.latent_files import decode_last_frame

    vae = RecordingVae()
    frame = decode_last_frame(vae, h3_latent(37))  # 124 frames
    (given,) = vae.decoded
    assert given.shape == (1, 24, 7, 4, 6)
    assert given[0, 0, :, 0, 0].tolist() == list(range(30, 37))
    assert frame.shape == (1, 8, 8, 3) and frame[0, 0, 0, 0] == 36


def test_short_h3_latent_is_decoded_whole():
    from cardamon_nodes.latent_files import last_frame_samples

    assert last_frame_samples(h3_latent(7)).shape[2] == 7
    assert last_frame_samples(h3_latent(2)).shape[2] == 2


def test_other_latents_decode_the_last_batch_item_whole():
    from cardamon_nodes.latent_files import decode_last_frame, last_frame_samples

    video = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1, 1).expand(3, 16, 10, 4, 4).clone()
    assert last_frame_samples({"samples": video}).shape == (1, 16, 10, 4, 4)
    images = torch.arange(3, dtype=torch.float32).view(3, 1, 1, 1).expand(3, 4, 8, 8).clone()
    vae = RecordingVae()
    frame = decode_last_frame(vae, {"samples": images})
    assert vae.decoded[0].shape == (1, 4, 8, 8) and frame[0, 0, 0, 0] == 2


def test_load_node_previews_only_with_a_vae(output_dir):
    save_latent(h3_latent(37), output_dir / "shot.latent")
    without = CardamonLoadLatent.execute("shot.latent")
    assert without.ui is None
    with_vae = CardamonLoadLatent.execute("shot.latent", vae=RecordingVae())
    assert with_vae.args[0]["samples"].tensors[0].shape == (2, 24, 37, 4, 6)
    images = with_vae.ui.as_dict()["images"]
    assert len(images) == 1 and images[0]["type"] == "temp"
