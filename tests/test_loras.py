import json

import pytest

import comfy.sd
import comfy.utils
import folder_paths
from cardamon_nodes.loras import CardamonLoraStack, parse_stack

AVAILABLE = ["style/film.safetensors", "detail.safetensors", "characters/prof/prof_v2.safetensors"]


@pytest.fixture
def applied(monkeypatch):
    calls = []
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda folder: AVAILABLE)
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda folder, name: f"/loras/{name}")
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda path, **kwargs: (f"weights:{path}", {"meta": path}))

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        calls.append((lora, strength_model, strength_clip, clip is not None))
        return f"{model}+{lora}", clip and f"{clip}+{lora}"

    monkeypatch.setattr(comfy.sd, "load_lora_for_models", load_lora_for_models)
    return calls


def stack(*entries):
    return json.dumps([{"name": name, "strength": strength} for name, strength in entries])


def test_applies_loras_in_order_and_skips_zero_strength(applied):
    loras = stack(("style/film.safetensors", 0.8), ("detail.safetensors", 0), ("characters/prof/prof_v2.safetensors", -0.5))
    model, clip = CardamonLoraStack.execute("model", loras).args
    assert applied == [
        ("weights:/loras/style/film.safetensors", 0.8, 0.8, False),
        ("weights:/loras/characters/prof/prof_v2.safetensors", -0.5, -0.5, False),
    ]
    assert model == "model+weights:/loras/style/film.safetensors+weights:/loras/characters/prof/prof_v2.safetensors"
    assert clip is None


def test_applies_to_clip_when_connected(applied):
    model, clip = CardamonLoraStack.execute("model", stack(("detail.safetensors", 1.0)), clip="clip").args
    assert applied == [("weights:/loras/detail.safetensors", 1.0, 1.0, True)]
    assert clip == "clip+weights:/loras/detail.safetensors"


def test_empty_stack_passes_model_through(applied):
    assert CardamonLoraStack.execute("model", "[]").args == ("model", None)
    assert CardamonLoraStack.execute("model", "").args == ("model", None)
    assert applied == []


def test_validation_reports_missing_loras(applied):
    assert CardamonLoraStack.validate_inputs(stack(("detail.safetensors", 1.0))) is True
    message = CardamonLoraStack.validate_inputs(stack(("gone.safetensors", 1.0), ("detail.safetensors", 1.0)))
    assert message == "LoRA not found: gone.safetensors"


def test_invalid_stack_is_reported():
    assert "not valid JSON" in CardamonLoraStack.validate_inputs("[{")
    with pytest.raises(ValueError):
        parse_stack("[{")


def test_disabled_loras_are_not_loaded_or_required(applied):
    loras = json.dumps([
        {"name": "style/film.safetensors", "strength": 0.8, "enabled": False},
        {"name": "gone.safetensors", "strength": 1.0, "enabled": False},
        {"name": "detail.safetensors", "strength": 0.5, "enabled": True},
    ])
    assert CardamonLoraStack.validate_inputs(loras) is True
    CardamonLoraStack.execute("model", loras)
    assert applied == [("weights:/loras/detail.safetensors", 0.5, 0.5, False)]


def test_entries_without_enabled_are_enabled(applied):
    # Stacks saved before the toggle existed
    assert parse_stack(stack(("detail.safetensors", 1.0))) == [("detail.safetensors", 1.0)]
