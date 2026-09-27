"""LoRA stack: apply a list of LoRAs, picked in a file browser popup (web/lora_stack.js)."""

import json

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io


def parse_stack(loras):
    """The enabled LoRAs as (name, strength) pairs, from the JSON the node's UI stores.

    Disabled LoRAs are left out entirely, so they are neither loaded nor required to exist.
    """
    try:
        entries = json.loads(loras or "[]")
    except json.JSONDecodeError as e:
        raise ValueError(f"the LoRA stack is not valid JSON: {e}") from e
    return [
        (entry["name"], float(entry["strength"]))
        for entry in entries
        if entry.get("enabled", True)
    ]


class CardamonLoraStack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonLoraStack",
            display_name="LoRA Stack",
            category="cardamon/loaders",
            description="Apply several LoRAs in order. Add them with the Add LoRA button, which browses the loras directory. "
            "Disabled LoRAs are not loaded at all.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", optional=True, tooltip="Also apply the LoRAs to the text encoder, at the same strength."),
                # Edited by the node's UI; hidden there.
                io.String.Input("loras", default="[]", socketless=True),
            ],
            outputs=[io.Model.Output(display_name="MODEL"), io.Clip.Output(display_name="CLIP")],
        )

    @classmethod
    def execute(cls, model, loras, clip=None) -> io.NodeOutput:
        for name, strength in parse_stack(loras):
            if strength == 0:
                continue  # no effect, so don't spend time loading it
            path = folder_paths.get_full_path_or_raise("loras", name)
            lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength, lora_metadata=metadata)
        return io.NodeOutput(model, clip)

    @classmethod
    def validate_inputs(cls, loras):
        try:
            stack = parse_stack(loras)
        except (ValueError, KeyError, TypeError) as e:
            return str(e)
        available = set(folder_paths.get_filename_list("loras"))
        missing = [name for name, _ in stack if name not in available]
        if missing:
            return f"LoRA not found: {', '.join(missing)}"
        return True


NODES = [CardamonLoraStack]
