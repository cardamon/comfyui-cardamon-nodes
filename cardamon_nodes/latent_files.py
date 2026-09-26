"""Save latents to, and load them straight back from, ComfyUI's output directory.

Unlike the built-in Save/Load Latent nodes, these handle nested latents (e.g. MiniMax H3 AV)
and keep simple extra values of the latent dict, and loading reads from the output directory,
so saved latents need not be moved to the input directory first.
"""

import json
import logging
import os
import re

import comfy.nested_tensor
import folder_paths
import safetensors
import safetensors.torch
import torch
from comfy.cli_args import args
from comfy_api.latest import io, ui

EXTENSION = ".latent"
FORMAT_KEY = "cardamon_latent"
FORMAT_VERSION = 1
# Save Latent names files "<prefix>_<counter>_.latent"
FILE_NUMBER = re.compile(r"_(\d+)_?\.latent$", re.IGNORECASE)

logger = logging.getLogger(__name__)


def empty_latent():
    """Placeholder for "no latent" in a latent list; nodes that take a guide skip it."""
    return {"samples": torch.zeros(0)}


def is_empty_latent(latent):
    samples = latent["samples"]
    return not getattr(samples, "is_nested", False) and samples.numel() == 0


def save_latent(latent, path, metadata=None):
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        tensors = {
            f"samples.{i}": t.contiguous().cpu() for i, t in enumerate(samples.tensors)
        }
        nested = len(samples.tensors)
    else:
        tensors = {"samples": samples.contiguous().cpu()}
        nested = 0
    # Keep the dict's simple values; tensors such as noise_mask belong to one sampling run.
    extras = {
        k: v
        for k, v in latent.items()
        if k != "samples" and isinstance(v, (bool, int, float, str))
    }
    metadata = dict(metadata or {})
    metadata[FORMAT_KEY] = json.dumps(
        {"version": FORMAT_VERSION, "nested": nested, "extras": extras}
    )
    safetensors.torch.save_file(tensors, path, metadata=metadata)


def load_latent(path):
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        info = (f.metadata() or {}).get(FORMAT_KEY)
        if info is None:
            # Built-in SaveLatent format
            if "latent_tensor" not in f.keys():
                raise ValueError(f"{path} is not a latent file")
            multiplier = 1.0 if "latent_format_version_0" in f.keys() else 1.0 / 0.18215
            return {"samples": f.get_tensor("latent_tensor").float() * multiplier}
        info = json.loads(info)
        if info["version"] > FORMAT_VERSION:
            raise ValueError(f"{path} was saved by a newer version of these nodes")
        if info["nested"]:
            samples = comfy.nested_tensor.NestedTensor(
                [f.get_tensor(f"samples.{i}").float() for i in range(info["nested"])]
            )
        else:
            samples = f.get_tensor("samples").float()
    return {"samples": samples, **info["extras"]}


def output_latent_files():
    files, _ = folder_paths.recursive_search(folder_paths.get_output_directory())
    return sorted(
        (f.replace(os.sep, "/") for f in files if f.lower().endswith(EXTENSION)),
        reverse=True,
    )


def output_file_path(name):
    """Absolute path of a file listed by output_latent_files(), refusing paths that leave the output directory."""
    output_dir = os.path.abspath(folder_paths.get_output_directory())
    path = os.path.abspath(os.path.join(output_dir, name))
    if os.path.commonpath((output_dir, path)) != output_dir:
        raise ValueError(f"{name} is outside the output directory")
    return path


def first_item(value):
    # List-input nodes get lists, but validation may pass the plain widget value.
    return value[0] if isinstance(value, list) else value


def numbered_latent_files(directory):
    """Paths of the numbered .latent files in `directory`, in order of their number."""
    if not os.path.isdir(directory):
        return []
    numbered = []
    for name in os.listdir(directory):
        match = FILE_NUMBER.search(name)
        if match and os.path.isfile(os.path.join(directory, name)):
            numbered.append((int(match.group(1)), name))
    return [os.path.join(directory, name) for _, name in sorted(numbered)]


def latest_latent_list(directory, count, start_with_empty):
    """The `count` latents to guide `count` shots, padded at the end with empty latents."""
    paths = numbered_latent_files(directory)[-count:] if count > 0 else []
    latents = [load_latent(path) for path in paths]
    if start_with_empty:
        latents = [empty_latent()] + latents[: count - 1]
    if len(latents) < count:
        logger.warning(
            f"{directory} has {len(paths)} numbered latents, "
            f"padding the list of {count} with {count - len(latents)} empty latents"
        )
        latents += [empty_latent() for _ in range(count - len(latents))]
    return latents


class CardamonSaveLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonSaveLatent",
            display_name="Save Latent (Output Dir)",
            category="cardamon/latent",
            description="Save a latent, including nested latents such as MiniMax H3 AV, to the output directory. "
            "Load it again with Load Latent (Output Dir).",
            inputs=[
                io.Latent.Input("latent"),
                io.String.Input("filename_prefix", default="latents/cardamon"),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, filename_prefix) -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
        )
        file = f"{filename}_{counter:05}_{EXTENSION}"

        metadata = {}
        if not args.disable_metadata and cls.hidden:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt)
            if cls.hidden.extra_pnginfo is not None:
                for k, v in cls.hidden.extra_pnginfo.items():
                    metadata[k] = json.dumps(v)

        save_latent(latent, os.path.join(full_output_folder, file), metadata)
        return io.NodeOutput(
            latent,
            ui={"latents": [ui.SavedResult(file, subfolder, io.FolderType.output)]},
        )


class CardamonLoadLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonLoadLatent",
            display_name="Load Latent (Output Dir)",
            category="cardamon/latent",
            description="Load a .latent file from the output directory, newest first. Also reads files from the built-in Save Latent node. "
            "Refresh the node definitions (press R) to list newly saved files.",
            inputs=[io.Combo.Input("latent", options=output_latent_files())],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        return io.NodeOutput(load_latent(output_file_path(latent)))

    @classmethod
    def fingerprint_inputs(cls, latent):
        return os.path.getmtime(output_file_path(latent))

    @classmethod
    def validate_inputs(cls, latent):
        try:
            path = output_file_path(latent)
        except ValueError as e:
            return str(e)
        if not os.path.isfile(path):
            return f"Latent file not found in the output directory: {latent}"
        return True


class CardamonLoadLatentList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonLoadLatentList",
            display_name="Load Latent List (Output Dir)",
            category="cardamon/latent",
            description="Load the latest latents from a directory in the output directory, one per prompt, "
            "as a list. The latest are the highest numbered files, as written by Save Latent (Output Dir). "
            "Missing latents are filled in with empty latents, which guide nodes skip.",
            inputs=[
                io.String.Input(
                    "prompts",
                    force_input=True,
                    tooltip="Only the number of prompts is used.",
                ),
                io.String.Input(
                    "subdir",
                    default="latents",
                    tooltip="Directory in the output directory to load from. It should hold one numbered series of latents.",
                ),
                io.Boolean.Input(
                    "start_with_empty",
                    default=False,
                    tooltip="Start the list with an empty latent and leave out the latest latent, "
                    "so each prompt gets the latent of the prompt before it and the first starts fresh.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="latents", is_output_list=True)],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, prompts, subdir, start_with_empty) -> io.NodeOutput:
        directory = output_file_path(subdir[0])
        return io.NodeOutput(
            latest_latent_list(directory, len(prompts), start_with_empty[0])
        )

    @classmethod
    def fingerprint_inputs(cls, subdir, **kwargs):
        # Rerun whenever latents are added to or replaced in the directory.
        paths = numbered_latent_files(output_file_path(first_item(subdir)))
        return [(path, os.path.getmtime(path)) for path in paths]

    @classmethod
    def validate_inputs(cls, subdir, **kwargs):
        try:
            output_file_path(first_item(subdir))
        except ValueError as e:
            return str(e)
        return True


NODES = [CardamonSaveLatent, CardamonLoadLatent, CardamonLoadLatentList]
