"""Save latents to, and load them straight back from, ComfyUI's output directory.

Unlike the built-in Save/Load Latent nodes, these handle nested latents (e.g. MiniMax H3 AV)
and keep simple extra values of the latent dict, and loading reads from the output directory,
so saved latents need not be moved to the input directory first.
"""

import json
import os

import comfy.nested_tensor
import folder_paths
import safetensors
import safetensors.torch
from comfy.cli_args import args
from comfy_api.latest import io, ui

EXTENSION = ".latent"
FORMAT_KEY = "cardamon_latent"
FORMAT_VERSION = 1


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


# MiniMax H3 video latents have 24 channels and are decoded in independent 17-frame chunks of 5
# tokens, so their last frames decode from the last two chunks (7 tokens, 22 frames) alone.
H3_VIDEO_CHANNELS = 24
H3_TAIL_TOKENS = 7


def last_frame_samples(latent):
    """The part of a latent needed to decode its last frame: the last batch item, and for MiniMax H3
    video only its last two chunks. Nested latents (e.g. H3 audio and video) use their video."""
    samples = latent["samples"]
    nested = getattr(samples, "is_nested", False)
    video = samples.tensors[0] if nested else samples
    video = video[-1:]
    tokens = video.shape[2] if video.ndim == 5 else 0
    if nested and video.ndim == 5 and video.shape[1] == H3_VIDEO_CHANNELS and tokens > H3_TAIL_TOKENS and (tokens - 2) % 5 == 0:
        video = video[:, :, tokens - H3_TAIL_TOKENS :]
    return video


def decode_last_frame(vae, latent):
    images = vae.decode(last_frame_samples(latent))
    if images.ndim == 5:  # [batch, frames, ...] -> [batch * frames, ...], as VAE Decode does
        images = images.reshape(-1, *images.shape[-3:])
    return images[-1:]


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
            "Refresh the node definitions (press R) to list newly saved files. "
            "With a VAE connected, selecting a latent shows its last frame.",
            inputs=[
                io.Combo.Input("latent", options=output_latent_files()),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Video or image VAE to preview the latent's last frame with. "
                    "Selecting a latent runs just this node and its VAE nodes to show it.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
            # An output node, so the preview can run this node on its own (a partial run).
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, vae=None) -> io.NodeOutput:
        loaded = load_latent(output_file_path(latent))
        if vae is None:
            return io.NodeOutput(loaded)
        return io.NodeOutput(loaded, ui=ui.PreviewImage(decode_last_frame(vae, loaded), cls=cls))

    @classmethod
    def fingerprint_inputs(cls, latent, **kwargs):
        return os.path.getmtime(output_file_path(latent))

    @classmethod
    def validate_inputs(cls, latent, **kwargs):
        try:
            path = output_file_path(latent)
        except ValueError as e:
            return str(e)
        if not os.path.isfile(path):
            return f"Latent file not found in the output directory: {latent}"
        return True


NODES = [CardamonSaveLatent, CardamonLoadLatent]
