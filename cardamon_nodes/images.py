"""Image nodes."""

import json
import os
import urllib.parse

import comfy.model_management
import comfy.utils
import node_helpers
import numpy as np
import torch
from comfy_api.latest import InputImpl, io
from PIL import Image, ImageOps, ImageSequence

DIRECTIONS = ["right", "down", "left", "up"]
# Load Images (Paths) declares this many image outputs; its UI shows one per path.
MAX_IMAGE_PATHS = 50


def parse_color(color):
    """(r, g, b, a) in 0..1 from "#RRGGBB" or "#RRGGBBAA"."""
    hex_digits = color.strip().lstrip("#")
    if len(hex_digits) not in (6, 8):
        raise ValueError(f"invalid color {color!r}, expected #RRGGBB or #RRGGBBAA")
    try:
        values = [
            int(hex_digits[i : i + 2], 16) / 255.0 for i in range(0, len(hex_digits), 2)
        ]
    except ValueError:
        raise ValueError(
            f"invalid color {color!r}, expected #RRGGBB or #RRGGBBAA"
        ) from None
    return tuple(values) if len(values) == 4 else (*values, 1.0)


def image_index(name):
    # Autogrow names end in the input's number: "image_0", "image_1", ... "image_10"
    return int(name.rsplit("_", 1)[-1])


def stitch_images(images, direction, match_image_size, spacing, color, wrap_after):
    """Place images in lines along `direction`, starting a new line after every `wrap_after` images.

    "right" and "down" start lines at the left and top, "left" and "up" run from image 1 at the
    right or bottom and align lines there. Rows stack top to bottom, columns left to right. Images
    are centered across their line, and the rest of the canvas has the spacing color.
    """
    batch = max(image.shape[0] for image in images)
    images = [
        torch.cat([image, image[-1:].repeat(batch - image.shape[0], 1, 1, 1)])
        if image.shape[0] < batch
        else image
        for image in images
    ]
    if match_image_size:
        height, width = images[0].shape[1:3]
        images = [images[0]] + [
            image
            if image.shape[1:3] == (height, width)
            else comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "lanczos", "disabled"
            ).movedim(1, -1)
            for image in images[1:]
        ]

    # RGBA when anything is transparent, so it stays transparent.
    channels = (
        4 if color[3] < 1.0 or any(image.shape[-1] == 4 for image in images) else 3
    )
    images = [
        torch.cat([image, torch.ones_like(image[..., :1])], dim=-1)
        if image.shape[-1] < channels
        else image
        for image in images
    ]

    horizontal = direction in ("right", "left")
    reverse = direction in ("left", "up")
    per_line = wrap_after if wrap_after > 0 else len(images)
    lines = [images[i : i + per_line] for i in range(0, len(images), per_line)]

    def along(image):
        return image.shape[2] if horizontal else image.shape[1]

    def across(image):
        return image.shape[1] if horizontal else image.shape[2]

    lengths = [
        sum(along(image) for image in line) + spacing * (len(line) - 1)
        for line in lines
    ]
    thicknesses = [max(across(image) for image in line) for line in lines]
    total_along = max(lengths)
    total_across = sum(thicknesses) + spacing * (len(lines) - 1)

    first = images[0]
    shape = (
        (batch, total_across, total_along)
        if horizontal
        else (batch, total_along, total_across)
    )
    canvas = torch.empty(shape + (channels,), dtype=first.dtype, device=first.device)
    canvas[...] = torch.tensor(color[:channels], dtype=first.dtype, device=first.device)

    line_start = 0
    for line, length, thickness in zip(lines, lengths, thicknesses):
        position = total_along - length if reverse else 0
        for image in reversed(line) if reverse else line:
            offset = line_start + (thickness - across(image)) // 2
            h, w = image.shape[1:3]
            if horizontal:
                canvas[:, offset : offset + h, position : position + w] = image.to(
                    canvas
                )
            else:
                canvas[:, position : position + h, offset : offset + w] = image.to(
                    canvas
                )
            position += along(image) + spacing
        line_start += thickness + spacing
    return canvas


class CardamonNodesImageStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonNodesImageStitch",
            display_name="Stitch Images (Grid)",
            search_aliases=[
                "combine images",
                "join images",
                "image grid",
                "concatenate images",
            ],
            category="Cardamon Nodes/image",
            description="Stitch one or more images in a direction, optionally wrapping onto new rows or columns. "
            "A new image input appears as soon as the last one is connected.",
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplatePrefix(
                        io.Image.Input("image"), prefix="image_", min=1, max=50
                    ),
                ),
                io.Combo.Input(
                    "direction",
                    options=DIRECTIONS,
                    default="right",
                    tooltip="right: image 1 at the left; left: image 1 at the right; down: image 1 at the top; up: image 1 at the bottom.",
                ),
                io.Boolean.Input(
                    "match_image_size",
                    default=True,
                    tooltip="Resize every other image to the width and height of image 1.",
                ),
                io.Int.Input(
                    "wrap_after",
                    default=0,
                    min=0,
                    max=1000,
                    tooltip="Images per row (right/left) or per column (down/up) before starting a new one. 0: never wrap.",
                ),
                io.Int.Input(
                    "spacing_width",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip="Pixels between images, rows and columns.",
                ),
                io.Color.Input(
                    "spacing_color",
                    default="#ffffff",
                    tooltip="Color of the spacing and any empty area. A transparent color gives an image with alpha.",
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        images,
        direction,
        match_image_size,
        wrap_after,
        spacing_width,
        spacing_color,
    ) -> io.NodeOutput:
        ordered = [
            images[name]
            for name in sorted(images, key=image_index)
            if images[name] is not None
        ]
        if not ordered:
            raise ValueError("connect at least one image")
        return io.NodeOutput(
            stitch_images(
                ordered,
                direction,
                match_image_size,
                spacing_width,
                parse_color(spacing_color),
                wrap_after,
            )
        )


def clean_path(text):
    """A path as typed or pasted: surrounding whitespace and quotes removed, file:// URLs and ~ resolved."""
    path = text.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "\"'":
        path = path[1:-1].strip()
    if path.startswith("file://"):
        path = urllib.parse.unquote(urllib.parse.urlparse(path).path)
    return os.path.expanduser(path)


def parse_paths(paths):
    """The paths the node's UI stores as a JSON list."""
    try:
        entries = json.loads(paths or "[]")
    except json.JSONDecodeError as e:
        raise ValueError(f"the image paths are not valid JSON: {e}") from e
    if not isinstance(entries, list) or len(entries) > MAX_IMAGE_PATHS:
        raise ValueError(f"expected a list of at most {MAX_IMAGE_PATHS} image paths")
    return [clean_path(str(entry)) for entry in entries]


def path_problems(paths):
    problems = []
    for i, path in enumerate(paths):
        if not path:
            problems.append(f"path {i + 1} is empty")
        elif not os.path.isfile(path):
            problems.append(f"image {i + 1} not found: {path}")
    return problems


def load_image_file(path):
    """An image file as an IMAGE batch, loaded like the core Load Image node does (frames of
    multi-frame images become the batch), but from any path."""
    dtype = comfy.model_management.intermediate_dtype()
    device = comfy.model_management.intermediate_device()
    components = InputImpl.VideoFromFile(path).get_components()
    if components.images.shape[0] > 0:
        return components.images.to(device=device, dtype=dtype)

    # PyAV can't read animated WebP; Pillow can.
    img = node_helpers.pillow(Image.open, path)
    frames = []
    for frame in ImageSequence.Iterator(img):
        frame = node_helpers.pillow(ImageOps.exif_transpose, frame).convert("RGB")
        if frames and frame.size != (frames[0].shape[1], frames[0].shape[0]):
            continue
        frames.append(torch.from_numpy(np.array(frame).astype(np.float32) / 255.0))
    return torch.stack(frames).to(device=device, dtype=dtype)


class CardamonNodesLoadImagePaths(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CardamonNodesLoadImagePaths",
            display_name="Load Images (Paths)",
            search_aliases=["load image from path", "image path", "load images"],
            category="Cardamon Nodes/image",
            description="Load images from any paths on this machine, one output per path. Typing in the last path "
            "field adds another field and output. Quotes around pasted paths and file:// URLs are fine.",
            inputs=[
                # Edited by the node's UI; hidden there.
                io.String.Input("paths", default="[]", socketless=True),
            ],
            outputs=[
                io.Image.Output(id=f"image_{i}", display_name=f"image {i + 1}")
                for i in range(MAX_IMAGE_PATHS)
            ],
        )

    @classmethod
    def execute(cls, paths) -> io.NodeOutput:
        entries = parse_paths(paths)
        problems = path_problems(entries)
        if problems:
            raise ValueError("; ".join(problems))
        images = [load_image_file(path) for path in entries]
        return io.NodeOutput(*images, *[None] * (MAX_IMAGE_PATHS - len(images)))

    @classmethod
    def fingerprint_inputs(cls, paths):
        # Rerun when a file changes.
        return [
            (path, os.path.getmtime(path) if os.path.isfile(path) else None)
            for path in parse_paths(paths)
        ]

    @classmethod
    def validate_inputs(cls, paths):
        try:
            problems = path_problems(parse_paths(paths))
        except ValueError as e:
            return str(e)
        return "; ".join(problems) if problems else True


NODES = [CardamonNodesImageStitch, CardamonNodesLoadImagePaths]
